"""
Main training loop for parallel model training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import math
import os
import time
import numpy as np
from tqdm import tqdm
import random
import itertools
import shutil
import logging

from metrics import evaluate
from losses import SequenceAccuracyLoss
from training import generate_evaluation_examples, setup_models_training

# Get the logger
logger = logging.getLogger(__name__)


def train_models_parallel(
    simple_model,
    latent_model,
    train_loader,
    val_loader,
    device,
    config,
    log_dir,
    dataset,
    max_steps=3000,
    accuracy_weight=0.0,
    tf_schedule="none",
    tf_start_step=0,
    args=None,
    models_params=None,
    start_step=0,
    simple_checkpoint=None,
    latent_checkpoint=None,
    continuation_steps=0,
):
    """Train both models in parallel for real-time comparison"""
    # Use max_steps from args if available
    if args is not None and hasattr(args, "max_steps"):
        max_steps = args.max_steps
        logger.info(f"Using max_steps from args: {max_steps}")

    # Calculate the true target step by adding continuation steps if provided
    target_step = max_steps + continuation_steps

    # Setup loggers
    os.makedirs(log_dir, exist_ok=True)
    simple_writer = SummaryWriter(log_dir=f"{log_dir}/simple")
    latent_writer = SummaryWriter(log_dir=f"{log_dir}/latent")

    vocab_size = dataset.vocab_size

    # Store parameter counts
    simple_params = (
        models_params["simple"]
        if models_params
        else sum(p.numel() for p in simple_model.parameters() if p.requires_grad)
    )
    latent_params = (
        models_params["latent"]
        if models_params
        else sum(p.numel() for p in latent_model.parameters() if p.requires_grad)
    )

    # Capture configuration for saving in checkpoints
    model_config = {}
    if args is not None:
        model_config = {
            "d_model": getattr(args, "d_model", None),
            "num_layers": getattr(args, "num_layers", None),
            "num_latent": getattr(args, "num_latent", None),
            "min_digits": getattr(args, "min_digits", None),
            "max_digits": getattr(args, "max_digits", None),
            "batch_size": getattr(args, "batch_size", None),
            "accuracy_weight": getattr(args, "accuracy_weight", None),
            "tf_schedule": getattr(args, "tf_schedule", None),
            "tf_start_step": getattr(args, "tf_start_step", None),
            "use_checkpointing": getattr(args, "use_checkpointing", None),
        }
        # Remove None values
        model_config = {k: v for k, v in model_config.items() if v is not None}

    # Optimize models with torch.compile if available (PyTorch 2.0+)
    try:
        if hasattr(torch, "compile") and device.type == "cuda":
            logger.info("Using torch.compile to optimize models")
            simple_model = torch.compile(simple_model)
            latent_model = torch.compile(latent_model)
    except Exception as e:
        logger.error(f"Could not compile models: {e}")

    # Enable cuDNN benchmarking for optimal performance
    torch.backends.cudnn.benchmark = True

    # Setup optimizers and schedulers
    (
        simple_optimizer,
        simple_scheduler,
        latent_optimizer,
        latent_scheduler,
    ) = setup_models_training(simple_model, latent_model, device, max_steps)

    # Gradient monitoring parameters - much more conservative for larger models
    MAX_GRAD_NORM = {
        "simple": 1.5,  # Reduced for SimpleTransformer
        "latent": 2.5,  # Reduced for LatentTransformer
    }

    # Gradient explosion thresholds - reduced for earlier detection
    GRAD_NORM_WARNING = {
        "simple": 1.2,  # Warning threshold for SimpleTransformer
        "latent": 2.0,  # Warning threshold for LatentTransformer
    }

    # Emergency LR reduction factors - more aggressive reduction
    LR_EMERGENCY_FACTOR = {
        "simple": 0.25,  # More aggressive reduction for SimpleTransformer
        "latent": 0.25,  # More aggressive reduction for LatentTransformer
    }

    # Early explosion detection thresholds - lowered for larger models
    LOSS_MAX_THRESHOLD = {
        "simple": 50,  # Lower maximum acceptable token loss for SimpleTransformer
        "latent": 75,  # Lower maximum acceptable token loss for LatentTransformer
    }

    # Dynamically adjust gradient norm based on learning rate
    def get_grad_clip_norm(model_type, lr, step):
        """Dynamically adjust gradient clipping based on learning rate and step"""
        base_norm = MAX_GRAD_NORM[model_type]

        # Reduce clip norm as learning rate increases (prevent explosion)
        lr_factor = 1.0
        if model_type == "simple" and lr > 2.8e-4:  # Lower threshold (was 3.0e-4)
            # More aggressive reduction for SimpleTransformer at high LRs
            lr_factor = (2.8e-4 / lr) ** 1.2  # Apply stronger scaling like latent
        elif model_type == "latent" and lr > 3.5e-4:  # Lower threshold for latent model
            # More aggressive reduction for LatentTransformer
            lr_factor = (3.5e-4 / lr) ** 1.2  # Apply stronger scaling

        # Add step-based adjustment (more conservative later in training)
        step_factor = 1.0
        if step > 2000:  # Earlier threshold (was 3000)
            step_factor = 0.8
        if step > 5000:
            step_factor = 0.7  # More conservative at later steps for both models

        return base_norm * lr_factor * step_factor

    # Setup models dict
    models = {
        "simple": {
            "name": "SimpleTransformer",
            "model": simple_model,
            "optimizer": simple_optimizer,
            "scheduler": simple_scheduler,
            "criterion": SequenceAccuracyLoss(
                padding_idx=0, ce_weight=0.9, seq_weight=0.1, smoothing=0.15
            ),
            "writer": simple_writer,
            "step": 0,
            "val_loss": float("inf"),
            "val_sequence_accuracy": 0.0,
            "recent_losses": [],
            "stability_window": 50,
            "no_improvement_count": 0,
            "latest_train_accuracy": 0.0,
            "val_accuracy_history": [],
            "best_accuracy": 0.0,
            "best_val_loss": float("inf"),
            "params": simple_params,
            "config": model_config,
        },
        "latent": {
            "name": "LatentTransformer",
            "model": latent_model,
            "optimizer": latent_optimizer,
            "scheduler": latent_scheduler,
            "criterion": SequenceAccuracyLoss(
                padding_idx=0, ce_weight=0.85, seq_weight=0.15, smoothing=0.15
            ),
            "writer": latent_writer,
            "step": 0,
            "val_loss": float("inf"),
            "val_sequence_accuracy": 0.0,
            "recent_losses": [],
            "stability_window": 50,
            "no_improvement_count": 0,
            "latest_train_accuracy": 0.0,
            "val_accuracy_history": [],
            "best_accuracy": 0.0,
            "best_val_loss": float("inf"),
            "params": latent_params,
            "config": model_config,
        },
    }

    # Load from checkpoints if provided
    if simple_checkpoint is not None:
        logger.info("Loading SimpleTransformer from checkpoint")
        models["simple"]["model"].load_state_dict(simple_checkpoint["model_state_dict"])
        if "optimizer_state_dict" in simple_checkpoint:
            models["simple"]["optimizer"].load_state_dict(
                simple_checkpoint["optimizer_state_dict"]
            )
            logger.info("Loaded SimpleTransformer optimizer state")
        if "scheduler_state_dict" in simple_checkpoint:
            models["simple"]["scheduler"].load_state_dict(
                simple_checkpoint["scheduler_state_dict"]
            )
            logger.info("Loaded SimpleTransformer scheduler state")
        # Set the step from checkpoint, but don't use it to limit training steps
        # We'll use start_step + global_step < max_steps instead
        models["simple"]["step"] = simple_checkpoint.get("step", 0)
        models["simple"]["val_loss"] = simple_checkpoint.get("val_loss", float("inf"))
        models["simple"]["val_sequence_accuracy"] = simple_checkpoint.get(
            "val_sequence_accuracy", 0.0
        )
        models["simple"]["best_val_loss"] = simple_checkpoint.get(
            "val_loss", float("inf")
        )

    if latent_checkpoint is not None:
        logger.info("Loading LatentTransformer from checkpoint")
        models["latent"]["model"].load_state_dict(latent_checkpoint["model_state_dict"])
        if "optimizer_state_dict" in latent_checkpoint:
            models["latent"]["optimizer"].load_state_dict(
                latent_checkpoint["optimizer_state_dict"]
            )
            logger.info("Loaded LatentTransformer optimizer state")
        if "scheduler_state_dict" in latent_checkpoint:
            models["latent"]["scheduler"].load_state_dict(
                latent_checkpoint["scheduler_state_dict"]
            )
            logger.info("Loaded LatentTransformer scheduler state")
        # Set the step from checkpoint, but don't use it to limit training steps
        # We'll use start_step + global_step < max_steps instead
        models["latent"]["step"] = latent_checkpoint.get("step", 0)
        models["latent"]["val_loss"] = latent_checkpoint.get("val_loss", float("inf"))
        models["latent"]["val_sequence_accuracy"] = latent_checkpoint.get(
            "val_sequence_accuracy", 0.0
        )
        models["latent"]["best_val_loss"] = latent_checkpoint.get(
            "val_loss", float("inf")
        )

    # Time tracking
    start_time = time.time()

    # Gradient scalers for mixed precision on CUDA
    if device.type == "cuda":
        for model_info in models.values():
            model_info["scaler"] = torch.amp.GradScaler()

    # Update tqdm function to also log to the logger
    def tqdm_with_logging(**kwargs):
        """Creates a tqdm progress bar that also logs to the logger"""
        desc = kwargs.get("desc", "")
        logger.info(f"Starting: {desc}")

        # Handle the case where we don't have an iterable (e.g., using total=max_steps)
        if "iterable" in kwargs:
            return tqdm(kwargs.pop("iterable"), **kwargs)
        else:
            return tqdm(**kwargs)

    # Generate evaluation examples with lower range for easier evaluation
    try:
        # Define min_digits based on args parameter
        min_digits = 1
        if args is not None:
            min_digits = args.min_digits

        eval_max_val = 999 if min_digits >= 3 else (99 if min_digits == 2 else 9)
        logger.info(f"Precomputing val set with range 10-{min(5, eval_max_val)}")
        eval_examples = generate_evaluation_examples(dataset, device, min_digits)
        logger.info(f"Created {len(eval_examples)} evaluation examples")

    except Exception as e:
        logger.error(f"Error generating evaluation examples: {e}")
        logger.info("Adding fallback examples")
        # Add some basic fallback examples
        eval_examples = []
        for a, b in [(2, 3), (4, 5), (3, 7)]:
            input_str = f"{a}*{b}"
            input_tokens = dataset.encode(input_str)
            input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(device)
            result = a * b
            result_str = str(result)
            eval_examples.append((input_tensor, result_str, a, b))

    # Main training loop
    global_step = 0  # Always start from 0 for the current run
    val_freq = 50  # Validate every 50 steps

    try:
        # Set models to training mode
        for model_info in models.values():
            model_info["model"].train()

        # Create progress bars
        pbar = tqdm_with_logging(total=max_steps - start_step, desc=f"Training (steps)")

        # Training loop
        train_iter = itertools.cycle(train_loader)
        while global_step + start_step < max_steps:
            # Get a batch
            inp, tgt, inp_lens, tgt_lens = next(train_iter)
            inp, tgt = inp.to(device), tgt.to(device)

            # Calculate teacher forcing probability based on schedule
            current_step = global_step + start_step
            if tf_schedule == "none" or current_step < tf_start_step:
                tf_prob = 1.0  # Always use teacher forcing
            elif tf_schedule == "linear":
                progress = (current_step - tf_start_step) / (max_steps - tf_start_step)
                tf_prob = 1.0 - progress  # Linear decay from 1.0 to 0.0
            elif tf_schedule == "exp":
                progress = (current_step - tf_start_step) / (max_steps - tf_start_step)
                tf_prob = math.exp(-10 * progress)  # Exponential decay
            else:
                tf_prob = 1.0

            # Train each model on the same batch
            for model_type, model_info in models.items():
                # Check if model already reached max steps
                current_model_step = model_info["step"]
                if current_model_step >= target_step:
                    continue

                model = model_info["model"]
                optimizer = model_info["optimizer"]
                criterion = model_info["criterion"]
                writer = model_info["writer"]

                # Clear gradients
                optimizer.zero_grad()

                # Decide whether to use teacher forcing for this batch
                use_teacher_forcing = random.random() < tf_prob

                if use_teacher_forcing:
                    # Standard teacher forcing - use target as decoder input
                    decoder_input = tgt[:, :-1]
                    decoder_target = tgt[:, 1:]

                    # Skip this batch if decoder input is empty
                    if decoder_input.size(1) == 0:
                        logger.warning(
                            f"Empty decoder input in batch with teacher forcing, skipping for {model_info['name']}"
                        )
                        continue
                else:
                    # Generate decoder input on-the-fly
                    batch_size = inp.size(0)
                    decoder_target = tgt[:, 1:]
                    max_len = decoder_target.size(1)

                    # Skip this batch if target is empty
                    if max_len == 0:
                        logger.warning(
                            f"Empty target sequence, skipping batch for {model_info['name']}"
                        )
                        continue

                    # Start with the start token
                    decoder_input = torch.ones(
                        batch_size, 1, dtype=torch.long, device=device
                    )

                    # Only generate additional tokens if target length > 1
                    if max_len > 1:
                        try:
                            # Generate partial sequence for decoder input
                            with torch.no_grad():
                                for i in range(max_len - 1):
                                    # Forward pass to get next token prediction
                                    temp_output = model(inp, decoder_input)
                                    next_token_logits = temp_output[:, -1, :]
                                    next_token = torch.argmax(
                                        next_token_logits, dim=-1, keepdim=True
                                    )
                                    # Append to decoder input
                                    decoder_input = torch.cat(
                                        [decoder_input, next_token], dim=1
                                    )

                            # Use all but the last token as decoder input
                            if decoder_input.size(1) > 1:
                                decoder_input = decoder_input[:, :-1]
                        except Exception as e:
                            logger.error(
                                f"Error during sequence generation for {model_info['name']}: {e}"
                            )
                            logger.info(
                                f"Falling back to teacher forcing for this batch"
                            )
                            decoder_input = tgt[:, :-1]
                            decoder_target = tgt[:, 1:]

                # Process batch with appropriate precision and error handling
                try:
                    # Process batch with cuda if available
                    if device.type == "cuda":
                        # Use mixed precision
                        with torch.amp.autocast(device_type="cuda"):
                            output = model(inp, decoder_input)

                            # Check for output/target shape mismatch
                            if output.size(0) != decoder_target.size(0) or output.size(
                                1
                            ) != decoder_target.size(1):
                                # Resize to compatible shapes
                                min_batch = min(output.size(0), decoder_target.size(0))
                                min_seq_len = min(
                                    output.size(1), decoder_target.size(1)
                                )
                                output = output[:min_batch, :min_seq_len, :]
                                decoder_target = decoder_target[
                                    :min_batch, :min_seq_len
                                ]

                            # Calculate loss
                            loss, batch_seq_accuracy = criterion(output, decoder_target)

                            # Handle loss explosions
                            if (
                                torch.isnan(loss).any()
                                or loss > LOSS_MAX_THRESHOLD[model_type]
                            ):
                                # Save emergency checkpoint
                                checkpoint_path = f"checkpoints/{model_info['name'].lower()}/{model_info['name'].lower()}_emergency.pt"
                                os.makedirs(
                                    os.path.dirname(checkpoint_path), exist_ok=True
                                )
                                torch.save(
                                    {
                                        "model_state_dict": model.state_dict(),
                                        "optimizer_state_dict": optimizer.state_dict(),
                                        "scheduler_state_dict": model_info[
                                            "scheduler"
                                        ].state_dict(),
                                        "step": model_info["step"],
                                        "val_loss": model_info["val_loss"],
                                        "config": model_info.get("config", {}),
                                    },
                                    checkpoint_path,
                                )

                                logger.warning(
                                    f"⚠️ Detected loss explosion for {model_info['name']} at step {model_info['step']}"
                                )
                                logger.warning(
                                    f"Maximum loss: {loss.item():.2f}, threshold: {LOSS_MAX_THRESHOLD[model_type]}"
                                )
                                logger.warning(
                                    f"Saved emergency checkpoint to {checkpoint_path}"
                                )

                                # Reduce learning rate
                                current_lr = optimizer.param_groups[0]["lr"]
                                for param_group in optimizer.param_groups:
                                    param_group["lr"] *= LR_EMERGENCY_FACTOR[model_type]

                                logger.warning(
                                    f"🚨 Emergency LR reduction from {current_lr:.6f} to {optimizer.param_groups[0]['lr']:.6f}"
                                )

                                # Mark explosion step for cooldown
                                model_info["last_explosion_step"] = model_info["step"]
                                continue

                        # Use scaler for backwards pass and step
                        scaler = model_info["scaler"]
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)

                        # Apply gradient clipping
                        clip_norm = get_grad_clip_norm(
                            model_type,
                            optimizer.param_groups[0]["lr"],
                            model_info["step"],
                        )
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

                        # Perform optimizer step and scaler update
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Standard CPU/MPS training
                        output = model(inp, decoder_input)

                        # Calculate loss
                        loss, batch_seq_accuracy = criterion(output, decoder_target)

                        # Handle loss explosions
                        if (
                            torch.isnan(loss).any()
                            or loss > LOSS_MAX_THRESHOLD[model_type]
                        ):
                            # Similar emergency handling as above
                            logger.warning(
                                f"⚠️ Detected loss explosion for {model_info['name']} at step {model_info['step']}"
                            )
                            continue

                        # Standard backwards pass
                        loss.backward()

                        # Gradient clipping
                        clip_norm = get_grad_clip_norm(
                            model_type,
                            optimizer.param_groups[0]["lr"],
                            model_info["step"],
                        )
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

                        # Optimizer step
                        optimizer.step()

                    # Log training metrics
                    with torch.no_grad():
                        # Calculate token and sequence accuracy
                        _, predicted = torch.max(output, dim=-1)
                        mask = (decoder_target != 0).float()
                        token_correct = (predicted == decoder_target).float() * mask
                        token_accuracy = token_correct.sum() / mask.sum().clamp(
                            min=1e-8
                        )

                        # Log metrics
                        writer.add_scalar("train/loss", loss.item(), model_info["step"])
                        writer.add_scalar(
                            "train/token_accuracy",
                            token_accuracy.item(),
                            model_info["step"],
                        )
                        writer.add_scalar(
                            "train/batch_size",
                            decoder_target.size(0),
                            model_info["step"],
                        )
                        writer.add_scalar(
                            "train/learning_rate",
                            optimizer.param_groups[0]["lr"],
                            model_info["step"],
                        )

                        # Track losses for stability monitoring
                        model_info["recent_losses"].append(loss.item())
                        if (
                            len(model_info["recent_losses"])
                            > model_info["stability_window"]
                        ):
                            model_info["recent_losses"].pop(0)

                    # Step the learning rate scheduler
                    model_info["scheduler"].step()

                    # Increment model step counter
                    model_info["step"] += 1

                    # Validation and checkpoint saving
                    save_checkpoint = False
                    perform_validation = False

                    # Check if custom checkpoint frequency is set
                    if (
                        args is not None
                        and hasattr(args, "save_every")
                        and args.save_every is not None
                    ):
                        # Use custom checkpoint frequency
                        if model_info["step"] % args.save_every == 0:
                            save_checkpoint = True
                            # Also perform validation when saving checkpoint
                            perform_validation = True
                    else:
                        # Use default validation frequency
                        if (
                            model_info["step"] > 0
                            and model_info["step"] % val_freq == 0
                        ):
                            perform_validation = True
                            save_checkpoint = True

                    if perform_validation:
                        # Determine if we should rotate validation examples
                        rotate_examples = (
                            model_info["step"] > 3000
                            and model_info["step"] % (val_freq * 4) == 0
                        )

                        val_loss, val_sequence_accuracy, val_digit_accuracy = evaluate(
                            model=model,
                            data_loader=val_loader,
                            criterion=criterion,
                            dataset=dataset,
                            device=device,
                            vocab_size=vocab_size,
                            desc=f"Val {model_info['name']}",
                            examples=eval_examples,
                            rotate_examples=rotate_examples,
                        )

                        # Store validation metrics
                        model_info["val_loss"] = val_loss
                        model_info["val_sequence_accuracy"] = val_sequence_accuracy
                        model_info["val_digit_accuracy"] = val_digit_accuracy

                        # Log validation metrics
                        writer.add_scalar("val/loss", val_loss, model_info["step"])
                        writer.add_scalar(
                            "val/sequence_accuracy",
                            val_sequence_accuracy,
                            model_info["step"],
                        )
                        writer.add_scalar(
                            "val/digit_accuracy", val_digit_accuracy, model_info["step"]
                        )

                        # Print validation results
                        logger.info(
                            f"\n{model_info['name']} - Step {model_info['step']}/{max_steps} - Val Loss: {val_loss:.6f} - Seq Acc: {val_sequence_accuracy:.2%}"
                        )

                        # Check for best model and save if improved
                        if val_loss < model_info["best_val_loss"]:
                            model_info["best_val_loss"] = val_loss
                            # Save best model checkpoint
                            model_checkpoint_dir = (
                                f"checkpoints/{model_info['name'].lower()}"
                            )
                            os.makedirs(model_checkpoint_dir, exist_ok=True)
                            torch.save(
                                {
                                    "step": model_info["step"],
                                    "model_state_dict": model.state_dict(),
                                    "optimizer_state_dict": optimizer.state_dict(),
                                    "scheduler_state_dict": model_info[
                                        "scheduler"
                                    ].state_dict(),
                                    "val_loss": val_loss,
                                    "val_sequence_accuracy": val_sequence_accuracy,
                                    "config": model_info.get("config", {}),
                                },
                                f"{model_checkpoint_dir}/{model_info['name'].lower()}_best.pt",
                            )
                            logger.info(
                                f"Saved new best model for {model_info['name']} (val_loss: {val_loss:.6f})"
                            )

                        # Return to training mode
                        model.train()

                    # Save latest checkpoint if needed
                    if save_checkpoint:
                        model_checkpoint_dir = (
                            f"checkpoints/{model_info['name'].lower()}"
                        )
                        os.makedirs(model_checkpoint_dir, exist_ok=True)
                        torch.save(
                            {
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": model_info[
                                    "scheduler"
                                ].state_dict(),
                                "step": model_info["step"],
                                "val_loss": model_info.get("val_loss", float("inf")),
                                "val_sequence_accuracy": model_info.get(
                                    "val_sequence_accuracy", 0.0
                                ),
                                "config": model_info.get("config", {}),
                            },
                            f"{model_checkpoint_dir}/{model_info['name'].lower()}_latest.pt",
                        )

                        if not perform_validation:
                            # Only log this if we haven't already logged validation results
                            logger.info(
                                f"Saved checkpoint for {model_info['name']} at step {model_info['step']}"
                            )
                except Exception as e:
                    logger.error(
                        f"Error in training step for {model_info['name']}: {e}"
                    )
                    continue

            # Update global step counter and progress bar
            global_step += 1
            pbar.update(1)

            # Update progress bar postfix with current loss values
            simple_loss = "N/A"
            latent_loss = "N/A"

            if models["simple"]["recent_losses"]:
                simple_loss = f"{models['simple']['recent_losses'][-1]:.4f}"
            if models["latent"]["recent_losses"]:
                latent_loss = f"{models['latent']['recent_losses'][-1]:.4f}"

            pbar.set_postfix({"simple_loss": simple_loss, "latent_loss": latent_loss})

            # Final check: If both models have completed their training steps, break
            current_max_step = start_step + global_step
            if current_max_step >= target_step:
                logger.info(
                    f"Target steps reached: {current_max_step}/{target_step}. Breaking training loop."
                )
                break

    except KeyboardInterrupt:
        logger.info("Training interrupted")

    # Final validation for both models
    for model_type, model_info in models.items():
        final_val_loss, final_sequence_accuracy, final_digit_accuracy = evaluate(
            model=model_info["model"],
            data_loader=val_loader,
            criterion=model_info["criterion"],
            dataset=dataset,
            device=device,
            vocab_size=vocab_size,
            desc=f"Final {model_info['name']} validation",
            examples=eval_examples,
        )

        model_info["final_loss"] = final_val_loss
        model_info["final_sequence_accuracy"] = final_sequence_accuracy
        model_info["final_digit_accuracy"] = final_digit_accuracy

    training_time = time.time() - start_time

    # Print summaries
    for model_type, model_info in models.items():
        logger.info(f"\n{model_info['name']} Training Summary:")
        logger.info(f"Steps completed: {model_info['step']}/{max_steps}")
        logger.info(f"Best validation loss: {model_info['best_val_loss']:.6f}")
        logger.info(f"Final validation loss: {model_info['final_loss']:.6f}")
        logger.info(
            f"Final sequence accuracy: {model_info['final_sequence_accuracy']:.2%}"
        )
        logger.info(f"Final digit accuracy: {model_info['final_digit_accuracy']:.2%}")

    # Close writers
    for model_info in models.values():
        model_info["writer"].close()

    return {
        "simple": {
            "loss": models["simple"]["final_loss"],
            "sequence_accuracy": models["simple"]["final_sequence_accuracy"],
            "digit_accuracy": models["simple"]["final_digit_accuracy"],
            "params": models["simple"]["params"],
            "steps": models["simple"]["step"],
        },
        "latent": {
            "loss": models["latent"]["final_loss"],
            "sequence_accuracy": models["latent"]["final_sequence_accuracy"],
            "digit_accuracy": models["latent"]["final_digit_accuracy"],
            "params": models["latent"]["params"],
            "steps": models["latent"]["step"],
        },
        "training_time": training_time,
    }
