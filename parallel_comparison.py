"""
Parallel comparison of SimpleTransformer and LatentTransformer training them simultaneously.
Includes device selection argument and real-time TensorBoard comparison.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import math
import os
import time
import numpy as np
from tqdm import tqdm
import random
import argparse
import itertools
import shutil  # For removing directory trees

from Dataset import MultiplicationDataset
from Collate import collate_fn
from config import TrainingConfig

# Import the stable model implementations
from stable_comparison_with_accuracy import StableSimpleTransformer, StableLatentTransformer
from stable_comparison_with_accuracy import generate_multiplication_examples, inference, calculate_accuracy

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def improved_inference(model, inp, dataset, device, max_len=10, print_debug=False):
    """More robust inference function specially for multiplication task"""
    model.eval()
    
    with torch.no_grad():
        # Start with start token (usually token 1)
        decoder_input = torch.tensor([[1]], dtype=torch.long).to(device)
        
        # Storage for output tokens and their probabilities
        output_tokens = []
        token_probs = []  # Store probabilities for debugging
        
        # Generate tokens one by one
        for i in range(max_len):
            # Forward pass
            output = model(inp, decoder_input)
            
            # Get probabilities for the next token
            logits = output[0, -1, :]
            probs = F.softmax(logits, dim=-1)
            
            # Extract inputs for better length control
            try:
                input_str = dataset.decode(inp[0].cpu().numpy())
                # Check if it's a simple multiplication problem
                if '*' in input_str:
                    parts = input_str.split('*')
                    if len(parts) == 2:
                        a, b = int(parts[0]), int(parts[1])
                        expected_length = len(str(a * b))
                        
                        # If we've generated enough tokens for the result, consider stopping
                        if len(output_tokens) >= expected_length:
                            # Strong bias towards stopping
                            break
            except:
                # If we can't determine the expected length, use heuristics
                pass
            
            # Get top 3 candidates
            topk_probs, topk_indices = torch.topk(probs, 3)
            
            # Select next token with some heuristics
            found_valid = False
            for j in range(len(topk_indices)):
                # Skip padding and tokens that would create 3+ consecutive repeats
                token_idx = topk_indices[j].item()
                token_prob = topk_probs[j].item()
                
                # Skip padding token
                if token_idx == 0:
                    continue
                
                # Skip non-digit tokens (assuming digits are tokens 2-11 representing 0-9)
                if not (2 <= token_idx <= 11):
                    continue
                    
                # Handle repeats - don't allow more than 2 consecutive repeats
                if len(output_tokens) >= 2:
                    if token_idx == output_tokens[-1] == output_tokens[-2]:
                        continue
                
                # If we reach here, we have a valid token
                found_valid = True
                next_token = token_idx
                next_prob = token_prob
                break
            
            # If no valid token found, just take the highest probability digit
            if not found_valid:
                for token_idx in range(2, 12):  # Tokens 2-11 are digits 0-9
                    if probs[token_idx] > 0:
                        next_token = token_idx
                        next_prob = probs[token_idx].item()
                        break
                else:
                    # If still no valid token, take the end token
                    next_token = 1  # Assuming 1 is the end/start token
                    next_prob = probs[1].item()
            
            # If we generated the end token or have enough digits, stop
            if next_token == 1 or len(output_tokens) >= max_len - 1:
                break
            
            # Add the token to our output
            output_tokens.append(next_token)
            token_probs.append(next_prob)
            
            # Prepare for the next iteration
            decoder_input = torch.cat([decoder_input, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=1)
        
        # Convert token IDs back to digits (token 2 = digit 0, token 3 = digit 1, etc.)
        result = ""
        for token in output_tokens:
            # Convert from token ID to digit
            digit = token - 2  # Token 2 = digit 0, etc.
            result += str(digit)
        
        # Remove leading zeros
        result = result.lstrip('0')
        if not result:  # Handle case of all zeros
            result = '0'
            
        if print_debug:
            input_str = ""
            if hasattr(dataset, 'decode'):
                # Try to decode the input tensor
                try:
                    input_str = dataset.decode(inp[0].cpu().numpy())
                except:
                    input_str = "[Decoding failed]"
            
            print(f"\nInput: {input_str}")
            print(f"Tokens: {output_tokens} (probabilities: {[f'{p:.4f}' for p in token_probs]})")
            print(f"Prediction: {result}")
            
        return result

def improved_accuracy(model, examples, dataset, device, print_debug=False, num_to_print=5):
    """Calculate accuracy using improved inference with detailed metrics"""
    if not examples:
        return 0.0
    
    correct = 0
    total = 0
    digit_correct = 0  # Track digit-level accuracy
    total_digits = 0
    
    # Display header for debug output
    if print_debug:
        print("\n" + "="*50)
        print(f"INFERENCE DIAGNOSTICS FOR {model.__class__.__name__}")
        print("="*50)
    
    # If we need to unpack a different number of values
    try:
        # The generate_multiplication_examples function returns tuples of 
        # (input_tensor, result_str, a, b)
        for idx, (input_tensor, expected_str, a, b) in enumerate(examples):
            try:
                # Whether to print debug info for this example
                should_print = print_debug and idx < num_to_print
                
                # Get prediction using improved inference
                pred = improved_inference(model, input_tensor, dataset, device, print_debug=should_print)
                
                # For digit-level accuracy
                min_len = min(len(pred), len(expected_str))
                for i in range(min_len):
                    if pred[i] == expected_str[i]:
                        digit_correct += 1
                total_digits += max(len(pred), len(expected_str))
                
                # Check if prediction matches expected
                is_correct = pred == expected_str
                if is_correct:
                    correct += 1
                
                if should_print:
                    print(f"Problem: {a} * {b} = {expected_str}")
                    print(f"Predicted: {pred} (Correct: {is_correct})")
                    if not is_correct:
                        # Show where the prediction is wrong
                        error_indices = []
                        for i in range(min_len):
                            if pred[i] != expected_str[i]:
                                error_indices.append(i)
                        if len(pred) != len(expected_str):
                            print(f"Length mismatch: Pred={len(pred)}, Expected={len(expected_str)}")
                        else:
                            print(f"Errors at positions: {error_indices}")
                    print("")
                
                total += 1
            except Exception as e:
                print(f"Error in inference: {e}")
                
        # Summary
        if print_debug:
            print(f"Total examples evaluated: {total}")
            print(f"Completely correct predictions: {correct} ({correct/total:.2%})")
            digit_accuracy = digit_correct / total_digits if total_digits > 0 else 0
            print(f"Digit-level accuracy: {digit_correct}/{total_digits} ({digit_accuracy:.2%})")
            print("="*50)
    except ValueError as e:
        # If there's a ValueError it might be a different format
        print(f"Warning: Example format error: {e}. Check example structure.")
        return 0.0
    
    # Return both full accuracy and digit accuracy
    sequence_accuracy = correct / total if total > 0 else 0.0
    digit_accuracy = digit_correct / total_digits if total_digits > 0 else 0.0
    return sequence_accuracy, digit_accuracy

def evaluate(model, data_loader, criterion, dataset, device, vocab_size, desc="Evaluating", examples=None):
    """Evaluate a model on a data loader with loss and accuracy"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for inp, tgt, inp_lens, tgt_lens in tqdm(data_loader, desc=desc, leave=False):
            inp, tgt = inp.to(device), tgt.to(device)
            decoder_input = tgt[:, :-1]
            decoder_target = tgt[:, 1:]
            
            output = model(inp, decoder_input)
            
            # Use length-aware loss calculation
            token_losses = criterion(output.reshape(-1, vocab_size), decoder_target.reshape(-1))
            token_losses = token_losses.view(decoder_target.size())
            
            # Create a mask for non-padding tokens
            mask = (decoder_target != 0).float()
            
            # Apply length penalty during evaluation
            length_penalty = 1.0
            
            try:
                batch_size = inp.size(0)
                length_penalties = []
                
                for b in range(batch_size):
                    try:
                        # Try to decode the input for this batch item
                        input_str = dataset.decode(inp[b].cpu().numpy())
                        if '*' in input_str:
                            parts = input_str.split('*')
                            if len(parts) == 2:
                                # Calculate expected result length
                                a, b = int(parts[0]), int(parts[1])
                                expected_length = len(str(a * b))
                                
                                # Get actual output length for this batch item
                                actual_length = mask[b].sum().item()
                                
                                # Calculate length penalty (quadratic penalty for length difference)
                                if actual_length > expected_length:
                                    item_penalty = 1.0 + 0.5 * ((actual_length - expected_length) / expected_length) ** 2
                                else:
                                    item_penalty = 1.0
                                
                                length_penalties.append(item_penalty)
                            else:
                                length_penalties.append(1.0)
                        else:
                            length_penalties.append(1.0)
                    except:
                        length_penalties.append(1.0)
                
                # Use the mean length penalty for the batch
                if length_penalties:
                    length_penalty = sum(length_penalties) / len(length_penalties)
            except Exception as e:
                print(f"Warning: Error calculating length penalty in evaluation: {e}")
                length_penalty = 1.0
            
            # Apply mask to get per-token losses (excluding padding)
            masked_losses = token_losses * mask
            
            # Get average loss per sequence
            sequence_losses = masked_losses.sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            
            # Apply length penalty to each sequence
            sequence_losses = sequence_losses * length_penalty
            
            # Get final batch loss
            loss = sequence_losses.mean()
            
            total_loss += loss.item() * decoder_target.size(0)
            total_samples += decoder_target.size(0)
    
    # Calculate accuracy if examples are provided
    sequence_accuracy = 0.0
    digit_accuracy = 0.0
    if examples:
        # Only print diagnostics during validation (not during final eval)
        print_debug = "Val" in desc  # Only print during validation
        sequence_accuracy, digit_accuracy = improved_accuracy(model, examples, dataset, device, print_debug=print_debug)
    
    return total_loss / max(1, total_samples), sequence_accuracy, digit_accuracy

def train_models_parallel(simple_model, latent_model, train_loader, val_loader, device, config, log_dir, dataset, max_steps=3000, accuracy_weight=0.0, tf_schedule="none", tf_start_step=0):
    """Train both models in parallel for real-time comparison"""
    os.makedirs(log_dir, exist_ok=True)
    simple_writer = SummaryWriter(log_dir=f"{log_dir}/simple")
    latent_writer = SummaryWriter(log_dir=f"{log_dir}/latent")
    
    # Setup models, criteria, and optimizers
    models = {
        "simple": {
            "model": simple_model,
            "writer": simple_writer,
            "name": "SimpleTransformer",
            "criterion": nn.CrossEntropyLoss(ignore_index=0, reduction='none'),  # Changed to 'none' for length penalty
            "optimizer": optim.AdamW(
                simple_model.parameters(), 
                lr=config.base_lr, 
                weight_decay=config.weight_decay,
                betas=(0.9, 0.98),
                eps=1e-8
            ),
            "best_val_loss": float('inf'),
            "final_loss": float('inf'),
            "final_sequence_accuracy": 0.0,
            "final_digit_accuracy": 0.0,
            "step": 0,
            "params": count_parameters(simple_model),
            "current_loss": 0.0  # Added for progress bar display
        },
        "latent": {
            "model": latent_model,
            "writer": latent_writer,
            "name": "LatentTransformer",
            "criterion": nn.CrossEntropyLoss(ignore_index=0, reduction='none'),  # Changed to 'none' for length penalty
            "optimizer": optim.AdamW(
                latent_model.parameters(), 
                lr=config.base_lr, 
                weight_decay=config.weight_decay,
                betas=(0.9, 0.98),
                eps=1e-8
            ),
            "best_val_loss": float('inf'),
            "final_loss": float('inf'),
            "final_sequence_accuracy": 0.0,
            "final_digit_accuracy": 0.0,
            "step": 0,
            "params": count_parameters(latent_model),
            "current_loss": 0.0  # Added for progress bar display
        }
    }
    
    # Create schedulers
    def warmup_cosine_schedule(step):
        if step < config.warmup_steps:
            return float(step) / float(max(1, config.warmup_steps))
        progress = float(step - config.warmup_steps) / float(max(1, max_steps - config.warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    for model_info in models.values():
        model_info["scheduler"] = optim.lr_scheduler.LambdaLR(
            model_info["optimizer"], 
            lr_lambda=warmup_cosine_schedule
        )
    
    # Get vocab size for loss computation
    vocab_size = train_loader.dataset.vocab_size
    
    # Time tracking
    start_time = time.time()
    
    # Gradient scalers for mixed precision on CUDA
    if device.type == 'cuda':
        for model_info in models.values():
            model_info["scaler"] = torch.cuda.amp.GradScaler()
    
    # Generate evaluation examples with lower range for easier evaluation
    try:
        # Create examples with smaller numbers for easier debugging
        print("Generating evaluation examples with range 1-5")
        eval_range = min(5, dataset.max_value)  # Use smaller numbers for evaluation
        
        # Create custom examples for easier debugging
        eval_examples = []
        for a in range(1, eval_range + 1):
            for b in range(1, eval_range + 1):
                # Format input
                input_str = f"{a}*{b}"
                input_tokens = dataset.encode(input_str)
                input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(device)
                
                # Format expected output
                result = a * b
                result_str = str(result)
                
                eval_examples.append((input_tensor, result_str, a, b))
                
                # Limit to 20 examples
                if len(eval_examples) >= 20:
                    break
            if len(eval_examples) >= 20:
                break
                
        print(f"Created {len(eval_examples)} evaluation examples")
    except Exception as e:
        print(f"Error generating evaluation examples: {e}")
        eval_examples = []
    
    # Main training loop
    global_step = 0
    val_freq = 50  # Validate every 50 steps
    
    try:
        # Set models to training mode
        for model_info in models.values():
            model_info["model"].train()
        
        # Create progress bars
        pbar = tqdm(total=max_steps, desc=f"Training (steps)")
        
        # Training loop
        train_iter = itertools.cycle(train_loader)
        while global_step < max_steps:
            # Get a batch
            inp, tgt, inp_lens, tgt_lens = next(train_iter)
            inp, tgt = inp.to(device), tgt.to(device)
            
            # Calculate teacher forcing probability based on schedule
            if tf_schedule == "none" or global_step < tf_start_step:
                tf_prob = 1.0  # Always use teacher forcing
            elif tf_schedule == "linear":
                progress = (global_step - tf_start_step) / (max_steps - tf_start_step)
                tf_prob = 1.0 - progress  # Linear decay from 1.0 to 0.0
            elif tf_schedule == "exp":
                progress = (global_step - tf_start_step) / (max_steps - tf_start_step)
                tf_prob = math.exp(-10 * progress)  # Exponential decay
            else:
                tf_prob = 1.0
                
            # Train each model on the same batch
            for model_type, model_info in models.items():
                if model_info["step"] >= max_steps:
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
                    
                    # Skip this batch if decoder input is empty (shouldn't happen, but just in case)
                    if decoder_input.size(1) == 0:
                        print(f"Warning: Empty decoder input in batch with teacher forcing, skipping for {model_info['name']}")
                        continue
                else:
                    # Generate decoder input on-the-fly
                    batch_size = inp.size(0)
                    decoder_target = tgt[:, 1:]
                    max_len = decoder_target.size(1)
                    
                    # Skip this batch if target is empty (shouldn't happen, but just in case)
                    if max_len == 0:
                        print(f"Warning: Empty target sequence, skipping batch for {model_info['name']}")
                        continue
                    
                    # Start with the start token (assumed to be at index 1 based on previous patterns)
                    decoder_input = torch.ones(batch_size, 1, dtype=torch.long, device=device)
                    
                    # Only generate additional tokens if target length > 1
                    if max_len > 1:
                        try:
                            # Generate partial sequence for decoder input
                            with torch.no_grad():
                                for i in range(max_len - 1):
                                    # Forward pass to get next token prediction
                                    temp_output = model(inp, decoder_input)
                                    next_token_logits = temp_output[:, -1, :]
                                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                                    # Append to decoder input
                                    decoder_input = torch.cat([decoder_input, next_token], dim=1)
                            
                            # Use all but the last token as decoder input (ensuring at least one token remains)
                            if decoder_input.size(1) > 1:
                                decoder_input = decoder_input[:, :-1]
                        except Exception as e:
                            print(f"Error during sequence generation for {model_info['name']}: {e}")
                            print(f"Falling back to teacher forcing for this batch")
                            decoder_input = tgt[:, :-1]
                            decoder_target = tgt[:, 1:]
                    
                # Skip this batch if decoder input is empty
                if decoder_input.size(1) == 0:
                    print(f"Warning: Empty decoder input in batch, skipping for {model_info['name']}")
                    continue
                
                # Log teacher forcing probability
                writer.add_scalar('train/tf_probability', tf_prob, model_info["step"])
                
                if device.type == 'cuda':
                    # Use mixed precision for faster training on GPU
                    with torch.cuda.amp.autocast():
                        output = model(inp, decoder_input)
                        
                        # Calculate base loss (per token)
                        token_losses = criterion(output.reshape(-1, vocab_size), decoder_target.reshape(-1))
                        
                        # Reshape token_losses to match batch shape
                        token_losses = token_losses.view(decoder_target.size())
                        
                        # Create a mask for non-padding tokens
                        mask = (decoder_target != 0).float()
                        
                        # Apply length penalty
                        length_penalty = 1.0
                        
                        # Extract inputs to calculate expected output length
                        try:
                            batch_size = inp.size(0)
                            length_penalties = []
                            
                            for b in range(batch_size):
                                # Try to decode the input for this batch item
                                input_str = dataset.decode(inp[b].cpu().numpy())
                                if '*' in input_str:
                                    parts = input_str.split('*')
                                    if len(parts) == 2:
                                        # Calculate expected result length
                                        a, b = int(parts[0]), int(parts[1])
                                        expected_length = len(str(a * b))
                                        
                                        # Get actual output length for this batch item
                                        actual_length = mask[b].sum().item()
                                        
                                        # Calculate length penalty (quadratic penalty for length difference)
                                        # No penalty if correct length, increasing penalty for longer sequences
                                        if actual_length > expected_length:
                                            item_penalty = 1.0 + 0.5 * ((actual_length - expected_length) / expected_length) ** 2
                                        else:
                                            item_penalty = 1.0
                                        
                                        length_penalties.append(item_penalty)
                                    else:
                                        length_penalties.append(1.0)
                                else:
                                    length_penalties.append(1.0)
                            
                            # Use the mean length penalty for the batch
                            if length_penalties:
                                length_penalty = sum(length_penalties) / len(length_penalties)
                        except Exception as e:
                            # If calculation fails, use default penalty
                            print(f"Warning: Error calculating length penalty: {e}")
                            length_penalty = 1.0
                        
                        # Apply mask to get per-token losses (excluding padding)
                        masked_losses = token_losses * mask
                        
                        # Get average loss per sequence
                        sequence_losses = masked_losses.sum(dim=1) / (mask.sum(dim=1) + 1e-8)
                        
                        # Apply length penalty to each sequence
                        sequence_losses = sequence_losses * length_penalty
                        
                        # Get final batch loss
                        loss = sequence_losses.mean()
                        
                        # Add accuracy component to loss if requested
                        if accuracy_weight > 0:
                            # Get predictions
                            _, predicted = torch.max(output, dim=-1)
                            
                            # Calculate token-level accuracy
                            correct = (predicted == decoder_target).float()
                            # Exclude padding tokens (0)
                            mask = (decoder_target != 0).float()
                            # Accuracy is number of correct tokens / number of non-padding tokens
                            batch_accuracy = (correct * mask).sum() / (mask.sum() + 1e-8)
                            
                            # IMPROVED: Use a sequence-level penalty
                            # Check if each sequence is completely correct
                            sequence_correct = torch.all(
                                torch.logical_or(predicted == decoder_target, decoder_target == 0),
                                dim=1
                            ).float()
                            sequence_accuracy = sequence_correct.mean()
                            
                            # Stronger penalty for sequence-level errors
                            # Loss multiplier: 
                            # - If all sequences are completely correct: multiply by 0.5
                            # - If no sequences are correct: multiply by 1.5
                            # This gives more weight to getting complete sequences right
                            seq_factor = 1.5 - sequence_accuracy
                            
                            # Combined factor (both token-level and sequence-level)
                            accuracy_factor = 1.0
                            
                            # Apply token-level factor (lower loss for higher token accuracy)
                            token_factor = 1.0 - (batch_accuracy * 0.5 * accuracy_weight)  # Scaled by accuracy_weight
                            accuracy_factor *= token_factor
                            
                            # Apply sequence-level factor (strongly lower loss for correct sequences)
                            # Scale by accuracy_weight for stronger effect
                            seq_factor = 1.5 - (sequence_accuracy * accuracy_weight)
                            accuracy_factor *= seq_factor
                            
                            # Apply the accuracy factor to the loss
                            original_loss = loss
                            loss = loss * accuracy_factor
                            
                            # Log the accuracy components
                            writer.add_scalar('train/token_accuracy', batch_accuracy.item(), model_info["step"])
                            writer.add_scalar('train/sequence_accuracy', sequence_accuracy.item(), model_info["step"])
                            writer.add_scalar('train/token_factor', token_factor, model_info["step"])
                            writer.add_scalar('train/sequence_factor', seq_factor, model_info["step"])
                            writer.add_scalar('train/combined_factor', accuracy_factor, model_info["step"])
                            writer.add_scalar('train/ce_loss', original_loss.item(), model_info["step"])
                            writer.add_scalar('train/modified_loss', loss.item(), model_info["step"])
                            
                        # Log length penalty
                        writer.add_scalar('train/length_penalty', length_penalty, model_info["step"])
                    
                    # Scale gradients
                    scaler = model_info["scaler"]
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard training for CPU/MPS
                    output = model(inp, decoder_input)
                    
                    # Check for NaN in output
                    if torch.isnan(output).any():
                        print(f"Warning: NaN in output for {model_info['name']} at step {model_info['step']}, skipping batch.")
                        continue
                    
                    # Calculate base loss (per token)
                    token_losses = criterion(output.reshape(-1, vocab_size), decoder_target.reshape(-1))
                    
                    # Reshape token_losses to match batch shape
                    token_losses = token_losses.view(decoder_target.size())
                    
                    # Create a mask for non-padding tokens
                    mask = (decoder_target != 0).float()
                    
                    # Apply length penalty
                    length_penalty = 1.0
                    
                    # Extract inputs to calculate expected output length
                    try:
                        batch_size = inp.size(0)
                        length_penalties = []
                        
                        for b in range(batch_size):
                            # Try to decode the input for this batch item
                            input_str = dataset.decode(inp[b].cpu().numpy())
                            if '*' in input_str:
                                parts = input_str.split('*')
                                if len(parts) == 2:
                                    # Calculate expected result length
                                    a, b = int(parts[0]), int(parts[1])
                                    expected_length = len(str(a * b))
                                    
                                    # Get actual output length for this batch item
                                    actual_length = mask[b].sum().item()
                                    
                                    # Calculate length penalty (quadratic penalty for length difference)
                                    # No penalty if correct length, increasing penalty for longer sequences
                                    if actual_length > expected_length:
                                        item_penalty = 1.0 + 0.5 * ((actual_length - expected_length) / expected_length) ** 2
                                    else:
                                        item_penalty = 1.0
                                    
                                    length_penalties.append(item_penalty)
                                else:
                                    length_penalties.append(1.0)
                            else:
                                length_penalties.append(1.0)
                        
                        # Use the mean length penalty for the batch
                        if length_penalties:
                            length_penalty = sum(length_penalties) / len(length_penalties)
                    except Exception as e:
                        # If calculation fails, use default penalty
                        print(f"Warning: Error calculating length penalty: {e}")
                        length_penalty = 1.0
                    
                    # Apply mask to get per-token losses (excluding padding)
                    masked_losses = token_losses * mask
                    
                    # Get average loss per sequence
                    sequence_losses = masked_losses.sum(dim=1) / (mask.sum(dim=1) + 1e-8)
                    
                    # Apply length penalty to each sequence
                    sequence_losses = sequence_losses * length_penalty
                    
                    # Get final batch loss
                    loss = sequence_losses.mean()
                    
                    # Add accuracy component to loss if requested
                    if accuracy_weight > 0:
                        # Get predictions
                        _, predicted = torch.max(output, dim=-1)
                        
                        # Calculate token-level accuracy
                        correct = (predicted == decoder_target).float()
                        # Exclude padding tokens (0)
                        mask = (decoder_target != 0).float()
                        # Accuracy is number of correct tokens / number of non-padding tokens
                        batch_accuracy = (correct * mask).sum() / (mask.sum() + 1e-8)
                        
                        # IMPROVED: Use a sequence-level penalty
                        # Check if each sequence is completely correct
                        sequence_correct = torch.all(
                            torch.logical_or(predicted == decoder_target, decoder_target == 0),
                            dim=1
                        ).float()
                        sequence_accuracy = sequence_correct.mean()
                        
                        # Stronger penalty for sequence-level errors
                        # Loss multiplier: 
                        # - If all sequences are completely correct: multiply by 0.5
                        # - If no sequences are correct: multiply by 1.5
                        # This gives more weight to getting complete sequences right
                        seq_factor = 1.5 - sequence_accuracy
                        
                        # Combined factor (both token-level and sequence-level)
                        accuracy_factor = 1.0
                        
                        # Apply token-level factor (lower loss for higher token accuracy)
                        token_factor = 1.0 - (batch_accuracy * 0.5 * accuracy_weight)  # Scaled by accuracy_weight
                        accuracy_factor *= token_factor
                        
                        # Apply sequence-level factor (strongly lower loss for correct sequences)
                        # Scale by accuracy_weight for stronger effect
                        seq_factor = 1.5 - (sequence_accuracy * accuracy_weight)
                        accuracy_factor *= seq_factor
                        
                        # Apply the accuracy factor to the loss
                        original_loss = loss
                        loss = loss * accuracy_factor
                        
                        # Log the accuracy components
                        writer.add_scalar('train/token_accuracy', batch_accuracy.item(), model_info["step"])
                        writer.add_scalar('train/sequence_accuracy', sequence_accuracy.item(), model_info["step"])
                        writer.add_scalar('train/token_factor', token_factor, model_info["step"])
                        writer.add_scalar('train/sequence_factor', seq_factor, model_info["step"])
                        writer.add_scalar('train/combined_factor', accuracy_factor, model_info["step"])
                        writer.add_scalar('train/ce_loss', original_loss.item(), model_info["step"])
                        writer.add_scalar('train/modified_loss', loss.item(), model_info["step"])
                    
                    # Log length penalty
                    writer.add_scalar('train/length_penalty', length_penalty, model_info["step"])
                    
                    # Check for NaN in loss
                    if torch.isnan(loss).any():
                        print(f"Warning: NaN in loss for {model_info['name']} at step {model_info['step']}, skipping batch.")
                        continue
                    
                    loss.backward()
                    
                    # Check for NaN in gradients
                    has_nan_grad = False
                    for name, param in model.named_parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            print(f"Warning: NaN gradient in {name} for {model_info['name']} at step {model_info['step']}")
                            has_nan_grad = True
                            break
                    
                    if has_nan_grad:
                        # Skip this batch if NaN gradients are detected
                        optimizer.zero_grad()
                        continue
                    
                    # Clip gradients to prevent explosions
                    nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                
                model_info["scheduler"].step()
                
                # Log metrics
                lr = model_info["scheduler"].get_last_lr()[0]
                writer.add_scalar('train/loss', loss.item(), model_info["step"])
                writer.add_scalar('train/lr', lr, model_info["step"])
                
                # Store current loss for display in progress bar
                model_info["current_loss"] = loss.item()
                
                # Increment step
                model_info["step"] += 1
                
                # Validate periodically
                if model_info["step"] > 0 and model_info["step"] % val_freq == 0:
                    val_loss, val_sequence_accuracy, val_digit_accuracy = evaluate(
                        model=model,
                        data_loader=val_loader,
                        criterion=criterion,
                        dataset=dataset,
                        device=device,
                        vocab_size=vocab_size,
                        desc=f"Val {model_info['name']}",
                        examples=eval_examples
                    )
                    
                    writer.add_scalar('val/loss', val_loss, model_info["step"])
                    writer.add_scalar('val/sequence_accuracy', val_sequence_accuracy, model_info["step"])
                    writer.add_scalar('val/digit_accuracy', val_digit_accuracy, model_info["step"])
                    print(f"\n{model_info['name']} - Step {model_info['step']}/{max_steps} - Val Loss: {val_loss:.6f} - Sequence Acc: {val_sequence_accuracy:.2%} - Digit Acc: {val_digit_accuracy:.2%}")
                    
                    # Save best model
                    if val_loss < model_info["best_val_loss"] and not np.isnan(val_loss):
                        model_info["best_val_loss"] = val_loss
                        os.makedirs(f"checkpoints/{model_info['name'].lower()}", exist_ok=True)
                        torch.save({
                            'step': model_info["step"],
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': val_loss,
                            'val_sequence_accuracy': val_sequence_accuracy,
                            'val_digit_accuracy': val_digit_accuracy,
                        }, f"checkpoints/{model_info['name'].lower()}/{model_info['name'].lower()}_best.pt")
                    
                    # Return to training mode
                    model.train()
                    
                    # Early stopping if we reach excellent performance
                    if val_loss < 0.01 and val_sequence_accuracy > 0.95:
                        print(f"Reached excellent validation performance for {model_info['name']}, stopping early")
                        model_info["step"] = max_steps  # Mark as completed
            
            # Update global step counter and progress bar
            global_step += 1
            pbar.update(1)
            pbar.set_postfix({
                "simple_loss": f"{models['simple']['current_loss']:.4f}", 
                "latent_loss": f"{models['latent']['current_loss']:.4f}"
            })
            
            # Check if both models have completed their steps
            if all(model_info["step"] >= max_steps for model_info in models.values()):
                break
                
    except KeyboardInterrupt:
        print("Training interrupted")
        
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
            examples=eval_examples
        )
        
        model_info["final_loss"] = final_val_loss
        model_info["final_sequence_accuracy"] = final_sequence_accuracy
        model_info["final_digit_accuracy"] = final_digit_accuracy
    
    training_time = time.time() - start_time
    
    # Print summaries
    for model_type, model_info in models.items():
        print(f"\n{model_info['name']} Training Summary:")
        print(f"Steps completed: {model_info['step']}/{max_steps}")
        print(f"Best validation loss: {model_info['best_val_loss']:.6f}")
        print(f"Final validation loss: {model_info['final_loss']:.6f}")
        print(f"Final sequence accuracy: {model_info['final_sequence_accuracy']:.2%}")
        print(f"Final digit accuracy: {model_info['final_digit_accuracy']:.2%}")
    
    # Close writers
    for model_info in models.values():
        model_info["writer"].close()
    
    return {
        "simple": {
            "loss": models["simple"]["final_loss"],
            "sequence_accuracy": models["simple"]["final_sequence_accuracy"],
            "digit_accuracy": models["simple"]["final_digit_accuracy"],
            "params": models["simple"]["params"],
            "steps": models["simple"]["step"]
        },
        "latent": {
            "loss": models["latent"]["final_loss"],
            "sequence_accuracy": models["latent"]["final_sequence_accuracy"],
            "digit_accuracy": models["latent"]["final_digit_accuracy"],
            "params": models["latent"]["params"],
            "steps": models["latent"]["step"]
        },
        "training_time": training_time
    }

def main():
    # Enable MPS fallback
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Parallel comparison of transformer architectures with device selection")
    parser.add_argument("--d-model", type=int, default=32, help="Model dimension (default: 32)")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of layers (default: 2)")
    parser.add_argument("--num-latent", type=int, default=4, help="Number of latent tokens (default: 4)")
    parser.add_argument("--bottleneck-factor", type=float, default=1.0, help="Bottleneck factor (default: 1.0)")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum training steps (default: 500)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--min-digits", type=int, default=1, help="Minimum number of digits (default: 1)")
    parser.add_argument("--device", type=str, default="auto", 
                        help="Device to use: 'cuda', 'mps', 'cpu', or 'auto' (default: auto-detect)")
    parser.add_argument("--accuracy-weight", type=float, default=0.5,
                        help="Weight for accuracy component in combined loss function (default: 0.5)")
    parser.add_argument("--tf-schedule", type=str, choices=["none", "linear", "exp"], default="linear",
                        help="Teacher forcing schedule: none=always use, linear=linearly decrease, exp=exponentially decrease (default: linear)")
    parser.add_argument("--tf-start-step", type=int, default=200,
                        help="Step at which to start scheduled teacher forcing (default: 200 - pure teacher forcing for first 200 steps)")
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load configuration
    config = TrainingConfig()
    
    # Override config for stability
    config.base_lr = 3e-4  # Standard learning rate
    config.max_grad_norm = 1.0  # Standard gradient clipping
    config.warmup_steps = 100  # Warmup period
    config.weight_decay = 0.01  # Regularization
    
    # Setup device based on arguments or auto-detect
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            device = torch.device('cpu')
        elif args.device == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS requested but not available. Falling back to CPU.")
            device = torch.device('cpu')
        else:
            device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Display info about accuracy-aware loss if enabled
    if args.accuracy_weight > 0:
        print(f"Using accuracy-aware loss with weight: {args.accuracy_weight}")
        print("This reduces loss for correct token predictions during training")
    
    # Create datasets
    min_val = 10 if args.min_digits >= 2 else 1
    max_val = 99 if args.min_digits == 2 else (9 if args.min_digits == 1 else 999)
    
    train_dataset = MultiplicationDataset(
        num_samples=8000,
        split='train',
        split_ratio=(0.8, 0.1, 0.1),
        min_value=min_val,
        max_value=max_val
    )
    
    val_dataset = MultiplicationDataset(
        num_samples=1000,
        split='val',
        split_ratio=(0.8, 0.1, 0.1),
        min_value=min_val,
        max_value=max_val
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,  # Standard batch size
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Initialize models
    d_model = args.d_model
    num_layers = args.num_layers
    num_latent = args.num_latent
    
    # Create SimpleTransformer (stable version)
    simple_transformer = StableSimpleTransformer(
        vocab_size=train_dataset.vocab_size,
        d_model=d_model,
        nhead=4,
        num_layers=num_layers,
        dropout=0.1
    ).to(device)
    
    # Create LatentTransformer (stable version)
    latent_transformer = StableLatentTransformer(
        vocab_size=train_dataset.vocab_size,
        d_model=d_model,
        nhead=4,
        num_layers=num_layers,
        num_latent=num_latent,
        dropout=0.1,
        bottleneck_factor=args.bottleneck_factor
    ).to(device)
    
    # Print parameter counts
    simple_params = count_parameters(simple_transformer)
    latent_params = count_parameters(latent_transformer)
    
    print(f"SimpleTransformer parameters: {simple_params:,}")
    print(f"LatentTransformer parameters: {latent_params:,}")
    print(f"Parameter ratio: {latent_params/simple_params:.2f}x")
    
    # Train both models in parallel
    print("\nTraining both models in parallel...")
    log_dir = f"runs/parallel_comparison/d{d_model}_l{num_layers}_n{num_latent}"
    
    # Delete existing runs directory if it exists to avoid TensorBoard confusion
    if os.path.exists(log_dir):
        print(f"Removing existing log directory: {log_dir}")
        shutil.rmtree(log_dir)
    
    results = train_models_parallel(
        simple_model=simple_transformer,
        latent_model=latent_transformer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        log_dir=log_dir,
        dataset=train_dataset,
        max_steps=args.max_steps,
        accuracy_weight=args.accuracy_weight,
        tf_schedule=args.tf_schedule,
        tf_start_step=args.tf_start_step
    )
    
    # Print comparison
    print("\nFinal Comparison:")
    print("=" * 50)
    
    simple_results = results["simple"]
    latent_results = results["latent"]
    training_time = results["training_time"]
    
    print(f"Training time: {training_time:.2f}s")
    print(f"SimpleTransformer: {simple_results['loss']:.6f} loss, {simple_results['sequence_accuracy']:.2%} sequence accuracy, {simple_results['digit_accuracy']:.2%} digit accuracy, {simple_results['params']:,} parameters")
    print(f"LatentTransformer: {latent_results['loss']:.6f} loss, {latent_results['sequence_accuracy']:.2%} sequence accuracy, {latent_results['digit_accuracy']:.2%} digit accuracy, {latent_results['params']:,} parameters")
    
    # Calculate efficiency metrics if possible
    if simple_results['loss'] != float('inf') and latent_results['loss'] != float('inf') and not np.isnan(simple_results['loss']) and not np.isnan(latent_results['loss']):
        # Parameter efficiency (lower is better: loss * num_params)
        param_efficiency_simple = simple_results['loss'] * simple_results['params']
        param_efficiency_latent = latent_results['loss'] * latent_results['params']
        
        # Accuracy efficiency (higher is better: accuracy / num_params)
        acc_param_efficiency_simple = simple_results['sequence_accuracy'] / simple_results['params'] if simple_results['params'] > 0 and simple_results['sequence_accuracy'] > 0 else 0
        acc_param_efficiency_latent = latent_results['sequence_accuracy'] / latent_results['params'] if latent_results['params'] > 0 and latent_results['sequence_accuracy'] > 0 else 0
        
        print("\nEfficiency Metrics:")
        
        if param_efficiency_latent < param_efficiency_simple:
            ratio = param_efficiency_simple / param_efficiency_latent
            print(f"LatentTransformer is {ratio:.2f}x more parameter-efficient (loss*params)")
        else:
            ratio = param_efficiency_latent / param_efficiency_simple
            print(f"SimpleTransformer is {ratio:.2f}x more parameter-efficient (loss*params)")
            
        # Only show accuracy efficiency if both models have non-zero accuracy
        if simple_results['sequence_accuracy'] > 0 and latent_results['sequence_accuracy'] > 0:
            if acc_param_efficiency_latent > acc_param_efficiency_simple:
                ratio = acc_param_efficiency_latent / acc_param_efficiency_simple
                print(f"LatentTransformer is {ratio:.2f}x more accuracy-per-parameter efficient")
            else:
                ratio = acc_param_efficiency_simple / acc_param_efficiency_latent
                print(f"SimpleTransformer is {ratio:.2f}x more accuracy-per-parameter efficient")
        else:
            print("Cannot calculate accuracy efficiency metrics: at least one model has 0% accuracy")
    else:
        print("Cannot calculate efficiency metrics due to invalid loss values")
    
    print("\nTo view parallel training curves, run:")
    print(f"tensorboard --logdir={log_dir}")

if __name__ == "__main__":
    main() 