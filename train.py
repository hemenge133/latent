"""
Training script for SimpleTransformer model
"""
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os
import time
from Dataset import MultiplicationDataset
from Collate import collate_fn
from Transformer import SimpleTransformer
from utils import (
    evaluate, save_checkpoint, load_checkpoint, 
    warmup_lr_scheduler, setup_training_dir, 
    calculate_model_size, format_time
)
from config import TrainingConfig

def main():
    # Load configuration
    config = TrainingConfig()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup tensorboard writer
    writer = SummaryWriter(log_dir=config.simple_transformer_logdir)
    
    # Create train, validation, and test datasets
    train_dataset = MultiplicationDataset(config.total_samples, split='train', split_ratio=config.split_ratio)
    val_dataset = MultiplicationDataset(config.total_samples, split='val', split_ratio=config.split_ratio)
    test_dataset = MultiplicationDataset(config.total_samples, split='test', split_ratio=config.split_ratio)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
    
    # Create model
    model = SimpleTransformer(
        vocab_size=train_dataset.vocab_size,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers
    ).to(device)
    
    model_size = calculate_model_size(model)
    print(f"Model parameters: {model_size:,}")
    
    # Setup loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.base_lr, 
        weight_decay=config.weight_decay, 
        betas=(0.9, 0.999)
    )
    
    # Setup learning rate scheduler
    update_lr = warmup_lr_scheduler(
        optimizer, 
        warmup_steps=config.warmup_steps, 
        base_lr=config.base_lr
    )
    
    # Setup directory for saving checkpoints
    save_dir = setup_training_dir(config.simple_transformer_name)
    
    # Training state tracking
    best_val_loss = float('inf')
    patience_counter = 0
    total_steps = config.max_epochs * len(train_loader)
    
    # Training loop
    for epoch in range(config.max_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.max_epochs}")
        
        for step, (inp, tgt, inp_lens, tgt_lens) in enumerate(pbar):
            global_step = epoch * len(train_loader) + step
            
            # Update learning rate for warmup
            update_lr(global_step)
            
            optimizer.zero_grad()
            
            inp, tgt = inp.to(device), tgt.to(device)
            decoder_input = tgt[:, :-1]
            decoder_target = tgt[:, 1:]
            
            output = model(inp, decoder_input)
            
            loss = criterion(output.reshape(-1, train_dataset.vocab_size), decoder_target.reshape(-1))
            
            # Check for NaN loss
            if torch.isnan(loss).item():
                print("NaN loss detected! Skipping this batch.")
                continue
                
            loss.backward()
            
            # Apply gradient clipping to prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            
            # Check for NaN gradients
            has_nan = False
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"NaN gradient detected in {name}!")
                    has_nan = True
                    break
            
            if has_nan:
                print("Skipping optimizer step due to NaN gradients")
                continue
            
            optimizer.step()
            
            total_loss += loss.item()
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], global_step)
            pbar.set_postfix(loss=loss.item())
        
        train_loss = total_loss / len(train_loader)
        
        # Evaluate on validation set
        val_loss = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            vocab_size=train_dataset.vocab_size,
            desc="Validation"
        )
        
        # Evaluate on test set every N epochs
        if (epoch + 1) % config.test_every == 0:
            test_loss = evaluate(
                model=model,
                data_loader=test_loader,
                criterion=criterion,
                device=device,
                vocab_size=train_dataset.vocab_size,
                desc="Test"
            )
            writer.add_scalar('Loss/test', test_loss, epoch)
            print(f"Test Loss: {test_loss:.4f}")
        
        # Log metrics
        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{config.max_epochs} - Time: {format_time(epoch_time)} - "
              f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        # Save model if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                save_path=f"{save_dir}/{config.simple_transformer_name}_best.pt",
                is_best=True
            )
            print(f"New best model saved with val loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"Early stopping after {config.patience} epochs without improvement")
                break
        
        # Save regular checkpoint every N epochs
        if (epoch + 1) % config.save_every == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                save_path=f"{save_dir}/{config.simple_transformer_name}_epoch_{epoch+1}.pt"
            )
    
    # Final evaluation on test set
    print("Evaluating best model on test set...")
    # Load the best model
    checkpoint = load_checkpoint(
        model=model,
        optimizer=None,
        checkpoint_path=f"{save_dir}/{config.simple_transformer_name}_best.pt",
        device=device
    )
    test_loss = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        vocab_size=train_dataset.vocab_size,
        desc="Final Test"
    )
    print(f"Final Test Loss: {test_loss:.4f}")
    
    writer.close()

if __name__ == "__main__":
    main()

