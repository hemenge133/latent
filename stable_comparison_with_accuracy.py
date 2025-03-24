"""
Stable comparison between SimpleTransformer and LatentTransformer with careful initialization
and numerical stability improvements. Also includes accuracy metrics for TensorBoard.
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

from Dataset import MultiplicationDataset
from utils import collate_fn
from config import TrainingConfig

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

# Modified SimpleTransformer implementation with stability improvements
class StableSimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Embedding with scaling - just like the original
        self.embed = nn.Embedding(vocab_size, d_model)
        self.embed_scale = d_model ** 0.5  # Important - keep this scaling
        
        # Learned positional encodings
        self.max_len = 20  # Maximum sequence length
        self.pos_encoder = nn.Parameter(torch.zeros(self.max_len, d_model))
        
        # Encoder - use GELU like the original
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",  # Match original
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decoder - use GELU like the original
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",  # Match original
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Layer normalization for additional stability
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize parameters using Xavier uniform like the original"""
        # Similar to original _init_parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)  # Use smaller gain for better stability
                
        # Initialize positional encodings
        nn.init.normal_(self.pos_encoder, mean=0, std=0.01)  # Smaller std for better stability
        
    def forward(self, src, tgt):
        # Apply embedding with scaling - just like original
        src_embed = self.embed(src) * self.embed_scale
        src_embed = src_embed + self.pos_encoder[:src.size(1)]
        
        # Generate padding mask
        src_padding_mask = (src == 0)
        
        # Apply encoder
        memory = self.encoder(
            src_embed, 
            src_key_padding_mask=src_padding_mask
        )
        memory = self.encoder_norm(memory)
        
        # Process target sequence - with scaling like original
        tgt_embed = self.embed(tgt) * self.embed_scale
        tgt_pos = self.pos_encoder[:tgt.size(1)]
        tgt_embed = tgt_embed + tgt_pos
        
        # Generate causal mask
        tgt_len = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(tgt.device)
        tgt_padding_mask = (tgt == 0)
        
        # Apply decoder
        output = self.decoder(
            tgt_embed,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        output = self.decoder_norm(output)
        
        return self.output_proj(output)

# Modified LatentTransformer for numerical stability
class StableLatentTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_latent=8, nhead=8, num_layers=4, dropout=0.1, bottleneck_factor=1.0):
        super().__init__()
        self.d_model = d_model
        self.num_latent = num_latent
        self.bottleneck_factor = bottleneck_factor  # Add this parameter back from original
        
        # Embeddings - match original with embed_scale
        self.embed = nn.Embedding(vocab_size, d_model)
        self.embed_scale = d_model ** 0.5
        
        # Positional encodings
        self.max_len = 20
        self.pos_encoder = nn.Parameter(torch.zeros(self.max_len, d_model))
        
        # Encoder - use GELU like original
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",  # Match original
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Latent tokens - same as original
        self.latent_tokens = nn.Parameter(torch.zeros(num_latent, d_model))
        
        # Latent attention projections - use same custom attention as original
        self.latent_proj_q = nn.Linear(d_model, d_model)
        self.latent_proj_k = nn.Linear(d_model, d_model)
        self.latent_proj_v = nn.Linear(d_model, d_model)
        
        # Latent processing layers
        self.latent_norm1 = nn.LayerNorm(d_model)
        self.latent_norm2 = nn.LayerNorm(d_model)
        
        # MLP with GELU like original
        self.latent_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),  # Match original
            nn.Linear(d_model * 4, d_model),
        )
        
        # Decoder - use GELU like original
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",  # Match original
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Additional norms for stability
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize like the original"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)  # Use smaller gain for better stability
                
        # Initialize latent tokens
        nn.init.normal_(self.latent_tokens, mean=0.0, std=0.01)  # Smaller std for better stability
        
        # Initialize positional encodings
        nn.init.normal_(self.pos_encoder, mean=0.0, std=0.01)  # Smaller std for better stability
        
    def _cross_attention(self, query, key, value, attn_mask=None):
        """Custom cross attention from latent tokens to encoder outputs - like original"""
        # Project queries, keys, values
        q = self.latent_proj_q(query)  # [batch_size, num_latent, d_model]
        k = self.latent_proj_k(key)    # [batch_size, seq_len, d_model]
        v = self.latent_proj_v(value)  # [batch_size, seq_len, d_model]
        
        # Calculate attention scores [batch_size, num_latent, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
        
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e4)  # Using -1e4 instead of -1e9 to avoid FP16 overflow
            
        # Apply softmax and get weighted values
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, num_latent, seq_len]
        out = torch.bmm(attn_weights, v)  # [batch_size, num_latent, d_model]
        
        return out
        
    def forward(self, src, tgt):
        # Encode input - with scaling like original
        src_embed = self.embed(src) * self.embed_scale
        src_embed = src_embed + self.pos_encoder[:src.size(1)]
        
        # Generate padding mask
        src_padding_mask = (src == 0)
        
        # Apply encoder
        memory = self.encoder(
            src_embed, 
            src_key_padding_mask=src_padding_mask
        )
        memory = self.encoder_norm(memory)
        
        # Process latent tokens (information bottleneck)
        batch_size = src.size(0)
        latent = self.latent_tokens.unsqueeze(0).expand(batch_size, self.num_latent, -1)
        
        # Apply custom cross-attention like original
        latent_attended = self._cross_attention(latent, memory, memory, 
                                               attn_mask=None if src_padding_mask is None else ~src_padding_mask.unsqueeze(1))
        latent = self.latent_norm1(latent + latent_attended)
        
        # Apply MLP
        latent_mlp_out = self.latent_mlp(latent)
        latent_memory = self.latent_norm2(latent + latent_mlp_out)
        
        # Control the bottleneck (like original)
        if self.bottleneck_factor < 1.0:
            # Mixed memory
            mixed_memory = torch.cat([
                latent_memory,  # [batch, num_latent, d_model]
                memory * (1.0 - self.bottleneck_factor)  # [batch, seq_len, d_model]
            ], dim=1)
        else:
            # Pure latent bottleneck
            mixed_memory = latent_memory
        
        # Process target sequence
        tgt_embed = self.embed(tgt) * self.embed_scale
        tgt_embed = tgt_embed + self.pos_encoder[:tgt.size(1)]
        
        # Generate masks
        tgt_len = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(tgt.device)
        tgt_padding_mask = (tgt == 0)
        
        # Apply decoder
        output = self.decoder(
            tgt_embed,
            mixed_memory,  # Use mixed memory like original
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        output = self.decoder_norm(output)
        
        return self.output_proj(output)

def generate_multiplication_examples(dataset, num_examples=5, device='cpu'):
    """Generate random multiplication examples for accuracy testing"""
    examples = []
    for _ in range(num_examples):
        a = random.randint(dataset.min_value, dataset.max_value)
        b = random.randint(dataset.min_value, dataset.max_value)
        
        # Format input
        input_str = f"{a}*{b}"
        input_tokens = dataset.encode(input_str)
        input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(device)
        
        # Format expected output
        result = a * b
        result_str = str(result)
        
        examples.append((input_tensor, result_str, a, b))
    return examples

def inference(model, input_tensor, dataset, max_length=10, device='cpu'):
    """Run inference to calculate accuracy"""
    model.eval()
    
    # Start with a single token (use the start token)
    output_sequence = [0]  # Assuming 0 is the start token
    
    # Generate tokens one by one
    for _ in range(max_length - 1):
        decoder_input = torch.tensor([output_sequence], dtype=torch.long).to(device)
        
        with torch.no_grad():
            # Forward pass
            output = model(input_tensor, decoder_input)
            
            # Get the next token prediction (last position)
            next_token_logits = output[0, -1, :]
            next_token = next_token_logits.argmax(dim=0).item()
            
            # Add to output sequence
            output_sequence.append(next_token)
            
            # Stop if we predict an end token (assuming 0 is both start and pad)
            if next_token == 0 and len(output_sequence) > 1:
                break
    
    # Remove start token and decode
    result_tokens = [t for t in output_sequence[1:] if t != 0]  # Remove start token and padding
    
    # Convert tokens to string
    result = ''.join([dataset.inv_tokenizer[t] for t in result_tokens])
    
    # Clean up result (remove non-digit characters)
    result = ''.join([c for c in result if c.isdigit()])
    
    return result if result else "0"  # Default to "0" if empty result

def calculate_accuracy(model, examples, dataset, device):
    """Calculate accuracy on a set of examples"""
    correct = 0
    
    for input_tensor, expected, a, b in examples:
        # Run inference
        predicted = inference(model, input_tensor, dataset, device=device)
        
        # Check if correct
        is_correct = predicted == expected
        if is_correct:
            correct += 1
            
    # Calculate accuracy
    accuracy = correct / len(examples) if examples else 0
    return accuracy

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
            
            loss = criterion(output.reshape(-1, vocab_size), decoder_target.reshape(-1))
            
            total_loss += loss.item() * decoder_target.size(0)
            total_samples += decoder_target.size(0)
    
    # Calculate accuracy if examples are provided
    accuracy = 0.0
    if examples:
        accuracy = calculate_accuracy(model, examples, dataset, device)
    
    return total_loss / max(1, total_samples), accuracy

def train_model(model_name, model, train_loader, val_loader, device, config, log_dir, dataset, max_steps=3000):
    """Train a model with stability enhancements and log accuracy metrics"""
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # Setup loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.base_lr, 
        weight_decay=config.weight_decay,
        betas=(0.9, 0.98),
        eps=1e-8
    )
    
    # Warmup and cosine decay scheduler
    def warmup_cosine_schedule(step):
        if step < config.warmup_steps:
            return float(step) / float(max(1, config.warmup_steps))
        progress = float(step - config.warmup_steps) / float(max(1, max_steps - config.warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_schedule)
    
    # Training state
    step = 0
    best_val_loss = float('inf')
    val_freq = 50  # Validate every 50 steps
    
    # Get vocab size for loss computation
    vocab_size = train_loader.dataset.vocab_size
    
    # Time tracking
    start_time = time.time()
    
    # Gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # Generate evaluation examples
    eval_examples = generate_multiplication_examples(dataset, num_examples=10, device=device)
    
    try:
        # Main training loop
        model.train()
        pbar = tqdm(total=max_steps, desc=f"Training {model_name}")
        
        for epoch in range(100):  # Max 100 epochs
            for batch_idx, (inp, tgt, inp_lens, tgt_lens) in enumerate(train_loader):
                if step >= max_steps:
                    break
                
                inp, tgt = inp.to(device), tgt.to(device)
                decoder_input = tgt[:, :-1]
                decoder_target = tgt[:, 1:]
                
                # Clear gradients
                optimizer.zero_grad()
                
                if device.type == 'cuda':
                    # Use mixed precision for faster training on GPU
                    with torch.cuda.amp.autocast():
                        output = model(inp, decoder_input)
                        loss = criterion(output.reshape(-1, vocab_size), decoder_target.reshape(-1))
                    
                    # Scale gradients
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
                        print(f"Warning: NaN in output at step {step}, skipping batch.")
                        continue
                    
                    loss = criterion(output.reshape(-1, vocab_size), decoder_target.reshape(-1))
                    
                    # Check for NaN in loss
                    if torch.isnan(loss).any():
                        print(f"Warning: NaN in loss at step {step}, skipping batch.")
                        continue
                    
                    loss.backward()
                    
                    # Check for NaN in gradients
                    has_nan_grad = False
                    for name, param in model.named_parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            print(f"Warning: NaN gradient in {name} at step {step}")
                            has_nan_grad = True
                            break
                    
                    if has_nan_grad:
                        # Skip this batch if NaN gradients are detected
                        optimizer.zero_grad()
                        continue
                    
                    # Clip gradients to prevent explosions
                    nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                
                scheduler.step()
                
                # Log metrics
                lr = scheduler.get_last_lr()[0]
                writer.add_scalar('train/loss', loss.item(), step)
                writer.add_scalar('train/lr', lr, step)
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.6f}")
                pbar.update(1)
                
                # Validate periodically
                if step > 0 and step % val_freq == 0:
                    val_loss, val_accuracy = evaluate(
                        model=model,
                        data_loader=val_loader,
                        criterion=criterion,
                        dataset=dataset,
                        device=device,
                        vocab_size=vocab_size,
                        desc=f"Val {model_name}",
                        examples=eval_examples
                    )
                    
                    writer.add_scalar('val/loss', val_loss, step)
                    writer.add_scalar('val/accuracy', val_accuracy, step)
                    print(f"\n{model_name} - Step {step}/{max_steps} - Val Loss: {val_loss:.6f} - Accuracy: {val_accuracy:.2%}")
                    
                    # Save best model
                    if val_loss < best_val_loss and not np.isnan(val_loss):
                        best_val_loss = val_loss
                        os.makedirs(f"checkpoints/{model_name}", exist_ok=True)
                        torch.save({
                            'step': step,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': val_loss,
                            'val_accuracy': val_accuracy,
                        }, f"checkpoints/{model_name}/{model_name}_best.pt")
                    
                    # Return to training mode
                    model.train()
                    
                    # Early stopping if we reach excellent performance
                    if val_loss < 0.01 and val_accuracy > 0.95:
                        print(f"Reached excellent validation performance, stopping early")
                        break
                
                step += 1
            
            if step >= max_steps:
                break
                
    except KeyboardInterrupt:
        print(f"Training {model_name} interrupted")
        
    # Final validation
    final_val_loss, final_accuracy = evaluate(
        model=model,
        data_loader=val_loader,
        criterion=criterion,
        dataset=dataset,
        device=device,
        vocab_size=vocab_size,
        desc=f"Final {model_name} validation",
        examples=eval_examples
    )
    
    training_time = time.time() - start_time
    
    # Print summary
    print(f"\n{model_name} Training Summary:")
    print(f"Steps completed: {step}/{max_steps}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final validation loss: {final_val_loss:.6f}")
    print(f"Final accuracy: {final_accuracy:.2%}")
    
    writer.close()
    return final_val_loss, final_accuracy, step, training_time

def main():
    # Enable MPS fallback
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Stable comparison between transformer architectures with accuracy metrics")
    parser.add_argument("--d-model", type=int, default=64, help="Model dimension (default: 64)")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of layers (default: 2)")
    parser.add_argument("--num-latent", type=int, default=4, help="Number of latent tokens (default: 4)")
    parser.add_argument("--bottleneck-factor", type=float, default=1.0, help="Bottleneck factor (default: 1.0)")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum training steps (default: 500)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--min-digits", type=int, default=1, help="Minimum number of digits (default: 1)")
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
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
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
    
    # Train SimpleTransformer
    print("\nTraining SimpleTransformer...")
    simple_log_dir = f"runs/stable_comparison_acc/simple_d{d_model}_l{num_layers}"
    simple_loss, simple_accuracy, simple_steps, simple_time = train_model(
        model_name=f"simple_d{d_model}_l{num_layers}",
        model=simple_transformer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        log_dir=simple_log_dir,
        dataset=train_dataset,
        max_steps=args.max_steps
    )
    
    # Train LatentTransformer
    print("\nTraining LatentTransformer...")
    latent_log_dir = f"runs/stable_comparison_acc/latent_d{d_model}_l{num_layers}_n{num_latent}"
    latent_loss, latent_accuracy, latent_steps, latent_time = train_model(
        model_name=f"latent_d{d_model}_l{num_layers}_n{num_latent}",
        model=latent_transformer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        log_dir=latent_log_dir,
        dataset=train_dataset,
        max_steps=args.max_steps
    )
    
    # Print comparison
    print("\nFinal Comparison:")
    print("=" * 50)
    
    print(f"SimpleTransformer: {simple_loss:.6f} loss, {simple_accuracy:.2%} accuracy, {simple_params:,} parameters, {simple_time:.2f}s training time")
    print(f"LatentTransformer: {latent_loss:.6f} loss, {latent_accuracy:.2%} accuracy, {latent_params:,} parameters, {latent_time:.2f}s training time")
    
    # Calculate efficiency metrics if possible
    if simple_loss != float('inf') and latent_loss != float('inf') and not np.isnan(simple_loss) and not np.isnan(latent_loss):
        # Parameter efficiency (lower is better: loss * num_params)
        param_efficiency_simple = simple_loss * simple_params
        param_efficiency_latent = latent_loss * latent_params
        
        # Time efficiency (lower is better: loss * training_time)
        time_efficiency_simple = simple_loss * simple_time
        time_efficiency_latent = latent_loss * latent_time
        
        # Accuracy efficiency (higher is better: accuracy / num_params)
        acc_param_efficiency_simple = simple_accuracy / simple_params if simple_params > 0 else 0
        acc_param_efficiency_latent = latent_accuracy / latent_params if latent_params > 0 else 0
        
        print("\nEfficiency Metrics:")
        
        if param_efficiency_latent < param_efficiency_simple:
            ratio = param_efficiency_simple / param_efficiency_latent
            print(f"LatentTransformer is {ratio:.2f}x more parameter-efficient (loss*params)")
        else:
            ratio = param_efficiency_latent / param_efficiency_simple
            print(f"SimpleTransformer is {ratio:.2f}x more parameter-efficient (loss*params)")
        
        if time_efficiency_latent < time_efficiency_simple:
            ratio = time_efficiency_simple / time_efficiency_latent
            print(f"LatentTransformer is {ratio:.2f}x more time-efficient (loss*time)")
        else:
            ratio = time_efficiency_latent / time_efficiency_simple
            print(f"SimpleTransformer is {ratio:.2f}x more time-efficient (loss*time)")
            
        if acc_param_efficiency_latent > acc_param_efficiency_simple:
            ratio = acc_param_efficiency_latent / acc_param_efficiency_simple
            print(f"LatentTransformer is {ratio:.2f}x more accuracy-per-parameter efficient")
        else:
            ratio = acc_param_efficiency_simple / acc_param_efficiency_latent
            print(f"SimpleTransformer is {ratio:.2f}x more accuracy-per-parameter efficient")
    else:
        print("Cannot calculate efficiency metrics due to invalid loss values")
    
    print("\nTo view training curves with accuracy metrics, run:")
    print(f"tensorboard --logdir=runs/stable_comparison_acc")

if __name__ == "__main__":
    main() 