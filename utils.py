import torch
import torch.nn as nn
from tqdm import tqdm
import os
import time
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Any, Optional, Callable, List

def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    vocab_size: int,
    desc: str = "Validation"
) -> float:
    """
    Evaluate the model on the given data loader.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader with evaluation data
        criterion: Loss function
        device: Device to run evaluation on
        vocab_size: Size of vocabulary for reshape
        desc: Description for progress bar
        
    Returns:
        float: Average loss on the dataset
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inp, tgt, inp_lens, tgt_lens in tqdm(data_loader, desc=desc):
            inp, tgt = inp.to(device), tgt.to(device)
            decoder_input = tgt[:, :-1]
            decoder_target = tgt[:, 1:]

            output = model(inp, decoder_input)
            loss = criterion(output.reshape(-1, vocab_size), decoder_target.reshape(-1))
            total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    save_path: str,
    is_best: bool = False
) -> None:
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch
        train_loss: Training loss
        val_loss: Validation loss
        save_path: Path to save the checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: str,
    device: torch.device
) -> Dict[str, Any]:
    """
    Load a model checkpoint
    
    Args:
        model: Model to load the state into
        optimizer: Optimizer to load the state into (optional)
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model onto
        
    Returns:
        Dict containing checkpoint information
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint

def warmup_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    base_lr: float
) -> Callable[[int], None]:
    """
    Create a learning rate scheduler with warmup
    
    Args:
        optimizer: Optimizer to adjust learning rate for
        warmup_steps: Number of warmup steps
        base_lr: Base learning rate after warmup
        
    Returns:
        Function to update learning rate based on step
    """
    def update_lr(step: int) -> None:
        if step < warmup_steps:
            lr = base_lr * (step + 1) / warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    return update_lr

def setup_training_dir(model_name: str) -> str:
    """
    Create a directory for saving model checkpoints
    
    Args:
        model_name: Name of the model
        
    Returns:
        Path to the created directory
    """
    save_dir = os.path.join("checkpoints", model_name)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def calculate_model_size(model: nn.Module) -> int:
    """
    Calculate and return the number of trainable parameters in the model
    
    Args:
        model: The model to analyze
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_time(seconds: float) -> str:
    """
    Format time in seconds to a readable string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    
    if h > 0:
        return f"{h:.0f}h {m:.0f}m {s:.0f}s"
    elif m > 0:
        return f"{m:.0f}m {s:.0f}s"
    else:
        return f"{s:.1f}s"

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    """
    Custom collate function for padding sequences in a batch
    
    Args:
        batch: List of (input, target) tensor pairs
        
    Returns:
        inputs_padded: Padded input sequences [batch_size, max_input_len]
        targets_padded: Padded target sequences [batch_size, max_target_len]
        input_lens: List of original input sequence lengths
        target_lens: List of original target sequence lengths
    """
    inputs, targets = zip(*batch)
    input_lens = [len(seq) for seq in inputs]
    target_lens = [len(seq) for seq in targets]

    inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    
    return inputs_padded, targets_padded, input_lens, target_lens 