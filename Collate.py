import torch
from typing import List, Tuple, Any

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

