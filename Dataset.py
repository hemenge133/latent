import torch
from torch.utils.data import Dataset, DataLoader
import random
from typing import Tuple, Dict, List
from config import TrainingConfig

class MultiplicationDataset(Dataset):
    def __init__(self, num_samples: int, split: str = 'train', split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42, min_value: int = None, max_value: int = None):
        """
        Create a dataset for multiplication task with train/val/test splits
        
        Args:
            num_samples: Total number of samples to generate
            split: One of 'train', 'val', 'test'
            split_ratio: Tuple of (train_ratio, val_ratio, test_ratio)
            seed: Random seed for reproducibility
            min_value: Minimum value for multiplication operands
            max_value: Maximum value for multiplication operands
        """
        self.num_samples = num_samples
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed
        
        # Determine split start and end indices
        assert sum(split_ratio) == 1.0, "Split ratios must sum to 1"
        train_end = int(num_samples * split_ratio[0])
        val_end = train_end + int(num_samples * split_ratio[1])
        
        if split == 'train':
            self.start_idx = 0
            self.end_idx = train_end
        elif split == 'val':
            self.start_idx = train_end
            self.end_idx = val_end
        elif split == 'test':
            self.start_idx = val_end
            self.end_idx = num_samples
        else:
            raise ValueError(f"Invalid split: {split}. Must be one of ['train', 'val', 'test']")
        
        # Calculate actual samples in this split
        self.split_samples = self.end_idx - self.start_idx
        
        # Define tokenization
        self.tokenizer = {c: i for i, c in enumerate("0123456789*")}
        self.vocab_size = len(self.tokenizer)
        self.inv_tokenizer = {v: k for k, v in self.tokenizer.items()}
        
        # Initialize the random generator with a seed based on split
        # This ensures deterministic behavior across different splits
        self.rng = random.Random(seed)
        
        # Get config for min/max values if not provided
        if min_value is None or max_value is None:
            config = TrainingConfig()
            self.min_value = config.min_value if min_value is None else min_value
            self.max_value = config.max_value if max_value is None else max_value
        else:
            self.min_value = min_value
            self.max_value = max_value
        
        print(f"Using {split} dataset with range {self.min_value}-{self.max_value}")
        
        # Precompute a few samples for validation and testing if split isn't train
        # This makes sure val and test sets remain consistent between runs
        self.precomputed_samples = None
        if split != 'train':
            # For validation and test sets, we precompute all samples to ensure consistency
            self._precompute_samples()
    
    def _precompute_samples(self):
        """Precompute samples for validation and test sets to ensure consistency"""
        # Save current random state
        state = self.rng.getstate()
        
        # Create a completely different seed for each split to avoid any chance of leakage
        if self.split == 'val':
            # For validation, use a different seed range 
            self.rng.seed(self.seed + 10000)
            # To further ensure separation, we can use different number ranges for different splits
            min_val_adjusted = self.min_value 
            max_val_adjusted = int(self.min_value + (self.max_value - self.min_value) * 0.5)
        elif self.split == 'test':
            # For test, use yet another seed range
            self.rng.seed(self.seed + 20000)
            # Use the upper half of the number range for test
            min_val_adjusted = int(self.min_value + (self.max_value - self.min_value) * 0.5)
            max_val_adjusted = self.max_value
        else:
            # Should never happen as we only precompute for val and test
            min_val_adjusted = self.min_value
            max_val_adjusted = self.max_value
            
        print(f"Precomputing {self.split} set with range {min_val_adjusted}-{max_val_adjusted}")
        
        # Generate samples for this split
        self.precomputed_samples = []
        for i in range(self.split_samples):
            a = self.rng.randint(min_val_adjusted, max_val_adjusted)
            b = self.rng.randint(min_val_adjusted, max_val_adjusted)
            input_str = f"{a}*{b}"
            output_str = str(a*b)
            self.precomputed_samples.append((input_str, output_str))
        
        # Restore random state
        self.rng.setstate(state)

    def encode(self, s: str) -> List[int]:
        """Convert string to token indices"""
        return [self.tokenizer[c] for c in s]

    def decode(self, tokens: List[int]) -> str:
        """Convert token indices to string"""
        # Ensure tokens are converted to Python integers if they are tensor elements
        return ''.join([self.inv_tokenizer[int(t) if isinstance(t, torch.Tensor) else t] for t in tokens])

    def __len__(self) -> int:
        """Return the number of samples in this split"""
        return self.split_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a specific sample by index"""
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self)} samples")
        
        # For validation and test, use precomputed samples
        if self.precomputed_samples is not None:
            inp, out = self.precomputed_samples[idx]
        else:
            # For training, generate samples on-the-fly
            # Use a seed based on the global index to ensure deterministic behavior
            sample_seed = self.seed + self.start_idx + idx
            saved_state = self.rng.getstate()
            self.rng.seed(sample_seed)
            
            a, b = self.rng.randint(self.min_value, self.max_value), self.rng.randint(self.min_value, self.max_value)
            inp = f"{a}*{b}"
            out = str(a*b)
            
            # Restore random state
            self.rng.setstate(saved_state)
        
        # Convert strings to tensors
        inp_tokens = torch.tensor(self.encode(inp), dtype=torch.long)
        out_tokens = torch.tensor(self.encode(out), dtype=torch.long)
        
        return inp_tokens, out_tokens

