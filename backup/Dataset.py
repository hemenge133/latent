import torch
from torch.utils.data import Dataset, DataLoader
import random
from typing import Tuple, Dict, List
from config import TrainingConfig
import numpy as np


class MultiplicationDataset(Dataset):
    # Class variable to store all generated problems for consistency
    _all_problems = {}
    _train_problems = {}
    _val_problems = {}
    _test_problems = {}

    @classmethod
    def _generate_systematic_problems(
        cls, min_value, max_value, split_ratio=(0.8, 0.1, 0.1), seed=42
    ):
        """Generate all possible multiplication problems and split them systematically"""
        # Create a unique key for this configuration
        config_key = f"{min_value}_{max_value}_{split_ratio}_{seed}"

        # If we've already generated problems for this configuration, return them
        if config_key in cls._all_problems:
            return (
                cls._train_problems[config_key],
                cls._val_problems[config_key],
                cls._test_problems[config_key],
            )

        # Generate all possible problems
        all_problems = []
        for a in range(min_value, max_value + 1):
            for b in range(min_value, max_value + 1):
                all_problems.append((a, b, a * b))

        # Set seed for reproducibility
        rng = random.Random(seed)

        # Calculate split sizes
        total_problems = len(all_problems)
        train_size = int(total_problems * split_ratio[0])
        val_size = int(total_problems * split_ratio[1])

        # First, sort by product magnitude (difficulty)
        all_problems.sort(key=lambda x: x[2])

        # Then create balanced splits by strided sampling
        train_problems = []
        val_problems = []
        test_problems = []

        # We want difficulty to be evenly distributed
        # To achieve this, we'll use a systematic sampling approach:
        # 1. Group problems by product magnitude buckets
        # 2. Sample from each bucket for each split

        # Create difficulty buckets
        num_buckets = 10  # Divide into 10 difficulty levels
        bucket_size = len(all_problems) // num_buckets
        buckets = [
            all_problems[i : i + bucket_size]
            for i in range(0, len(all_problems), bucket_size)
        ]

        # Ensure last bucket has remaining problems
        if len(buckets) > 1 and len(buckets[-1]) < bucket_size // 2:
            buckets[-2].extend(buckets[-1])
            buckets.pop()

        # Now sample from each bucket to create splits
        for bucket in buckets:
            # Shuffle within each bucket
            rng.shuffle(bucket)

            # Calculate sizes for this bucket
            bucket_train_size = int(len(bucket) * split_ratio[0])
            bucket_val_size = int(len(bucket) * split_ratio[1])

            # Split bucket
            train_problems.extend(bucket[:bucket_train_size])
            val_problems.extend(
                bucket[bucket_train_size : bucket_train_size + bucket_val_size]
            )
            test_problems.extend(bucket[bucket_train_size + bucket_val_size :])

        # Final shuffle to mix difficulties
        rng.shuffle(train_problems)
        rng.shuffle(val_problems)
        rng.shuffle(test_problems)

        # Store for future use
        cls._all_problems[config_key] = all_problems
        cls._train_problems[config_key] = train_problems
        cls._val_problems[config_key] = val_problems
        cls._test_problems[config_key] = test_problems

        return train_problems, val_problems, test_problems

    def __init__(
        self,
        num_samples: int,
        split: str = "train",
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        min_value: int = None,
        max_value: int = None,
    ):
        """
        Create a dataset for multiplication task with train/val/test splits

        Args:
            num_samples: Total number of samples to generate (may be overridden by systematic generation)
            split: One of 'train', 'val', 'test'
            split_ratio: Tuple of (train_ratio, val_ratio, test_ratio)
            seed: Random seed for reproducibility
            min_value: Minimum value for multiplication operands
            max_value: Maximum value for multiplication operands
        """
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed

        # Get config for min/max values if not provided
        if min_value is None or max_value is None:
            config = TrainingConfig()
            self.min_value = config.min_value if min_value is None else min_value
            self.max_value = config.max_value if max_value is None else max_value
        else:
            self.min_value = min_value
            self.max_value = max_value

        # Define tokenization
        self.tokenizer = {c: i for i, c in enumerate("0123456789*")}
        self.vocab_size = len(self.tokenizer)
        self.inv_tokenizer = {v: k for k, v in self.tokenizer.items()}

        # Generate systematic problems
        train_probs, val_probs, test_probs = self._generate_systematic_problems(
            self.min_value, self.max_value, split_ratio, seed
        )

        # Select problems based on split
        if split == "train":
            self.problems = train_probs
        elif split == "val":
            self.problems = val_probs
        elif split == "test":
            self.problems = test_probs
        else:
            raise ValueError(
                f"Invalid split: {split}. Must be one of ['train', 'val', 'test']"
            )

        # Handle num_samples if it's less than the full set
        self.num_samples = min(num_samples, len(self.problems))
        if num_samples < len(self.problems):
            # Use a deterministic sample if requested size is smaller
            rng = random.Random(seed)
            self.problems = rng.sample(self.problems, self.num_samples)

        # Initialize RNG for any remaining random operations
        self.rng = random.Random(seed)

        print(
            f"Using {split} dataset with range {self.min_value}-{self.max_value}: {self.num_samples} problems"
        )

    def encode(self, s: str) -> List[int]:
        """Convert string to token indices"""
        return [self.tokenizer[c] for c in s]

    def decode(self, tokens: List[int]) -> str:
        """Convert token indices to string"""
        # Ensure tokens are converted to Python integers if they are tensor elements
        return "".join(
            [
                self.inv_tokenizer[int(t) if isinstance(t, torch.Tensor) else t]
                for t in tokens
            ]
        )

    def __len__(self) -> int:
        """Return the number of samples in this split"""
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a specific sample by index"""
        if idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset with {len(self)} samples"
            )

        # Get problem from precomputed list
        a, b, result = self.problems[idx]
        inp = f"{a}*{b}"
        out = str(result)

        # Convert strings to tensors
        inp_tokens = torch.tensor(self.encode(inp), dtype=torch.long)
        out_tokens = torch.tensor(self.encode(out), dtype=torch.long)

        return inp_tokens, out_tokens
