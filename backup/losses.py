"""
Loss functions for transformer-based models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# Create label smoothing loss for better generalization
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, ignore_index=0, reduction="none"):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.criterion = nn.KLDivLoss(reduction="none")

    def forward(self, inputs, targets):
        # Flatten the inputs and targets
        inputs = inputs.view(-1, inputs.size(-1))
        targets = targets.view(-1)

        # Create mask for non-padding tokens
        mask = (targets != self.ignore_index).float()

        # Create smoothed targets
        targets_non_pad = targets * mask
        n_class = inputs.size(1)
        one_hot = torch.zeros_like(inputs).scatter(
            1, targets_non_pad.unsqueeze(1).long(), 1
        )
        one_hot = one_hot * (1 - self.smoothing) + (1 - one_hot) * self.smoothing / (
            n_class - 1
        )

        # Apply log softmax to inputs
        log_probs = F.log_softmax(inputs, dim=1)

        # Calculate KL divergence loss
        loss = self.criterion(log_probs, one_hot)
        loss = loss.sum(dim=1)

        # Apply mask to exclude padding tokens
        loss = loss * mask

        # Return per-token loss for later reshaping
        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return loss.sum()
        else:  # mean
            return loss.sum() / mask.sum().clamp(min=1e-8)


class SequenceAccuracyLoss(nn.Module):
    """Loss function that directly optimizes for exact sequence matches"""

    def __init__(self, padding_idx=0, ce_weight=0.7, seq_weight=0.3, smoothing=0.05):
        super(SequenceAccuracyLoss, self).__init__()
        self.padding_idx = padding_idx
        self.ce_weight = ce_weight
        self.seq_weight = seq_weight
        self.label_smoothing = LabelSmoothingLoss(
            smoothing=smoothing, ignore_index=padding_idx, reduction="none"
        )

    def forward(self, outputs, targets):
        batch_size = outputs.size(0)
        seq_len = outputs.size(1)

        # Reshape for token predictions
        flat_outputs = outputs.reshape(-1, outputs.size(-1))
        flat_targets = targets.reshape(-1)

        # Get base token-level CE loss
        token_loss = self.label_smoothing(flat_outputs, flat_targets)
        token_loss = token_loss.reshape(batch_size, seq_len)

        # Create mask for non-padding tokens
        mask = (targets != self.padding_idx).float()

        # Get predictions
        _, predicted = torch.max(outputs, dim=-1)

        # Calculate per-sequence correctness (1 if all tokens correct, 0 otherwise)
        token_correct = (predicted == targets).float() * mask
        seq_lengths = mask.sum(dim=1)
        seq_correct_tokens = token_correct.sum(dim=1)

        # Calculate sequence correctness as percentage of tokens correct,
        # which is more gradual than binary correct/incorrect
        seq_correctness = seq_correct_tokens / seq_lengths.clamp(min=1e-8)

        # For metrics/logging, still calculate binary sequence correctness
        seq_correct_binary = (seq_correct_tokens == seq_lengths).float()

        # Compute per-token CE loss (averaged per sequence)
        masked_token_loss = (token_loss * mask).sum(dim=1) / seq_lengths.clamp(min=1e-8)

        # Use a smoother penalty that scales with sequence correctness
        # Instead of binary 1.0/2.0, use a continuous scale
        seq_penalty = 1.0 + self.seq_weight * (1.0 - seq_correctness)

        # Combined loss: token-level CE loss adjusted by sequence correctness
        combined_loss = masked_token_loss * seq_penalty

        # Return mean loss across the batch and binary sequence accuracy for metrics
        return combined_loss.mean(), seq_correct_binary.mean()
