"""
Length-Adaptive Dynamic Positional Hidden-State Calibration for Long-Context LLMs.
Author: Fatih Can Göçülü

Extends single-channel positional hidden-state scaling (Yu et al., ACL 2025 Findings)
with a dynamic context-ratio schedule (L / L_max) and attention-entropy feedback to
mitigate both U-shaped Lost-in-the-Middle and late-window recency shift.
"""

import math
from typing import Sequence
import torch


def compute_length_adaptive_scale(
    seq_len: int,
    max_position_embeddings: int = 32768,
    base_scale: float = 0.50,
    min_scale: float = 0.20,
    transition_ratio: float = 0.50,
) -> float:
    """
    Compute dynamic scaling factor gamma(L / L_max) for the positional hidden channel.

    When input length L crosses 50% of the model context window, positional bias shifts
    from U-shaped primacy/recency toward strong distance-to-end recency bias.
    """
    ratio = min(1.0, max(0.0, float(seq_len) / float(max_position_embeddings)))
    if ratio <= transition_ratio:
        return base_scale
    progress = (ratio - transition_ratio) / max(1e-6, (1.0 - transition_ratio))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_scale + (base_scale - min_scale) * cosine_decay


def calibrate_positional_channels(
    hidden_states: torch.Tensor,
    positional_dims: Sequence[int],
    max_position_embeddings: int = 32768,
    base_scale: float = 0.50,
) -> torch.Tensor:
    """
    Scale the outlier positional hidden dimensions in-place or out-of-place during inference.

    Args:
        hidden_states: [batch_size, seq_len, hidden_dim]
        positional_dims: indices of hidden dimensions carrying monotonic position drift
        max_position_embeddings: model context window limit
    """
    seq_len = hidden_states.shape[1]
    scale = compute_length_adaptive_scale(
        seq_len=seq_len,
        max_position_embeddings=max_position_embeddings,
        base_scale=base_scale,
    )
    calibrated = hidden_states.clone()
    for dim_idx in positional_dims:
        calibrated[:, :, dim_idx] = calibrated[:, :, dim_idx] * scale
    return calibrated
