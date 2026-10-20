"""
Anchor-Aware & Entropy-Guided Training-Free KV Cache Compression for Reasoning LLMs.
Author: Fatih Can Göçülü
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Tuple
import torch
import torch.nn.functional as F


DEFAULT_REASONING_ANCHORS: Tuple[str, ...] = (
    "Wait",
    "Therefore",
    "Thus",
    "Hence",
    "Let's",
    " check",
    " verify",
    "Ancak",
    "Sonuç",
    "Dolayısıyla",
    "Öyleyse",
    "=",
    "\\frac",
    "\\boxed",
)


@dataclass
class AnchorAwarePress:
    """
    Training-free KV cache pruning press designed for long Chain-of-Thought (CoT) decoding.

    Preserves:
      1. Initial attention sink tokens (first `num_sink_tokens`)
      2. Recent local observation window (last `window_size` tokens)
      3. Structural reasoning anchors (numeric tokens, equations, discourse markers)
      4. High cumulative attention tokens with low cosine key redundancy
    """

    compression_ratio: float = 0.80
    num_sink_tokens: int = 4
    window_size: int = 64
    alpha_attention: float = 1.0
    beta_anchor: float = 2.5
    gamma_redundancy: float = 0.65

    def detect_anchor_mask(
        self,
        input_ids: torch.Tensor,
        anchor_token_ids: Optional[Set[int]] = None,
        digit_token_ids: Optional[Set[int]] = None,
    ) -> torch.Tensor:
        """Return a binary mask [batch, seq_len] marking structural anchor tokens."""
        mask = torch.zeros_like(input_ids, dtype=torch.float32)
        if anchor_token_ids:
            for tid in anchor_token_ids:
                mask = torch.where(input_ids == tid, torch.ones_like(mask), mask)
        if digit_token_ids:
            for tid in digit_token_ids:
                mask = torch.where(input_ids == tid, torch.ones_like(mask), mask)
        return mask

    def score_kv(
        self,
        keys: torch.Tensor,
        attentions: torch.Tensor,
        anchor_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute composite retention score for each token position.

        Args:
            keys: [batch, num_kv_heads, seq_len, head_dim]
            attentions: [batch, num_heads, q_len, seq_len]
            anchor_mask: [batch, seq_len] in {0.0, 1.0}
        Returns:
            scores: [batch, num_kv_heads, seq_len]
        """
        bsz, num_kv_heads, seq_len, _ = keys.shape
        num_heads = attentions.shape[1]
        group_size = max(1, num_heads // num_kv_heads)

        # Aggregate attention over recent queries and head groups
        attn_mean = attentions.mean(dim=2)  # [B, H, S]
        if group_size > 1:
            attn_mean = attn_mean.view(bsz, num_kv_heads, group_size, seq_len).mean(dim=2)

        # Normalize keys to measure cosine redundancy against the recent window
        k_norm = F.normalize(keys.float(), p=2, dim=-1)
        win = min(self.window_size, seq_len)
        recent_k = k_norm[:, :, -win:, :]  # [B, H_kv, W, D]
        sim_matrix = torch.matmul(k_norm, recent_k.transpose(-1, -2))  # [B, H_kv, S, W]
        max_redundancy, _ = sim_matrix.max(dim=-1)  # [B, H_kv, S]

        anchor_bonus = anchor_mask.unsqueeze(1).expand(bsz, num_kv_heads, seq_len)
        scores = (
            self.alpha_attention * attn_mean
            + self.beta_anchor * anchor_bonus
            - self.gamma_redundancy * max_redundancy
        )

        # Always retain attention sinks and the most recent sliding window
        scores[:, :, : self.num_sink_tokens] = 1e4
        scores[:, :, -win:] = 1e4
        return scores

    def compress(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        anchor_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prune KV cache along sequence dimension to (1 - compression_ratio) budget."""
        bsz, num_kv_heads, seq_len, head_dim = keys.shape
        keep_len = max(
            self.num_sink_tokens + self.window_size,
            int(seq_len * (1.0 - self.compression_ratio)),
        )
        if keep_len >= seq_len:
            idx = torch.arange(seq_len, device=keys.device).view(1, 1, seq_len).expand(bsz, num_kv_heads, seq_len)
            return keys, values, idx

        scores = self.score_kv(keys, attentions, anchor_mask)
        _, top_indices = torch.topk(scores, k=keep_len, dim=-1, largest=True)
        top_indices, _ = torch.sort(top_indices, dim=-1)

        gather_idx = top_indices.unsqueeze(-1).expand(bsz, num_kv_heads, keep_len, head_dim)
        pruned_keys = torch.gather(keys, dim=2, index=gather_idx)
        pruned_values = torch.gather(values, dim=2, index=gather_idx)
        return pruned_keys, pruned_values, top_indices
