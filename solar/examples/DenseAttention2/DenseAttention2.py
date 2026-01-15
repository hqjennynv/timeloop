"""Dense Attention example model.

This implements standard dense attention which computes the full attention matrix:
    attention = softmax(Q @ K^T / sqrt(d_k)) @ V

Shape: (batch_size, num_heads, seq_len, head_dim) for Q, K, V
Output: (batch_size, num_heads, seq_len, head_dim)

Implements standard dense attention pattern
"""

import torch
import torch.nn as nn


class DenseAttention(nn.Module):
    """Standard dense attention - computes full attention matrix."""

    def __init__(self, causal: bool = False):
        super().__init__()
        self.causal = causal

    def forward(
        self,
        inp: torch.Tensor,
        wq: torch.Tensor,
        wk: torch.Tensor,
    ) -> torch.Tensor:
        """Apply dense attention.

        Args:
            query: (batch_size, num_heads, seq_len, head_dim)
            key: (batch_size, num_heads, seq_len, head_dim)
            value: (batch_size, num_heads, seq_len, head_dim)

        Returns:
            output: (batch_size, num_heads, seq_len, head_dim)
        """
        q = torch.matmul(inp, wq)
        k = torch.matmul(inp, wk)
        k = k.transpose(-2, -1)
        qk = torch.matmul(q, k)

        return qk


class Model(nn.Module):
    """Wrapper model for dense attention."""

    def __init__(self):
        super().__init__()
        self.attention = DenseAttention(causal=False)

    def forward(self, i: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor) -> torch.Tensor:
        return self.attention(i, wq, wk)


def get_inputs():
    """Get sample inputs for the model."""
    B, H, M, E, D = 2, 4, 32, 64, 64  # batch, heads, seq_len, weight_hidden_dim, head_dim
    inp = torch.randn(B, M, D, dtype=torch.float32)
    wq = torch.randn(H, 1, D, E, dtype=torch.float32)
    wk = torch.randn(H, 1, D, E, dtype=torch.float32)
    return [inp, wq, wk]


def get_init_inputs():
    """Get initialization inputs (none needed)."""
    return []

