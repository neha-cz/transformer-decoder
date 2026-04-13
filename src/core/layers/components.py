"""Reusable neural network primitives: RMSNorm, SwiGLU."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Normalises the last dimension without re-centering (no bias subtraction).

    Args:
        dim: Dimension of the last axis to normalise.
        eps: Small constant for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.scale


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network (Shazeer, 2020).

    Three linear projections with SiLU-gated activation and no bias,
    following LLaMA / Mistral conventions.

    Args:
        d_model: Input and output dimension.
        d_ff: Nominal intermediate dimension (actual hidden size is
            ``round_up(2/3 * d_ff, 8)``).
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        d_hidden = (int(2 * d_ff / 3) + 7) // 8 * 8
        self.w_gate = nn.Linear(d_model, d_hidden, bias=False)
        self.w_up = nn.Linear(d_model, d_hidden, bias=False)
        self.w_down = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


__all__ = ["RMSNorm", "SwiGLU"]
