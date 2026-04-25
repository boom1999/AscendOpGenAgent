"""AscendC-optimised RotaryPositionEmbedding (RoPE).

Supports mode 0 (half-rotation) and mode 1 (interleave).
Mode 1 is handled by deinterleaving inputs in the host, calling the same
half-rotation kernel, then reinterleaving the output.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import _rotary_pos_emb_ext as _ext  # noqa: E402


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mode: int = 0,
    ) -> torch.Tensor:
        B, H, S, D = x.shape

        # Broadcast cos/sin from [1, 1, S, D] to [B, H, S, D]
        cos_exp = cos.expand(B, H, S, D).contiguous()
        sin_exp = sin.expand(B, H, S, D).contiguous()

        # For mode 1 (interleave): deinterleave even/odd into contiguous halves
        if mode == 1:
            x = x.view(B, H, S, D // 2, 2).permute(0, 1, 2, 4, 3).contiguous().view(B, H, S, D)
            cos_exp = cos_exp.view(B, H, S, D // 2, 2).permute(0, 1, 2, 4, 3).contiguous().view(B, H, S, D)
            sin_exp = sin_exp.view(B, H, S, D // 2, 2).permute(0, 1, 2, 4, 3).contiguous().view(B, H, S, D)

        # Flatten to [M, D]
        M = B * H * S
        x_flat = x.reshape(M, D).contiguous()
        cos_flat = cos_exp.reshape(M, D).contiguous()
        sin_flat = sin_exp.reshape(M, D).contiguous()

        # Call kernel
        out_flat = _ext.run_rotary_pos_emb(x_flat, cos_flat, sin_flat)

        # Reshape back to [B, H, S, D]
        out = out_flat.reshape(B, H, S, D)

        # For mode 1: reinterleave result
        if mode == 1:
            out = out.view(B, H, S, 2, D // 2).permute(0, 1, 2, 4, 3).contiguous().view(B, H, S, D)

        return out
