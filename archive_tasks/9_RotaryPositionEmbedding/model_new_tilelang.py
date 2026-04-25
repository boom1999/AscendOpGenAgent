"""TileLang-optimised RotaryPositionEmbedding (RoPE).

Supports mode 0 (half-rotation) and mode 1 (interleave).
Mode 1 is handled by deinterleaving inputs in the host, calling the same
half-rotation kernel, then reinterleaving the output.
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from design.tile_level.rotary_pos_emb import rotary_pos_emb as _tl_kernel

_DTYPE_TO_STR = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}


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
        dtype_str = _DTYPE_TO_STR[x.dtype]
        B, H, S, D = x.shape

        # Broadcast cos/sin from [1, 1, S, D] to [B, H, S, D]
        cos_exp = cos.expand(B, H, S, D).contiguous()
        sin_exp = sin.expand(B, H, S, D).contiguous()

        # For mode 1 (interleave): deinterleave even/odd into contiguous halves
        if mode == 1:
            # [B, H, S, D] -> [B, H, S, D//2, 2] -> [B, H, S, 2, D//2] -> [B, H, S, D]
            x = x.view(B, H, S, D // 2, 2).permute(0, 1, 2, 4, 3).contiguous().view(B, H, S, D)
            cos_exp = cos_exp.view(B, H, S, D // 2, 2).permute(0, 1, 2, 4, 3).contiguous().view(B, H, S, D)
            sin_exp = sin_exp.view(B, H, S, D // 2, 2).permute(0, 1, 2, 4, 3).contiguous().view(B, H, S, D)

        # Flatten to [M, D]
        M = B * H * S
        x_flat = x.reshape(M, D).contiguous()
        cos_flat = cos_exp.reshape(M, D).contiguous()
        sin_flat = sin_exp.reshape(M, D).contiguous()

        # Call kernel
        kernel = _tl_kernel(M, D, dtype=dtype_str)
        out_flat = kernel(x_flat, cos_flat, sin_flat)

        # Reshape back to [B, H, S, D]
        out = out_flat.reshape(B, H, S, D)

        # For mode 1: reinterleave result
        if mode == 1:
            # [B, H, S, D] -> [B, H, S, 2, D//2] -> [B, H, S, D//2, 2] -> [B, H, S, D]
            out = out.view(B, H, S, 2, D // 2).permute(0, 1, 2, 4, 3).contiguous().view(B, H, S, D)

        return out
