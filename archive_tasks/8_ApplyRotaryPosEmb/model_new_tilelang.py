"""TileLang-optimised ApplyRotaryPosEmb (fused query + key).

Normalises both BSND and TND layouts, expands cos/sin per head row,
calls the row-parallel TileLang RoPE kernel for query and key, then
reshapes back to the original layout.
"""
import os
import sys
from typing import List

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from design.tile_level.apply_rotary_pos_emb import apply_rotary_pos_emb as _tl_kernel

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
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layout: str = "BSND",
    ) -> List[torch.Tensor]:
        dtype_str = _DTYPE_TO_STR[query.dtype]

        # ── normalise layout to [total_tokens, N, D] ───────────
        if layout == "BSND":
            B, S, N_q, D = query.shape
            _, _, N_k, _ = key.shape
            total_tokens = B * S
            q = query.reshape(total_tokens, N_q, D)
            k = key.reshape(total_tokens, N_k, D)
            D_rot = cos.shape[-1]
            c = cos.reshape(total_tokens, 1, D_rot).squeeze(1)
            s = sin.reshape(total_tokens, 1, D_rot).squeeze(1)
        else:  # TND
            total_tokens, N_q, D = query.shape
            _, N_k, _ = key.shape
            q = query
            k = key
            D_rot = cos.shape[-1]
            c = cos.squeeze(-2) if cos.ndim == 3 else cos
            s = sin.squeeze(-2) if sin.ndim == 3 else sin

        M_q = total_tokens * N_q
        M_k = total_tokens * N_k

        # ── flatten query / key to [M, D] ──────────────────────
        q_flat = q.reshape(M_q, D).contiguous()
        k_flat = k.reshape(M_k, D).contiguous()

        # ── expand cos/sin per head row ────────────────────────
        cos_q = c.unsqueeze(1).expand(-1, N_q, -1).contiguous().view(-1, D_rot)
        sin_q = s.unsqueeze(1).expand(-1, N_q, -1).contiguous().view(-1, D_rot)
        cos_k = c.unsqueeze(1).expand(-1, N_k, -1).contiguous().view(-1, D_rot)
        sin_k = s.unsqueeze(1).expand(-1, N_k, -1).contiguous().view(-1, D_rot)

        # ── call kernel for query ──────────────────────────────
        kernel_q = _tl_kernel(M_q, D, D_rot, dtype=dtype_str)
        q_out_flat = kernel_q(q_flat, cos_q, sin_q)

        # ── call kernel for key ────────────────────────────────
        kernel_k = _tl_kernel(M_k, D, D_rot, dtype=dtype_str)
        k_out_flat = kernel_k(k_flat, cos_k, sin_k)

        # ── reshape back ───────────────────────────────────────
        if layout == "BSND":
            q_out = q_out_flat.reshape(B, S, N_q, D)
            k_out = k_out_flat.reshape(B, S, N_k, D)
        else:
            q_out = q_out_flat.reshape(total_tokens, N_q, D)
            k_out = k_out_flat.reshape(total_tokens, N_k, D)

        return [q_out, k_out]
