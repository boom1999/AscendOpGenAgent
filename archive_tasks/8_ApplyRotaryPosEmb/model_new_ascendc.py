"""AscendC-optimised ApplyRotaryPosEmb (fused query + key).

Normalises both BSND and TND layouts, expands cos/sin per head row,
calls the AscendC RoPE kernel for query and key, then
reshapes back to the original layout.
"""
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import _apply_rotary_pos_emb_ext as _ext  # noqa: E402


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
        q_out_flat = _ext.run_apply_rotary_pos_emb(q_flat, cos_q, sin_q)

        # ── call kernel for key ────────────────────────────────
        k_out_flat = _ext.run_apply_rotary_pos_emb(k_flat, cos_k, sin_k)

        # ── reshape back ───────────────────────────────────────
        if layout == "BSND":
            q_out = q_out_flat.reshape(B, S, N_q, D)
            k_out = k_out_flat.reshape(B, S, N_k, D)
        else:
            q_out = q_out_flat.reshape(total_tokens, N_q, D)
            k_out = k_out_flat.reshape(total_tokens, N_k, D)

        return [q_out, k_out]
