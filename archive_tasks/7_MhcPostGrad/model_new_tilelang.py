import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

_TASK_DIR = Path(__file__).resolve().parent
if str(_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(_TASK_DIR))

from design.tile_level.mhc_post_grad import mhc_post_grad as tl_mhc_post_grad


def _choose_block_D(D: int, min_block: int = 64) -> int:
    """Choose block_D that evenly divides D (or padded D) and is >= min_block."""
    for bd in [256, 128, 64]:
        if D % bd == 0:
            return bd
    return min_block


def _pad_to_multiple(tensor: torch.Tensor, dim: int, multiple: int) -> torch.Tensor:
    """Pad tensor along `dim` to next multiple of `multiple` with zeros."""
    size = tensor.shape[dim]
    remainder = size % multiple
    if remainder == 0:
        return tensor
    pad_size = multiple - remainder
    pad_shape = list(tensor.shape)
    pad_shape[dim] = pad_size
    padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, padding], dim=dim)


def _compute_n_pad(n: int) -> int:
    """Pad n to next multiple of 8 for 32-byte alignment (float32)."""
    return ((n + 7) // 8) * 8


class ModelNew(nn.Module):
    """TileLang implementation of mHC Post backward gradients."""

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(
        self,
        grad_output: torch.Tensor,
        x: torch.Tensor,
        h_res: torch.Tensor,
        h_out: torch.Tensor,
        h_post: torch.Tensor,
    ) -> List[torch.Tensor]:
        orig_dtype = grad_output.dtype
        assert orig_dtype in (torch.float16, torch.bfloat16)

        shape = grad_output.shape
        n = shape[-2]
        D_orig = shape[-1]
        n_pad = _compute_n_pad(n)

        # Choose block_D and compute padded D
        block_D = _choose_block_D(D_orig)
        D_pad = ((D_orig + block_D - 1) // block_D) * block_D

        # Flatten all leading dims into a single batch dim B
        B = 1
        for s in shape[:-2]:
            B *= s

        # Reshape inputs to (B, n, D) / (B, n, n) / (B, D) / (B, n)
        go_3d = grad_output.reshape(B, n, D_orig).contiguous()
        x_3d = x.reshape(B, n, D_orig).contiguous()
        hres_3d = h_res.reshape(B, n, n).contiguous()
        hout_2d = h_out.reshape(B, D_orig).contiguous()
        hpost_2d = h_post.reshape(B, n).contiguous()

        # Pad D dimension if needed
        if D_pad != D_orig:
            go_3d = _pad_to_multiple(go_3d, dim=2, multiple=block_D)
            x_3d = _pad_to_multiple(x_3d, dim=2, multiple=block_D)
            hout_2d = _pad_to_multiple(hout_2d, dim=1, multiple=block_D)

        # Pad n-related dims to n_pad for 32B DMA alignment
        if n_pad != n:
            # h_res: (B, n, n) -> (B, n, n_pad) pad last dim
            hres_3d = _pad_to_multiple(hres_3d, dim=2, multiple=n_pad)
            # h_post: (B, n) -> (B, n_pad) pad last dim
            hpost_2d = _pad_to_multiple(hpost_2d, dim=1, multiple=n_pad)

        dtype_str = "bfloat16" if orig_dtype == torch.bfloat16 else "float16"

        kernel = tl_mhc_post_grad(B, n, n_pad, D_pad, block_D=block_D, dtype=dtype_str)

        # Kernel outputs: grad_x (B,n,D_pad), grad_h_res (B,n,n_pad),
        #                 grad_h_out (B,D_pad), grad_h_post (B,n_pad)
        grad_x_pad, grad_h_res_pad, grad_h_out_pad, grad_h_post_pad = kernel(
            go_3d, x_3d, hres_3d, hout_2d, hpost_2d
        )

        # Slice back to original dimensions
        grad_x_out = grad_x_pad[:, :, :D_orig]
        grad_h_res_out = grad_h_res_pad[:, :, :n]
        grad_h_out_out = grad_h_out_pad[:, :D_orig]
        grad_h_post_out = grad_h_post_pad[:, :n]

        # Reshape outputs back to original leading dims
        leading_shape = shape[:-2]
        grad_x_out = grad_x_out.reshape(*leading_shape, n, D_orig)
        grad_h_res_out = grad_h_res_out.reshape(*leading_shape, n, n)
        grad_h_out_out = grad_h_out_out.reshape(*leading_shape, D_orig)
        grad_h_post_out = grad_h_post_out.reshape(*leading_shape, n)

        return [grad_x_out, grad_h_res_out, grad_h_out_out, grad_h_post_out]
