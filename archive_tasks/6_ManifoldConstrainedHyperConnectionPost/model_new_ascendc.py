import sys
from pathlib import Path

import torch
import torch.nn as nn

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import _mhc_post_ext as _ext  # noqa: E402


def _choose_block_D(D: int, min_block: int = 64) -> int:
    for bd in [256, 128, 64]:
        if D % bd == 0:
            return bd
    return min_block


def _pad_to_multiple(tensor: torch.Tensor, dim: int, multiple: int) -> torch.Tensor:
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
    return ((n + 7) // 8) * 8


class ModelNew(nn.Module):
    """AscendC implementation of ManifoldConstrainedHyperConnectionPost."""

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(
        self,
        x: torch.Tensor,
        h_res: torch.Tensor,
        h_out: torch.Tensor,
        h_post: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        shape = x.shape
        n = shape[-2]
        D_orig = shape[-1]
        n_pad = _compute_n_pad(n)

        block_D = _choose_block_D(D_orig)
        D_pad = ((D_orig + block_D - 1) // block_D) * block_D

        B = 1
        for s in shape[:-2]:
            B *= s

        x_3d = x.reshape(B, n, D_orig).contiguous()
        hres_3d = h_res.reshape(B, n, n).contiguous()
        hout_2d = h_out.reshape(B, D_orig).contiguous()
        hpost_2d = h_post.reshape(B, n).contiguous()

        if D_pad != D_orig:
            x_3d = _pad_to_multiple(x_3d, dim=2, multiple=block_D)
            hout_2d = _pad_to_multiple(hout_2d, dim=1, multiple=block_D)

        if n_pad != n:
            hres_3d = _pad_to_multiple(hres_3d, dim=2, multiple=n_pad)
            hpost_2d = _pad_to_multiple(hpost_2d, dim=1, multiple=n_pad)

        y_pad = _ext.run_mhc_post(x_3d, hres_3d, hout_2d, hpost_2d, n, n_pad, block_D)

        y_out = y_pad[:, :, :D_orig]

        leading_shape = shape[:-2]
        y_out = y_out.reshape(*leading_shape, n, D_orig)

        return y_out
