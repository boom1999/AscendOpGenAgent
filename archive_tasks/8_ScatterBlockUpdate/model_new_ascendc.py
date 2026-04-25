import sys
from pathlib import Path

import torch
import torch.nn as nn

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import _scatter_block_update_ext as _ext  # noqa: E402

_DTYPE_SIZE = {
    torch.bfloat16: 2,
    torch.float16: 2,
    torch.float32: 4,
    torch.int8: 1,
}


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input: torch.Tensor,
        indices: torch.Tensor,
        update: torch.Tensor,
    ) -> torch.Tensor:
        D0, D1, D2 = input.shape
        K = indices.shape[0]

        if K == 0:
            return input.clone()

        if indices.dtype != torch.int32:
            indices = indices.to(torch.int32)
        indices = indices.contiguous()
        update = update.contiguous()

        elemSize = _DTYPE_SIZE[input.dtype]

        output = _ext.run_scatter_block_update(
            input.contiguous(), indices, update, D0, D1, D2, K, elemSize
        )
        return output
