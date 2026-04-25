import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from design.tile_level.scatter_block_update import scatter_block_update


_DTYPE_TO_STR = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.int8: "int8",
}

_IDX_DTYPE_TO_STR = {
    torch.int32: "int32",
    torch.int64: "int64",
}


class Model(nn.Module):
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
        dtype_str = _DTYPE_TO_STR[input.dtype]
        idx_dtype_str = _IDX_DTYPE_TO_STR[indices.dtype]

        input_flat = input.reshape(D0 * D1, D2).contiguous()

        kernel = scatter_block_update(D0, D1, D2, K, dtype=dtype_str, idx_dtype=idx_dtype_str)
        output_flat = kernel(input_flat, indices, update)

        return output_flat.reshape(D0, D1, D2)
