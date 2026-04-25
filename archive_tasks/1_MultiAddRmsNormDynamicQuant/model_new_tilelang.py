import sys
from pathlib import Path
from typing import List
import json as _json
import os as _os
from pathlib import Path as _Path
import torch
import torch.nn as nn

_TASK_DIR = Path(__file__).resolve().parent
if str(_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(_TASK_DIR))

from design.tile_level.multi_add_rms_norm_dynamic_quant import (
    multi_add_rms_norm_quant_no_smooth,
    multi_add_rms_norm_quant_smooth1,
    multi_add_rms_norm_quant_dual_smooth,
)


def _to_2d(t, M, N):
    """Reshape tensor to 2D [M, N], handling 1D broadcast."""
    if t.ndim == 1:
        return t.unsqueeze(0).expand(M, N).contiguous()
    return t.reshape(M, N).contiguous()


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        gamma: torch.Tensor,
        smooth_scale1: torch.Tensor = None,
        smooth_scale2: torch.Tensor = None,
        epsilon: float = 1e-5,
    ) -> List[torch.Tensor]:
        orig_shape = x1.shape
        N = orig_shape[-1]
        M = x1[..., 0].numel()
        dtype_str = str(x1.dtype).split(".")[-1]

        x1_2d = x1.reshape(M, N).contiguous()
        x2_2d = x2.reshape(M, N).contiguous()
        gamma_2d = _to_2d(gamma, M, N)

        if smooth_scale1 is not None and smooth_scale2 is not None:
            ss1_2d = _to_2d(smooth_scale1, M, N)
            ss2_2d = _to_2d(smooth_scale2, M, N)
            kernel = multi_add_rms_norm_quant_dual_smooth(
                M, N, eps=epsilon, dtype=dtype_str,
            )
            x_sum_2d, y_norm_2d, y1_2d, scale1_1d, y2_2d, scale2_1d = kernel(
                x1_2d, x2_2d, gamma_2d, ss1_2d, ss2_2d,
            )
        elif smooth_scale1 is not None:
            ss1_2d = _to_2d(smooth_scale1, M, N)
            kernel = multi_add_rms_norm_quant_smooth1(
                M, N, eps=epsilon, dtype=dtype_str,
            )
            x_sum_2d, y_norm_2d, y1_2d, scale1_1d = kernel(
                x1_2d, x2_2d, gamma_2d, ss1_2d,
            )
            y2_2d = y1_2d
            scale2_1d = scale1_1d
        else:
            kernel = multi_add_rms_norm_quant_no_smooth(
                M, N, eps=epsilon, dtype=dtype_str,
            )
            x_sum_2d, y_norm_2d, y1_2d, scale1_1d = kernel(
                x1_2d, x2_2d, gamma_2d,
            )
            y2_2d = y1_2d
            scale2_1d = scale1_1d

        return [
            x_sum_2d.reshape(orig_shape),
            y_norm_2d.reshape(orig_shape),
            y1_2d.reshape(orig_shape),
            scale1_1d.reshape(orig_shape[:-1]),
            y2_2d.reshape(orig_shape),
            scale2_1d.reshape(orig_shape[:-1]),
        ]
