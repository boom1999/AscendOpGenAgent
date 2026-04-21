"""Golden for L2/3 GaussianFilter — hand-written reference (no torch_npu wrapper).

Per OPERATOR_TORCH_NPU_MAPPING.md classification: meta_gauss_render not installed
in this environment, so golden mirrors prompt_reference.py exactly. If/when
meta_gauss_render._C is packaged, swap forward() to call the real kernel.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Model(nn.Module):
    """
    3DGS Gaussian Filter: filters valid Gaussians per camera view.

    Computes valid Gaussian positions based on depth, determinant, and screen bounds,
    then culls and compacts the inputs accordingly.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        means: torch.Tensor,
        colors: torch.Tensor,
        det: torch.Tensor,
        opacities: torch.Tensor,
        means2d: torch.Tensor,
        depths: torch.Tensor,
        radius: torch.Tensor,
        conics: torch.Tensor,
        covars2d: torch.Tensor,
        near_plane: float = 0.0,
        far_plane: float = 2.0,
        width: int = 200,
        height: int = 600,
    ) -> List[torch.Tensor]:
        """
        Args:
            means: (B, 3, N) float32
            colors: (B, 3, N) float32
            det: (B, C, N) float32
            opacities: (B, N) float32
            means2d: (B, C, 2, N) float32
            depths: (B, C, N) float32
            radius: (B, C, 2, N) float32
            conics: (B, C, 3, N) float32
            covars2d: (B, C, 3, N) float32
            near_plane, far_plane, width, height: scalar attrs

        Returns:
            List of culled tensors + filter_uint8 + cnt.
        """
        B, C, N = det.shape

        opacities_exp = opacities.unsqueeze(1).expand(B, C, N)

        means_t = means.permute(0, 2, 1).contiguous()
        means2d_t = means2d.permute(0, 1, 3, 2).contiguous()
        radius_t = radius.permute(0, 1, 3, 2).contiguous()
        radius_out = radius.permute(0, 1, 3, 2).contiguous()
        conics_t = conics.permute(0, 1, 3, 2).contiguous()
        colors_t = colors.permute(0, 2, 1).contiguous()
        covars2d_t = covars2d.permute(0, 1, 3, 2).contiguous()

        valid = (det > 0) & (depths > near_plane) & (depths < far_plane)
        radius_t[~valid] = 0.0
        inside = (
            (means2d_t[..., 0] + radius_t[..., 0] > 0)
            & (means2d_t[..., 0] - radius_t[..., 0] < width)
            & (means2d_t[..., 1] + radius_t[..., 1] > 0)
            & (means2d_t[..., 1] - radius_t[..., 1] < height)
        )
        radius_t[~inside] = 0.0
        filter_mask = torch.logical_and(inside, valid)

        means_culling = torch.ones_like(conics_t)
        radius_culling = torch.ones_like(radius_t)
        means2d_culling = torch.ones_like(means2d_t)
        depths_culling = torch.ones_like(depths)
        opacities_culling = torch.ones_like(depths)
        conics_culling = torch.ones_like(conics_t)
        colors_culling = torch.ones_like(conics_t)
        covars2d_culling = torch.ones_like(covars2d_t)

        for b in range(B):
            for c in range(C):
                cnt = filter_mask[b, c].sum()
                radius_culling[b, c, :cnt] = radius_out[b, c, filter_mask[b, c]]
                means_culling[b, c, :cnt] = means_t[b, filter_mask[b, c]]
                means2d_culling[b, c, :cnt] = means2d_t[b, c, filter_mask[b, c]]
                depths_culling[b, c, :cnt] = depths[b, c, filter_mask[b, c]]
                conics_culling[b, c, :cnt] = conics_t[b, c, filter_mask[b, c]]
                colors_culling[b, c, :cnt] = colors_t[b, filter_mask[b, c]]
                covars2d_culling[b, c, :cnt] = covars2d_t[b, c, filter_mask[b, c]]
                opacities_culling[b, c, :cnt] = opacities_exp[b, c, filter_mask[b, c]]

        cnt_out = filter_mask.sum(-1)

        filter_bool = filter_mask.bool()
        remainder = N % 8
        if remainder != 0:
            pad_size = 8 - remainder
            filter_bool = F.pad(filter_bool, (0, pad_size), mode='constant', value=False)
        M = (N + 7) // 8
        filter_reshaped = filter_bool.reshape(B, C, M, 8)
        powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=filter_mask.device)
        filter_uint8 = (filter_reshaped.to(torch.uint8) * powers).sum(dim=-1, dtype=torch.uint8)

        means_culling = means_culling.permute(0, 1, 3, 2).contiguous()
        radius_culling = radius_culling.permute(0, 1, 3, 2).contiguous()
        means2d_culling = means2d_culling.permute(0, 1, 3, 2).contiguous()
        conics_culling = conics_culling.permute(0, 1, 3, 2).contiguous()
        colors_culling = colors_culling.permute(0, 1, 3, 2).contiguous()
        covars2d_culling = covars2d_culling.permute(0, 1, 3, 2).contiguous()

        return [means_culling, colors_culling, means2d_culling,
                depths_culling, radius_culling, covars2d_culling,
                conics_culling, opacities_culling, filter_uint8, cnt_out.to(torch.int32)]


def _load_jsonl_cases(jsonl_path):
    """Load INPUT_CASES from a JSONL file at runtime, converting dtype abbreviations."""
    import json as _json
    from pathlib import Path as _Path

    _DTYPE_ALIAS = {
        "bf16": "bfloat16", "fp16": "float16", "fp32": "float32",
        "fp64": "float64",
    }
    p = _Path(jsonl_path)
    if not p.exists():
        return []
    cases = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = _json.loads(line)
            for inp in raw["inputs"]:
                inp["dtype"] = _DTYPE_ALIAS.get(inp["dtype"], inp["dtype"])
            cases.append(raw)
    return cases


import os as _os
_JSONL_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "3_GaussianFilter.json")
INPUT_CASES_FULL = _load_jsonl_cases(_JSONL_PATH)
# 默认 smoke：硬编码前 N 条用例，避免 1000 条全量跑炸；
# 设置环境变量 AIINFRABENCH_FULL_CASES=1 切回 .json 全量。
INPUT_CASES_SMOKE = [{'inputs': [{'name': 'means',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [32, 3, 5941],
              'range': [1.0, 2.0]},
             {'name': 'colors',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [32, 3, 5941],
              'range': [1.0, 2.0]},
             {'name': 'det',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [32, 6, 5941],
              'range': [1.0, 2.0]},
             {'name': 'opacities',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [32, 5941],
              'range': [1.0, 2.0]},
             {'name': 'means2d',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [32, 6, 2, 5941],
              'range': [1.0, 2.0]},
             {'name': 'depths',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [32, 6, 5941],
              'range': [1.0, 2.0]},
             {'name': 'radius',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [32, 6, 2, 5941],
              'range': [1.0, 2.0]},
             {'name': 'conics',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [32, 6, 3, 5941],
              'range': [1.0, 2.0]},
             {'name': 'covars2d',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [32, 6, 3, 5941],
              'range': [1.0, 2.0]},
             {'name': 'width', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 200},
             {'name': 'height', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 600},
             {'name': 'near_plane', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 0.0},
             {'name': 'far_plane', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 2.0}]},
 {'inputs': [{'name': 'means',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [26, 3, 79041],
              'range': [1.0, 2.0]},
             {'name': 'colors',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [26, 3, 79041],
              'range': [1.0, 2.0]},
             {'name': 'det',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [26, 8, 79041],
              'range': [1.0, 2.0]},
             {'name': 'opacities',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [26, 79041],
              'range': [1.0, 2.0]},
             {'name': 'means2d',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [26, 8, 2, 79041],
              'range': [1.0, 2.0]},
             {'name': 'depths',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [26, 8, 79041],
              'range': [1.0, 2.0]},
             {'name': 'radius',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [26, 8, 2, 79041],
              'range': [1.0, 2.0]},
             {'name': 'conics',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [26, 8, 3, 79041],
              'range': [1.0, 2.0]},
             {'name': 'covars2d',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [26, 8, 3, 79041],
              'range': [1.0, 2.0]},
             {'name': 'width', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 200},
             {'name': 'height', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 600},
             {'name': 'near_plane', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 0.0},
             {'name': 'far_plane', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 2.0}]},
 {'inputs': [{'name': 'means',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [38, 3, 100707],
              'range': [1.0, 2.0]},
             {'name': 'colors',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [38, 3, 100707],
              'range': [1.0, 2.0]},
             {'name': 'det',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [38, 1, 100707],
              'range': [1.0, 2.0]},
             {'name': 'opacities',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [38, 100707],
              'range': [1.0, 2.0]},
             {'name': 'means2d',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [38, 1, 2, 100707],
              'range': [1.0, 2.0]},
             {'name': 'depths',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [38, 1, 100707],
              'range': [1.0, 2.0]},
             {'name': 'radius',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [38, 1, 2, 100707],
              'range': [1.0, 2.0]},
             {'name': 'conics',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [38, 1, 3, 100707],
              'range': [1.0, 2.0]},
             {'name': 'covars2d',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [38, 1, 3, 100707],
              'range': [1.0, 2.0]},
             {'name': 'width', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 200},
             {'name': 'height', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 600},
             {'name': 'near_plane', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 0.0},
             {'name': 'far_plane', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 2.0}]},
 {'inputs': [{'name': 'means',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [59, 3, 9024],
              'range': [1.0, 2.0]},
             {'name': 'colors',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [59, 3, 9024],
              'range': [1.0, 2.0]},
             {'name': 'det',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [59, 8, 9024],
              'range': [1.0, 2.0]},
             {'name': 'opacities',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [59, 9024],
              'range': [1.0, 2.0]},
             {'name': 'means2d',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [59, 8, 2, 9024],
              'range': [1.0, 2.0]},
             {'name': 'depths',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [59, 8, 9024],
              'range': [1.0, 2.0]},
             {'name': 'radius',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [59, 8, 2, 9024],
              'range': [1.0, 2.0]},
             {'name': 'conics',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [59, 8, 3, 9024],
              'range': [1.0, 2.0]},
             {'name': 'covars2d',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [59, 8, 3, 9024],
              'range': [1.0, 2.0]},
             {'name': 'width', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 200},
             {'name': 'height', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 600},
             {'name': 'near_plane', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 0.0},
             {'name': 'far_plane', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 2.0}]},
 {'inputs': [{'name': 'means',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [57, 3, 54800],
              'range': [1.0, 2.0]},
             {'name': 'colors',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [57, 3, 54800],
              'range': [1.0, 2.0]},
             {'name': 'det',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [57, 8, 54800],
              'range': [1.0, 2.0]},
             {'name': 'opacities',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [57, 54800],
              'range': [1.0, 2.0]},
             {'name': 'means2d',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [57, 8, 2, 54800],
              'range': [1.0, 2.0]},
             {'name': 'depths',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [57, 8, 54800],
              'range': [1.0, 2.0]},
             {'name': 'radius',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [57, 8, 2, 54800],
              'range': [1.0, 2.0]},
             {'name': 'conics',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [57, 8, 3, 54800],
              'range': [1.0, 2.0]},
             {'name': 'covars2d',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [57, 8, 3, 54800],
              'range': [1.0, 2.0]},
             {'name': 'width', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 200},
             {'name': 'height', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 600},
             {'name': 'near_plane', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 0.0},
             {'name': 'far_plane', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 2.0}]},
 {'inputs': [{'name': 'means',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [21, 3, 408864],
              'range': [1.0, 2.0]},
             {'name': 'colors',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [21, 3, 408864],
              'range': [1.0, 2.0]},
             {'name': 'det',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [21, 8, 408864],
              'range': [1.0, 2.0]},
             {'name': 'opacities',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [21, 408864],
              'range': [1.0, 2.0]},
             {'name': 'means2d',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [21, 8, 2, 408864],
              'range': [1.0, 2.0]},
             {'name': 'depths',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [21, 8, 408864],
              'range': [1.0, 2.0]},
             {'name': 'radius',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [21, 8, 2, 408864],
              'range': [1.0, 2.0]},
             {'name': 'conics',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [21, 8, 3, 408864],
              'range': [1.0, 2.0]},
             {'name': 'covars2d',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [21, 8, 3, 408864],
              'range': [1.0, 2.0]},
             {'name': 'width', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 200},
             {'name': 'height', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 600},
             {'name': 'near_plane', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 0.0},
             {'name': 'far_plane', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 2.0}]},
 {'inputs': [{'name': 'means',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [62, 3, 8972],
              'range': [1.0, 2.0]},
             {'name': 'colors',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [62, 3, 8972],
              'range': [1.0, 2.0]},
             {'name': 'det',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [62, 8, 8972],
              'range': [1.0, 2.0]},
             {'name': 'opacities',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [62, 8972],
              'range': [1.0, 2.0]},
             {'name': 'means2d',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [62, 8, 2, 8972],
              'range': [1.0, 2.0]},
             {'name': 'depths',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [62, 8, 8972],
              'range': [1.0, 2.0]},
             {'name': 'radius',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [62, 8, 2, 8972],
              'range': [1.0, 2.0]},
             {'name': 'conics',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [62, 8, 3, 8972],
              'range': [1.0, 2.0]},
             {'name': 'covars2d',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [62, 8, 3, 8972],
              'range': [1.0, 2.0]},
             {'name': 'width', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 200},
             {'name': 'height', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 600},
             {'name': 'near_plane', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 0.0},
             {'name': 'far_plane', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 2.0}]},
 {'inputs': [{'name': 'means',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [11, 3, 50971],
              'range': [1.0, 2.0]},
             {'name': 'colors',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [11, 3, 50971],
              'range': [1.0, 2.0]},
             {'name': 'det',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [11, 11, 50971],
              'range': [1.0, 2.0]},
             {'name': 'opacities',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [11, 50971],
              'range': [1.0, 2.0]},
             {'name': 'means2d',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [11, 11, 2, 50971],
              'range': [1.0, 2.0]},
             {'name': 'depths',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [11, 11, 50971],
              'range': [1.0, 2.0]},
             {'name': 'radius',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [11, 11, 2, 50971],
              'range': [1.0, 2.0]},
             {'name': 'conics',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [11, 11, 3, 50971],
              'range': [1.0, 2.0]},
             {'name': 'covars2d',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [11, 11, 3, 50971],
              'range': [1.0, 2.0]},
             {'name': 'width', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 200},
             {'name': 'height', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 600},
             {'name': 'near_plane', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 0.0},
             {'name': 'far_plane', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 2.0}]},
 {'inputs': [{'name': 'means',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [47, 3, 8020],
              'range': [1.0, 2.0]},
             {'name': 'colors',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [47, 3, 8020],
              'range': [1.0, 2.0]},
             {'name': 'det',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [47, 49, 8020],
              'range': [1.0, 2.0]},
             {'name': 'opacities',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [47, 8020],
              'range': [1.0, 2.0]},
             {'name': 'means2d',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [47, 49, 2, 8020],
              'range': [1.0, 2.0]},
             {'name': 'depths',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [47, 49, 8020],
              'range': [1.0, 2.0]},
             {'name': 'radius',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [47, 49, 2, 8020],
              'range': [1.0, 2.0]},
             {'name': 'conics',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [47, 49, 3, 8020],
              'range': [1.0, 2.0]},
             {'name': 'covars2d',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [47, 49, 3, 8020],
              'range': [1.0, 2.0]},
             {'name': 'width', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 200},
             {'name': 'height', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 600},
             {'name': 'near_plane', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 0.0},
             {'name': 'far_plane', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 2.0}]},
 {'inputs': [{'name': 'means',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [3, 3, 787836],
              'range': [1.0, 2.0]},
             {'name': 'colors',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [3, 3, 787836],
              'range': [1.0, 2.0]},
             {'name': 'det',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [3, 49, 787836],
              'range': [1.0, 2.0]},
             {'name': 'opacities',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [3, 787836],
              'range': [1.0, 2.0]},
             {'name': 'means2d',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [3, 49, 2, 787836],
              'range': [1.0, 2.0]},
             {'name': 'depths',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [3, 49, 787836],
              'range': [1.0, 2.0]},
             {'name': 'radius',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [3, 49, 2, 787836],
              'range': [1.0, 2.0]},
             {'name': 'conics',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [3, 49, 3, 787836],
              'range': [1.0, 2.0]},
             {'name': 'covars2d',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [3, 49, 3, 787836],
              'range': [1.0, 2.0]},
             {'name': 'width', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 200},
             {'name': 'height', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 600},
             {'name': 'near_plane', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 0.0},
             {'name': 'far_plane', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 2.0}]}]
INPUT_CASES = INPUT_CASES_FULL if _os.environ.get("AIINFRABENCH_FULL_CASES") == "1" else INPUT_CASES_SMOKE
_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


def _make_boxes(shape, dtype):
    leading_shape = tuple(shape[:-1])
    mins = torch.rand(*leading_shape, 2, dtype=torch.float32)
    sizes = torch.rand(*leading_shape, 2, dtype=torch.float32) + 0.05
    maxs = mins + sizes
    boxes = torch.cat([mins, maxs], dim=-1)
    return boxes.to(dtype=dtype)


def _make_tensor(spec):
    dtype = _DTYPE_MAP[spec["dtype"]]
    shape = spec["shape"]
    name = spec["name"]
    value_range = spec.get("range")

    if dtype == torch.bool:
        return torch.randint(0, 2, tuple(shape), dtype=torch.int64).to(torch.bool)

    if name in {"boxes", "bboxes", "gtboxes"} and shape and shape[-1] == 4 and dtype in {
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
    }:
        return _make_boxes(shape, dtype)

    if value_range is not None:
        low, high = value_range
        if dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
            high_exclusive = high + 1
            return torch.randint(low, high_exclusive, tuple(shape), dtype=dtype)
        return torch.empty(tuple(shape), dtype=dtype).uniform_(low, high)

    if dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        return torch.randint(0, 17, tuple(shape), dtype=dtype)

    return torch.randn(*shape, dtype=dtype)


def _make_tensor_list(spec):
    dtype = _DTYPE_MAP[spec["dtype"]]
    return [torch.randn(*shape, dtype=dtype) for shape in spec["shapes"]]


def _make_arg(spec):
    spec_type = spec["type"]
    if spec_type == "tensor":
        return _make_tensor(spec)
    if spec_type == "tensor_list":
        return _make_tensor_list(spec)
    if spec_type == "attr":
        return spec["value"]
    raise ValueError(f"Unsupported input spec type: {spec_type}")


def get_input_groups():
    for case in INPUT_CASES:
        yield [_make_arg(spec) for spec in case["inputs"]]


def get_init_inputs():
    return []
