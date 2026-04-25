"""AscendC optimized implementation for GaussianFilter.

Wrapper: reshape inputs to kernel layout (feature-major), pad N to TILE_N
multiple, call compiled AscendC kernel, slice outputs back and reshape to
reference format.

Allowed operations: tensor creation, tensor transforms (reshape, contiguous,
expand, clone), and calling the compiled AscendC kernel extension.
No torch compute ops.
"""

import json as _json
import os as _os
import sys as _sys
from pathlib import Path as _Path
from typing import List

import torch
import torch.nn as nn

_KERNEL_BUILD = _Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in _sys.path:
    _sys.path.insert(0, str(_KERNEL_BUILD))

import _gaussian_filter_ext as _ext  # noqa: E402

_DTYPE_ALIAS = {
    "bf16": "bfloat16", "fp16": "float16", "fp32": "float32",
    "fp64": "float64",
}


def _pad_dim(tensor, dim, target):
    """Pad tensor along `dim` from current size to `target` with zeros."""
    cur = tensor.shape[dim]
    if cur >= target:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[dim] = target - cur
    padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, padding], dim=dim)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

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
        B, C, N = det.shape
        BC = B * C
        M = (N + 7) // 8

        TILE_N = 256
        n_tiles = (N + TILE_N - 1) // TILE_N
        N_padded = n_tiles * TILE_N
        M_padded = n_tiles * (TILE_N // 8)

        # Feature-major input layout — no permutes needed
        means_p = _pad_dim(means.contiguous(), 2, N_padded)
        colors_p = _pad_dim(colors.contiguous(), 2, N_padded)
        det_flat = _pad_dim(det.reshape(BC, N).contiguous(), 1, N_padded)
        opac_p = _pad_dim(opacities.contiguous(), 1, N_padded)
        means2d_flat = _pad_dim(means2d.reshape(BC, 2, N).contiguous(), 2, N_padded)
        depths_flat = _pad_dim(depths.reshape(BC, N).contiguous(), 1, N_padded)
        radius_flat = _pad_dim(radius.reshape(BC, 2, N).contiguous(), 2, N_padded)
        conics_flat = _pad_dim(conics.reshape(BC, 3, N).contiguous(), 2, N_padded)
        covars2d_flat = _pad_dim(covars2d.reshape(BC, 3, N).contiguous(), 2, N_padded)

        b_map = torch.arange(B, dtype=torch.int32, device=det.device
                             ).unsqueeze(1).expand(B, C).reshape(-1).contiguous()

        results = _ext.run_gaussian_filter(
            means_p, colors_p, det_flat, opac_p,
            means2d_flat, depths_flat, radius_flat, conics_flat, covars2d_flat,
            b_map,
            BC, B, N_padded, M_padded,
            float(near_plane), float(far_plane), float(width), float(height),
        )

        (means_out, colors_out, means2d_out, depths_out,
         radius_out, covars2d_out, conics_out, opacities_out,
         filter_uint8, cnt_out) = results

        # Slice N_padded → N, reshape to reference format
        means_culling = means_out[:, :, :N].reshape(B, C, 3, N)
        colors_culling = colors_out[:, :, :N].reshape(B, C, 3, N)
        means2d_culling = means2d_out[:, :, :N].reshape(B, C, 2, N)
        depths_culling = depths_out[:, :N].reshape(B, C, N)
        radius_culling = radius_out[:, :, :N].reshape(B, C, 2, N)
        covars2d_culling = covars2d_out[:, :, :N].reshape(B, C, 3, N)
        conics_culling = conics_out[:, :, :N].reshape(B, C, 3, N)
        opacities_culling = opacities_out[:, :N].reshape(B, C, N)
        filter_uint8_out = filter_uint8[:, :M].reshape(B, C, M)
        cnt_out_final = cnt_out.reshape(B, C)

        return [
            means_culling, colors_culling, means2d_culling,
            depths_culling, radius_culling, covars2d_culling,
            conics_culling, opacities_culling,
            filter_uint8_out, cnt_out_final,
        ]


def _load_jsonl_cases(path):
    p = _Path(path)
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


_JSONL_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "3_GaussianFilter.json")
INPUT_CASES = _load_jsonl_cases(_JSONL_PATH)
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
        torch.float16, torch.float32, torch.float64, torch.bfloat16,
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
    return [[_make_arg(spec) for spec in case["inputs"]] for case in INPUT_CASES]


def get_init_inputs():
    return []
