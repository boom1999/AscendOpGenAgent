import sys
import json as _json
import os as _os
from pathlib import Path as _Path
from typing import Optional

import torch
import torch.nn as nn

_KERNEL_BUILD = _Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import _clipped_swiglu_ext as _ext  # noqa: E402


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        group_index: Optional[torch.Tensor],
        dim: int,
        alpha: float,
        limit: float,
        bias: float,
        interleaved: bool,
    ) -> torch.Tensor:
        input_x = x
        orig_dtype = input_x.dtype

        # Merge dims around `dim` into [pre, cut]
        before_total = 1
        for s in input_x.shape[:dim]:
            before_total *= s
        after_total = 1
        for s in input_x.shape[dim:]:
            after_total *= s
        x_2d = input_x.reshape(before_total, after_total)

        # Cast to float32 for computation
        if orig_dtype != torch.float32:
            x_2d = x_2d.to(torch.float32)

        # Group filtering
        if group_index is not None:
            _gi_list = group_index.tolist()
            _acc = 0
            for _v in _gi_list:
                _acc += _v
            group_sum = min(int(_acc), x_2d.shape[0])
        else:
            group_sum = x_2d.shape[0]
        x_filtered = x_2d[:group_sum]

        # Split into A and B
        if interleaved:
            a_tensor = x_filtered[:, ::2].contiguous()
            b_tensor = x_filtered[:, 1::2].contiguous()
        else:
            half = x_filtered.shape[1] // 2
            a_tensor = x_filtered[:, :half].contiguous()
            b_tensor = x_filtered[:, half:].contiguous()

        M, N = a_tensor.shape

        # Move to NPU and call kernel
        a_npu = a_tensor.npu()
        b_npu = b_tensor.npu()
        result_npu = _ext.run_clipped_swiglu(a_npu, b_npu, M, N, alpha, limit, bias)
        result = result_npu.cpu()

        # Cast back to original dtype
        result = result.to(orig_dtype)

        # Build output with zeros for filtered rows
        y = torch.zeros((x_2d.shape[0], x_2d.shape[1] // 2), dtype=orig_dtype)
        y[:group_sum] = result

        # Reshape to original dims with dim halved
        shape = list(input_x.shape)
        shape[dim] = shape[dim] // 2
        return y.reshape(shape)


# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------
_DTYPE_ALIAS = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32", "fp64": "float64"}
_DTYPE_MAP = {
    "float16": torch.float16, "float32": torch.float32, "float64": torch.float64,
    "bfloat16": torch.bfloat16, "int8": torch.int8, "int16": torch.int16,
    "int32": torch.int32, "int64": torch.int64, "uint8": torch.uint8, "bool": torch.bool,
}


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


_JSONL_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "2_ClippedSwiglu.json")
INPUT_CASES = _load_jsonl_cases(_JSONL_PATH)


def _make_tensor(spec):
    dtype = _DTYPE_MAP[spec["dtype"]]
    shape = spec["shape"]
    value_range = spec.get("range")
    if dtype == torch.bool:
        return torch.randint(0, 2, tuple(shape), dtype=torch.int64).to(torch.bool)
    if value_range is not None:
        low, high = value_range
        if dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
            return torch.randint(low, high + 1, tuple(shape), dtype=dtype)
        return torch.empty(tuple(shape), dtype=dtype).uniform_(low, high)
    if dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        return torch.randint(0, 17, tuple(shape), dtype=dtype)
    return torch.randn(*shape, dtype=dtype)


def _make_arg(spec):
    t = spec["type"]
    if t == "tensor":
        return _make_tensor(spec)
    if t == "attr":
        return spec["value"]
    raise ValueError(f"Unsupported input spec type: {t}")


def get_input_groups():
    results = []
    for case in INPUT_CASES:
        args = []
        has_group_index = False
        for spec in case["inputs"]:
            if spec["name"] == "group_index":
                has_group_index = True
            args.append(_make_arg(spec))
        if not has_group_index:
            args.insert(1, None)
        results.append(args)
    return results


def get_init_inputs():
    return []
