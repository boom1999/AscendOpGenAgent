"""TileLang optimized implementation for AttentionUpdate.

Host side: casts inputs to float32, pads tensor lists to 3,
calls kernel, casts output back to original dtype.
"""
import json as _json
import os as _os
import sys as _sys
from pathlib import Path as _Path
from typing import List

import torch
import torch.nn as nn

_TASK_DIR = _Path(__file__).resolve().parent
if str(_TASK_DIR) not in _sys.path:
    _sys.path.insert(0, str(_TASK_DIR))

from design.tile_level.attention_update import attention_update_kernel


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        lse_list: List[torch.Tensor],
        local_out_list: List[torch.Tensor],
        update_type: int = 0,
    ) -> List[torch.Tensor]:
        out_dtype = local_out_list[0].dtype
        K = len(lse_list)
        N = lse_list[0].shape[0]
        H = local_out_list[0].shape[1]

        # Cast to float32
        lse_f32 = [t.float().contiguous() for t in lse_list]
        out_f32 = [t.float().contiguous() for t in local_out_list]

        # Pad to 3 tensors (kernel always takes 3 lse + 3 out)
        while len(lse_f32) < 3:
            lse_f32.append(torch.zeros(N, dtype=torch.float32))
        while len(out_f32) < 3:
            out_f32.append(torch.zeros(N, H, dtype=torch.float32))

        # Build and call kernel
        kernel = attention_update_kernel(K, N, H)
        result_out, lse_out = kernel(
            lse_f32[0], lse_f32[1], lse_f32[2],
            out_f32[0], out_f32[1], out_f32[2],
        )

        return [result_out.to(out_dtype), lse_out]


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


_JSONL_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "3_AttentionUpdate.json")
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


def _make_tensor_list(spec):
    dtype = _DTYPE_MAP[spec["dtype"]]
    return [torch.randn(*shape, dtype=dtype) for shape in spec["shapes"]]


def _make_arg(spec):
    t = spec["type"]
    if t == "tensor":
        return _make_tensor(spec)
    if t == "tensor_list":
        return _make_tensor_list(spec)
    if t == "attr":
        return spec["value"]
    raise ValueError(f"Unsupported input spec type: {t}")


def get_input_groups():
    return [[_make_arg(spec) for spec in case["inputs"]] for case in INPUT_CASES]


def get_init_inputs():
    return []
