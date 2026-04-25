"""TileLang optimized implementation for ScatterPaKvCache.

Host side handles: clone caches, reshape key/value to 2D, reshape caches to 3D.
Kernel computes: scatter key/value tokens into PA KV cache based on slot_mapping.
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

from design.tile_level.scatter_pa_kv_cache import scatter_pa_kv_cache


_DTYPE_NAME = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int8: "int8",
}


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self._kernel_cache = {}

    def forward(
        self,
        key: torch.Tensor,
        key_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        value: torch.Tensor,
        value_cache: torch.Tensor,
        cache_mode: str = "PA_NZ",
    ) -> List[torch.Tensor]:
        n_tokens = key.shape[0]
        num_blocks = key_cache.shape[0]
        num_kv_slices = key_cache.shape[1]
        block_size = key_cache.shape[2]
        last_dim_k = key_cache.shape[3]
        num_vv_slices = value_cache.shape[1]
        last_dim_v = value_cache.shape[3]

        key_dtype = _DTYPE_NAME[key.dtype]
        val_dtype = _DTYPE_NAME[value.dtype]

        # Clone caches (reference returns updated copies)
        key_cache_out = key_cache.clone()
        value_cache_out = value_cache.clone()

        # Reshape key/value to 2D: [n_tokens, flat_dim]
        key_flat = key.reshape(n_tokens, -1).contiguous()
        val_flat = value.reshape(n_tokens, -1).contiguous()

        # Reshape caches to 3D: [num_blocks * slices, block_size, last_dim]
        key_cache_3d = key_cache_out.reshape(-1, block_size, last_dim_k)
        val_cache_3d = value_cache_out.reshape(-1, block_size, last_dim_v)

        # Ensure slot_mapping is int32
        slot_mapping_i32 = slot_mapping.to(torch.int32).contiguous()

        # Build kernel (cache by shape signature)
        cache_key = (n_tokens, num_kv_slices, num_vv_slices, num_blocks,
                     block_size, last_dim_k, last_dim_v, key_dtype, val_dtype)
        kernel = self._kernel_cache.get(cache_key)
        if kernel is None:
            kernel = scatter_pa_kv_cache(
                n_tokens, num_kv_slices, num_vv_slices,
                num_blocks, block_size, last_dim_k, last_dim_v,
                key_dtype, val_dtype,
            )
            self._kernel_cache[cache_key] = kernel

        kernel(key_flat, key_cache_3d, slot_mapping_i32, val_flat, val_cache_3d)

        return [key_cache_out, value_cache_out]


# ---------------------------------------------------------------------------
# Case loading (copied from model.py)
# ---------------------------------------------------------------------------

_DTYPE_ALIAS = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32", "fp64": "float64"}
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


_JSONL_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "2_ScatterPaKvCache.json")
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


def _make_scatter_kv_inputs(case):
    """Custom input generation for ScatterPaKvCache with valid slot_mapping."""
    inputs = case["inputs"]
    args = []
    for inp in inputs:
        if inp["type"] == "attr":
            args.append(inp["value"])
            continue
        if inp["name"] == "slotMapping":
            key_cache_spec = next(i for i in inputs if i["name"] == "keyCache")
            num_blocks = key_cache_spec["shape"][0]
            block_size = key_cache_spec["shape"][2]
            total_slots = num_blocks * block_size
            n_tokens = inp["shape"][0]
            slot_mapping = torch.randperm(total_slots)[:n_tokens].to(torch.int32)
            args.append(slot_mapping)
        elif inp["name"] in ("keyCache", "valueCache"):
            dtype = _DTYPE_MAP[inp["dtype"]]
            args.append(torch.zeros(*inp["shape"], dtype=dtype))
        else:
            args.append(_make_tensor(inp))
    return args


def get_input_groups():
    results = []
    for case in INPUT_CASES:
        results.append(_make_scatter_kv_inputs(case))
    return results


def get_init_inputs():
    return []
