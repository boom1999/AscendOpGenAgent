"""Golden for L3/5 MoeFinalizeRoutingV2 -- direct wrapper around `torch_npu.npu_moe_finalize_routing`.

Schema (runtime-confirmed via `torch.ops.npu.npu_moe_finalize_routing.default._schema`):
    npu_moe_finalize_routing(
        Tensor expanded_permuted_rows,
        Tensor? skip1, Tensor? skip2, Tensor? bias, Tensor? scales,
        Tensor expanded_src_to_dst_row,
        Tensor? export_for_source_row,
        int? drop_pad_mode=0
    ) -> Tensor

Parameter name mapping (reference -> schema):
    expanded_x         -> expanded_permuted_rows
    expanded_row_idx   -> expanded_src_to_dst_row
    x1                 -> skip1
    x2                 -> skip2
    expert_idx         -> export_for_source_row

Note: Schema positional order differs from reference forward() — we remap.
"""
import json as _json
import os as _os
from pathlib import Path as _Path
from typing import Optional

import torch
import torch.nn as nn
import torch_npu  # noqa: F401


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        expanded_x: torch.Tensor,
        expanded_row_idx: torch.Tensor,
        x1: torch.Tensor,
        x2: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        scales: Optional[torch.Tensor] = None,
        expert_idx: Optional[torch.Tensor] = None,
        drop_pad_mode: int = 0,
    ) -> torch.Tensor:
        _to_npu = lambda t: t.npu() if isinstance(t, torch.Tensor) else t
        result = torch_npu.npu_moe_finalize_routing(
            _to_npu(expanded_x),            # expanded_permuted_rows
            _to_npu(x1),                    # skip1
            _to_npu(x2),                    # skip2
            _to_npu(bias),                  # bias
            _to_npu(scales),                # scales
            _to_npu(expanded_row_idx),      # expanded_src_to_dst_row
            _to_npu(expert_idx),            # export_for_source_row
            drop_pad_mode=drop_pad_mode,
        )
        return result.cpu() if isinstance(result, torch.Tensor) else result


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


_JSONL_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "5_MoeFinalizeRoutingV2.json")
INPUT_CASES_FULL = _load_jsonl_cases(_JSONL_PATH)
# 默认 smoke：硬编码前 N 条用例，避免 1000 条全量跑炸；
# 设置环境变量 AIINFRABENCH_FULL_CASES=1 切回 .json 全量。
INPUT_CASES_SMOKE = [{'inputs': [{'name': 'expanded_x',
              'type': 'tensor',
              'required': True,
              'dtype': 'bfloat16',
              'shape': [3120, 721],
              'range': [-1.0, 1.0]},
             {'name': 'expanded_row_idx',
              'type': 'tensor',
              'required': True,
              'dtype': 'int32',
              'shape': [3120],
              'range': [0, 3119]},
             {'name': 'x1',
              'type': 'tensor',
              'required': True,
              'dtype': 'bfloat16',
              'shape': [24, 721],
              'range': [-1.0, 1.0]},
             {'name': 'x2',
              'type': 'tensor',
              'required': True,
              'dtype': 'bfloat16',
              'shape': [24, 721],
              'range': [-1.0, 1.0]},
             {'name': 'bias',
              'type': 'tensor',
              'required': True,
              'dtype': 'bfloat16',
              'shape': [482, 721],
              'range': [-1.0, 1.0]},
             {'name': 'scales',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [24, 130],
              'range': [-1.0, 1.0]},
             {'name': 'expert_idx',
              'type': 'tensor',
              'required': True,
              'dtype': 'int32',
              'shape': [24, 130],
              'range': [0, 481]},
             {'name': 'drop_pad_mode', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 0}]},
 {'inputs': [{'name': 'expanded_x',
              'type': 'tensor',
              'required': True,
              'dtype': 'bfloat16',
              'shape': [88, 359],
              'range': [-1.0, 1.0]},
             {'name': 'expanded_row_idx',
              'type': 'tensor',
              'required': True,
              'dtype': 'int32',
              'shape': [88],
              'range': [0, 87]},
             {'name': 'x1',
              'type': 'tensor',
              'required': True,
              'dtype': 'bfloat16',
              'shape': [22, 359],
              'range': [-1.0, 1.0]},
             {'name': 'x2',
              'type': 'tensor',
              'required': True,
              'dtype': 'bfloat16',
              'shape': [22, 359],
              'range': [-1.0, 1.0]},
             {'name': 'scales',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [22, 4],
              'range': [-1.0, 1.0]},
             {'name': 'expert_idx',
              'type': 'tensor',
              'required': True,
              'dtype': 'int32',
              'shape': [22, 4],
              'range': [0, 477]},
             {'name': 'drop_pad_mode', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 0}]},
 {'inputs': [{'name': 'expanded_x',
              'type': 'tensor',
              'required': True,
              'dtype': 'bfloat16',
              'shape': [148, 532],
              'range': [-1.0, 1.0]},
             {'name': 'expanded_row_idx',
              'type': 'tensor',
              'required': True,
              'dtype': 'int32',
              'shape': [148],
              'range': [0, 147]},
             {'name': 'x1',
              'type': 'tensor',
              'required': True,
              'dtype': 'bfloat16',
              'shape': [37, 532],
              'range': [-1.0, 1.0]},
             {'name': 'scales',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [37, 4],
              'range': [-1.0, 1.0]},
             {'name': 'expert_idx',
              'type': 'tensor',
              'required': True,
              'dtype': 'int32',
              'shape': [37, 4],
              'range': [0, 322]},
             {'name': 'drop_pad_mode', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 0}]},
 {'inputs': [{'name': 'expanded_x',
              'type': 'tensor',
              'required': True,
              'dtype': 'bfloat16',
              'shape': [64, 540],
              'range': [-1.0, 1.0]},
             {'name': 'expanded_row_idx',
              'type': 'tensor',
              'required': True,
              'dtype': 'int32',
              'shape': [64],
              'range': [0, 63]},
             {'name': 'x1',
              'type': 'tensor',
              'required': True,
              'dtype': 'bfloat16',
              'shape': [32, 540],
              'range': [-1.0, 1.0]},
             {'name': 'bias',
              'type': 'tensor',
              'required': True,
              'dtype': 'bfloat16',
              'shape': [10, 540],
              'range': [-1.0, 1.0]},
             {'name': 'scales',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [32, 2],
              'range': [-1.0, 1.0]},
             {'name': 'expert_idx',
              'type': 'tensor',
              'required': True,
              'dtype': 'int32',
              'shape': [32, 2],
              'range': [0, 9]},
             {'name': 'drop_pad_mode', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 0}]},
 {'inputs': [{'name': 'expanded_x',
              'type': 'tensor',
              'required': True,
              'dtype': 'bfloat16',
              'shape': [228, 537],
              'range': [-1.0, 1.0]},
             {'name': 'expanded_row_idx',
              'type': 'tensor',
              'required': True,
              'dtype': 'int32',
              'shape': [228],
              'range': [0, 227]},
             {'name': 'x1',
              'type': 'tensor',
              'required': True,
              'dtype': 'bfloat16',
              'shape': [57, 537],
              'range': [-1.0, 1.0]},
             {'name': 'x2',
              'type': 'tensor',
              'required': True,
              'dtype': 'bfloat16',
              'shape': [57, 537],
              'range': [-1.0, 1.0]},
             {'name': 'scales',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [57, 4],
              'range': [-1.0, 1.0]},
             {'name': 'expert_idx',
              'type': 'tensor',
              'required': True,
              'dtype': 'int32',
              'shape': [57, 4],
              'range': [0, 183]},
             {'name': 'drop_pad_mode', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 2}]},
 {'inputs': [{'name': 'expanded_x',
              'type': 'tensor',
              'required': True,
              'dtype': 'bfloat16',
              'shape': [208, 174, 1017],
              'range': [-1.0, 1.0]},
             {'name': 'expanded_row_idx',
              'type': 'tensor',
              'required': True,
              'dtype': 'int32',
              'shape': [304],
              'range': [-1, 36191]},
             {'name': 'x1',
              'type': 'tensor',
              'required': True,
              'dtype': 'bfloat16',
              'shape': [76, 1017],
              'range': [-1.0, 1.0]},
             {'name': 'bias',
              'type': 'tensor',
              'required': True,
              'dtype': 'bfloat16',
              'shape': [208, 1017],
              'range': [-1.0, 1.0]},
             {'name': 'scales',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [76, 4],
              'range': [-1.0, 1.0]},
             {'name': 'expert_idx',
              'type': 'tensor',
              'required': True,
              'dtype': 'int32',
              'shape': [76, 4],
              'range': [0, 207]},
             {'name': 'drop_pad_mode', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 1}]},
 {'inputs': [{'name': 'expanded_x',
              'type': 'tensor',
              'required': True,
              'dtype': 'bfloat16',
              'shape': [15522, 1003],
              'range': [-1.0, 1.0]},
             {'name': 'expanded_row_idx',
              'type': 'tensor',
              'required': True,
              'dtype': 'int32',
              'shape': [15522],
              'range': [0, 15521]},
             {'name': 'x1',
              'type': 'tensor',
              'required': True,
              'dtype': 'bfloat16',
              'shape': [78, 1003],
              'range': [-1.0, 1.0]},
             {'name': 'bias',
              'type': 'tensor',
              'required': True,
              'dtype': 'bfloat16',
              'shape': [383, 1003],
              'range': [-1.0, 1.0]},
             {'name': 'scales',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [78, 199],
              'range': [-1.0, 1.0]},
             {'name': 'expert_idx',
              'type': 'tensor',
              'required': True,
              'dtype': 'int32',
              'shape': [78, 199],
              'range': [0, 382]},
             {'name': 'drop_pad_mode', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 2}]},
 {'inputs': [{'name': 'expanded_x',
              'type': 'tensor',
              'required': True,
              'dtype': 'bfloat16',
              'shape': [92, 584],
              'range': [-1.0, 1.0]},
             {'name': 'expanded_row_idx',
              'type': 'tensor',
              'required': True,
              'dtype': 'int32',
              'shape': [92],
              'range': [0, 91]},
             {'name': 'x1',
              'type': 'tensor',
              'required': True,
              'dtype': 'bfloat16',
              'shape': [23, 584],
              'range': [-1.0, 1.0]},
             {'name': 'scales',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [23, 4],
              'range': [-1.0, 1.0]},
             {'name': 'expert_idx',
              'type': 'tensor',
              'required': True,
              'dtype': 'int32',
              'shape': [23, 4],
              'range': [0, 254]},
             {'name': 'drop_pad_mode', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 0}]},
 {'inputs': [{'name': 'expanded_x',
              'type': 'tensor',
              'required': True,
              'dtype': 'bfloat16',
              'shape': [149, 27, 4],
              'range': [-1.0, 1.0]},
             {'name': 'expanded_row_idx',
              'type': 'tensor',
              'required': True,
              'dtype': 'int32',
              'shape': [142],
              'range': [-1, 4022]},
             {'name': 'x1',
              'type': 'tensor',
              'required': True,
              'dtype': 'bfloat16',
              'shape': [71, 4],
              'range': [-1.0, 1.0]},
             {'name': 'scales',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [71, 2],
              'range': [-1.0, 1.0]},
             {'name': 'expert_idx',
              'type': 'tensor',
              'required': True,
              'dtype': 'int32',
              'shape': [71, 2],
              'range': [0, 148]},
             {'name': 'drop_pad_mode', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 1}]},
 {'inputs': [{'name': 'expanded_x',
              'type': 'tensor',
              'required': True,
              'dtype': 'bfloat16',
              'shape': [497, 22, 708],
              'range': [-1.0, 1.0]},
             {'name': 'expanded_row_idx',
              'type': 'tensor',
              'required': True,
              'dtype': 'int32',
              'shape': [158],
              'range': [-1, 10933]},
             {'name': 'x1',
              'type': 'tensor',
              'required': True,
              'dtype': 'bfloat16',
              'shape': [79, 708],
              'range': [-1.0, 1.0]},
             {'name': 'x2',
              'type': 'tensor',
              'required': True,
              'dtype': 'bfloat16',
              'shape': [79, 708],
              'range': [-1.0, 1.0]},
             {'name': 'scales',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [79, 2],
              'range': [-1.0, 1.0]},
             {'name': 'expert_idx',
              'type': 'tensor',
              'required': True,
              'dtype': 'int32',
              'shape': [79, 2],
              'range': [0, 496]},
             {'name': 'drop_pad_mode', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 3}]}]
INPUT_CASES = INPUT_CASES_FULL if _os.environ.get("AIINFRABENCH_FULL_CASES") == "1" else INPUT_CASES_SMOKE
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
    _PARAM_ORDER = [
        "expanded_x", "expanded_row_idx", "x1", "x2",
        "bias", "scales", "expert_idx", "drop_pad_mode",
    ]
    _PARAM_DEFAULTS = {
        "x2": None, "bias": None, "scales": None,
        "expert_idx": None, "drop_pad_mode": 0,
    }
    for case in INPUT_CASES:
        kwargs = {}
        for spec in case["inputs"]:
            kwargs[spec["name"]] = _make_arg(spec)
        args = []
        for p in _PARAM_ORDER:
            if p in kwargs:
                args.append(kwargs[p])
            elif p in _PARAM_DEFAULTS:
                args.append(_PARAM_DEFAULTS[p])
        yield args


def get_init_inputs():
    return []
