"""Golden for L1/2 ClippedSwiglu — direct wrapper around `torch_npu.npu_clipped_swiglu`.

Schema (runtime-confirmed via `torch.ops.npu.npu_clipped_swiglu.default._schema`):
    npu_clipped_swiglu(
        Tensor x, *,
        Tensor? group_index=None,
        int dim=-1, float alpha=1.702, float limit=7., float bias=1.,
        bool interleaved=True
    ) -> Tensor

Cases come from neighboring `2_ClippedSwiglu.json` (one JSON record per line).
"""
import json as _json
import os as _os
from pathlib import Path as _Path
from typing import Optional

import torch
import torch.nn as nn
import torch_npu  # noqa: F401  (registers torch.ops.npu.npu_clipped_swiglu)


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
        x_npu = x.npu()
        gi_npu = group_index.npu() if group_index is not None else None
        out = torch_npu.npu_clipped_swiglu(
            x_npu,
            group_index=gi_npu,
            dim=dim,
            alpha=alpha,
            limit=limit,
            bias=bias,
            interleaved=interleaved,
        )
        # NPU kernel does not write rows beyond `Σ group_index` in the merged
        # [pre, cut*after] layout, leaving uninitialized memory there. The
        # reference contract zero-fills those rows. Mask them here so the
        # padding region is well-defined and matches the reference bit-for-bit.
        if group_index is not None:
            ndim = x.dim()
            d = dim if dim >= 0 else dim + ndim
            pre = 1
            for s in x.shape[:d]:
                pre *= s
            after = 1
            for s in out.shape[d + 1:]:
                after *= s
            cut_half = out.shape[d]
            flat = out.reshape(pre, cut_half * after)
            group_sum = min(int(group_index.sum().item()), pre)
            if group_sum < pre:
                flat[group_sum:].zero_()
        return out.cpu()


# ---------------------------------------------------------------------------
# Case loading (mirrors prompt_reference.py contract for test_golden.py)
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


_JSONL_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "2_ClippedSwiglu.json")
INPUT_CASES_FULL = _load_jsonl_cases(_JSONL_PATH)
# 默认 smoke：硬编码前 N 条用例，避免 1000 条全量跑炸；
# 设置环境变量 AIINFRABENCH_FULL_CASES=1 切回 .json 全量。
INPUT_CASES_SMOKE = [{'inputs': [{'name': 'x',
              'type': 'tensor',
              'required': True,
              'dtype': 'float16',
              'shape': [5017, 2528],
              'range': [-1, 1]},
             {'name': 'group_index',
              'type': 'tensor',
              'required': False,
              'dtype': 'int64',
              'shape': [21],
              'range': [0, 239]},
             {'name': 'dim', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 1},
             {'name': 'alpha', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 1.2835},
             {'name': 'limit', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 4.6055},
             {'name': 'bias', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 2.8631},
             {'name': 'interleaved', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': True}]},
 {'inputs': [{'name': 'x',
              'type': 'tensor',
              'required': True,
              'dtype': 'float16',
              'shape': [674, 4384],
              'range': [-1, 1]},
             {'name': 'group_index',
              'type': 'tensor',
              'required': False,
              'dtype': 'int64',
              'shape': [63],
              'range': [0, 11]},
             {'name': 'dim', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': -1},
             {'name': 'alpha', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 1.7872},
             {'name': 'limit', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 4.4461},
             {'name': 'bias', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 1.8227},
             {'name': 'interleaved', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': True}]},
 {'inputs': [{'name': 'x',
              'type': 'tensor',
              'required': True,
              'dtype': 'float16',
              'shape': [3, 3, 281, 1, 12, 20, 20],
              'range': [-1, 1]},
             {'name': 'group_index',
              'type': 'tensor',
              'required': False,
              'dtype': 'int64',
              'shape': [15],
              'range': [0, 169]},
             {'name': 'dim', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': -3},
             {'name': 'alpha', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 2.7845},
             {'name': 'limit', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 4.795},
             {'name': 'bias', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 1.0532},
             {'name': 'interleaved', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': True}]},
 {'inputs': [{'name': 'x',
              'type': 'tensor',
              'required': True,
              'dtype': 'float16',
              'shape': [251, 4, 6, 94, 2],
              'range': [-1, 1]},
             {'name': 'group_index',
              'type': 'tensor',
              'required': False,
              'dtype': 'int64',
              'shape': [38],
              'range': [0, 7]},
             {'name': 'dim', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 1},
             {'name': 'alpha', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 1.5433},
             {'name': 'limit', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 5.0322},
             {'name': 'bias', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 1.1462},
             {'name': 'interleaved', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': True}]},
 {'inputs': [{'name': 'x',
              'type': 'tensor',
              'required': True,
              'dtype': 'float16',
              'shape': [2, 2, 2, 2, 2, 53, 1, 1],
              'range': [-1, 1]},
             {'name': 'group_index',
              'type': 'tensor',
              'required': False,
              'dtype': 'int64',
              'shape': [5],
              'range': [0, 630]},
             {'name': 'dim', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 0},
             {'name': 'alpha', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 2.7455},
             {'name': 'limit', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 4.5959},
             {'name': 'bias', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 2.0436},
             {'name': 'interleaved', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': True}]},
 {'inputs': [{'name': 'x',
              'type': 'tensor',
              'required': True,
              'dtype': 'float16',
              'shape': [3955, 1024],
              'range': [-1, 1]},
             {'name': 'group_index',
              'type': 'tensor',
              'required': False,
              'dtype': 'int64',
              'shape': [10],
              'range': [0, 396]},
             {'name': 'dim', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': -1},
             {'name': 'alpha', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 1.3645},
             {'name': 'limit', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 5.4685},
             {'name': 'bias', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 1.6365},
             {'name': 'interleaved', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': True}]},
 {'inputs': [{'name': 'x',
              'type': 'tensor',
              'required': True,
              'dtype': 'float16',
              'shape': [1623, 3, 2304],
              'range': [-1, 1]},
             {'name': 'group_index',
              'type': 'tensor',
              'required': False,
              'dtype': 'int64',
              'shape': [21],
              'range': [0, 232]},
             {'name': 'dim', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 2},
             {'name': 'alpha', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 2.8386},
             {'name': 'limit', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 4.1957},
             {'name': 'bias', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 2.8952},
             {'name': 'interleaved', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': True}]},
 {'inputs': [{'name': 'x', 'type': 'tensor', 'required': True, 'dtype': 'float16', 'shape': [1056], 'range': [-1, 1]},
             {'name': 'group_index',
              'type': 'tensor',
              'required': False,
              'dtype': 'int64',
              'shape': [16],
              'range': [0, 393]},
             {'name': 'dim', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 0},
             {'name': 'alpha', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 2.6715},
             {'name': 'limit', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 6.7149},
             {'name': 'bias', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 1.8338},
             {'name': 'interleaved', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': True}]},
 {'inputs': [{'name': 'x',
              'type': 'tensor',
              'required': True,
              'dtype': 'float16',
              'shape': [3, 3, 11, 19, 1, 5280],
              'range': [-1, 1]},
             {'name': 'group_index',
              'type': 'tensor',
              'required': False,
              'dtype': 'int64',
              'shape': [55],
              'range': [0, 35]},
             {'name': 'dim', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': -1},
             {'name': 'alpha', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 2.976},
             {'name': 'limit', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 4.5355},
             {'name': 'bias', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 2.0448},
             {'name': 'interleaved', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': True}]},
 {'inputs': [{'name': 'x',
              'type': 'tensor',
              'required': True,
              'dtype': 'float16',
              'shape': [13, 31, 1, 1, 8, 8, 8, 8],
              'range': [-1, 1]},
             {'name': 'group_index',
              'type': 'tensor',
              'required': False,
              'dtype': 'int64',
              'shape': [27],
              'range': [0, 15]},
             {'name': 'dim', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 4},
             {'name': 'alpha', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 2.1455},
             {'name': 'limit', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 3.3602},
             {'name': 'bias', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 1.506},
             {'name': 'interleaved', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': True}]}]
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
    for case in INPUT_CASES:
        args = []
        has_group_index = False
        for spec in case["inputs"]:
            if spec["name"] == "group_index":
                has_group_index = True
            args.append(_make_arg(spec))
        if not has_group_index:
            args.insert(1, None)
        yield args


def get_init_inputs():
    return []
