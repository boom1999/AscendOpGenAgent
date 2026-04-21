"""Golden for L1/7 ManifoldConstrainedHyperConnectionPostGrad — direct wrapper around
`torch.ops.custom.npu_ai_infra_manifold_constrained_hyper_connection_post_grad`.

Schema (runtime-confirmed):
    custom::npu_ai_infra_manifold_constrained_hyper_connection_post_grad(
        Tensor grad_output, Tensor x, Tensor h_res, Tensor h_out, Tensor h_post
    ) -> (Tensor, Tensor, Tensor, Tensor)

Reference signature: forward(grad_output, x, h_res, h_out, h_post) -> [grad_x, grad_h_res, grad_h_out, grad_h_post]

Cases come from neighboring `7_ManifoldConstrainedHyperConnectionPostGrad.json` (one JSON record per line).
"""
import json as _json
import os as _os
from pathlib import Path as _Path
from typing import List

import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import omni_training_custom_ops  # noqa: F401  (registers torch.ops.custom.npu_ai_infra_manifold_constrained_hyper_connection_post_grad)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        grad_output: torch.Tensor,
        x: torch.Tensor,
        h_res: torch.Tensor,
        h_out: torch.Tensor,
        h_post: torch.Tensor,
    ) -> List[torch.Tensor]:
        go_npu = grad_output.npu()
        x_npu = x.npu()
        hr_npu = h_res.npu()
        ho_npu = h_out.npu()
        hp_npu = h_post.npu()

        result = torch.ops.custom.npu_ai_infra_manifold_constrained_hyper_connection_post_grad(
            go_npu, x_npu, hr_npu, ho_npu, hp_npu,
        )
        return [r.cpu() for r in result]


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


_JSONL_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "7_ManifoldConstrainedHyperConnectionPostGrad.json")
INPUT_CASES_FULL = _load_jsonl_cases(_JSONL_PATH)
# 默认 smoke：硬编码前 N 条用例，避免 1000 条全量跑炸；
# 设置环境变量 AIINFRABENCH_FULL_CASES=1 切回 .json 全量。
INPUT_CASES_SMOKE = [{'inputs': [{'name': 'grad_output', 'type': 'tensor', 'required': True, 'dtype': 'bfloat16', 'shape': [1, 8, 384]},
             {'name': 'x', 'type': 'tensor', 'required': True, 'dtype': 'bfloat16', 'shape': [1, 8, 384]},
             {'name': 'h_res', 'type': 'tensor', 'required': True, 'dtype': 'float32', 'shape': [1, 8, 8]},
             {'name': 'h_out', 'type': 'tensor', 'required': True, 'dtype': 'bfloat16', 'shape': [1, 384]},
             {'name': 'h_post', 'type': 'tensor', 'required': True, 'dtype': 'float32', 'shape': [1, 8]}]},
 {'inputs': [{'name': 'grad_output', 'type': 'tensor', 'required': True, 'dtype': 'float16', 'shape': [1, 4, 762]},
             {'name': 'x', 'type': 'tensor', 'required': True, 'dtype': 'float16', 'shape': [1, 4, 762]},
             {'name': 'h_res', 'type': 'tensor', 'required': True, 'dtype': 'float32', 'shape': [1, 4, 4]},
             {'name': 'h_out', 'type': 'tensor', 'required': True, 'dtype': 'float16', 'shape': [1, 762]},
             {'name': 'h_post', 'type': 'tensor', 'required': True, 'dtype': 'float32', 'shape': [1, 4]}]},
 {'inputs': [{'name': 'grad_output', 'type': 'tensor', 'required': True, 'dtype': 'bfloat16', 'shape': [2, 4, 386]},
             {'name': 'x', 'type': 'tensor', 'required': True, 'dtype': 'bfloat16', 'shape': [2, 4, 386]},
             {'name': 'h_res', 'type': 'tensor', 'required': True, 'dtype': 'float32', 'shape': [2, 4, 4]},
             {'name': 'h_out', 'type': 'tensor', 'required': True, 'dtype': 'bfloat16', 'shape': [2, 386]},
             {'name': 'h_post', 'type': 'tensor', 'required': True, 'dtype': 'float32', 'shape': [2, 4]}]},
 {'inputs': [{'name': 'grad_output', 'type': 'tensor', 'required': True, 'dtype': 'float16', 'shape': [1, 1, 6, 576]},
             {'name': 'x', 'type': 'tensor', 'required': True, 'dtype': 'float16', 'shape': [1, 1, 6, 576]},
             {'name': 'h_res', 'type': 'tensor', 'required': True, 'dtype': 'float32', 'shape': [1, 1, 6, 6]},
             {'name': 'h_out', 'type': 'tensor', 'required': True, 'dtype': 'float16', 'shape': [1, 1, 576]},
             {'name': 'h_post', 'type': 'tensor', 'required': True, 'dtype': 'float32', 'shape': [1, 1, 6]}]},
 {'inputs': [{'name': 'grad_output', 'type': 'tensor', 'required': True, 'dtype': 'bfloat16', 'shape': [1, 4, 960]},
             {'name': 'x', 'type': 'tensor', 'required': True, 'dtype': 'bfloat16', 'shape': [1, 4, 960]},
             {'name': 'h_res', 'type': 'tensor', 'required': True, 'dtype': 'float32', 'shape': [1, 4, 4]},
             {'name': 'h_out', 'type': 'tensor', 'required': True, 'dtype': 'bfloat16', 'shape': [1, 960]},
             {'name': 'h_post', 'type': 'tensor', 'required': True, 'dtype': 'float32', 'shape': [1, 4]}]},
 {'inputs': [{'name': 'grad_output', 'type': 'tensor', 'required': True, 'dtype': 'float16', 'shape': [1, 4, 966]},
             {'name': 'x', 'type': 'tensor', 'required': True, 'dtype': 'float16', 'shape': [1, 4, 966]},
             {'name': 'h_res', 'type': 'tensor', 'required': True, 'dtype': 'float32', 'shape': [1, 4, 4]},
             {'name': 'h_out', 'type': 'tensor', 'required': True, 'dtype': 'float16', 'shape': [1, 966]},
             {'name': 'h_post', 'type': 'tensor', 'required': True, 'dtype': 'float32', 'shape': [1, 4]}]},
 {'inputs': [{'name': 'grad_output', 'type': 'tensor', 'required': True, 'dtype': 'float16', 'shape': [1, 1, 6, 766]},
             {'name': 'x', 'type': 'tensor', 'required': True, 'dtype': 'float16', 'shape': [1, 1, 6, 766]},
             {'name': 'h_res', 'type': 'tensor', 'required': True, 'dtype': 'float32', 'shape': [1, 1, 6, 6]},
             {'name': 'h_out', 'type': 'tensor', 'required': True, 'dtype': 'float16', 'shape': [1, 1, 766]},
             {'name': 'h_post', 'type': 'tensor', 'required': True, 'dtype': 'float32', 'shape': [1, 1, 6]}]},
 {'inputs': [{'name': 'grad_output', 'type': 'tensor', 'required': True, 'dtype': 'float16', 'shape': [1, 6, 778]},
             {'name': 'x', 'type': 'tensor', 'required': True, 'dtype': 'float16', 'shape': [1, 6, 778]},
             {'name': 'h_res', 'type': 'tensor', 'required': True, 'dtype': 'float32', 'shape': [1, 6, 6]},
             {'name': 'h_out', 'type': 'tensor', 'required': True, 'dtype': 'float16', 'shape': [1, 778]},
             {'name': 'h_post', 'type': 'tensor', 'required': True, 'dtype': 'float32', 'shape': [1, 6]}]},
 {'inputs': [{'name': 'grad_output', 'type': 'tensor', 'required': True, 'dtype': 'bfloat16', 'shape': [1, 4, 1152]},
             {'name': 'x', 'type': 'tensor', 'required': True, 'dtype': 'bfloat16', 'shape': [1, 4, 1152]},
             {'name': 'h_res', 'type': 'tensor', 'required': True, 'dtype': 'float32', 'shape': [1, 4, 4]},
             {'name': 'h_out', 'type': 'tensor', 'required': True, 'dtype': 'bfloat16', 'shape': [1, 1152]},
             {'name': 'h_post', 'type': 'tensor', 'required': True, 'dtype': 'float32', 'shape': [1, 4]}]},
 {'inputs': [{'name': 'grad_output', 'type': 'tensor', 'required': True, 'dtype': 'bfloat16', 'shape': [1, 4, 1334]},
             {'name': 'x', 'type': 'tensor', 'required': True, 'dtype': 'bfloat16', 'shape': [1, 4, 1334]},
             {'name': 'h_res', 'type': 'tensor', 'required': True, 'dtype': 'float32', 'shape': [1, 4, 4]},
             {'name': 'h_out', 'type': 'tensor', 'required': True, 'dtype': 'bfloat16', 'shape': [1, 1334]},
             {'name': 'h_post', 'type': 'tensor', 'required': True, 'dtype': 'float32', 'shape': [1, 4]}]}]
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
    for case in INPUT_CASES:
        yield [_make_arg(spec) for spec in case["inputs"]]


def get_init_inputs():
    return []
