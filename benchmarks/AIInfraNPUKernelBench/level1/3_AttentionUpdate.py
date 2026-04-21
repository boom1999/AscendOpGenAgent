"""Golden for L1/3 AttentionUpdate — direct wrapper around `torch_npu.npu_attention_update`.

Schema (runtime-confirmed via `torch.ops.npu.npu_attention_update.default._schema`):
    npu_attention_update(Tensor[] lse, Tensor[] local_out, int update_type) -> (Tensor, Tensor)

Returns (all_out, lse_out) tuple; reference returns [all_out, lse_out] list.

Cases come from neighboring `3_AttentionUpdate.json` (one JSON record per line).
"""
import json as _json
import os as _os
from pathlib import Path as _Path
from typing import List

import torch
import torch.nn as nn
import torch_npu  # noqa: F401


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        lse_list: List[torch.Tensor],
        local_out_list: List[torch.Tensor],
        update_type: int = 0,
    ) -> List[torch.Tensor]:
        # NPU kernel only supports fp32 output; cast fp16/bf16 inputs to fp32
        out_dtype = local_out_list[0].dtype
        lse_npu = [t.float().npu() for t in lse_list]
        out_npu = [t.float().npu() for t in local_out_list]

        all_out, lse_out = torch_npu.npu_attention_update(lse_npu, out_npu, update_type)
        result = [all_out.cpu().to(out_dtype)]
        # lse_out may be None for update_type=0
        if lse_out is not None:
            result.append(lse_out.cpu())
        else:
            # Reference always returns lse_out; compute it via log-sum-exp on CPU
            lse_cpu = [t.detach().float() for t in lse_list]
            all_lse = torch.stack(lse_cpu, dim=0)
            lse_max, _ = torch.max(all_lse, dim=0)
            lse_sub_exp = torch.exp(all_lse - lse_max.unsqueeze(0))
            lse_sum = torch.sum(lse_sub_exp, dim=0)
            result.append(lse_max + torch.log(lse_sum))
        return result


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


_JSONL_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "3_AttentionUpdate.json")
INPUT_CASES_FULL = _load_jsonl_cases(_JSONL_PATH)
# 默认 smoke：硬编码前 N 条用例，避免 1000 条全量跑炸；
# 设置环境变量 AIINFRABENCH_FULL_CASES=1 切回 .json 全量。
INPUT_CASES_SMOKE = [{'inputs': [{'name': 'lse_list', 'type': 'tensor_list', 'required': True, 'dtype': 'float32', 'shapes': [[33776]]},
             {'name': 'local_out_list',
              'type': 'tensor_list',
              'required': True,
              'dtype': 'float32',
              'shapes': [[33776, 8]]},
             {'name': 'update_type', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 0}]},
 {'inputs': [{'name': 'lse_list', 'type': 'tensor_list', 'required': True, 'dtype': 'float32', 'shapes': [[24108]]},
             {'name': 'local_out_list',
              'type': 'tensor_list',
              'required': True,
              'dtype': 'float32',
              'shapes': [[24108, 8]]},
             {'name': 'update_type', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 1}]},
 {'inputs': [{'name': 'lse_list', 'type': 'tensor_list', 'required': True, 'dtype': 'float32', 'shapes': [[57066]]},
             {'name': 'local_out_list',
              'type': 'tensor_list',
              'required': True,
              'dtype': 'float16',
              'shapes': [[57066, 8]]},
             {'name': 'update_type', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 0}]},
 {'inputs': [{'name': 'lse_list', 'type': 'tensor_list', 'required': True, 'dtype': 'float32', 'shapes': [[47960]]},
             {'name': 'local_out_list',
              'type': 'tensor_list',
              'required': True,
              'dtype': 'float16',
              'shapes': [[47960, 8]]},
             {'name': 'update_type', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 1}]},
 {'inputs': [{'name': 'lse_list', 'type': 'tensor_list', 'required': True, 'dtype': 'float32', 'shapes': [[31776]]},
             {'name': 'local_out_list',
              'type': 'tensor_list',
              'required': True,
              'dtype': 'bfloat16',
              'shapes': [[31776, 8]]},
             {'name': 'update_type', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 0}]},
 {'inputs': [{'name': 'lse_list', 'type': 'tensor_list', 'required': True, 'dtype': 'float32', 'shapes': [[3586]]},
             {'name': 'local_out_list',
              'type': 'tensor_list',
              'required': True,
              'dtype': 'bfloat16',
              'shapes': [[3586, 8]]},
             {'name': 'update_type', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 1}]},
 {'inputs': [{'name': 'lse_list', 'type': 'tensor_list', 'required': True, 'dtype': 'float32', 'shapes': [[33021]]},
             {'name': 'local_out_list',
              'type': 'tensor_list',
              'required': True,
              'dtype': 'float32',
              'shapes': [[33021, 16]]},
             {'name': 'update_type', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 0}]},
 {'inputs': [{'name': 'lse_list', 'type': 'tensor_list', 'required': True, 'dtype': 'float32', 'shapes': [[19739]]},
             {'name': 'local_out_list',
              'type': 'tensor_list',
              'required': True,
              'dtype': 'float32',
              'shapes': [[19739, 16]]},
             {'name': 'update_type', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 1}]},
 {'inputs': [{'name': 'lse_list', 'type': 'tensor_list', 'required': True, 'dtype': 'float32', 'shapes': [[30598]]},
             {'name': 'local_out_list',
              'type': 'tensor_list',
              'required': True,
              'dtype': 'float16',
              'shapes': [[30598, 16]]},
             {'name': 'update_type', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 0}]},
 {'inputs': [{'name': 'lse_list', 'type': 'tensor_list', 'required': True, 'dtype': 'float32', 'shapes': [[18334]]},
             {'name': 'local_out_list',
              'type': 'tensor_list',
              'required': True,
              'dtype': 'float16',
              'shapes': [[18334, 16]]},
             {'name': 'update_type', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 1}]}]
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
