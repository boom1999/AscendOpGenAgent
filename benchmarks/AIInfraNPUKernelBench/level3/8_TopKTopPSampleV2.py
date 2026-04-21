"""Golden for L3/8 TopKTopPSampleV2 -- direct wrapper around `torch_npu.npu_top_k_top_p_sample`.

Schema (runtime-confirmed via `torch.ops.npu.npu_top_k_top_p_sample.default._schema`):
    npu_top_k_top_p_sample(
        Tensor logits, Tensor top_k, Tensor top_p,
        Tensor? q=None,
        float? eps=1e-08, bool? is_need_logits=False,
        int? top_k_guess=32,
        Tensor? min_ps=None, int? ks_max=1024,
        bool? input_is_logits=True,
        str? post_sample="qSample",
        Generator? generator=None
    ) -> (Tensor, Tensor)

Parameter name mapping (reference -> schema):
    is_need_sample_result -> post_sample="multiNomial"  (when True)

Note: V2 shares the same kernel as V1 (`npu_top_k_top_p_sample`).
      When `is_need_sample_result=True`, schema uses `post_sample="multiNomial"`.
      Schema always returns 2-tuple; reference returns 4 elements when
      `is_need_sample_result=True` ([rs_index, rs_value, logits_idx, logits_sort_masked]).
      Golden returns 2 elements always — test harness should compare only the
      first 2 when `is_need_sample_result=True`.
"""
import json as _json
import os as _os
from pathlib import Path as _Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch_npu  # noqa: F401


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        q: Optional[torch.Tensor] = None,
        min_ps: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
        is_need_logits: bool = False,
        top_k_guess: int = 32,
        ks_max: int = 1024,
        input_is_logits: bool = True,
        is_need_sample_result: bool = False,
    ) -> List[torch.Tensor]:
        # Map is_need_sample_result to post_sample string
        if is_need_sample_result:
            post_sample = "multiNomial"
        elif q is not None:
            post_sample = "qSample"
        else:
            post_sample = "qSample"

        _to_npu = lambda t: t.npu() if isinstance(t, torch.Tensor) else t
        result = torch_npu.npu_top_k_top_p_sample(
            _to_npu(logits), _to_npu(top_k), _to_npu(top_p),
            q=_to_npu(q),
            eps=eps,
            is_need_logits=is_need_logits,
            top_k_guess=top_k_guess,
            min_ps=_to_npu(min_ps),
            ks_max=ks_max,
            input_is_logits=input_is_logits,
            post_sample=post_sample,
        )
        out = [t.cpu() if isinstance(t, torch.Tensor) else t for t in result]
        # When is_need_logits=False, reference returns torch.empty(0) for rs_value
        if not is_need_logits:
            out[1] = torch.empty(0)
        # In multiNomial mode, reference keeps rs_index as zeros (no sampling);
        # kernel performs real sampling. Zero out to align.
        if is_need_sample_result and isinstance(out[0], torch.Tensor):
            out[0] = torch.zeros_like(out[0])
        return out


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


_JSONL_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "8_TopKTopPSampleV2.json")
INPUT_CASES_FULL = _load_jsonl_cases(_JSONL_PATH)
# 默认 smoke：硬编码前 N 条用例，避免 1000 条全量跑炸；
# 设置环境变量 AIINFRABENCH_FULL_CASES=1 切回 .json 全量。
INPUT_CASES_SMOKE = [{'inputs': [{'name': 'logits',
              'type': 'tensor',
              'required': True,
              'dtype': 'float16',
              'shape': [72, 982151],
              'range': [0, 1]},
             {'name': 'top_k', 'type': 'tensor', 'required': True, 'dtype': 'int32', 'shape': [72], 'range': [1, 576]},
             {'name': 'top_p', 'type': 'tensor', 'required': True, 'dtype': 'float16', 'shape': [72], 'range': [-1, 0]},
             {'name': 'min_ps',
              'type': 'tensor',
              'required': True,
              'dtype': 'float16',
              'shape': [72],
              'range': [-1, 0]},
             {'name': 'eps', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 1e-08},
             {'name': 'is_need_logits', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': True},
             {'name': 'top_k_guess', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 0},
             {'name': 'ks_max', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 576},
             {'name': 'input_is_logits', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': True},
             {'name': 'is_need_sample_result', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': True}]},
 {'inputs': [{'name': 'logits',
              'type': 'tensor',
              'required': True,
              'dtype': 'float16',
              'shape': [72, 982151],
              'range': [0, 1]},
             {'name': 'top_k', 'type': 'tensor', 'required': True, 'dtype': 'int32', 'shape': [72], 'range': [1, 576]},
             {'name': 'top_p', 'type': 'tensor', 'required': True, 'dtype': 'float16', 'shape': [72], 'range': [-1, 0]},
             {'name': 'min_ps',
              'type': 'tensor',
              'required': True,
              'dtype': 'float16',
              'shape': [72],
              'range': [-1, 0]},
             {'name': 'eps', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 1e-08},
             {'name': 'is_need_logits', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': True},
             {'name': 'top_k_guess', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 0},
             {'name': 'ks_max', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 576},
             {'name': 'input_is_logits', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': True},
             {'name': 'is_need_sample_result', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': False}]},
 {'inputs': [{'name': 'logits',
              'type': 'tensor',
              'required': True,
              'dtype': 'float16',
              'shape': [72, 982151],
              'range': [0, 10]},
             {'name': 'top_k', 'type': 'tensor', 'required': True, 'dtype': 'int32', 'shape': [72], 'range': [1, 576]},
             {'name': 'top_p', 'type': 'tensor', 'required': True, 'dtype': 'float16', 'shape': [72], 'range': [-1, 0]},
             {'name': 'min_ps',
              'type': 'tensor',
              'required': True,
              'dtype': 'float16',
              'shape': [72],
              'range': [-1, 0]},
             {'name': 'eps', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 1e-08},
             {'name': 'is_need_logits', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': True},
             {'name': 'top_k_guess', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 0},
             {'name': 'ks_max', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 576},
             {'name': 'input_is_logits', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': False},
             {'name': 'is_need_sample_result', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': True}]},
 {'inputs': [{'name': 'logits',
              'type': 'tensor',
              'required': True,
              'dtype': 'float16',
              'shape': [72, 982151],
              'range': [0, 10]},
             {'name': 'top_k', 'type': 'tensor', 'required': True, 'dtype': 'int32', 'shape': [72], 'range': [1, 576]},
             {'name': 'top_p', 'type': 'tensor', 'required': True, 'dtype': 'float16', 'shape': [72], 'range': [-1, 0]},
             {'name': 'q',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [72, 982151],
              'range': [0, 1]},
             {'name': 'min_ps',
              'type': 'tensor',
              'required': True,
              'dtype': 'float16',
              'shape': [72],
              'range': [-1, 0]},
             {'name': 'eps', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 1e-08},
             {'name': 'is_need_logits', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': True},
             {'name': 'top_k_guess', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 0},
             {'name': 'ks_max', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 576},
             {'name': 'input_is_logits', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': False},
             {'name': 'is_need_sample_result', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': False}]},
 {'inputs': [{'name': 'logits',
              'type': 'tensor',
              'required': True,
              'dtype': 'float16',
              'shape': [72, 982151],
              'range': [0, 1]},
             {'name': 'top_k', 'type': 'tensor', 'required': True, 'dtype': 'int32', 'shape': [72], 'range': [1, 576]},
             {'name': 'top_p', 'type': 'tensor', 'required': True, 'dtype': 'float16', 'shape': [72], 'range': [-1, 0]},
             {'name': 'min_ps',
              'type': 'tensor',
              'required': True,
              'dtype': 'float16',
              'shape': [72],
              'range': [-1, 0]},
             {'name': 'eps', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 1e-08},
             {'name': 'is_need_logits', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': False},
             {'name': 'top_k_guess', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 0},
             {'name': 'ks_max', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 576},
             {'name': 'input_is_logits', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': True},
             {'name': 'is_need_sample_result', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': True}]},
 {'inputs': [{'name': 'logits',
              'type': 'tensor',
              'required': True,
              'dtype': 'float16',
              'shape': [72, 982151],
              'range': [0, 1]},
             {'name': 'top_k', 'type': 'tensor', 'required': True, 'dtype': 'int32', 'shape': [72], 'range': [1, 576]},
             {'name': 'top_p', 'type': 'tensor', 'required': True, 'dtype': 'float16', 'shape': [72], 'range': [-1, 0]},
             {'name': 'q',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [72, 982151],
              'range': [0, 1]},
             {'name': 'min_ps',
              'type': 'tensor',
              'required': True,
              'dtype': 'float16',
              'shape': [72],
              'range': [-1, 0]},
             {'name': 'eps', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 1e-08},
             {'name': 'is_need_logits', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': False},
             {'name': 'top_k_guess', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 0},
             {'name': 'ks_max', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 576},
             {'name': 'input_is_logits', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': True},
             {'name': 'is_need_sample_result', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': False}]},
 {'inputs': [{'name': 'logits',
              'type': 'tensor',
              'required': True,
              'dtype': 'float16',
              'shape': [72, 982151],
              'range': [0, 10]},
             {'name': 'top_k', 'type': 'tensor', 'required': True, 'dtype': 'int32', 'shape': [72], 'range': [1, 576]},
             {'name': 'top_p', 'type': 'tensor', 'required': True, 'dtype': 'float16', 'shape': [72], 'range': [-1, 0]},
             {'name': 'min_ps',
              'type': 'tensor',
              'required': True,
              'dtype': 'float16',
              'shape': [72],
              'range': [-1, 0]},
             {'name': 'eps', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 1e-08},
             {'name': 'is_need_logits', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': False},
             {'name': 'top_k_guess', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 0},
             {'name': 'ks_max', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 576},
             {'name': 'input_is_logits', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': False},
             {'name': 'is_need_sample_result', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': True}]},
 {'inputs': [{'name': 'logits',
              'type': 'tensor',
              'required': True,
              'dtype': 'float16',
              'shape': [72, 982151],
              'range': [0, 10]},
             {'name': 'top_k', 'type': 'tensor', 'required': True, 'dtype': 'int32', 'shape': [72], 'range': [1, 576]},
             {'name': 'top_p', 'type': 'tensor', 'required': True, 'dtype': 'float16', 'shape': [72], 'range': [-1, 0]},
             {'name': 'min_ps',
              'type': 'tensor',
              'required': True,
              'dtype': 'float16',
              'shape': [72],
              'range': [-1, 0]},
             {'name': 'eps', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 1e-08},
             {'name': 'is_need_logits', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': False},
             {'name': 'top_k_guess', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 0},
             {'name': 'ks_max', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 576},
             {'name': 'input_is_logits', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': False},
             {'name': 'is_need_sample_result', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': False}]},
 {'inputs': [{'name': 'logits',
              'type': 'tensor',
              'required': True,
              'dtype': 'bfloat16',
              'shape': [72, 982151],
              'range': [0, 1]},
             {'name': 'top_k', 'type': 'tensor', 'required': True, 'dtype': 'int32', 'shape': [72], 'range': [1, 576]},
             {'name': 'top_p',
              'type': 'tensor',
              'required': True,
              'dtype': 'bfloat16',
              'shape': [72],
              'range': [-1, 0]},
             {'name': 'min_ps',
              'type': 'tensor',
              'required': True,
              'dtype': 'bfloat16',
              'shape': [72],
              'range': [-1, 0]},
             {'name': 'eps', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 1e-08},
             {'name': 'is_need_logits', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': True},
             {'name': 'top_k_guess', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 0},
             {'name': 'ks_max', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 576},
             {'name': 'input_is_logits', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': True},
             {'name': 'is_need_sample_result', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': True}]},
 {'inputs': [{'name': 'logits',
              'type': 'tensor',
              'required': True,
              'dtype': 'bfloat16',
              'shape': [72, 982151],
              'range': [0, 1]},
             {'name': 'top_k', 'type': 'tensor', 'required': True, 'dtype': 'int32', 'shape': [72], 'range': [1, 576]},
             {'name': 'top_p',
              'type': 'tensor',
              'required': True,
              'dtype': 'bfloat16',
              'shape': [72],
              'range': [-1, 0]},
             {'name': 'q',
              'type': 'tensor',
              'required': True,
              'dtype': 'float32',
              'shape': [72, 982151],
              'range': [0, 1]},
             {'name': 'min_ps',
              'type': 'tensor',
              'required': True,
              'dtype': 'bfloat16',
              'shape': [72],
              'range': [-1, 0]},
             {'name': 'eps', 'type': 'attr', 'required': False, 'dtype': 'float', 'value': 1e-08},
             {'name': 'is_need_logits', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': True},
             {'name': 'top_k_guess', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 0},
             {'name': 'ks_max', 'type': 'attr', 'required': False, 'dtype': 'int', 'value': 576},
             {'name': 'input_is_logits', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': True},
             {'name': 'is_need_sample_result', 'type': 'attr', 'required': False, 'dtype': 'bool', 'value': False}]}]
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
        "logits", "top_k", "top_p", "q", "min_ps",
        "eps", "is_need_logits", "top_k_guess", "ks_max",
        "input_is_logits", "is_need_sample_result",
    ]
    _PARAM_DEFAULTS = {
        "q": None, "min_ps": None, "eps": 1e-8,
        "is_need_logits": False, "top_k_guess": 32, "ks_max": 1024,
        "input_is_logits": True, "is_need_sample_result": False,
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
