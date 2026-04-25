import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import _causal_conv1d_ext as _ext  # noqa: E402


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        conv_states: torch.Tensor,
        query_start_loc: torch.Tensor,
        cache_indices: torch.Tensor,
        initial_state_mode: torch.Tensor,
        pad_slot_id: int = -1,
        residual_connection: int = 1,
    ) -> List[torch.Tensor]:
        orig_dtype = x.dtype
        cu_seq_len, dim = x.shape
        num_states, state_len, _ = conv_states.shape
        batch_count = query_start_loc.shape[0] - 1

        # Cast to float32 and move to NPU
        x_npu = x.to(torch.float32).npu().contiguous()
        w_npu = weight.to(torch.float32).npu().contiguous()
        cs_f32 = conv_states.to(torch.float32)

        # Flatten conv_states: [num_states, 2, dim] -> [num_states*2, dim]
        cs_flat_npu = cs_f32.reshape(-1, dim).contiguous().npu()

        qsl_npu = query_start_loc.to(torch.int32).npu().contiguous()
        ci_npu = cache_indices.to(torch.int32).npu().contiguous()
        ism_npu = initial_state_mode.to(torch.int32).npu().contiguous()

        results = _ext.run_causal_conv1d(
            x_npu, w_npu, cs_flat_npu,
            qsl_npu, ci_npu, ism_npu,
            residual_connection, pad_slot_id,
        )

        output_npu = results[0]
        cache_updates_npu = results[1]

        # Move results to CPU and cast back to original dtype
        output_cpu = output_npu.cpu().to(orig_dtype)
        cache_updates_cpu = cache_updates_npu.cpu().to(orig_dtype)

        # Scatter cache updates back into conv_states clone (all on CPU)
        cache_updates_reshaped = cache_updates_cpu.reshape(batch_count, state_len, dim)
        conv_states_out = conv_states.cpu().to(orig_dtype).clone()
        ci_cpu = cache_indices.cpu().long()
        conv_states_out[ci_cpu] = cache_updates_reshaped

        return [output_cpu, conv_states_out]
