"""TileLang optimized implementation for CausalConv1dFn.

Host side handles: conv_states flatten/unflatten, cache index scatter.
Kernel computes: causal 1D depthwise conv with cached states, residual add.
"""
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from design.tile_level.causal_conv1d import causal_conv1d_fn


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._kernel_cache = {}

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
        cu_seq_len, dim = x.shape
        num_states, state_len, _ = conv_states.shape
        batch_count = query_start_loc.shape[0] - 1

        # Use cu_seq_len as upper bound for max_seq_len (safe: total tokens >= any single seq)
        max_seq_len = max(cu_seq_len, 1)

        # Flatten conv_states: [num_states, 2, dim] -> [num_states*2, dim]
        conv_states_flat = conv_states.reshape(-1, dim).contiguous()
        num_states_x_sl = conv_states_flat.shape[0]

        dtype_str = {
            torch.bfloat16: "bfloat16",
            torch.float16: "float16",
            torch.float32: "float32",
        }[x.dtype]

        cache_key = (cu_seq_len, dim, num_states_x_sl, batch_count,
                     max_seq_len, residual_connection, dtype_str)
        kernel = self._kernel_cache.get(cache_key)
        if kernel is None:
            kernel = causal_conv1d_fn(
                cu_seq_len, dim, num_states_x_sl, batch_count,
                max_seq_len,
                residual=residual_connection,
                dtype=dtype_str,
            )
            self._kernel_cache[cache_key] = kernel

        output, cache_updates_flat = kernel(
            x.contiguous(),
            weight.contiguous(),
            conv_states_flat,
            query_start_loc.contiguous(),
            cache_indices.contiguous(),
            initial_state_mode.contiguous(),
        )

        # Scatter cache updates back into conv_states clone
        cache_updates = cache_updates_flat.reshape(batch_count, state_len, dim)
        conv_states_out = conv_states.clone()
        conv_states_out[cache_indices.long()] = cache_updates

        return [output, conv_states_out]
