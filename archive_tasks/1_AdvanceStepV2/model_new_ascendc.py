import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import _advance_step_v2_ext as _ext  # noqa: E402


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input_tokens: torch.Tensor,
        sampled_tokens: torch.Tensor,
        input_positions: torch.Tensor,
        seq_lens: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
        spec_tokens: torch.Tensor,
        accepted_num: torch.Tensor,
        num_seqs: int,
        num_queries: int,
        block_size: int,
    ) -> List[torch.Tensor]:
        num_reqs = num_seqs
        token_each_reqs = 1 + spec_tokens.shape[1]
        sampled_cols = sampled_tokens.shape[1]
        max_num_blocks = block_table.shape[1]

        # Flatten 2D tensors for kernel
        block_table_flat = block_table.contiguous().reshape(-1)
        spec_num = spec_tokens.shape[1]
        spec_tokens_flat = spec_tokens.contiguous().reshape(-1) if spec_num > 0 else torch.zeros(1, dtype=torch.int64, device=input_tokens.device)

        # Ensure all inputs are on NPU and contiguous
        results = _ext.run_advance_step_v2(
            input_tokens.contiguous(),
            sampled_tokens.contiguous(),
            input_positions.contiguous(),
            accepted_num.contiguous(),
            block_table_flat,
            spec_tokens_flat,
            num_reqs,
            token_each_reqs,
            sampled_cols,
            max_num_blocks,
            block_size,
        )

        return [results[0], results[1], results[2], results[3]]
