import torch
import torch.nn as nn
from typing import List

from design.tile_level.advance_step_v2 import advance_step_v2_kernel


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._kernel_cache = {}

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
        total_elements = num_reqs * token_each_reqs
        max_num_blocks = block_table.shape[1]
        spec_num = spec_tokens.shape[1]
        sampled_cols = sampled_tokens.shape[1]

        # Flatten 2D inputs for kernel
        block_table_flat = block_table.contiguous().reshape(-1)
        spec_tokens_flat = spec_tokens.contiguous().reshape(-1) if spec_num > 0 else torch.zeros(1, dtype=torch.int64, device=input_tokens.device)

        # Build or retrieve kernel
        cache_key = (num_reqs, token_each_reqs, sampled_cols, max_num_blocks, block_size)
        kernel = self._kernel_cache.get(cache_key)
        if kernel is None:
            block_dim = min(num_reqs, 40)
            kernel = advance_step_v2_kernel(
                num_reqs=num_reqs,
                token_each_reqs=token_each_reqs,
                sampled_cols=sampled_cols,
                max_num_blocks=max_num_blocks,
                block_size=block_size,
                block_dim=block_dim,
            )
            self._kernel_cache[cache_key] = kernel

        # Allocate outputs
        out_input_tokens = torch.empty(total_elements, dtype=torch.int64, device=input_tokens.device)
        out_input_positions = torch.empty(total_elements, dtype=torch.int64, device=input_positions.device)
        out_seq_lens = torch.empty(total_elements, dtype=torch.int64, device=seq_lens.device)
        out_slot_mapping = torch.empty(total_elements, dtype=torch.int64, device=slot_mapping.device)

        # Run kernel - all computation happens inside
        kernel(
            input_tokens.contiguous(),
            sampled_tokens.contiguous(),
            input_positions.contiguous(),
            accepted_num.contiguous(),
            block_table_flat,
            spec_tokens_flat,
            out_input_tokens,
            out_input_positions,
            out_seq_lens,
            out_slot_mapping,
        )

        return [out_input_tokens, out_input_positions, out_seq_lens, out_slot_mapping]
