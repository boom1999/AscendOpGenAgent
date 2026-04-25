"""
AdvanceStepV2 TileLang Tile-Level Design

This kernel computes the advance step for vLLM speculative decoding.
All computation including argmin (last_token finding) is done inside the kernel.

Per-request processing:
  1. Find last_token: scan sampled_tokens row, find first -1, take previous token
  2. Compute out_input_positions = input_positions + accepted_num + 1
  3. Compute out_seq_lens = out_input_positions + 1
  4. Assemble out_input_tokens: [last_token, spec_tokens...]
  5. Compute out_slot_mapping via block_table gather
"""

import tilelang
import tilelang.language as T
from tilelang import Profiler


def advance_step_v2_kernel(
    num_reqs: int,
    token_each_reqs: int,
    sampled_cols: int,
    max_num_blocks: int,
    block_size: int,
    block_dim: int = 1,
):
    total_elements = num_reqs * token_each_reqs
    spec_num = token_each_reqs - 1

    @T.prim_func
    def main(
        input_tokens: T.Buffer((total_elements,), "int64"),
        sampled_tokens: T.Buffer((num_reqs, sampled_cols), "int64"),
        input_positions: T.Buffer((total_elements,), "int64"),
        accepted_num: T.Buffer((num_reqs,), "int64"),
        block_table_flat: T.Buffer((num_reqs * max_num_blocks,), "int64"),
        spec_tokens_flat: T.Buffer((num_reqs * max(spec_num, 1),), "int64"),
        out_input_tokens: T.Buffer((total_elements,), "int64"),
        out_input_positions: T.Buffer((total_elements,), "int64"),
        out_seq_lens: T.Buffer((total_elements,), "int64"),
        out_slot_mapping: T.Buffer((total_elements,), "int64"),
    ):
        for block_idx in T.thread_binding(0, block_dim, thread="blockIdx.x"):
            reqs_per_core = (num_reqs + block_dim - 1) // block_dim
            req_start = block_idx * reqs_per_core
            req_end = T.min(req_start + reqs_per_core, num_reqs)

            for req_idx in T.serial(req_start, req_end):
                elem_start = req_idx * token_each_reqs

                # --- Step 1: Find last_token via argmin logic ---
                sampled_row = T.alloc_ub((sampled_cols,), "int64")
                T.copy(sampled_tokens[req_idx, 0:sampled_cols], sampled_row)

                min_idx = T.alloc_ub((1,), "int64")
                min_idx[0] = T.cast(sampled_cols - 1, "int64")
                found_flag = T.alloc_ub((1,), "int64")
                found_flag[0] = T.cast(0, "int64")
                for col in T.serial(sampled_cols):
                    if sampled_row[col] < T.cast(0, "int64"):
                        if found_flag[0] == T.cast(0, "int64"):
                            min_idx[0] = T.cast(col - 1, "int64")
                            found_flag[0] = T.cast(1, "int64")

                last_token_val = T.alloc_ub((1,), "int64")
                idx = min_idx[0]
                last_token_val[0] = sampled_row[idx]

                # --- Step 2: Compute output positions ---
                pos_ub = T.alloc_ub((token_each_reqs,), "int64")
                acc_val = T.alloc_ub((token_each_reqs,), "int64")
                ones_ub = T.alloc_ub((token_each_reqs,), "int64")
                out_pos_ub = T.alloc_ub((token_each_reqs,), "int64")
                out_sl_ub = T.alloc_ub((token_each_reqs,), "int64")

                T.copy(input_positions[elem_start:elem_start + token_each_reqs], pos_ub)
                T.tile.fill(acc_val, accepted_num[req_idx])
                T.tile.fill(ones_ub, T.cast(1, "int64"))

                T.tile.add(pos_ub, acc_val, out_pos_ub)
                T.tile.add(out_pos_ub, ones_ub, out_pos_ub)
                T.tile.add(out_pos_ub, ones_ub, out_sl_ub)

                # --- Step 3: Assemble output tokens ---
                out_tok_ub = T.alloc_ub((token_each_reqs,), "int64")
                out_tok_ub[0] = last_token_val[0]
                if spec_num > 0:
                    spec_ub = T.alloc_ub((spec_num,), "int64")
                    T.copy(spec_tokens_flat[req_idx * spec_num:(req_idx + 1) * spec_num], spec_ub)
                    for ti in T.serial(spec_num):
                        out_tok_ub[ti + 1] = spec_ub[ti]

                # --- Step 4: Compute slot_mapping ---
                out_sm_ub = T.alloc_ub((token_each_reqs,), "int64")
                bs_ub = T.alloc_ub((token_each_reqs,), "int64")
                T.tile.fill(bs_ub, T.cast(block_size, "int64"))

                div_ub = T.alloc_ub((token_each_reqs,), "int64")
                mod_ub = T.alloc_ub((token_each_reqs,), "int64")
                T.tile.div(out_pos_ub, bs_ub, div_ub)
                T.tile.mod(out_pos_ub, bs_ub, mod_ub)

                bn_ub = T.alloc_ub((token_each_reqs,), "int64")
                base_offset = req_idx * max_num_blocks
                for ti in T.serial(token_each_reqs):
                    bt_idx = base_offset + div_ub[ti]
                    bn_ub[ti] = block_table_flat[bt_idx]

                T.tile.mul(bn_ub, bs_ub, out_sm_ub)
                T.tile.add(out_sm_ub, mod_ub, out_sm_ub)

                # --- Write outputs ---
                T.copy(out_tok_ub, out_input_tokens[elem_start:elem_start + token_each_reqs])
                T.copy(out_pos_ub, out_input_positions[elem_start:elem_start + token_each_reqs])
                T.copy(out_sl_ub, out_seq_lens[elem_start:elem_start + token_each_reqs])
                T.copy(out_sm_ub, out_slot_mapping[elem_start:elem_start + token_each_reqs])

    return main
