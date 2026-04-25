"""
AdvanceStepV2 Block-Level Design
================================

Operator: AdvanceStepV2 (vLLM speculative decoding advance step)

Inputs (all int64):
  - input_tokens:   (num_reqs * token_each_reqs,)   — flat 1D
  - sampled_tokens: (num_reqs, token_each_reqs)       — 2D
  - input_positions:(num_reqs * token_each_reqs,)     — flat 1D
  - seq_lens:       (num_reqs * token_each_reqs,)     — flat 1D
  - slot_mapping:   (num_reqs * token_each_reqs,)     — flat 1D
  - block_table:    (num_reqs, max_blocks)            — 2D
  - spec_tokens:    (num_reqs, spec_num)              — 2D, spec_num = token_each_reqs - 1
  - accepted_num:   (num_reqs,)                       — 1D

Attrs:
  - num_seqs:    int
  - num_queries: int
  - block_size:  int

Outputs:
  - out_input_tokens:   (num_reqs * token_each_reqs,)
  - out_input_positions:(num_reqs * token_each_reqs,)
  - out_seq_lens:       (num_reqs * token_each_reqs,)
  - out_slot_mapping:   (num_reqs * token_each_reqs,)

Computation (per element i in [0, num_reqs * token_each_reqs)):
  req_idx = i // token_each_reqs
  tok_idx = i % token_each_reqs

  # Step 1: Update input_positions
  out_input_positions[i] = input_positions[i] + accepted_num[req_idx] + 1

  # Step 2: Update seq_lens
  out_seq_lens[i] = out_input_positions[i] + 1

  # Step 3: Update input_tokens
  if token_each_reqs == 1:
      # Find last token from sampled_tokens
      index = argmin(cat(sampled_tokens[req_idx], [-1])) - 1
      out_input_tokens[i] = sampled_tokens[req_idx, index]
  else:
      if tok_idx == 0:
          index = argmin(cat(sampled_tokens[req_idx], [-1])) - 1
          out_input_tokens[i] = sampled_tokens[req_idx, index]
      else:
          out_input_tokens[i] = spec_tokens[req_idx, tok_idx - 1]

  # Step 4: Update slot_mapping
  max_num_blocks = block_table.shape[1]
  block_table_idx = req_idx * max_num_blocks + out_input_positions[i] // block_size
  block_number = block_table.flatten()[block_table_idx]
  block_offset = out_input_positions[i] % block_size
  out_slot_mapping[i] = block_number * block_size + block_offset

Block-level parallelism:
  - total_elements = num_reqs * token_each_reqs
  - Each AI core processes a contiguous chunk of the flat output arrays
  - block_dim = min(total_elements, available_cores)
  - Each core: elements [core_id * chunk_size, (core_id+1) * chunk_size)

Key constraints:
  - int64 data: NPU vector unit handles int64 natively for basic ops
  - block_table gather: requires indirect indexing into flattened block_table
  - accepted_num broadcast: each req_idx maps to token_each_reqs elements
  - argmin computation: per-request, produces last_tokens array

Strategy:
  - Pre-compute last_tokens (num_reqs,) on host or as a separate kernel pass
  - Main kernel: vectorized element-wise computation for positions, seq_lens,
    input_tokens, slot_mapping
  - Use int32 internally where possible (positions, block indices) to leverage
    wider vector unit, cast back to int64 for output
"""
