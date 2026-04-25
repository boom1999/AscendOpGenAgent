"""Tile-level design for CausalConv1dFn.

Fills TODO(tile-level) from block-level design.
Causal 1D depthwise convolution (K=3) for variable-length sequences.

Per batch:
  - Sliding window conv: y[i,d] = w0[d]*padded[i,d] + w1[d]*padded[i+1,d] + w2[d]*padded[i+2,d]
  - padded = cat(cache_state, x_slice)
  - Optional mode-2 zeroing of first 2 output rows
  - Optional residual: y += x
  - Cache update: new_cache = padded[-2:]

Uses T.tile.mul/add for vectorized compute, T.copy for GM<->UB data movement.
Sliding window maintained via buffer rotation (prev2, prev1, curr).
"""

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


@tilelang.jit(out_idx=[6, 7], pass_configs=pass_configs)
def causal_conv1d_fn(cu_seq_len, dim, num_states_x_sl, batch_count, max_seq_len,
                     residual=1, dtype="bfloat16"):
    K = 3
    state_len = 2  # K - 1
    block_N = min(dim, 1024)
    n_tiles = T.ceildiv(dim, block_N)

    num_physical_cores = 20
    usedCoreNum = min(num_physical_cores, max(batch_count, 1))
    tasksPerCore = T.ceildiv(batch_count, usedCoreNum)

    @T.prim_func
    def main(
        x: T.Tensor((cu_seq_len, dim), dtype),
        weight: T.Tensor((K, dim), dtype),
        conv_states_flat: T.Tensor((num_states_x_sl, dim), dtype),
        query_start_loc: T.Tensor((batch_count + 1,), "int32"),
        cache_indices: T.Tensor((batch_count,), "int32"),
        initial_state_mode: T.Tensor((batch_count,), "int32"),
        output: T.Tensor((cu_seq_len, dim), dtype),
        cache_updates: T.Tensor((batch_count * state_len, dim), dtype),
    ):
        with T.Kernel(usedCoreNum, is_npu=True) as (cid, vid):
            # UB allocations — weight rows
            w0_ub = T.alloc_ub((1, block_N), dtype)
            w1_ub = T.alloc_ub((1, block_N), dtype)
            w2_ub = T.alloc_ub((1, block_N), dtype)
            # Sliding window buffers: prev2=padded[i], prev1=padded[i+1], curr=padded[i+2]
            prev2_ub = T.alloc_ub((1, block_N), dtype)
            prev1_ub = T.alloc_ub((1, block_N), dtype)
            curr_ub = T.alloc_ub((1, block_N), dtype)
            # Compute temporaries
            out_ub = T.alloc_ub((1, block_N), dtype)
            tmp_ub = T.alloc_ub((1, block_N), dtype)

            for localIdx in T.serial(tasksPerCore):
                batch_idx = cid * tasksPerCore + localIdx
                with T.Scope("V"):
                    if batch_idx < batch_count:
                        # --- 1. Read per-batch metadata from GM ---
                        start = T.cast(query_start_loc[batch_idx], "int32")
                        end = T.cast(query_start_loc[batch_idx + 1], "int32")
                        cache_idx = T.cast(cache_indices[batch_idx], "int32")
                        mode = T.cast(initial_state_mode[batch_idx], "int32")

                        # --- 2. Tile along dim ---
                        for dt in T.serial(n_tiles):
                            d_start = dt * block_N
                            valid_n = T.if_then_else(
                                d_start + block_N <= dim, block_N, dim - d_start
                            )

                            # (a) Load weight rows for this dim tile
                            # Zero-init first to handle tail tile padding
                            T.tile.fill(w0_ub, 0.0)
                            T.tile.fill(w1_ub, 0.0)
                            T.tile.fill(w2_ub, 0.0)
                            T.copy(weight[0:1, d_start:d_start + valid_n],
                                   w0_ub[:, 0:valid_n])
                            T.copy(weight[1:2, d_start:d_start + valid_n],
                                   w1_ub[:, 0:valid_n])
                            T.copy(weight[2:3, d_start:d_start + valid_n],
                                   w2_ub[:, 0:valid_n])

                            # (b) Load or zero-init cache for this dim tile
                            cache_base = cache_idx * state_len
                            if mode == T.cast(1, "int32"):
                                T.tile.fill(prev2_ub, 0.0)
                                T.tile.fill(prev1_ub, 0.0)
                                T.copy(conv_states_flat[cache_base:cache_base + 1,
                                       d_start:d_start + valid_n],
                                       prev2_ub[:, 0:valid_n])
                                T.copy(conv_states_flat[cache_base + 1:cache_base + 2,
                                       d_start:d_start + valid_n],
                                       prev1_ub[:, 0:valid_n])
                            else:
                                T.tile.fill(prev2_ub, 0.0)
                                T.tile.fill(prev1_ub, 0.0)

                            # (c) Process sequence positions with sliding window
                            for i in T.serial(max_seq_len):
                                pos = start + i
                                if pos < end:
                                    # Load x[pos] = padded[i+2]
                                    T.tile.fill(curr_ub, 0.0)
                                    T.copy(x[pos:pos + 1, d_start:d_start + valid_n],
                                           curr_ub[:, 0:valid_n])

                                    # Conv: out = w0*prev2 + w1*prev1 + w2*curr
                                    # Use full block_N for T.tile.* (tail padding is zero, harmless)
                                    T.tile.mul(out_ub, w0_ub, prev2_ub)
                                    T.tile.mul(tmp_ub, w1_ub, prev1_ub)
                                    T.tile.add(out_ub, out_ub, tmp_ub)
                                    T.tile.mul(tmp_ub, w2_ub, curr_ub)
                                    T.tile.add(out_ub, out_ub, tmp_ub)

                                    # Mode 2: zero first 2 output rows of batch
                                    if mode == T.cast(2, "int32"):
                                        if i < 2:
                                            T.tile.fill(out_ub, 0.0)

                                    # Residual connection: out += x[pos]
                                    if residual == 1:
                                        T.tile.add(out_ub, out_ub, curr_ub)

                                    # Write output (only valid_n columns)
                                    T.copy(out_ub[:, 0:valid_n],
                                           output[pos:pos + 1, d_start:d_start + valid_n])

                                    # Rotate sliding window: prev2 <- prev1, prev1 <- curr
                                    T.copy(prev1_ub, prev2_ub)
                                    T.copy(curr_ub, prev1_ub)

                            # (d) Write cache updates for this dim tile
                            # After loop: prev2 = padded[seq_len], prev1 = padded[seq_len+1]
                            cache_out_base = batch_idx * state_len
                            T.copy(prev2_ub[:, 0:valid_n],
                                   cache_updates[cache_out_base:cache_out_base + 1,
                                   d_start:d_start + valid_n])
                            T.copy(prev1_ub[:, 0:valid_n],
                                   cache_updates[cache_out_base + 1:cache_out_base + 2,
                                   d_start:d_start + valid_n])

    return main
