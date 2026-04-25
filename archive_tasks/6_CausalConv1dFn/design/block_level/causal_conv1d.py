"""Block-level design for CausalConv1dFn.

Causal 1D depthwise convolution for variable-length sequences (CuSeqLen format).
K=3 (kernel width fixed), state_len=2 (cache rows per batch).

Algorithm per batch:
  1. Read cached state [2, dim] (or zeros if mode != 1)
  2. Form padded sequence: padded = cat(cache, x_slice)  shape [seq_len+2, dim]
  3. Conv: y[i, d] = sum_{k=0..2} w[k, d] * padded[i+k, d]   for i in [0, seq_len)
  4. If mode == 2: zero first 2 output rows
  5. If residual: y += x_slice
  6. Cache update: new_cache = padded[-2:]

Block partitioning:
  - Pure Vector computation (no Cube/MMA needed)
  - Persistent kernel: 20 cores distribute batches round-robin
  - Within each batch: tile along dim (block_N=1024)
  - Sequence positions processed serially (K=3 sliding window)

Inputs (flattened in host):
  - x [cu_seq_len, dim]
  - weight [K=3, dim]
  - conv_states_flat [num_states * state_len, dim]   (host reshapes from [num_states, 2, dim])
  - query_start_loc [batch_count + 1]  int32
  - cache_indices [batch_count]  int32
  - initial_state_mode [batch_count]  int32

Outputs:
  - output [cu_seq_len, dim]
  - cache_updates [batch_count * state_len, dim]   (host scatters back to conv_states)
"""

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


@tilelang.jit(out_idx=[6, 7], pass_configs=pass_configs)
def causal_conv1d_fn(cu_seq_len, dim, num_states_x_sl, batch_count, residual=1, dtype="bfloat16"):
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
            coreIdx = cid
            for localIdx in T.serial(tasksPerCore):
                batch_idx = coreIdx * tasksPerCore + localIdx
                with T.Scope("V"):
                    if batch_idx < batch_count:
                        # TODO(tile-level):
                        # 1. Read per-batch metadata from GM:
                        #    qsl_ub = alloc_ub((2,), "int32")
                        #    T.copy(query_start_loc[batch_idx:batch_idx+2], qsl_ub)
                        #    start = qsl_ub[0], end = qsl_ub[1], seq_len = end - start
                        #    ci_ub = alloc_ub((1,), "int32")
                        #    T.copy(cache_indices[batch_idx:batch_idx+1], ci_ub)
                        #    cache_idx = ci_ub[0]
                        #    ism_ub = alloc_ub((1,), "int32")
                        #    T.copy(initial_state_mode[batch_idx:batch_idx+1], ism_ub)
                        #    mode = ism_ub[0]
                        #
                        # 2. For each dim tile dt in range(n_tiles):
                        #    d_start = dt * block_N
                        #    valid_n = min(block_N, dim - d_start)
                        #
                        #    a. Alloc UB:
                        #       w_ub    [K, block_N]      weight tile
                        #       cache_ub[state_len, block_N]  cache or zeros
                        #       row0_ub [1, block_N]      sliding window row 0
                        #       row1_ub [1, block_N]      sliding window row 1
                        #       row2_ub [1, block_N]      sliding window row 2
                        #       out_ub  [1, block_N]      output accumulator
                        #       tmp_ub  [1, block_N]      temp for mul-add
                        #
                        #    b. Load weight:
                        #       T.copy(weight[0:K, d_start:d_start+valid_n], w_ub[:, 0:valid_n])
                        #
                        #    c. Load/init cache:
                        #       cache_base = cache_idx * state_len
                        #       if mode == 1:
                        #         T.copy(conv_states_flat[cache_base:cache_base+2, d_start:d_start+valid_n],
                        #                cache_ub[:, 0:valid_n])
                        #       else:
                        #         T.tile.fill(cache_ub, 0.0)
                        #
                        #    d. Position 0 (if seq_len >= 1):
                        #       row0 = cache[0], row1 = cache[1], row2 = x[start]
                        #       out = w[0]*row0 + w[1]*row1 + w[2]*row2
                        #       if mode == 2: out = 0
                        #       if residual: out += x[start]
                        #       write output[start, d_start:d_end]
                        #
                        #    e. Position 1 (if seq_len >= 2):
                        #       row0 = cache[1], row1 = x[start], row2 = x[start+1]
                        #       out = w[0]*row0 + w[1]*row1 + w[2]*row2
                        #       if mode == 2: out = 0
                        #       if residual: out += x[start+1]
                        #       write output[start+1, d_start:d_end]
                        #
                        #    f. Positions 2..seq_len-1 (sliding window, no cache dependency):
                        #       for i in range(2, seq_len):
                        #         row0 = x[start+i-2], row1 = x[start+i-1], row2 = x[start+i]
                        #         out = w[0]*row0 + w[1]*row1 + w[2]*row2
                        #         if residual: out += x[start+i]
                        #         write output[start+i, d_start:d_end]
                        #
                        #    g. Cache update:
                        #       cache_out_base = batch_idx * state_len
                        #       Determine padded[-2:] based on seq_len:
                        #         seq_len >= 2: cache_updates rows = x[end-2], x[end-1]
                        #         seq_len == 1: cache_updates rows = cache[1], x[start]
                        #         seq_len == 0: cache_updates rows = cache[0], cache[1]
                        #       write cache_updates[cache_out_base:cache_out_base+2, d_start:d_end]
                        pass

    return main
