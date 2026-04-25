"""Block-level TileLang design for RotaryPositionEmbedding (RoPE).

Algorithm (half-rotation pattern, used for both mode 0 and mode 1):
  split = D // 2
  result[0:split]     = x[0:split]*cos[0:split]     - x[split:D]*sin[0:split]
  result[split:D]     = x[split:D]*cos[split:D]     + x[0:split]*sin[split:D]

For mode 1 (interleave), the host wrapper deinterleaves x/cos/sin before
calling this kernel and reinterleaves the result after, so the kernel
always operates in half-rotation mode.

Design:
  - model_new_tilelang.py flattens x from [B, H, S, D] to [M, D]
    and broadcasts+flattens cos/sin from [1, 1, S, D] to [M, D].
  - For mode 1, the host rearranges even/odd elements into contiguous halves.
  - Pure Vector compute: D <= 128 fits entirely in UB, no Cube needed.
  - Persistent kernel: 20 cores, block_M=64, 2 vector sub-blocks.
"""

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def rotary_pos_emb(M, D, dtype="float16"):
    """Row-parallel half-rotation RoPE kernel.

    Parameters
    ----------
    M : int   - total rows (B * H * S after flattening)
    D : int   - head dimension (last dim of x, cos, sin)
    """
    split = D // 2

    block_M = 64
    num_physical_cores = 20
    m_num = T.ceildiv(M, block_M)
    usedCoreNum = min(num_physical_cores, m_num)
    tasksPerCore = T.ceildiv(m_num, usedCoreNum)
    vec_num = 2
    sub_block_M = block_M // vec_num

    @T.prim_func
    def main(
        x:   T.Tensor((M, D), dtype),
        cos: T.Tensor((M, D), dtype),
        sin: T.Tensor((M, D), dtype),
        out: T.Tensor((M, D), dtype),
    ):
        with T.Kernel(usedCoreNum, is_npu=True) as (cid, vid):
            for localIdx in T.serial(tasksPerCore):
                bx = cid * tasksPerCore + localIdx
                if bx < m_num:
                    with T.Scope("V"):
                        for row in T.serial(sub_block_M):
                            row_idx = bx * block_M + vid * sub_block_M + row
                            if row_idx < M:
                                # TODO(tile-level):
                                # 1. Load x[row_idx, 0:split] and x[row_idx, split:D]
                                #    into separate UB buffers.
                                # 2. Load cos[row_idx, 0:split], cos[row_idx, split:D],
                                #    sin[row_idx, 0:split], sin[row_idx, split:D].
                                # 3. If dtype != float32, cast all to float32 for compute.
                                # 4. Compute first half:
                                #    res1 = x1 * cos1 - x2 * sin1
                                # 5. Compute second half:
                                #    res2 = x2 * cos2 + x1 * sin2
                                # 6. Cast results back to original dtype if needed.
                                # 7. Write res1 to out[row_idx, 0:split],
                                #    res2 to out[row_idx, split:D].
                                _ = x
                                _ = cos
                                _ = sin
                                _ = out

    return main
