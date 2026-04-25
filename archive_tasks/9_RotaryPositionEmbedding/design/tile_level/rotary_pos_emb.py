"""Tile-level TileLang design for RotaryPositionEmbedding (RoPE).

Algorithm per row (half-rotation RoPE):
  split = D // 2
  result[0:split]     = x[0:split]*cos[0:split]     - x[split:D]*sin[0:split]
  result[split:D]     = x[split:D]*cos[split:D]     + x[0:split]*sin[split:D]

The host (model_new_tilelang.py) flattens x to [M, D] and broadcasts+flattens
cos/sin to [M, D]. For mode 1 (interleave), the host deinterleaves even/odd
into contiguous halves so this kernel always operates in half-rotation mode.

NOTE: T.tile.* operations do NOT accept UB-slice arguments.
      We load each half directly from GM into separate full UB buffers.
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
    M : int   - total rows (B * H * S)
    D : int   - head dimension
    """
    split = D // 2

    block_M = 64
    num_physical_cores = 20
    m_num = T.ceildiv(M, block_M)
    usedCoreNum = min(num_physical_cores, m_num)
    tasksPerCore = T.ceildiv(m_num, usedCoreNum)
    vec_num = 2
    sub_block_M = block_M // vec_num

    need_cast = dtype != "float32"
    out_cast_mode = "CAST_ROUND" if dtype == "bfloat16" else "CAST_NONE"

    @T.prim_func
    def main(
        x:   T.Tensor((M, D), dtype),
        cos: T.Tensor((M, D), dtype),
        sin: T.Tensor((M, D), dtype),
        out: T.Tensor((M, D), dtype),
    ):
        with T.Kernel(usedCoreNum, is_npu=True) as (cid, vid):
            # -- UB buffers for two halves (native dtype) --
            x1_in_ub   = T.alloc_ub((1, split), dtype)
            x2_in_ub   = T.alloc_ub((1, split), dtype)
            cos1_in_ub = T.alloc_ub((1, split), dtype)
            cos2_in_ub = T.alloc_ub((1, split), dtype)
            sin1_in_ub = T.alloc_ub((1, split), dtype)
            sin2_in_ub = T.alloc_ub((1, split), dtype)
            out1_ub    = T.alloc_ub((1, split), dtype)
            out2_ub    = T.alloc_ub((1, split), dtype)

            # -- UB buffers for float32 compute --
            x1_ub   = T.alloc_ub((1, split), "float32")
            x2_ub   = T.alloc_ub((1, split), "float32")
            cos1_ub = T.alloc_ub((1, split), "float32")
            cos2_ub = T.alloc_ub((1, split), "float32")
            sin1_ub = T.alloc_ub((1, split), "float32")
            sin2_ub = T.alloc_ub((1, split), "float32")
            res1_ub = T.alloc_ub((1, split), "float32")
            res2_ub = T.alloc_ub((1, split), "float32")
            tmp_ub  = T.alloc_ub((1, split), "float32")

            # -- persistent kernel loop --
            for localIdx in T.serial(tasksPerCore):
                bx = cid * tasksPerCore + localIdx
                if bx < m_num:
                    with T.Scope("V"):
                        for row in T.serial(sub_block_M):
                            row_idx = bx * block_M + vid * sub_block_M + row
                            if row_idx < M:
                                # -- load two halves of x --
                                if need_cast:
                                    T.copy(x[row_idx:row_idx + 1, 0:split],
                                           x1_in_ub)
                                    T.copy(x[row_idx:row_idx + 1, split:D],
                                           x2_in_ub)
                                    T.tile.cast(x1_ub, x1_in_ub,
                                                mode="CAST_NONE", count=split)
                                    T.tile.cast(x2_ub, x2_in_ub,
                                                mode="CAST_NONE", count=split)

                                    T.copy(cos[row_idx:row_idx + 1, 0:split],
                                           cos1_in_ub)
                                    T.copy(cos[row_idx:row_idx + 1, split:D],
                                           cos2_in_ub)
                                    T.tile.cast(cos1_ub, cos1_in_ub,
                                                mode="CAST_NONE", count=split)
                                    T.tile.cast(cos2_ub, cos2_in_ub,
                                                mode="CAST_NONE", count=split)

                                    T.copy(sin[row_idx:row_idx + 1, 0:split],
                                           sin1_in_ub)
                                    T.copy(sin[row_idx:row_idx + 1, split:D],
                                           sin2_in_ub)
                                    T.tile.cast(sin1_ub, sin1_in_ub,
                                                mode="CAST_NONE", count=split)
                                    T.tile.cast(sin2_ub, sin2_in_ub,
                                                mode="CAST_NONE", count=split)
                                else:
                                    T.copy(x[row_idx:row_idx + 1, 0:split],
                                           x1_ub)
                                    T.copy(x[row_idx:row_idx + 1, split:D],
                                           x2_ub)
                                    T.copy(cos[row_idx:row_idx + 1, 0:split],
                                           cos1_ub)
                                    T.copy(cos[row_idx:row_idx + 1, split:D],
                                           cos2_ub)
                                    T.copy(sin[row_idx:row_idx + 1, 0:split],
                                           sin1_ub)
                                    T.copy(sin[row_idx:row_idx + 1, split:D],
                                           sin2_ub)

                                # -- first half: x1*cos1 - x2*sin1 --
                                T.tile.mul(res1_ub, x1_ub, cos1_ub)
                                T.tile.mul(tmp_ub,  x2_ub, sin1_ub)
                                T.tile.sub(res1_ub, res1_ub, tmp_ub)

                                # -- second half: x2*cos2 + x1*sin2 --
                                T.tile.mul(res2_ub, x2_ub, cos2_ub)
                                T.tile.mul(tmp_ub,  x1_ub, sin2_ub)
                                T.tile.add(res2_ub, res2_ub, tmp_ub)

                                # -- store results --
                                if need_cast:
                                    T.tile.cast(out1_ub, res1_ub,
                                                mode=out_cast_mode,
                                                count=split)
                                    T.tile.cast(out2_ub, res2_ub,
                                                mode=out_cast_mode,
                                                count=split)
                                    T.copy(out1_ub,
                                           out[row_idx:row_idx + 1, 0:split])
                                    T.copy(out2_ub,
                                           out[row_idx:row_idx + 1, split:D])
                                else:
                                    T.copy(res1_ub,
                                           out[row_idx:row_idx + 1, 0:split])
                                    T.copy(res2_ub,
                                           out[row_idx:row_idx + 1, split:D])

    return main
