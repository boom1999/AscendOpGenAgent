"""Tile-level design for ClippedSwiglu.

Kernel: A[M, N], B[M, N] -> Y[M, N] (all float32)
  A_clamped = clamp(A, max=limit)
  B_clamped = clamp(B, -limit, limit)
  y = A_clamped * sigmoid(alpha * A_clamped) * (B_clamped + bias)

sigmoid(x) = 1 / (1 + exp(-x))

Uses T.tile.* vectorized ops: min, max, mul, add, exp, reciprocal.
"""

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def clipped_swiglu_kernel(M, N, alpha=1.702, limit=7.0, bias_val=1.0, dtype="float32"):
    block_M = 1
    num_physical_cores = 20
    m_num = M
    usedCoreNum = min(num_physical_cores, m_num)
    tasksPerCore = (m_num + usedCoreNum - 1) // usedCoreNum

    block_N = min(N, 1024)
    n_loops = (N + block_N - 1) // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(usedCoreNum, is_npu=True) as (cid, vid):
            # UB allocations
            a_ub = T.alloc_ub((1, block_N), dtype)
            b_ub = T.alloc_ub((1, block_N), dtype)
            y_ub = T.alloc_ub((1, block_N), dtype)
            tmp_ub = T.alloc_ub((1, block_N), dtype)
            limit_ub = T.alloc_ub((1, block_N), dtype)
            neg_limit_ub = T.alloc_ub((1, block_N), dtype)
            one_ub = T.alloc_ub((1, block_N), dtype)

            coreIdx = cid
            for localIdx in T.serial(tasksPerCore):
                row = coreIdx * tasksPerCore + localIdx
                with T.Scope("V"):
                    if row < m_num:
                        for ni in T.serial(n_loops):
                            col_start = ni * block_N

                            # Load A and B tiles
                            T.copy(A[row, col_start], a_ub)
                            T.copy(B[row, col_start], b_ub)

                            # Clamp A: A_clamped = min(A, limit)
                            T.tile.min(a_ub, a_ub, T.float32(limit))

                            # Clamp B: B_clamped = clamp(B, -limit, limit)
                            T.tile.min(b_ub, b_ub, T.float32(limit))
                            T.tile.max(b_ub, b_ub, T.float32(-limit))

                            # sigmoid(alpha * A_clamped):
                            # tmp = alpha * a_ub
                            T.tile.mul(tmp_ub, a_ub, T.float32(alpha))
                            # tmp = -tmp (negate for exp(-x))
                            T.tile.mul(tmp_ub, tmp_ub, T.float32(-1.0))
                            # tmp = exp(-alpha * a_ub)
                            T.tile.exp(tmp_ub, tmp_ub)
                            # tmp = 1 + exp(-alpha * a_ub)
                            T.tile.add(tmp_ub, tmp_ub, T.float32(1.0))
                            # tmp = 1 / (1 + exp(-alpha * a_ub)) = sigmoid
                            T.tile.reciprocal(tmp_ub, tmp_ub)

                            # y = a_ub * sigmoid
                            T.tile.mul(y_ub, a_ub, tmp_ub)

                            # b_ub = b_ub + bias
                            T.tile.add(b_ub, b_ub, T.float32(bias_val))

                            # y = y * (b_ub + bias)
                            T.tile.mul(y_ub, y_ub, b_ub)

                            # Write back
                            T.copy(y_ub, Y[row, col_start])

    return main
