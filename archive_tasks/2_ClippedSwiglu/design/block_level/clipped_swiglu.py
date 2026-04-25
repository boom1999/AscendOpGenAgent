"""Block-level design for ClippedSwiglu.

Host side (model_new_tilelang.py) handles:
  - reshape x to 2D [pre, cut]
  - group_index filtering
  - split A/B (interleaved or halved)
  - cast to float32 if needed

Kernel receives: A[M, N], B[M, N] (both float32), outputs Y[M, N]
Computation per element:
  A_clamped = clamp(A, max=limit)
  B_clamped = clamp(B, -limit, limit)
  y = A_clamped * sigmoid(alpha * A_clamped) * (B_clamped + bias)

Pure Vector operation, no Cube needed.
Task partition: split M rows across cores, each core processes block_N columns per iteration.
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
            coreIdx = cid
            for localIdx in T.serial(tasksPerCore):
                row = coreIdx * tasksPerCore + localIdx
                with T.Scope("V"):
                    if row < m_num:
                        for ni in T.serial(n_loops):
                            col_start = ni * block_N
                            # TODO(tile-level):
                            # - load A[row, col_start:col_start+block_N] into a_ub
                            # - load B[row, col_start:col_start+block_N] into b_ub
                            # - clamp a_ub: min(a_ub, limit)
                            # - clamp b_ub: max(min(b_ub, limit), -limit)
                            # - compute sigmoid: sig = 1/(1+exp(-alpha*a_ub))
                            # - y = a_ub * sig * (b_ub + bias)
                            # - write y_ub to Y[row, col_start:col_start+block_N]

    return main
