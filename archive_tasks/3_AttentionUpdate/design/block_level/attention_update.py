"""Block-level design for AttentionUpdate (log-sum-exp merge).

Host side (model_new_tilelang.py) handles:
  - Cast lse_list and local_out_list tensors to float32
  - Pad lists to always 3 tensors (dummy zeros for unused K slots)
  - Pass 3 lse tensors (N,) and 3 out tensors (N, H) to kernel

Kernel receives: lse_0/1/2 (N,), out_0/1/2 (N,H) — all float32
Outputs: result_out (N,H), lse_out (N,)
K is trace-time constant controlling which inputs are used.

Computation per row n:
  lse_max = max(lse_k[n] for k in 0..K-1)
  lse_sum = sum(exp(lse_k[n] - lse_max) for k in 0..K-1)
  lse_out[n] = lse_max + log(lse_sum)
  result_out[n, :] = sum(out_k[n, :] * exp(lse_k[n] - lse_out[n]) for k in 0..K-1)

Pure Vector operation, no Cube needed.
Task partition: N rows split across cores, one row at a time,
H dimension fully vectorized per row.
"""

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


@tilelang.jit(out_idx=[6, 7], pass_configs=pass_configs)
def attention_update_kernel(K, N, H, dtype="float32"):
    num_physical_cores = 20
    usedCoreNum = min(num_physical_cores, N)
    tasksPerCore = (N + usedCoreNum - 1) // usedCoreNum

    @T.prim_func
    def main(
        lse_0: T.Tensor((N,), dtype),
        lse_1: T.Tensor((N,), dtype),
        lse_2: T.Tensor((N,), dtype),
        out_0: T.Tensor((N, H), dtype),
        out_1: T.Tensor((N, H), dtype),
        out_2: T.Tensor((N, H), dtype),
        result_out: T.Tensor((N, H), dtype),
        lse_out: T.Tensor((N,), dtype),
    ):
        with T.Kernel(usedCoreNum, is_npu=True) as (cid, vid):
            coreIdx = cid
            for localIdx in T.serial(tasksPerCore):
                row = coreIdx * tasksPerCore + localIdx
                with T.Scope("V"):
                    if row < N:
                        # TODO(tile-level):
                        # Phase A: lse reduction (scalar operations on (1,) UB buffers)
                        # - Load lse_k[row] for each k in 0..K-1
                        # - Compute lse_max = max over K values
                        # - Compute exp(lse_k - lse_max) for each k
                        # - Compute lse_sum = sum of exp values
                        # - lse_out[row] = lse_max + log(lse_sum)
                        # - Recompute weights: exp(lse_k - lse_out) for each k
                        #
                        # Phase B: weighted output sum (vectorized over H)
                        # - acc_ub = out_0[row, :] * weight_0
                        # - For k=1..K-1: acc_ub += out_k[row, :] * weight_k
                        # - Store acc_ub → result_out[row, :]
                        # - Store lse_out_val → lse_out[row]

    return main
