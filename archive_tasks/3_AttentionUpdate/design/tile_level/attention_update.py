"""Tile-level design for AttentionUpdate (log-sum-exp merge).

Kernel: lse_0/1/2 (N,), out_0/1/2 (N,H) -> result_out (N,H), lse_out (N,)
All float32. K (1-3) is trace-time constant.

Per row n:
  lse_max = max(lse_k[n])
  lse_out[n] = lse_max + log(sum(exp(lse_k[n] - lse_max)))
  result_out[n,:] = sum(out_k[n,:] * exp(lse_k[n] - lse_out[n]))

Uses T.tile.* vectorized ops: max, sub, exp, ln, add, mul.
Scalar lse ops via (1,) UB buffers; H-dim vectorized for weighted sum.
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
            # UB allocations for lse scalar computation (1 element each)
            l0_ub = T.alloc_ub((1,), dtype)
            l1_ub = T.alloc_ub((1,), dtype)
            l2_ub = T.alloc_ub((1,), dtype)
            lmax_ub = T.alloc_ub((1,), dtype)
            lout_ub = T.alloc_ub((1,), dtype)
            w0_ub = T.alloc_ub((1,), dtype)
            w1_ub = T.alloc_ub((1,), dtype)
            w2_ub = T.alloc_ub((1,), dtype)

            # UB allocations for H-vectorized output computation
            out_ub = T.alloc_ub((1, H), dtype)
            acc_ub = T.alloc_ub((1, H), dtype)

            coreIdx = cid
            for localIdx in T.serial(tasksPerCore):
                row = coreIdx * tasksPerCore + localIdx
                with T.Scope("V"):
                    if row < N:
                        # ── Phase A: lse reduction ──────────────────
                        # Load lse values for this row
                        T.copy(lse_0[row], l0_ub)
                        if K >= 2:
                            T.copy(lse_1[row], l1_ub)
                        if K >= 3:
                            T.copy(lse_2[row], l2_ub)

                        # Compute lse_max
                        T.copy(l0_ub, lmax_ub)
                        if K >= 2:
                            T.tile.max(lmax_ub, lmax_ub, l1_ub)
                        if K >= 3:
                            T.tile.max(lmax_ub, lmax_ub, l2_ub)

                        lse_max_val = lmax_ub[0]

                        # exp(lse_k - lse_max)
                        T.tile.sub(l0_ub, l0_ub, lse_max_val)
                        T.tile.exp(l0_ub, l0_ub)
                        if K >= 2:
                            T.tile.sub(l1_ub, l1_ub, lse_max_val)
                            T.tile.exp(l1_ub, l1_ub)
                        if K >= 3:
                            T.tile.sub(l2_ub, l2_ub, lse_max_val)
                            T.tile.exp(l2_ub, l2_ub)

                        # lse_sum = sum of exp values (accumulate into l0_ub)
                        if K >= 2:
                            T.tile.add(l0_ub, l0_ub, l1_ub)
                        if K >= 3:
                            T.tile.add(l0_ub, l0_ub, l2_ub)

                        # lse_out = lse_max + log(lse_sum)
                        T.tile.ln(l0_ub, l0_ub)
                        T.tile.add(lout_ub, lmax_ub, l0_ub)

                        # Store lse_out
                        T.copy(lout_ub, lse_out[row])

                        lse_out_val = lout_ub[0]

                        # ── Phase B: weighted output sum ────────────
                        # Recompute weights: exp(lse_k - lse_out)
                        T.copy(lse_0[row], w0_ub)
                        T.tile.sub(w0_ub, w0_ub, lse_out_val)
                        T.tile.exp(w0_ub, w0_ub)

                        if K >= 2:
                            T.copy(lse_1[row], w1_ub)
                            T.tile.sub(w1_ub, w1_ub, lse_out_val)
                            T.tile.exp(w1_ub, w1_ub)

                        if K >= 3:
                            T.copy(lse_2[row], w2_ub)
                            T.tile.sub(w2_ub, w2_ub, lse_out_val)
                            T.tile.exp(w2_ub, w2_ub)

                        # Weighted sum over K, vectorized over H
                        # acc = out_0 * w0
                        T.copy(out_0[row, 0], acc_ub)
                        T.tile.mul(acc_ub, acc_ub, w0_ub[0])

                        if K >= 2:
                            T.copy(out_1[row, 0], out_ub)
                            T.tile.mul(out_ub, out_ub, w1_ub[0])
                            T.tile.add(acc_ub, acc_ub, out_ub)

                        if K >= 3:
                            T.copy(out_2[row, 0], out_ub)
                            T.tile.mul(out_ub, out_ub, w2_ub[0])
                            T.tile.add(acc_ub, acc_ub, out_ub)

                        # Store result
                        T.copy(acc_ub, result_out[row, 0])

    return main
