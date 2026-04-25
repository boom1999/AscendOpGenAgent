"""
Tile-level design for ScatterBlockUpdate.

Operation:
    output = input.clone()
    output[indices[k,0], indices[k,1], :] = update[k, :]

Flatten input/output to (D0*D1, D2).
Phase 1: Copy input -> output row by row via UB staging (parallel by rows).
Phase 2: For each scatter index k, read (idx0, idx1), compute target_row = idx0*D1+idx1,
         copy update[k, :] -> output[target_row, :] via UB staging (parallel by K).

NOTE: Phase 1 and Phase 2 require cross-core sync to guarantee all copies finish
before any scatter writes begin. AscendC translation must add cross_core_sync().

Purely Vector work -- no Cube computation.
"""
import tilelang
import tilelang.language as T

pass_configs = {
    "tl.ConvertForLoopToSerial": {},
}


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def scatter_block_update(D0, D1, D2, K, dtype="bfloat16", idx_dtype="int32"):
    total_rows = D0 * D1
    num_physical_cores = 20
    usedCoreNum = min(num_physical_cores, max(1, total_rows))

    rows_per_core = (total_rows + usedCoreNum - 1) // usedCoreNum
    k_per_core = (K + usedCoreNum - 1) // usedCoreNum

    @T.prim_func
    def main(
        input_data: T.Tensor((total_rows, D2), dtype),
        indices: T.Tensor((K, 2), idx_dtype),
        update: T.Tensor((K, D2), dtype),
        output: T.Tensor((total_rows, D2), dtype),
    ):
        with T.Kernel(usedCoreNum, is_npu=True) as (cid, vid):
            # UB allocations
            row_ub = T.alloc_ub((1, D2), dtype)
            idx_ub = T.alloc_ub((1, 2), idx_dtype)

            coreIdx = cid

            # ---- Phase 1: Copy input to output (row-parallel) ----
            with T.Scope("V"):
                for r in T.serial(rows_per_core):
                    row = coreIdx * rows_per_core + r
                    if row < total_rows:
                        T.copy(input_data[row, 0], row_ub)
                        T.copy(row_ub, output[row, 0])

            # NOTE: cross-core sync needed here
            # AscendC: CrossBarrier / cross_core_sync()

            # ---- Phase 2: Scatter updates (K-parallel) ----
            with T.Scope("V"):
                for ki in T.serial(k_per_core):
                    k = coreIdx * k_per_core + ki
                    if k < K:
                        # Read index pair from GM -> UB
                        T.copy(indices[k, 0], idx_ub)
                        # Scalar read: compute target row
                        idx0 = idx_ub[0, 0]
                        idx1 = idx_ub[0, 1]
                        target_row = idx0 * D1 + idx1
                        # Load update row from GM -> UB
                        T.copy(update[k, 0], row_ub)
                        # Write update to target location in output
                        T.copy(row_ub, output[target_row, 0])

    return main
