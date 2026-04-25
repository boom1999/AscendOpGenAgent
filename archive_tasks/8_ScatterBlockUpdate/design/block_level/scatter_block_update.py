"""
Block-level design for ScatterBlockUpdate.

Operation:
    output = input.clone()
    output[indices[k,0], indices[k,1], :] = update[k, :]

Design decisions:
- Flatten input/output from (D0, D1, D2) to (D0*D1, D2) for row-based parallelization.
  Each row of D2 contiguous elements corresponds to a unique (i, j) position.
- Phase 1: Copy all rows from input to output (parallel by rows across cores).
- Phase 2: Overwrite scattered rows with update data (parallel by K across cores).
- Requires cross-core sync between Phase 1 and Phase 2.
  In AscendC translation: use cross_core_sync() or IPC barrier.
- Purely Vector work (no Cube): scatter/gather and data copy.
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
    usedCoreNum = min(num_physical_cores, total_rows)

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
            coreIdx = cid

            # Phase 1: Copy input to output (row-parallel)
            with T.Scope("V"):
                for r in T.serial(rows_per_core):
                    row = coreIdx * rows_per_core + r
                    if row < total_rows:
                        # TODO(tile-level): vector copy D2 elements
                        # T.copy(input_data[row, 0:D2], output[row, 0:D2])
                        pass

            # Cross-core sync required here
            # AscendC translation: cross_core_sync()

            # Phase 2: Scatter updates (K-parallel)
            with T.Scope("V"):
                for ki in T.serial(k_per_core):
                    k = coreIdx * k_per_core + ki
                    if k < K:
                        # TODO(tile-level):
                        # Read idx0 = indices[k, 0], idx1 = indices[k, 1]
                        # target_row = idx0 * D1 + idx1
                        # T.copy(update[k, 0:D2], output[target_row, 0:D2])
                        pass

    return main
