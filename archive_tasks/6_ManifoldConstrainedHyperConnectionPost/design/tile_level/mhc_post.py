"""Tile-level TileLang design for ManifoldConstrainedHyperConnectionPost.

Computes:
  y[i, d] = h_post[i] * h_out[d] + sum_j(h_res[j, i] * x[j, d])

This is an outer-product (h_post outer h_out) plus a small matmul (h_res^T @ x),
fused into a single kernel.

Strategy:
- Pure Vector kernel (n is tiny: 4-8, D is large: 384-24359).
- Persistent-kernel partitioning over batch dim B.
- D tiled in chunks of block_D (D must be a multiple of block_D -- host pads).
- h_res and h_post padded to n_pad for 32B DMA alignment.
- For each batch element, for each D-tile:
    1. Load h_out[d_start : d_start+block_D] once, cast to fp32
    2. For each output row i (0..n-1):
       a. y_row = h_post[i] * h_out_tile   (scalar-vector mul)
       b. For each j (0..n-1):
            Load h_res[j, :] row, get scalar h_res[j,i]
            Load x[j, d_tile], cast to fp32
            y_row += h_res[j,i] * x_tile
       c. Cast y_row to orig dtype, write to y[i, d_start:]
"""

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def mhc_post(B, n, n_pad, D, block_D=256, dtype="float16"):
    num_physical_cores = 20
    total_blocks = B
    usedCoreNum = min(num_physical_cores, total_blocks)
    tasksPerCore = (total_blocks + usedCoreNum - 1) // usedCoreNum
    assert D % block_D == 0
    d_tiles = D // block_D

    need_cast = dtype != "float32"
    out_cast_mode = "CAST_ROUND" if dtype == "bfloat16" else "CAST_NONE"

    @T.prim_func
    def main(
        x: T.Tensor((B, n, D), dtype),
        h_res: T.Tensor((B, n, n_pad), "float32"),
        h_out: T.Tensor((B, D), dtype),
        h_post: T.Tensor((B, n_pad), "float32"),
        y: T.Tensor((B, n, D), dtype),
    ):
        with T.Kernel(usedCoreNum, is_npu=True) as (cid, vid):
            # ---- UB allocations ----
            hout_tile_ub = T.alloc_ub((1, block_D), "float32")
            x_row_ub = T.alloc_ub((1, block_D), "float32")
            y_row_ub = T.alloc_ub((1, block_D), "float32")
            tmp_mul_ub = T.alloc_ub((1, block_D), "float32")

            # Cast IO buffers
            row_in_ub = T.alloc_ub((1, block_D), dtype)
            row_out_ub = T.alloc_ub((1, block_D), dtype)

            # Small buffers: padded to n_pad for 32B alignment
            hres_row_ub = T.alloc_ub((n_pad,), "float32")
            hpost_ub = T.alloc_ub((n_pad,), "float32")

            for localIdx in T.serial(tasksPerCore):
                batch_id = cid * tasksPerCore + localIdx

                with T.Scope("V"):
                    if batch_id < total_blocks:
                        # Load h_post once per batch element
                        T.copy(h_post[batch_id, 0], hpost_ub)

                        for dt in T.serial(d_tiles):
                            d_start = dt * block_D

                            # Load h_out tile, cast to fp32
                            if need_cast:
                                T.copy(h_out[batch_id, d_start], row_in_ub)
                                T.tile.cast(hout_tile_ub, row_in_ub, mode="CAST_NONE", count=block_D)
                            else:
                                T.copy(h_out[batch_id, d_start], hout_tile_ub)

                            # For each output row i
                            for i in T.serial(n):
                                # y_row = h_post[i] * h_out_tile
                                hpost_i = hpost_ub[i]
                                T.tile.mul(y_row_ub, hout_tile_ub, hpost_i)

                                # Accumulate h_res^T @ x: sum_j(h_res[j,i] * x[j, d_tile])
                                for j in T.serial(n):
                                    # Load h_res row j
                                    T.copy(h_res[batch_id, j, 0], hres_row_ub)
                                    hres_ji = hres_row_ub[i]

                                    # Load x[j, d_tile], cast to fp32
                                    if need_cast:
                                        T.copy(x[batch_id, j, d_start], row_in_ub)
                                        T.tile.cast(x_row_ub, row_in_ub, mode="CAST_NONE", count=block_D)
                                    else:
                                        T.copy(x[batch_id, j, d_start], x_row_ub)

                                    # y_row += h_res[j,i] * x[j, d_tile]
                                    T.tile.mul(tmp_mul_ub, x_row_ub, hres_ji)
                                    T.tile.add(y_row_ub, y_row_ub, tmp_mul_ub)

                                # Cast and write output
                                if need_cast:
                                    T.tile.cast(row_out_ub, y_row_ub, mode=out_cast_mode, count=block_D)
                                    T.copy(row_out_ub, y[batch_id, i, d_start])
                                else:
                                    T.copy(y_row_ub, y[batch_id, i, d_start])

    return main
