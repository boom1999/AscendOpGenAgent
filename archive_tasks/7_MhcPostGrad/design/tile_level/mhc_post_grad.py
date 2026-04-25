"""Tile-level TileLang design for MhcPostGrad.

Computes four backward gradients fused into a single kernel:
  grad_x      = h_res @ grad_output          -> (B, n, D), cast to orig_dtype
  grad_h_res  = x @ grad_output^T            -> (B, n, n), float32
  grad_h_out  = sum(grad_output * h_post, -2) -> (B, D), cast to orig_dtype
  grad_h_post = sum(grad_output * h_out, -1)  -> (B, n), float32

Strategy:
- Pure Vector kernel (n is tiny: 4-8, D is large: 384-5120).
- Persistent-kernel partitioning over batch dim B.
- D tiled in chunks of block_D (D must be a multiple of block_D -- host pads).
- h_res and h_post padded to n_pad for 32B DMA alignment.
- Accumulate grad_h_res row by row in UB, one complete row per full D pass.
  No workspace spill needed.
"""

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[5, 6, 7, 8], pass_configs=pass_configs)
def mhc_post_grad(B, n, n_pad, D, block_D=256, dtype="float16"):
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
        grad_output: T.Tensor((B, n, D), dtype),
        x: T.Tensor((B, n, D), dtype),
        h_res: T.Tensor((B, n, n_pad), "float32"),
        h_out: T.Tensor((B, D), dtype),
        h_post: T.Tensor((B, n_pad), "float32"),
        grad_x: T.Tensor((B, n, D), dtype),
        grad_h_res: T.Tensor((B, n, n_pad), "float32"),
        grad_h_out: T.Tensor((B, D), dtype),
        grad_h_post: T.Tensor((B, n_pad), "float32"),
    ):
        with T.Kernel(usedCoreNum, is_npu=True) as (cid, vid):
            # ---- UB allocations ----
            go_row_ub = T.alloc_ub((1, block_D), "float32")
            x_row_ub = T.alloc_ub((1, block_D), "float32")
            hout_tile_ub = T.alloc_ub((1, block_D), "float32")
            gx_row_ub = T.alloc_ub((1, block_D), "float32")
            ghout_tile_ub = T.alloc_ub((1, block_D), "float32")
            tmp_mul_ub = T.alloc_ub((1, block_D), "float32")

            # Cast IO buffers
            row_in_ub = T.alloc_ub((1, block_D), dtype)
            row_out_ub = T.alloc_ub((1, block_D), dtype)

            # Small buffers: padded to n_pad for 32B alignment
            hres_row_ub = T.alloc_ub((n_pad,), "float32")
            hpost_ub = T.alloc_ub((n_pad,), "float32")
            ghres_row_ub = T.alloc_ub((n_pad,), "float32")
            ghpost_ub = T.alloc_ub((n_pad,), "float32")

            # Reduce tmp for dot products
            reduce_tmp = T.alloc_ub((2 * block_D,), "uint8")

            for localIdx in T.serial(tasksPerCore):
                batch_id = cid * tasksPerCore + localIdx

                with T.Scope("V"):
                    if batch_id < total_blocks:
                        # Load h_post once per batch element
                        T.copy(h_post[batch_id, 0], hpost_ub)

                        # Zero grad_h_post accumulator
                        T.tile.fill(ghpost_ub, T.float32(0))

                        # ==== Phase 1: grad_h_out, grad_h_post, and grad_x ====
                        for dt in T.serial(d_tiles):
                            d_start = dt * block_D

                            # Load h_out tile
                            if need_cast:
                                T.copy(h_out[batch_id, d_start], row_in_ub)
                                T.tile.cast(hout_tile_ub, row_in_ub, mode="CAST_NONE", count=block_D)
                            else:
                                T.copy(h_out[batch_id, d_start], hout_tile_ub)

                            # Zero grad_h_out tile
                            T.tile.fill(ghout_tile_ub, T.float32(0))

                            # --- grad_h_out and grad_h_post ---
                            for j in T.serial(n):
                                if need_cast:
                                    T.copy(grad_output[batch_id, j, d_start], row_in_ub)
                                    T.tile.cast(go_row_ub, row_in_ub, mode="CAST_NONE", count=block_D)
                                else:
                                    T.copy(grad_output[batch_id, j, d_start], go_row_ub)

                                hpost_j = hpost_ub[j]
                                T.tile.mul(tmp_mul_ub, go_row_ub, hpost_j)
                                T.tile.add(ghout_tile_ub, ghout_tile_ub, tmp_mul_ub)

                                T.tile.mul(tmp_mul_ub, go_row_ub, hout_tile_ub)
                                T.reduce_sum(tmp_mul_ub, tmp_mul_ub[:, 0], reduce_tmp, dim=-1)
                                ghpost_ub[j] = ghpost_ub[j] + tmp_mul_ub[0, 0]

                            # --- grad_x (row by row for this D-tile) ---
                            for i in T.serial(n):
                                T.copy(h_res[batch_id, i, 0], hres_row_ub)
                                T.tile.fill(gx_row_ub, T.float32(0))

                                for j in T.serial(n):
                                    if need_cast:
                                        T.copy(grad_output[batch_id, j, d_start], row_in_ub)
                                        T.tile.cast(go_row_ub, row_in_ub, mode="CAST_NONE", count=block_D)
                                    else:
                                        T.copy(grad_output[batch_id, j, d_start], go_row_ub)

                                    hres_ij = hres_row_ub[j]
                                    T.tile.mul(tmp_mul_ub, go_row_ub, hres_ij)
                                    T.tile.add(gx_row_ub, gx_row_ub, tmp_mul_ub)

                                if need_cast:
                                    T.tile.cast(row_out_ub, gx_row_ub, mode=out_cast_mode, count=block_D)
                                    T.copy(row_out_ub, grad_x[batch_id, i, d_start])
                                else:
                                    T.copy(gx_row_ub, grad_x[batch_id, i, d_start])

                            # Write grad_h_out tile
                            if need_cast:
                                T.tile.cast(row_out_ub, ghout_tile_ub, mode=out_cast_mode, count=block_D)
                                T.copy(row_out_ub, grad_h_out[batch_id, d_start])
                            else:
                                T.copy(ghout_tile_ub, grad_h_out[batch_id, d_start])

                        # ==== Phase 2: grad_h_res (row by row, full D pass) ====
                        for i in T.serial(n):
                            T.tile.fill(ghres_row_ub, T.float32(0))

                            for dt in T.serial(d_tiles):
                                d_start = dt * block_D

                                # Load x[batch_id, i, d_tile]
                                if need_cast:
                                    T.copy(x[batch_id, i, d_start], row_in_ub)
                                    T.tile.cast(x_row_ub, row_in_ub, mode="CAST_NONE", count=block_D)
                                else:
                                    T.copy(x[batch_id, i, d_start], x_row_ub)

                                for j in T.serial(n):
                                    if need_cast:
                                        T.copy(grad_output[batch_id, j, d_start], row_in_ub)
                                        T.tile.cast(go_row_ub, row_in_ub, mode="CAST_NONE", count=block_D)
                                    else:
                                        T.copy(grad_output[batch_id, j, d_start], go_row_ub)

                                    T.tile.mul(tmp_mul_ub, x_row_ub, go_row_ub)
                                    T.reduce_sum(tmp_mul_ub, tmp_mul_ub[:, 0], reduce_tmp, dim=-1)
                                    ghres_row_ub[j] = ghres_row_ub[j] + tmp_mul_ub[0, 0]

                            # Write completed row to GM
                            T.copy(ghres_row_ub, grad_h_res[batch_id, i, 0])

                        # Write accumulated grad_h_post
                        T.copy(ghpost_ub, grad_h_post[batch_id, 0])

    return main
