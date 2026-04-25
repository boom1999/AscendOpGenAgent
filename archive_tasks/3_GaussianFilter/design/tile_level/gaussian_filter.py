"""Tile-level design for GaussianFilter.

3DGS Gaussian Filter: per (b, c) pair, compute validity mask,
stream-compact valid elements, and pack mask to uint8.

Layout: Feature-major (BC, F, N_padded) for all multi-feature tensors.
All T.copy operations are 1D on TILE_N contiguous elements (DMA-aligned).

Algorithm per (b, c) work item:
  1. Vectorized mask computation
  2. Stream compaction (scalar — unavoidable)
  3. Uint8 bit packing (scalar)

Hardware: All Vector work. No Cube. TILE_N=256 tiling over N dimension.
"""

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=list(range(10, 20)), pass_configs=pass_configs)
def gaussian_filter_kernel(
    BC, B, N_padded, M_padded,
    near_plane, far_plane, width, height,
    dtype="float32",
):
    TILE_N = 256
    n_tiles = N_padded // TILE_N
    bytes_per_tile = TILE_N // 8

    num_physical_cores = 20
    num_cores = min(num_physical_cores, BC)
    tasks_per_core = (BC + num_cores - 1) // num_cores

    @T.prim_func
    def main(
        # --- Inputs (0..9): feature-major ---
        means:     T.Tensor((B, 3, N_padded), dtype),
        colors:    T.Tensor((B, 3, N_padded), dtype),
        det:       T.Tensor((BC, N_padded), dtype),
        opacities: T.Tensor((B, N_padded), dtype),
        means2d:   T.Tensor((BC, 2, N_padded), dtype),
        depths:    T.Tensor((BC, N_padded), dtype),
        radius:    T.Tensor((BC, 2, N_padded), dtype),
        conics:    T.Tensor((BC, 3, N_padded), dtype),
        covars2d:  T.Tensor((BC, 3, N_padded), dtype),
        b_map:     T.Tensor((BC,), "int32"),
        # --- Outputs (10..19): feature-major ---
        means_out:     T.Tensor((BC, 3, N_padded), dtype),
        colors_out:    T.Tensor((BC, 3, N_padded), dtype),
        means2d_out:   T.Tensor((BC, 2, N_padded), dtype),
        depths_out:    T.Tensor((BC, N_padded), dtype),
        radius_out_t:  T.Tensor((BC, 2, N_padded), dtype),
        covars2d_out:  T.Tensor((BC, 3, N_padded), dtype),
        conics_out:    T.Tensor((BC, 3, N_padded), dtype),
        opacities_out: T.Tensor((BC, N_padded), dtype),
        filter_uint8:  T.Tensor((BC, M_padded), "uint8"),
        cnt_out:       T.Tensor((BC,), "int32"),
    ):
        with T.Kernel(num_cores, is_npu=True) as (cid, vid):
            # ========= UB Allocations =========

            # Mask computation
            det_ub     = T.alloc_ub((TILE_N,), dtype)
            depths_ub  = T.alloc_ub((TILE_N,), dtype)
            m2d_x_ub   = T.alloc_ub((TILE_N,), dtype)
            m2d_y_ub   = T.alloc_ub((TILE_N,), dtype)
            rad_x_ub   = T.alloc_ub((TILE_N,), dtype)
            rad_y_ub   = T.alloc_ub((TILE_N,), dtype)
            valid_ub   = T.alloc_ub((TILE_N,), dtype)
            inside_ub  = T.alloc_ub((TILE_N,), dtype)
            mask_ub    = T.alloc_ub((TILE_N,), dtype)
            tmp_ub     = T.alloc_ub((TILE_N,), dtype)
            ones_ub    = T.alloc_ub((TILE_N,), dtype)
            zeros_ub   = T.alloc_ub((TILE_N,), dtype)

            # Compaction data: all feature channels loaded as 1D
            m0_ub   = T.alloc_ub((TILE_N,), dtype)
            m1_ub   = T.alloc_ub((TILE_N,), dtype)
            m2_ub   = T.alloc_ub((TILE_N,), dtype)
            c0_ub   = T.alloc_ub((TILE_N,), dtype)
            c1_ub   = T.alloc_ub((TILE_N,), dtype)
            c2_ub   = T.alloc_ub((TILE_N,), dtype)
            con0_ub = T.alloc_ub((TILE_N,), dtype)
            con1_ub = T.alloc_ub((TILE_N,), dtype)
            con2_ub = T.alloc_ub((TILE_N,), dtype)
            cov0_ub = T.alloc_ub((TILE_N,), dtype)
            cov1_ub = T.alloc_ub((TILE_N,), dtype)
            cov2_ub = T.alloc_ub((TILE_N,), dtype)
            ro0_ub  = T.alloc_ub((TILE_N,), dtype)
            ro1_ub  = T.alloc_ub((TILE_N,), dtype)
            opac_ub = T.alloc_ub((TILE_N,), dtype)

            # Fill and output element buffers
            fill_ub = T.alloc_ub((TILE_N,), dtype)
            out1    = T.alloc_ub((1,), dtype)

            # Uint8 packing
            pack_ub  = T.alloc_ub((bytes_per_tile,), "uint8")
            pack_int = T.alloc_ub((1,), "int32")
            pow_val  = T.alloc_ub((1,), "int32")

            # Counter
            cnt_local = T.alloc_ub((1,), "int32")
            cnt_tmp   = T.alloc_ub((1,), "int32")

            with T.Scope("V"):
                T.tile.fill(ones_ub, T.float32(1.0))
                T.tile.fill(zeros_ub, T.float32(0.0))
                T.tile.fill(fill_ub, T.float32(1.0))

                for local_idx in T.serial(tasks_per_core):
                    bc = cid * tasks_per_core + local_idx
                    if bc < BC:
                        b_idx = T.cast(b_map[bc], "int32")

                        # ====== Phase A: Initialize outputs to 1.0 ======
                        for t in T.serial(n_tiles):
                            ts = t * TILE_N
                            T.copy(fill_ub, means_out[bc, 0, ts])
                            T.copy(fill_ub, means_out[bc, 1, ts])
                            T.copy(fill_ub, means_out[bc, 2, ts])
                            T.copy(fill_ub, colors_out[bc, 0, ts])
                            T.copy(fill_ub, colors_out[bc, 1, ts])
                            T.copy(fill_ub, colors_out[bc, 2, ts])
                            T.copy(fill_ub, means2d_out[bc, 0, ts])
                            T.copy(fill_ub, means2d_out[bc, 1, ts])
                            T.copy(fill_ub, depths_out[bc, ts])
                            T.copy(fill_ub, radius_out_t[bc, 0, ts])
                            T.copy(fill_ub, radius_out_t[bc, 1, ts])
                            T.copy(fill_ub, covars2d_out[bc, 0, ts])
                            T.copy(fill_ub, covars2d_out[bc, 1, ts])
                            T.copy(fill_ub, covars2d_out[bc, 2, ts])
                            T.copy(fill_ub, conics_out[bc, 0, ts])
                            T.copy(fill_ub, conics_out[bc, 1, ts])
                            T.copy(fill_ub, conics_out[bc, 2, ts])
                            T.copy(fill_ub, opacities_out[bc, ts])

                        # ====== Phase B: Mask + Compact + Pack ======
                        cnt_local[0] = T.cast(0, "int32")

                        for tile_idx in T.serial(n_tiles):
                            tile_start = tile_idx * TILE_N

                            # -- B.1: Load mask inputs (1D per channel) --
                            T.copy(det[bc, tile_start], det_ub)
                            T.copy(depths[bc, tile_start], depths_ub)
                            T.copy(means2d[bc, 0, tile_start], m2d_x_ub)
                            T.copy(means2d[bc, 1, tile_start], m2d_y_ub)
                            T.copy(radius[bc, 0, tile_start], rad_x_ub)
                            T.copy(radius[bc, 1, tile_start], rad_y_ub)

                            # -- B.2: Vectorized mask computation --
                            T.tile.compare(valid_ub, det_ub, T.float32(0.0), "GT")
                            T.tile.select(valid_ub, valid_ub, ones_ub, zeros_ub,
                                          "VSEL_CMPMASK_SPR")

                            T.tile.compare(tmp_ub, depths_ub,
                                           T.float32(near_plane), "GT")
                            T.tile.select(tmp_ub, tmp_ub, ones_ub, zeros_ub,
                                          "VSEL_CMPMASK_SPR")
                            T.tile.mul(valid_ub, valid_ub, tmp_ub)

                            T.tile.compare(tmp_ub, depths_ub,
                                           T.float32(far_plane), "LT")
                            T.tile.select(tmp_ub, tmp_ub, ones_ub, zeros_ub,
                                          "VSEL_CMPMASK_SPR")
                            T.tile.mul(valid_ub, valid_ub, tmp_ub)

                            T.tile.mul(rad_x_ub, rad_x_ub, valid_ub)
                            T.tile.mul(rad_y_ub, rad_y_ub, valid_ub)

                            T.tile.add(tmp_ub, m2d_x_ub, rad_x_ub)
                            T.tile.compare(inside_ub, tmp_ub,
                                           T.float32(0.0), "GT")
                            T.tile.select(inside_ub, inside_ub, ones_ub,
                                          zeros_ub, "VSEL_CMPMASK_SPR")

                            T.tile.sub(tmp_ub, m2d_x_ub, rad_x_ub)
                            T.tile.compare(tmp_ub, tmp_ub,
                                           T.float32(float(width)), "LT")
                            T.tile.select(tmp_ub, tmp_ub, ones_ub, zeros_ub,
                                          "VSEL_CMPMASK_SPR")
                            T.tile.mul(inside_ub, inside_ub, tmp_ub)

                            T.tile.add(tmp_ub, m2d_y_ub, rad_y_ub)
                            T.tile.compare(tmp_ub, tmp_ub,
                                           T.float32(0.0), "GT")
                            T.tile.select(tmp_ub, tmp_ub, ones_ub, zeros_ub,
                                          "VSEL_CMPMASK_SPR")
                            T.tile.mul(inside_ub, inside_ub, tmp_ub)

                            T.tile.sub(tmp_ub, m2d_y_ub, rad_y_ub)
                            T.tile.compare(tmp_ub, tmp_ub,
                                           T.float32(float(height)), "LT")
                            T.tile.select(tmp_ub, tmp_ub, ones_ub, zeros_ub,
                                          "VSEL_CMPMASK_SPR")
                            T.tile.mul(inside_ub, inside_ub, tmp_ub)

                            T.tile.mul(rad_x_ub, rad_x_ub, inside_ub)
                            T.tile.mul(rad_y_ub, rad_y_ub, inside_ub)

                            T.tile.mul(mask_ub, valid_ub, inside_ub)

                            # -- B.3: Load ALL compaction data --
                            T.copy(means[b_idx, 0, tile_start], m0_ub)
                            T.copy(means[b_idx, 1, tile_start], m1_ub)
                            T.copy(means[b_idx, 2, tile_start], m2_ub)
                            T.copy(colors[b_idx, 0, tile_start], c0_ub)
                            T.copy(colors[b_idx, 1, tile_start], c1_ub)
                            T.copy(colors[b_idx, 2, tile_start], c2_ub)
                            T.copy(conics[bc, 0, tile_start], con0_ub)
                            T.copy(conics[bc, 1, tile_start], con1_ub)
                            T.copy(conics[bc, 2, tile_start], con2_ub)
                            T.copy(covars2d[bc, 0, tile_start], cov0_ub)
                            T.copy(covars2d[bc, 1, tile_start], cov1_ub)
                            T.copy(covars2d[bc, 2, tile_start], cov2_ub)
                            T.copy(radius[bc, 0, tile_start], ro0_ub)
                            T.copy(radius[bc, 1, tile_start], ro1_ub)
                            T.copy(opacities[b_idx, tile_start], opac_ub)

                            # -- B.4: Scalar compaction (single pass) --
                            for elem in T.serial(TILE_N):
                                if mask_ub[elem] > T.float32(0.5):
                                    wp = cnt_local[0]

                                    out1[0] = m0_ub[elem]
                                    T.copy(out1, means_out[bc, 0, wp])
                                    out1[0] = m1_ub[elem]
                                    T.copy(out1, means_out[bc, 1, wp])
                                    out1[0] = m2_ub[elem]
                                    T.copy(out1, means_out[bc, 2, wp])

                                    out1[0] = c0_ub[elem]
                                    T.copy(out1, colors_out[bc, 0, wp])
                                    out1[0] = c1_ub[elem]
                                    T.copy(out1, colors_out[bc, 1, wp])
                                    out1[0] = c2_ub[elem]
                                    T.copy(out1, colors_out[bc, 2, wp])

                                    out1[0] = con0_ub[elem]
                                    T.copy(out1, conics_out[bc, 0, wp])
                                    out1[0] = con1_ub[elem]
                                    T.copy(out1, conics_out[bc, 1, wp])
                                    out1[0] = con2_ub[elem]
                                    T.copy(out1, conics_out[bc, 2, wp])

                                    out1[0] = cov0_ub[elem]
                                    T.copy(out1, covars2d_out[bc, 0, wp])
                                    out1[0] = cov1_ub[elem]
                                    T.copy(out1, covars2d_out[bc, 1, wp])
                                    out1[0] = cov2_ub[elem]
                                    T.copy(out1, covars2d_out[bc, 2, wp])

                                    out1[0] = m2d_x_ub[elem]
                                    T.copy(out1, means2d_out[bc, 0, wp])
                                    out1[0] = m2d_y_ub[elem]
                                    T.copy(out1, means2d_out[bc, 1, wp])

                                    out1[0] = ro0_ub[elem]
                                    T.copy(out1, radius_out_t[bc, 0, wp])
                                    out1[0] = ro1_ub[elem]
                                    T.copy(out1, radius_out_t[bc, 1, wp])

                                    out1[0] = depths_ub[elem]
                                    T.copy(out1, depths_out[bc, wp])

                                    out1[0] = opac_ub[elem]
                                    T.copy(out1, opacities_out[bc, wp])

                                    cnt_local[0] = cnt_local[0] + T.cast(1, "int32")

                            # -- B.5: Uint8 bit packing --
                            for g in T.serial(bytes_per_tile):
                                pack_int[0] = T.cast(0, "int32")
                                pow_val[0] = T.cast(1, "int32")
                                for bit in T.serial(8):
                                    idx = g * 8 + bit
                                    if mask_ub[idx] > T.float32(0.5):
                                        pack_int[0] = pack_int[0] + pow_val[0]
                                    pow_val[0] = pow_val[0] + pow_val[0]
                                pack_ub[g] = T.cast(pack_int[0], "uint8")

                            byte_offset = tile_idx * bytes_per_tile
                            T.copy(pack_ub, filter_uint8[bc, byte_offset])

                        # Write final count
                        cnt_tmp[0] = cnt_local[0]
                        T.copy(cnt_tmp, cnt_out[bc])

    return main
