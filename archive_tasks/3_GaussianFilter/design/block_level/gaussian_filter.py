"""Block-level design for GaussianFilter.

3DGS Gaussian Filter: per (b, c) pair, compute validity mask, stream-compact
valid elements, and pack mask to uint8.

Hardware mapping:
  - All Vector work (mask computation, gather/compact, bitpack). No Cube.
  - Total work items = BC = B * C, distributed across cores.
  - Each work item tiles over N dimension in chunks of TILE_N.

Input layout (after wrapper permute/flatten):
  means:     (B, N, 3)    fp32  — shared across C, indexed by b_map
  colors:    (B, N, 3)    fp32  — shared across C, indexed by b_map
  det:       (BC, N)      fp32
  opacities: (B, N)       fp32  — shared across C, indexed by b_map
  means2d:   (BC, N, 2)   fp32
  depths:    (BC, N)      fp32
  radius:    (BC, N, 2)   fp32  — also used as radius_out source
  conics:    (BC, N, 3)   fp32
  covars2d:  (BC, N, 3)   fp32
  b_map:     (BC,)        int32 — maps bc index to b index

Output layout:
  means_out:     (BC, N, 3)   fp32  — initialized to 1.0, first cnt rows filled
  colors_out:    (BC, N, 3)   fp32
  means2d_out:   (BC, N, 2)   fp32
  depths_out:    (BC, N)      fp32
  radius_out:    (BC, N, 2)   fp32
  covars2d_out:  (BC, N, 3)   fp32
  conics_out:    (BC, N, 3)   fp32
  opacities_out: (BC, N)      fp32
  filter_uint8:  (BC, M)      uint8  — M = (N+7)//8
  cnt_out:       (BC,)        int32
"""

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def gaussian_filter_kernel(
    BC, B, N, M,
    near_plane, far_plane, width, height,
    dtype="float32",
):
    """Block-level kernel generator.

    Args:
        BC: B * C (total work items)
        B:  batch size (for b_map indexing into means/colors/opacities)
        N:  number of Gaussians per (b, c) pair
        M:  (N + 7) // 8  (packed uint8 length)
        near_plane, far_plane, width, height: scalar filter params
    """
    TILE_N = min(N, 512)
    n_tiles = (N + TILE_N - 1) // TILE_N

    num_physical_cores = 20
    usedCoreNum = min(num_physical_cores, BC)
    tasksPerCore = (BC + usedCoreNum - 1) // usedCoreNum

    @T.prim_func
    def main(
        # --- Inputs (indices 0..9) ---
        means:     T.Tensor((B, N, 3), dtype),       # 0
        colors:    T.Tensor((B, N, 3), dtype),        # 1
        det:       T.Tensor((BC, N), dtype),           # 2
        opacities: T.Tensor((B, N), dtype),            # 3
        means2d:   T.Tensor((BC, N, 2), dtype),        # 4
        depths:    T.Tensor((BC, N), dtype),            # 5
        radius:    T.Tensor((BC, N, 2), dtype),         # 6
        conics:    T.Tensor((BC, N, 3), dtype),         # 7
        covars2d:  T.Tensor((BC, N, 3), dtype),         # 8
        b_map:     T.Tensor((BC,), "int32"),            # 9
        # --- Outputs (indices 10..19) ---
        means_out:     T.Tensor((BC, N, 3), dtype),    # 10
        colors_out:    T.Tensor((BC, N, 3), dtype),    # 11
        means2d_out:   T.Tensor((BC, N, 2), dtype),    # 12
        depths_out:    T.Tensor((BC, N), dtype),        # 13
        radius_out:    T.Tensor((BC, N, 2), dtype),     # 14
        covars2d_out:  T.Tensor((BC, N, 3), dtype),     # 15
        conics_out:    T.Tensor((BC, N, 3), dtype),     # 16
        opacities_out: T.Tensor((BC, N), dtype),        # 17
        filter_uint8:  T.Tensor((BC, M), "uint8"),      # 18
        cnt_out:       T.Tensor((BC,), "int32"),         # 19
    ):
        with T.Kernel(usedCoreNum, is_npu=True) as (cid, vid):
            with T.Scope("V"):
                for localIdx in T.serial(tasksPerCore):
                    bc = cid * tasksPerCore + localIdx
                    if bc < BC:
                        b_idx = T.cast(b_map[bc], "int32")

                        # TODO(tile-level):
                        # Per (b, c) pair, tile over N:
                        #   1. Load det, depths, means2d, radius tiles
                        #   2. Vectorized mask: valid = (det>0) & (depths>near) & (depths<far)
                        #   3. Set radius=0 where ~valid
                        #   4. Vectorized inside check with screen bounds
                        #   5. Set radius=0 where ~inside
                        #   6. filter_mask = valid & inside
                        #   7. Stream compact: gather valid from all inputs, write sequentially
                        #   8. Pack filter_mask bits into uint8
                        #   9. Write cnt_out[bc]

    return main
