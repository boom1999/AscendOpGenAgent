"""Block-level TileLang design for ManifoldConstrainedHyperConnectionPost.

Computes:
  y[i, d] = h_post[i] * h_out[d] + sum_j(h_res[j, i] * x[j, d])

This is an outer-product (h_post outer h_out) plus a small matmul (h_res^T @ x),
fused into a single kernel.

Inputs:
  x:      (B, n, D)     dtype (fp16/bf16)
  h_res:  (B, n, n_pad) float32
  h_out:  (B, D)        dtype (fp16/bf16)
  h_post: (B, n_pad)    float32
Output:
  y:      (B, n, D)     dtype (fp16/bf16)

Strategy:
- Pure Vector kernel (n is tiny: 4-8, D is large: 384-24359).
- Persistent-kernel partitioning over batch dim B.
- D tiled in chunks of block_D (D must be a multiple of block_D -- host pads).
- h_res and h_post padded to n_pad for 32B DMA alignment.
- For each batch element, for each D-tile:
    1. Load h_out[d_start : d_start+block_D] once
    2. For each output row i (0..n-1):
       a. y_row = h_post[i] * h_out_tile   (scalar-vector mul)
       b. For each j (0..n-1):
            y_row += h_res[j, i] * x[j, d_start : d_start+block_D]
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
            # TODO(tile-level): allocate UB buffers for D-tile computation
            # TODO(tile-level): allocate cast IO buffers
            # TODO(tile-level): allocate small buffers for h_res column, h_post

            for localIdx in T.serial(tasksPerCore):
                batch_id = cid * tasksPerCore + localIdx

                with T.Scope("V"):
                    if batch_id < total_blocks:
                        # TODO(tile-level): Load h_post once per batch element

                        for dt in T.serial(d_tiles):
                            d_start = dt * block_D
                            # TODO(tile-level): Load h_out tile, cast to fp32
                            # TODO(tile-level): For each output row i:
                            #   1. y_row = h_post[i] * h_out_tile
                            #   2. For each j: y_row += h_res[j,i] * x[j, d_tile]
                            #   3. Cast y_row, write to y[i, d_start:]

    return main
