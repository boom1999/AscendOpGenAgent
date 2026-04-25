"""Block-level TileLang design for MhcPostGrad.

MhcPostGrad computes four backward gradients from the mHC Post forward:
  grad_x      = h_res @ grad_output          (B, n, n) @ (B, n, D) -> (B, n, D)
  grad_h_res  = x @ grad_output^T            (B, n, D) @ (B, D, n) -> (B, n, n)
  grad_h_out  = sum(grad_output * h_post, -2) (B, n, D) * (B, n, 1) -> (B, D)
  grad_h_post = sum(grad_output * h_out, -1)  (B, n, D) * (B, 1, D) -> (B, n)

Key properties:
- n is very small (4, 6, 8), D is large (384-5120).
- All computation is element-wise or small-matrix multiply -- Vector only, no Cube.
- Each batch element is fully independent, so we partition along the batch dim.
- D is the large dimension; we tile along D to fit in UB.
- h_res and h_post padded to n_pad (multiple of 8) for 32B DMA alignment.

Block-level decisions:
- Flatten all leading dims into a single batch B (done in model_new_tilelang.py).
- Each block handles one batch element.
- Persistent-kernel: each core iterates over a contiguous batch range.
- Pure Vector scope -- no Cube/Vector pipeline needed.
- Two-phase design per batch element:
  Phase 1: D-tiled loop computes grad_h_out, grad_h_post, and grad_x.
  Phase 2: For each row i, a separate D-tiled loop computes grad_h_res row i,
           keeping the accumulator in UB across the entire D pass.
"""

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[5, 6, 7, 8], pass_configs=pass_configs)
def mhc_post_grad(B, n, n_pad, D, block_D=256, dtype="float16"):
    """Block-level kernel for MhcPostGrad.

    Args:
        B: batch size (flattened leading dims)
        n: small head dim (4, 6, 8)
        n_pad: n padded to multiple of 8 for 32B alignment
        D: large hidden dim, padded to multiple of block_D
        block_D: tile size along D dimension
        dtype: input/output dtype string
    """
    num_physical_cores = 20
    total_blocks = B
    usedCoreNum = min(num_physical_cores, total_blocks)
    tasksPerCore = (total_blocks + usedCoreNum - 1) // usedCoreNum
    d_tiles = D // block_D

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
            coreIdx = cid

            for localIdx in T.serial(tasksPerCore):
                batch_id = coreIdx * tasksPerCore + localIdx

                with T.Scope("V"):
                    if batch_id < total_blocks:
                        # ==== Phase 1: grad_h_out, grad_h_post, grad_x ====
                        # Tile along D dimension.
                        for dt in T.serial(d_tiles):
                            # TODO(tile-level):
                            # - Load grad_output rows, h_out tile, h_post
                            # - grad_h_out[d] = sum_j(h_post[j] * grad_output[j, d])
                            # - grad_h_post[j] += sum_d(grad_output[j, d] * h_out[d])
                            # - For each i: grad_x[i, d] = sum_j(h_res[i,j] * go[j, d])
                            _ = grad_output
                            _ = x
                            _ = h_res
                            _ = h_out
                            _ = h_post
                            _ = grad_x
                            _ = grad_h_out

                        # ==== Phase 2: grad_h_res (row by row, full D pass) ====
                        # For each row i, iterate over all D-tiles and accumulate
                        # the dot product sum_d(x[i,d] * grad_output[j,d]) for each j.
                        # Keep accumulator in UB across all D-tiles.
                        for i in T.serial(n):
                            for dt in T.serial(d_tiles):
                                # TODO(tile-level):
                                # - Load x[i, d_tile], grad_output[j, d_tile]
                                # - ghres_row[j] += dot(x[i, :], grad_output[j, :])
                                pass
                            # Write completed row to GM
                            _ = grad_h_res

                        # Write grad_h_post to GM
                        _ = grad_h_post

    return main
