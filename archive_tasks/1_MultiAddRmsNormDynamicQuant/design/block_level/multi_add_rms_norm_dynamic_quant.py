"""Block-level TileLang design for MultiAddRmsNormDynamicQuant.

Computation per row:
  1. x_sum = x1 + x2
  2. RmsNorm: y = (x_sum / rms) * gamma, rms = sqrt(mean(x_sum^2) + eps)
  3. DynamicQuant: scale = max(abs(input))/127, y_quant = round(input/scale).clamp(-128,127).int8
     - input = y if no smooth_scale, or y * smooth_scale

All N values in test cases (4096-7680) fall in the single-row regime (1024 < N <= 8192).
Only a single_row template is needed.

Three kernel variants based on smooth_scale presence:
  - no_smooth: (x1, x2, gamma) -> (x_sum, y_norm, y1, scale1)
  - smooth1:   (x1, x2, gamma, ss1) -> (x_sum, y_norm, y1, scale1)
  - dual_smooth: (x1, x2, gamma, ss1, ss2) -> (x_sum, y_norm, y1, scale1, y2, scale2)

Gamma and smooth_scales are always passed as 2D [M, N] (Python wrapper handles broadcasting).
"""

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[3, 4, 5, 6], pass_configs=pass_configs)
def multi_add_rms_norm_quant_no_smooth(M, N, eps=1e-5, dtype="bfloat16"):
    """No smooth scale: x1, x2, gamma -> x_sum, y_norm, y1_quant, scale1."""
    block_M = 64
    num_physical_cores = 20
    m_num = (M + block_M - 1) // block_M
    usedCoreNum = min(num_physical_cores, m_num)
    tasksPerCore = (m_num + usedCoreNum - 1) // usedCoreNum
    vec_num = 2
    sub_block_M = block_M // vec_num

    @T.prim_func
    def main(
        X1: T.Tensor((M, N), dtype),
        X2: T.Tensor((M, N), dtype),
        Gamma: T.Tensor((M, N), dtype),
        XSum: T.Tensor((M, N), dtype),
        YNorm: T.Tensor((M, N), dtype),
        Y1: T.Tensor((M, N), "int8"),
        Scale1: T.Tensor((M,), "float32"),
    ):
        with T.Kernel(usedCoreNum, is_npu=True) as (cid, vid):
            coreIdx = cid

            for localIdx in T.serial(tasksPerCore):
                bx = coreIdx * tasksPerCore + localIdx

                with T.Scope("V"):
                    if bx < m_num:
                        for row in T.serial(sub_block_M):
                            row_idx = bx * block_M + vid * sub_block_M + row
                            if row_idx < M:
                                # TODO(tile-level):
                                # 1. Load X1[row_idx, :], X2[row_idx, :] -> add -> x_sum_f32
                                # 2. Cast x_sum_f32 to dtype -> store XSum[row_idx, :]
                                # 3. sum_sq = reduce_sum(x_sum_f32 * x_sum_f32)
                                # 4. inv_rms = rsqrt(sum_sq / N + eps)
                                # 5. Load Gamma[row_idx, :], cast to f32
                                # 6. y_norm_f32 = x_sum_f32 * inv_rms * gamma_f32
                                # 7. Cast y_norm_f32 to dtype -> store YNorm[row_idx, :]
                                # 8. abs_max = reduce_max(abs(y_norm_f32))
                                # 9. scale1 = abs_max / 127.0 -> store Scale1[row_idx]
                                # 10. y1_f32 = round(y_norm_f32 / scale1)
                                # 11. clamp [-128, 127], cast to int8 -> store Y1[row_idx, :]
                                pass

    _ = eps
    return main


@tilelang.jit(out_idx=[4, 5, 6, 7], pass_configs=pass_configs)
def multi_add_rms_norm_quant_smooth1(M, N, eps=1e-5, dtype="bfloat16"):
    """Single smooth scale: x1, x2, gamma, ss1 -> x_sum, y_norm, y1_quant, scale1."""
    block_M = 64
    num_physical_cores = 20
    m_num = (M + block_M - 1) // block_M
    usedCoreNum = min(num_physical_cores, m_num)
    tasksPerCore = (m_num + usedCoreNum - 1) // usedCoreNum
    vec_num = 2
    sub_block_M = block_M // vec_num

    @T.prim_func
    def main(
        X1: T.Tensor((M, N), dtype),
        X2: T.Tensor((M, N), dtype),
        Gamma: T.Tensor((M, N), dtype),
        SS1: T.Tensor((M, N), dtype),
        XSum: T.Tensor((M, N), dtype),
        YNorm: T.Tensor((M, N), dtype),
        Y1: T.Tensor((M, N), "int8"),
        Scale1: T.Tensor((M,), "float32"),
    ):
        with T.Kernel(usedCoreNum, is_npu=True) as (cid, vid):
            coreIdx = cid

            for localIdx in T.serial(tasksPerCore):
                bx = coreIdx * tasksPerCore + localIdx

                with T.Scope("V"):
                    if bx < m_num:
                        for row in T.serial(sub_block_M):
                            row_idx = bx * block_M + vid * sub_block_M + row
                            if row_idx < M:
                                # TODO(tile-level):
                                # 1-7: Same as no_smooth (add, rmsnorm, store x_sum + y_norm)
                                # 8. Load SS1[row_idx, :], cast to f32
                                # 9. input1_f32 = y_norm_f32 * ss1_f32
                                # 10. abs_max = reduce_max(abs(input1_f32))
                                # 11. scale1 = abs_max / 127.0 -> store Scale1[row_idx]
                                # 12. y1_f32 = round(input1_f32 / scale1)
                                # 13. clamp [-128, 127], cast int8 -> store Y1[row_idx, :]
                                pass

    _ = eps
    return main


@tilelang.jit(out_idx=[5, 6, 7, 8, 9, 10], pass_configs=pass_configs)
def multi_add_rms_norm_quant_dual_smooth(M, N, eps=1e-5, dtype="bfloat16"):
    """Dual smooth scale: x1, x2, gamma, ss1, ss2 -> x_sum, y_norm, y1, scale1, y2, scale2."""
    block_M = 64
    num_physical_cores = 20
    m_num = (M + block_M - 1) // block_M
    usedCoreNum = min(num_physical_cores, m_num)
    tasksPerCore = (m_num + usedCoreNum - 1) // usedCoreNum
    vec_num = 2
    sub_block_M = block_M // vec_num

    @T.prim_func
    def main(
        X1: T.Tensor((M, N), dtype),
        X2: T.Tensor((M, N), dtype),
        Gamma: T.Tensor((M, N), dtype),
        SS1: T.Tensor((M, N), dtype),
        SS2: T.Tensor((M, N), dtype),
        XSum: T.Tensor((M, N), dtype),
        YNorm: T.Tensor((M, N), dtype),
        Y1: T.Tensor((M, N), "int8"),
        Scale1: T.Tensor((M,), "float32"),
        Y2: T.Tensor((M, N), "int8"),
        Scale2: T.Tensor((M,), "float32"),
    ):
        with T.Kernel(usedCoreNum, is_npu=True) as (cid, vid):
            coreIdx = cid

            for localIdx in T.serial(tasksPerCore):
                bx = coreIdx * tasksPerCore + localIdx

                with T.Scope("V"):
                    if bx < m_num:
                        for row in T.serial(sub_block_M):
                            row_idx = bx * block_M + vid * sub_block_M + row
                            if row_idx < M:
                                # TODO(tile-level):
                                # 1-7: Same as no_smooth (add, rmsnorm, store x_sum + y_norm)
                                # --- Quant path 1 ---
                                # 8. input1_f32 = y_norm_f32 * SS1[row_idx, :].f32
                                # 9. scale1 = reduce_max(abs(input1_f32)) / 127
                                # 10. y1 = round(input1_f32 / scale1), clamp, int8
                                # 11. Store Y1[row_idx, :], Scale1[row_idx]
                                # --- Quant path 2 ---
                                # 12. input2_f32 = y_norm_f32 * SS2[row_idx, :].f32
                                # 13. scale2 = reduce_max(abs(input2_f32)) / 127
                                # 14. y2 = round(input2_f32 / scale2), clamp, int8
                                # 15. Store Y2[row_idx, :], Scale2[row_idx]
                                pass

    _ = eps
    return main
