"""Tile-level TileLang design for MultiAddRmsNormDynamicQuant.

Extends the rms_norm single_row template with:
  1. Add: x_sum = x1 + x2
  2. DynamicQuant: scale = max(abs(input))/127, y_quant = round(input/scale).int8
  3. Optional smooth scale multiplication before quantization

All N values (4096-7680) fall in the single-row regime (1024 < N <= 8192).
Computation is in float32 internally, with bf16/fp16 for input/output.
"""

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[3, 4, 5, 6], pass_configs=pass_configs)
def multi_add_rms_norm_quant_no_smooth(M, N, eps=1e-5, dtype="bfloat16"):
    """No smooth scale variant."""
    block_M = 64
    num_physical_cores = 20
    m_num = T.ceildiv(M, block_M)
    used_core_num = min(num_physical_cores, m_num)
    tasks_per_core = T.ceildiv(m_num, used_core_num)
    vec_num = 2
    sub_block_M = block_M // vec_num

    need_cast = dtype != "float32"
    out_cast_mode = "CAST_ROUND" if dtype == "bfloat16" else "CAST_NONE"

    eps_const = T.float32(eps)
    inv_n_const = T.float32(1.0 / N)
    inv_127_const = T.float32(1.0 / 127.0)

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
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            # Input/output cast buffers
            in_ub = T.alloc_ub((1, N), dtype)
            out_cast_ub = T.alloc_ub((1, N), dtype)
            # Float32 compute buffers
            x_f32_ub = T.alloc_ub((1, N), "float32")
            xsum_f32_ub = T.alloc_ub((1, N), "float32")
            gamma_f32_ub = T.alloc_ub((1, N), "float32")
            ynorm_f32_ub = T.alloc_ub((1, N), "float32")
            sq_ub = T.alloc_ub((1, N), "float32")
            # Quant buffers
            quant_f16_ub = T.alloc_ub((1, N), "float16")
            quant_i8_ub = T.alloc_ub((1, N), "int8")
            # Scalar buffers
            scalar_ub = T.alloc_ub((1, 1), "float32")
            scale_ub = T.alloc_ub((1, 1), "float32")
            # Reduce scratch
            reduce_tmp = T.alloc_ub((2 * N,), "uint8")

            with T.Scope("V"):
                for local_idx in T.serial(tasks_per_core):
                    bx = cid * tasks_per_core + local_idx
                    if bx < m_num:
                        for row in T.serial(sub_block_M):
                            row_idx = bx * block_M + vid * sub_block_M + row
                            if row_idx < M:
                                # --- Add: x_sum = x1 + x2 ---
                                if need_cast:
                                    T.copy(X1[row_idx, :], in_ub)
                                    T.tile.cast(x_f32_ub, in_ub, mode="CAST_NONE", count=N)
                                    T.copy(X2[row_idx, :], in_ub)
                                    T.tile.cast(xsum_f32_ub, in_ub, mode="CAST_NONE", count=N)
                                else:
                                    T.copy(X1[row_idx, :], x_f32_ub)
                                    T.copy(X2[row_idx, :], xsum_f32_ub)
                                T.tile.add(xsum_f32_ub, x_f32_ub, xsum_f32_ub)

                                # Store x_sum
                                if need_cast:
                                    T.tile.cast(out_cast_ub, xsum_f32_ub, mode=out_cast_mode, count=N)
                                    T.copy(out_cast_ub, XSum[row_idx, :])
                                else:
                                    T.copy(xsum_f32_ub, XSum[row_idx, :])

                                # --- RmsNorm ---
                                T.tile.mul(sq_ub, xsum_f32_ub, xsum_f32_ub)
                                T.reduce_sum(sq_ub, sq_ub[:, 0], reduce_tmp, dim=-1)
                                sum_sq = sq_ub[0, 0] * inv_n_const + eps_const
                                sq_ub[0, 0] = sum_sq
                                T.tile.rsqrt(scalar_ub[:, 0], sq_ub[:, 0])
                                inv_rms = scalar_ub[0, 0]

                                # Load gamma, cast to f32
                                if need_cast:
                                    T.copy(Gamma[row_idx, :], in_ub)
                                    T.tile.cast(gamma_f32_ub, in_ub, mode="CAST_NONE", count=N)
                                else:
                                    T.copy(Gamma[row_idx, :], gamma_f32_ub)

                                # y_norm = x_sum * inv_rms * gamma
                                T.tile.mul(ynorm_f32_ub, xsum_f32_ub, inv_rms)
                                T.tile.mul(ynorm_f32_ub, ynorm_f32_ub, gamma_f32_ub)

                                # Store y_norm
                                if need_cast:
                                    T.tile.cast(out_cast_ub, ynorm_f32_ub, mode=out_cast_mode, count=N)
                                    T.copy(out_cast_ub, YNorm[row_idx, :])
                                else:
                                    T.copy(ynorm_f32_ub, YNorm[row_idx, :])

                                # --- Dynamic Quant (no smooth) ---
                                T.tile.abs(sq_ub, ynorm_f32_ub)
                                T.reduce_max(sq_ub, scale_ub, reduce_tmp, dim=-1)
                                scale_val = scale_ub[0, 0] * inv_127_const
                                scale_ub[0, 0] = scale_val
                                T.copy(scale_ub[:, 0], Scale1[row_idx:row_idx + 1])

                                T.tile.div(ynorm_f32_ub, ynorm_f32_ub, scale_val)
                                T.tile.cast(quant_f16_ub, ynorm_f32_ub, mode="CAST_NONE", count=N)
                                T.tile.cast(quant_i8_ub, quant_f16_ub, mode="CAST_ROUND", count=N)
                                T.copy(quant_i8_ub, Y1[row_idx, :])

    return main


@tilelang.jit(out_idx=[4, 5, 6, 7], pass_configs=pass_configs)
def multi_add_rms_norm_quant_smooth1(M, N, eps=1e-5, dtype="bfloat16"):
    """Single smooth scale variant."""
    block_M = 64
    num_physical_cores = 20
    m_num = T.ceildiv(M, block_M)
    used_core_num = min(num_physical_cores, m_num)
    tasks_per_core = T.ceildiv(m_num, used_core_num)
    vec_num = 2
    sub_block_M = block_M // vec_num

    need_cast = dtype != "float32"
    out_cast_mode = "CAST_ROUND" if dtype == "bfloat16" else "CAST_NONE"

    eps_const = T.float32(eps)
    inv_n_const = T.float32(1.0 / N)
    inv_127_const = T.float32(1.0 / 127.0)

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
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            in_ub = T.alloc_ub((1, N), dtype)
            out_cast_ub = T.alloc_ub((1, N), dtype)
            x_f32_ub = T.alloc_ub((1, N), "float32")
            xsum_f32_ub = T.alloc_ub((1, N), "float32")
            gamma_f32_ub = T.alloc_ub((1, N), "float32")
            ynorm_f32_ub = T.alloc_ub((1, N), "float32")
            sq_ub = T.alloc_ub((1, N), "float32")
            quant_f16_ub = T.alloc_ub((1, N), "float16")
            quant_i8_ub = T.alloc_ub((1, N), "int8")
            scalar_ub = T.alloc_ub((1, 1), "float32")
            scale_ub = T.alloc_ub((1, 1), "float32")
            reduce_tmp = T.alloc_ub((2 * N,), "uint8")

            with T.Scope("V"):
                for local_idx in T.serial(tasks_per_core):
                    bx = cid * tasks_per_core + local_idx
                    if bx < m_num:
                        for row in T.serial(sub_block_M):
                            row_idx = bx * block_M + vid * sub_block_M + row
                            if row_idx < M:
                                # --- Add ---
                                if need_cast:
                                    T.copy(X1[row_idx, :], in_ub)
                                    T.tile.cast(x_f32_ub, in_ub, mode="CAST_NONE", count=N)
                                    T.copy(X2[row_idx, :], in_ub)
                                    T.tile.cast(xsum_f32_ub, in_ub, mode="CAST_NONE", count=N)
                                else:
                                    T.copy(X1[row_idx, :], x_f32_ub)
                                    T.copy(X2[row_idx, :], xsum_f32_ub)
                                T.tile.add(xsum_f32_ub, x_f32_ub, xsum_f32_ub)

                                if need_cast:
                                    T.tile.cast(out_cast_ub, xsum_f32_ub, mode=out_cast_mode, count=N)
                                    T.copy(out_cast_ub, XSum[row_idx, :])
                                else:
                                    T.copy(xsum_f32_ub, XSum[row_idx, :])

                                # --- RmsNorm ---
                                T.tile.mul(sq_ub, xsum_f32_ub, xsum_f32_ub)
                                T.reduce_sum(sq_ub, sq_ub[:, 0], reduce_tmp, dim=-1)
                                sum_sq = sq_ub[0, 0] * inv_n_const + eps_const
                                sq_ub[0, 0] = sum_sq
                                T.tile.rsqrt(scalar_ub[:, 0], sq_ub[:, 0])
                                inv_rms = scalar_ub[0, 0]

                                if need_cast:
                                    T.copy(Gamma[row_idx, :], in_ub)
                                    T.tile.cast(gamma_f32_ub, in_ub, mode="CAST_NONE", count=N)
                                else:
                                    T.copy(Gamma[row_idx, :], gamma_f32_ub)

                                T.tile.mul(ynorm_f32_ub, xsum_f32_ub, inv_rms)
                                T.tile.mul(ynorm_f32_ub, ynorm_f32_ub, gamma_f32_ub)

                                if need_cast:
                                    T.tile.cast(out_cast_ub, ynorm_f32_ub, mode=out_cast_mode, count=N)
                                    T.copy(out_cast_ub, YNorm[row_idx, :])
                                else:
                                    T.copy(ynorm_f32_ub, YNorm[row_idx, :])

                                # --- Quant with smooth1 ---
                                # input1 = y_norm * ss1
                                if need_cast:
                                    T.copy(SS1[row_idx, :], in_ub)
                                    T.tile.cast(gamma_f32_ub, in_ub, mode="CAST_NONE", count=N)
                                else:
                                    T.copy(SS1[row_idx, :], gamma_f32_ub)
                                T.tile.mul(sq_ub, ynorm_f32_ub, gamma_f32_ub)

                                T.tile.abs(x_f32_ub, sq_ub)
                                T.reduce_max(x_f32_ub, scale_ub, reduce_tmp, dim=-1)
                                scale_val = scale_ub[0, 0] * inv_127_const
                                scale_ub[0, 0] = scale_val
                                T.copy(scale_ub[:, 0], Scale1[row_idx:row_idx + 1])

                                T.tile.div(sq_ub, sq_ub, scale_val)
                                T.tile.cast(quant_f16_ub, sq_ub, mode="CAST_NONE", count=N)
                                T.tile.cast(quant_i8_ub, quant_f16_ub, mode="CAST_ROUND", count=N)
                                T.copy(quant_i8_ub, Y1[row_idx, :])

    return main


@tilelang.jit(out_idx=[5, 6, 7, 8, 9, 10], pass_configs=pass_configs)
def multi_add_rms_norm_quant_dual_smooth(M, N, eps=1e-5, dtype="bfloat16"):
    """Dual smooth scale variant."""
    block_M = 64
    num_physical_cores = 20
    m_num = T.ceildiv(M, block_M)
    used_core_num = min(num_physical_cores, m_num)
    tasks_per_core = T.ceildiv(m_num, used_core_num)
    vec_num = 2
    sub_block_M = block_M // vec_num

    need_cast = dtype != "float32"
    out_cast_mode = "CAST_ROUND" if dtype == "bfloat16" else "CAST_NONE"

    eps_const = T.float32(eps)
    inv_n_const = T.float32(1.0 / N)
    inv_127_const = T.float32(1.0 / 127.0)

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
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            in_ub = T.alloc_ub((1, N), dtype)
            out_cast_ub = T.alloc_ub((1, N), dtype)
            x_f32_ub = T.alloc_ub((1, N), "float32")
            xsum_f32_ub = T.alloc_ub((1, N), "float32")
            gamma_f32_ub = T.alloc_ub((1, N), "float32")
            ynorm_f32_ub = T.alloc_ub((1, N), "float32")
            sq_ub = T.alloc_ub((1, N), "float32")
            quant_f16_ub = T.alloc_ub((1, N), "float16")
            quant_i8_ub = T.alloc_ub((1, N), "int8")
            scalar_ub = T.alloc_ub((1, 1), "float32")
            scale_ub = T.alloc_ub((1, 1), "float32")
            reduce_tmp = T.alloc_ub((2 * N,), "uint8")

            with T.Scope("V"):
                for local_idx in T.serial(tasks_per_core):
                    bx = cid * tasks_per_core + local_idx
                    if bx < m_num:
                        for row in T.serial(sub_block_M):
                            row_idx = bx * block_M + vid * sub_block_M + row
                            if row_idx < M:
                                # --- Add ---
                                if need_cast:
                                    T.copy(X1[row_idx, :], in_ub)
                                    T.tile.cast(x_f32_ub, in_ub, mode="CAST_NONE", count=N)
                                    T.copy(X2[row_idx, :], in_ub)
                                    T.tile.cast(xsum_f32_ub, in_ub, mode="CAST_NONE", count=N)
                                else:
                                    T.copy(X1[row_idx, :], x_f32_ub)
                                    T.copy(X2[row_idx, :], xsum_f32_ub)
                                T.tile.add(xsum_f32_ub, x_f32_ub, xsum_f32_ub)

                                if need_cast:
                                    T.tile.cast(out_cast_ub, xsum_f32_ub, mode=out_cast_mode, count=N)
                                    T.copy(out_cast_ub, XSum[row_idx, :])
                                else:
                                    T.copy(xsum_f32_ub, XSum[row_idx, :])

                                # --- RmsNorm ---
                                T.tile.mul(sq_ub, xsum_f32_ub, xsum_f32_ub)
                                T.reduce_sum(sq_ub, sq_ub[:, 0], reduce_tmp, dim=-1)
                                sum_sq = sq_ub[0, 0] * inv_n_const + eps_const
                                sq_ub[0, 0] = sum_sq
                                T.tile.rsqrt(scalar_ub[:, 0], sq_ub[:, 0])
                                inv_rms = scalar_ub[0, 0]

                                if need_cast:
                                    T.copy(Gamma[row_idx, :], in_ub)
                                    T.tile.cast(gamma_f32_ub, in_ub, mode="CAST_NONE", count=N)
                                else:
                                    T.copy(Gamma[row_idx, :], gamma_f32_ub)

                                T.tile.mul(ynorm_f32_ub, xsum_f32_ub, inv_rms)
                                T.tile.mul(ynorm_f32_ub, ynorm_f32_ub, gamma_f32_ub)

                                if need_cast:
                                    T.tile.cast(out_cast_ub, ynorm_f32_ub, mode=out_cast_mode, count=N)
                                    T.copy(out_cast_ub, YNorm[row_idx, :])
                                else:
                                    T.copy(ynorm_f32_ub, YNorm[row_idx, :])

                                # --- Quant path 1: y_norm * ss1 ---
                                if need_cast:
                                    T.copy(SS1[row_idx, :], in_ub)
                                    T.tile.cast(gamma_f32_ub, in_ub, mode="CAST_NONE", count=N)
                                else:
                                    T.copy(SS1[row_idx, :], gamma_f32_ub)
                                T.tile.mul(sq_ub, ynorm_f32_ub, gamma_f32_ub)

                                T.tile.abs(x_f32_ub, sq_ub)
                                T.reduce_max(x_f32_ub, scale_ub, reduce_tmp, dim=-1)
                                scale_val1 = scale_ub[0, 0] * inv_127_const
                                scale_ub[0, 0] = scale_val1
                                T.copy(scale_ub[:, 0], Scale1[row_idx:row_idx + 1])

                                T.tile.div(sq_ub, sq_ub, scale_val1)
                                T.tile.cast(quant_f16_ub, sq_ub, mode="CAST_NONE", count=N)
                                T.tile.cast(quant_i8_ub, quant_f16_ub, mode="CAST_ROUND", count=N)
                                T.copy(quant_i8_ub, Y1[row_idx, :])

                                # --- Quant path 2: y_norm * ss2 ---
                                if need_cast:
                                    T.copy(SS2[row_idx, :], in_ub)
                                    T.tile.cast(gamma_f32_ub, in_ub, mode="CAST_NONE", count=N)
                                else:
                                    T.copy(SS2[row_idx, :], gamma_f32_ub)
                                T.tile.mul(sq_ub, ynorm_f32_ub, gamma_f32_ub)

                                T.tile.abs(x_f32_ub, sq_ub)
                                T.reduce_max(x_f32_ub, scale_ub, reduce_tmp, dim=-1)
                                scale_val2 = scale_ub[0, 0] * inv_127_const
                                scale_ub[0, 0] = scale_val2
                                T.copy(scale_ub[:, 0], Scale2[row_idx:row_idx + 1])

                                T.tile.div(sq_ub, sq_ub, scale_val2)
                                T.tile.cast(quant_f16_ub, sq_ub, mode="CAST_NONE", count=N)
                                T.tile.cast(quant_i8_ub, quant_f16_ub, mode="CAST_ROUND", count=N)
                                T.copy(quant_i8_ub, Y2[row_idx, :])

    return main
