"""Block-level design for AggregateHidden.

Algorithm (W=3 fixed):
  Forward:  output[s,b,h] = mask[b,s] * (w[2,h]*inp[s,b,h] + w[1,h]*inp[s-1,b,h] + w[0,h]*inp[s-2,b,h])
  GradInp:  grad_input[s,b,h] = w[0,h]*ge[s+2,b,h] + w[1,h]*ge[s+1,b,h] + w[2,h]*ge[s,b,h]
            where ge[s,b,h] = grad_out[s,b,h] * mask[b,s]
  GradWt:   grad_weight[w,h] = sum_{s,b} ge[s,b,h] * inp[s-(W-1-w), b, h]

Input layout: (S, B, H) — H innermost, contiguous.
Weight: (W=3, H). Mask: (B, S) bool -> (B, S) float on host.

Pure Vector operation, no Cube needed.
Task partition: split H across cores, each core iterates B then S.
Single-pass pipeline: forward + grad_weight accumulation + delayed grad_input (2-step delay).

Host side (model_new_tilelang.py):
  - Convert mask bool to float dtype for kernel-side vectorized masking
  - No layout transpose needed since H is already contiguous
"""

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


@tilelang.jit(out_idx=[4, 5, 6], pass_configs=pass_configs)
def aggregate_hidden_kernel(S, B, H, dtype="bfloat16"):
    W = 3
    block_H = min(H, 1024)
    num_physical_cores = 20
    h_num = (H + block_H - 1) // block_H
    usedCoreNum = min(num_physical_cores, h_num)
    tasksPerCore = (h_num + usedCoreNum - 1) // usedCoreNum
    accum_dtype = "float32"

    @T.prim_func
    def main(
        grad_out: T.Tensor((S, B, H), dtype),
        input_t: T.Tensor((S, B, H), dtype),
        weight: T.Tensor((W, H), dtype),
        mask_float: T.Tensor((B, S), dtype),
        output: T.Tensor((S, B, H), dtype),
        grad_input: T.Tensor((S, B, H), dtype),
        grad_weight: T.Tensor((W, H), dtype),
    ):
        with T.Kernel(usedCoreNum, is_npu=True) as (cid, vid):
            coreIdx = cid
            for localIdx in T.serial(tasksPerCore):
                bx = coreIdx * tasksPerCore + localIdx
                with T.Scope("V"):
                    if bx < h_num:
                        h_start = bx * block_H
                        # TODO(tile-level):
                        # Alloc UB buffers:
                        #   w_ub[3][block_H]     — weight rows
                        #   inp_cur/prev1/prev2  — input sliding window (block_H each)
                        #   ge_cur/prev1/prev2   — grad_eff sliding window (block_H each)
                        #   go_ub, out_ub, gi_ub  — grad_out, output, grad_input (block_H each)
                        #   gw_accum[3][block_H] — grad_weight accumulators in float32
                        #   mask_scalar_ub, mask_vec_ub — for broadcasting mask
                        #   tmp buffers for cast / mul
                        #
                        # Load weight[0, h_start], weight[1, h_start], weight[2, h_start]
                        # Zero-init gw_accum[0..2]
                        #
                        # For b in range(B):
                        #   Zero-init inp_prev1, inp_prev2, ge_prev1, ge_prev2
                        #
                        #   For s = 0..S-1:
                        #     ---- Forward ----
                        #     Load inp_cur = input_t[s, b, h_start]
                        #     fwd = w[2]*inp_cur + w[1]*inp_prev1 + w[0]*inp_prev2
                        #     Load mask_val from mask_float[b, s], broadcast to block_H
                        #     out = fwd * mask_val
                        #     Store output[s, b, h_start] = out
                        #
                        #     ---- Grad_eff ----
                        #     Load go = grad_out[s, b, h_start]
                        #     ge = go * mask_val
                        #
                        #     ---- Accumulate grad_weight (in float32) ----
                        #     gw_accum[2] += cast_f32(ge * inp_cur)
                        #     gw_accum[1] += cast_f32(ge * inp_prev1)
                        #     gw_accum[0] += cast_f32(ge * inp_prev2)
                        #
                        #     ---- Grad_input (delayed by 2 steps) ----
                        #     if s >= 2:
                        #       gi = w[0]*ge + w[1]*ge_prev1 + w[2]*ge_prev2
                        #       Store grad_input[s-2, b, h_start] = gi
                        #
                        #     ---- Shift sliding windows ----
                        #     inp_prev2 = inp_prev1; inp_prev1 = inp_cur
                        #     ge_prev2 = ge_prev1; ge_prev1 = ge
                        #
                        #   ---- Tail: last 2 grad_input positions ----
                        #   if S >= 2:
                        #     gi = w[1]*ge_prev1 + w[2]*ge_prev2
                        #     Store grad_input[S-2, b, h_start] = gi
                        #   gi = w[2]*ge_prev1
                        #   Store grad_input[S-1, b, h_start] = gi
                        #
                        # Cast gw_accum to dtype, store grad_weight[0..2, h_start]

    return main
