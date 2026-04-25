"""Tile-level design for AggregateHidden.

Forward:  output[s,b,h] = mask[b,s] * (w[2,h]*inp[s] + w[1,h]*inp[s-1] + w[0,h]*inp[s-2])
GradInp:  grad_input[s,b,h] = w[0,h]*ge[s+2] + w[1,h]*ge[s+1] + w[2,h]*ge[s]
          where ge[s,b,h] = grad_out[s,b,h] * mask[b,s]
GradWt:   grad_weight[w,h] = sum_{s,b} ge[s] * inp[s-(W-1-w)]

All computation in float32. Host casts bf16/fp16 inputs to float32 before calling.
Tensors reshaped to 2D on host: (S,B,H) -> (S*B,H). Mask: (B,S) float32.

Single-pass pipeline over S with 2-step delayed grad_input.
Partition H across cores. Iterate B, S within each core.
"""

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


@tilelang.jit(out_idx=[4, 5, 6], pass_configs=pass_configs)
def aggregate_hidden_kernel(S, B, H, dtype="float32"):
    W = 3
    block_H = min(H, 1024)
    num_physical_cores = 20
    h_num = (H + block_H - 1) // block_H
    usedCoreNum = min(num_physical_cores, h_num)
    tasksPerCore = (h_num + usedCoreNum - 1) // usedCoreNum

    @T.prim_func
    def main(
        grad_out: T.Tensor((S * B, H), dtype),
        input_t: T.Tensor((S * B, H), dtype),
        weight: T.Tensor((W, H), dtype),
        mask_float: T.Tensor((B, S), dtype),
        output: T.Tensor((S * B, H), dtype),
        grad_input: T.Tensor((S * B, H), dtype),
        grad_weight: T.Tensor((W, H), dtype),
    ):
        with T.Kernel(usedCoreNum, is_npu=True) as (cid, vid):
            # UB allocations — sliding windows for input and grad_eff
            inp_cur = T.alloc_ub((1, block_H), dtype)
            inp_prev1 = T.alloc_ub((1, block_H), dtype)
            inp_prev2 = T.alloc_ub((1, block_H), dtype)
            ge_cur = T.alloc_ub((1, block_H), dtype)
            ge_prev1 = T.alloc_ub((1, block_H), dtype)
            ge_prev2 = T.alloc_ub((1, block_H), dtype)

            # Weight rows
            w0_ub = T.alloc_ub((1, block_H), dtype)
            w1_ub = T.alloc_ub((1, block_H), dtype)
            w2_ub = T.alloc_ub((1, block_H), dtype)

            # Working buffers
            go_ub = T.alloc_ub((1, block_H), dtype)
            out_ub = T.alloc_ub((1, block_H), dtype)
            gi_ub = T.alloc_ub((1, block_H), dtype)
            tmp_ub = T.alloc_ub((1, block_H), dtype)

            # Mask buffer: load a row of mask[b, :] per batch
            mask_row_ub = T.alloc_ub((1, S), dtype)
            mask_vec_ub = T.alloc_ub((1, block_H), dtype)

            # grad_weight accumulators
            gw0_ub = T.alloc_ub((1, block_H), dtype)
            gw1_ub = T.alloc_ub((1, block_H), dtype)
            gw2_ub = T.alloc_ub((1, block_H), dtype)

            coreIdx = cid
            for localIdx in T.serial(tasksPerCore):
                bx = coreIdx * tasksPerCore + localIdx
                with T.Scope("V"):
                    if bx < h_num:
                        h_start = bx * block_H

                        # Load weight rows
                        T.copy(weight[0, h_start], w0_ub)
                        T.copy(weight[1, h_start], w1_ub)
                        T.copy(weight[2, h_start], w2_ub)

                        # Zero-init grad_weight accumulators
                        T.tile.fill(gw0_ub, T.float32(0.0))
                        T.tile.fill(gw1_ub, T.float32(0.0))
                        T.tile.fill(gw2_ub, T.float32(0.0))

                        for bi in T.serial(B):
                            # Load mask row for this batch
                            T.copy(mask_float[bi, 0], mask_row_ub)

                            # Zero-init sliding windows
                            T.tile.fill(inp_prev1, T.float32(0.0))
                            T.tile.fill(inp_prev2, T.float32(0.0))
                            T.tile.fill(ge_prev1, T.float32(0.0))
                            T.tile.fill(ge_prev2, T.float32(0.0))

                            for si in T.serial(S):
                                flat_idx = si * B + bi

                                # Load input and grad_out
                                T.copy(input_t[flat_idx, h_start], inp_cur)
                                T.copy(grad_out[flat_idx, h_start], go_ub)

                                # Broadcast mask value to block_H vector
                                T.tile.fill(mask_vec_ub, mask_row_ub[0, si])

                                # ---- Forward convolution ----
                                # out = w2 * inp_cur + w1 * inp_prev1 + w0 * inp_prev2
                                T.tile.mul(out_ub, w2_ub, inp_cur)
                                T.tile.mul(tmp_ub, w1_ub, inp_prev1)
                                T.tile.add(out_ub, out_ub, tmp_ub)
                                T.tile.mul(tmp_ub, w0_ub, inp_prev2)
                                T.tile.add(out_ub, out_ub, tmp_ub)
                                # Apply mask
                                T.tile.mul(out_ub, out_ub, mask_vec_ub)
                                T.copy(out_ub, output[flat_idx, h_start])

                                # ---- Grad_eff = grad_out * mask ----
                                T.tile.mul(ge_cur, go_ub, mask_vec_ub)

                                # ---- Accumulate grad_weight ----
                                # gw2 += ge * inp_cur
                                T.tile.mul(tmp_ub, ge_cur, inp_cur)
                                T.tile.add(gw2_ub, gw2_ub, tmp_ub)
                                # gw1 += ge * inp_prev1
                                T.tile.mul(tmp_ub, ge_cur, inp_prev1)
                                T.tile.add(gw1_ub, gw1_ub, tmp_ub)
                                # gw0 += ge * inp_prev2
                                T.tile.mul(tmp_ub, ge_cur, inp_prev2)
                                T.tile.add(gw0_ub, gw0_ub, tmp_ub)

                                # ---- Grad_input (delayed by 2) ----
                                # gi[s-2] = w0*ge[s] + w1*ge[s-1] + w2*ge[s-2]
                                if si >= 2:
                                    T.tile.mul(gi_ub, w0_ub, ge_cur)
                                    T.tile.mul(tmp_ub, w1_ub, ge_prev1)
                                    T.tile.add(gi_ub, gi_ub, tmp_ub)
                                    T.tile.mul(tmp_ub, w2_ub, ge_prev2)
                                    T.tile.add(gi_ub, gi_ub, tmp_ub)
                                    T.copy(gi_ub, grad_input[(si - 2) * B + bi, h_start])

                                # ---- Shift sliding windows ----
                                T.copy(inp_prev1, inp_prev2)
                                T.copy(inp_cur, inp_prev1)
                                T.copy(ge_prev1, ge_prev2)
                                T.copy(ge_cur, ge_prev1)

                            # ---- Tail: last 2 grad_input positions ----
                            # grad_input[S-2] = w1*ge[S-1] + w2*ge[S-2]
                            if S >= 2:
                                T.tile.mul(gi_ub, w1_ub, ge_prev1)
                                T.tile.mul(tmp_ub, w2_ub, ge_prev2)
                                T.tile.add(gi_ub, gi_ub, tmp_ub)
                                T.copy(gi_ub, grad_input[(S - 2) * B + bi, h_start])
                            # grad_input[S-1] = w2*ge[S-1]
                            T.tile.mul(gi_ub, w2_ub, ge_prev1)
                            T.copy(gi_ub, grad_input[(S - 1) * B + bi, h_start])

                        # Store grad_weight
                        T.copy(gw0_ub, grad_weight[0, h_start])
                        T.copy(gw1_ub, grad_weight[1, h_start])
                        T.copy(gw2_ub, grad_weight[2, h_start])

    return main
