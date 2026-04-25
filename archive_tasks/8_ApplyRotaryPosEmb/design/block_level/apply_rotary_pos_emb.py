"""Block-level TileLang design for fused ApplyRotaryPosEmb (query + key).

Algorithm (half-rotation RoPE):
  For each input tensor x (query or key), with rotation dimension D_rot:
    split = D_rot // 2
    x1 = x[..., :split]
    x2 = x[..., split:D_rot]
    rotated = cat(-x2, x1)                   # length D_rot
    result[..., :D_rot] = x[..., :D_rot] * cos + rotated * sin
    result[..., D_rot:] = x[..., D_rot:]     # passthrough (partial rotary)

Design:
  - model_new_tilelang.py normalises both BSND and TND layouts into
    [total_tokens, N, D] and squeezes cos/sin to [total_tokens, D_rot].
  - The kernel fuses query and key processing in a single launch.
  - Each block handles one token per vector sub-block (vid).
    For each token the kernel loads cos/sin once and iterates over
    N_q query heads then N_k key heads, applying the same rotation.
  - Since D <= 128, the full head row fits in UB; no Cube compute needed.
"""

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[4, 5], pass_configs=pass_configs)
def apply_rotary_pos_emb(total_tokens, N_q, N_k, D, D_rot, dtype="float16"):
    """Fused RoPE kernel for query and key.

    Parameters
    ----------
    total_tokens : int
        Number of tokens (B*S for BSND, T for TND).
    N_q, N_k : int
        Number of query / key heads.
    D : int
        Head dimension.
    D_rot : int
        Rotary dimension (<= D).  When D_rot < D, partial rotary.
    dtype : str
        Element type of all tensors.
    """
    num_physical_cores = 20
    block_T = 2                     # tokens per block (one per vid)
    num_blocks = (total_tokens + block_T - 1) // block_T
    usedCoreNum = min(num_physical_cores, num_blocks)
    tasksPerCore = (num_blocks + usedCoreNum - 1) // usedCoreNum
    vec_num = 2
    sub_block_T = block_T // vec_num   # = 1 token per vid

    @T.prim_func
    def main(
        query:  T.Tensor((total_tokens, N_q, D), dtype),
        key:    T.Tensor((total_tokens, N_k, D), dtype),
        cos:    T.Tensor((total_tokens, D_rot), dtype),
        sin:    T.Tensor((total_tokens, D_rot), dtype),
        q_out:  T.Tensor((total_tokens, N_q, D), dtype),
        k_out:  T.Tensor((total_tokens, N_k, D), dtype),
    ):
        with T.Kernel(usedCoreNum, is_npu=True) as (cid, vid):
            for localIdx in T.serial(tasksPerCore):
                bx = cid * tasksPerCore + localIdx
                if bx < num_blocks:
                    with T.Scope("V"):
                        token_idx = bx * block_T + vid * sub_block_T
                        if token_idx < total_tokens:
                            # TODO(tile-level):
                            # 1. Load cos[token_idx, :D_rot] and sin[token_idx, :D_rot]
                            #    into UB buffers (loaded once, reused for all heads).
                            #
                            # 2. For h in [0 .. N_q):
                            #      Load query[token_idx, h, :D] into UB.
                            #      Apply half-rotation RoPE:
                            #        split = D_rot // 2
                            #        For [:D_rot]: result = x * cos + rotate_half(x) * sin
                            #        For [D_rot:D]: passthrough copy
                            #      Store q_out[token_idx, h, :D].
                            #
                            # 3. For h in [0 .. N_k):
                            #      Load key[token_idx, h, :D] into UB.
                            #      Apply the same half-rotation RoPE.
                            #      Store k_out[token_idx, h, :D].
                            _ = query
                            _ = key
                            _ = cos
                            _ = sin
                            _ = q_out
                            _ = k_out
                            _ = token_idx

    return main
