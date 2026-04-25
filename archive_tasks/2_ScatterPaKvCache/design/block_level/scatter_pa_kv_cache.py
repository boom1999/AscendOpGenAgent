"""Block-level design for ScatterPaKvCache.

Algorithm: For each token, scatter key/value into PA KV cache based on slot_mapping.
- Pure Vector operation (no Cube needed)
- Each AI Core processes a subset of tokens
- For each token: read slot_mapping -> compute cache address -> write key/value slices

Cache layout (PA_NZ):
  key_cache:   [num_blocks, num_kv_slices, block_size, last_dim_k]
  value_cache: [num_blocks, num_vv_slices, block_size, last_dim_v]
  where num_kv_slices = num_head * k_head_size // last_dim_k
        last_dim_k = 16 for fp16/bf16, 32 for int8

Flattened view for kernel addressing:
  key_cache:   [num_blocks * num_kv_slices, block_size, last_dim_k]
  value_cache: [num_blocks * num_vv_slices, block_size, last_dim_v]

Task partition:
  - n_tokens tasks total, one per token
  - Distributed round-robin to AI Cores
  - No write conflicts: each token writes to a unique slot
"""
import tilelang
import tilelang.language as T


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


@tilelang.jit(out_idx=[1, 4], pass_configs=pass_configs)
def scatter_pa_kv_cache(
    n_tokens,
    num_kv_slices,
    num_vv_slices,
    num_blocks,
    block_size_param,
    last_dim_k,
    last_dim_v,
    key_dtype="float16",
    val_dtype="float16",
):
    num_physical_cores = 20
    usedCoreNum = min(num_physical_cores, n_tokens)
    tokensPerCore = T.ceildiv(n_tokens, usedCoreNum)

    total_kv_rows = num_blocks * num_kv_slices
    total_vv_rows = num_blocks * num_vv_slices
    key_flat_dim = num_kv_slices * last_dim_k
    val_flat_dim = num_vv_slices * last_dim_v

    @T.prim_func
    def main(
        key: T.Tensor((n_tokens, key_flat_dim), key_dtype),
        key_cache: T.Tensor((total_kv_rows, block_size_param, last_dim_k), key_dtype),
        slot_mapping: T.Tensor((n_tokens,), "int32"),
        value: T.Tensor((n_tokens, val_flat_dim), val_dtype),
        value_cache: T.Tensor((total_vv_rows, block_size_param, last_dim_v), val_dtype),
    ):
        with T.Kernel(usedCoreNum, is_npu=True) as (cid, vid):
            coreIdx = cid
            for localIdx in T.serial(tokensPerCore):
                token_idx = coreIdx * tokensPerCore + localIdx
                with T.Scope("V"):
                    if token_idx < n_tokens:
                        # TODO(tile-level):
                        # 1. Read slot = slot_mapping[token_idx] (scalar from GM)
                        # 2. Compute block_index = slot // block_size_param
                        #    Compute block_offset = slot % block_size_param
                        # 3. Load key[token_idx, :] (key_flat_dim elements) into UB
                        # 4. For each kv_slice s in [0, num_kv_slices):
                        #    T.copy: key_slice_ub -> key_cache[block_index*num_kv_slices+s, block_offset, :]
                        # 5. Load value[token_idx, :] (val_flat_dim elements) into UB
                        # 6. For each vv_slice s in [0, num_vv_slices):
                        #    T.copy: val_slice_ub -> value_cache[block_index*num_vv_slices+s, block_offset, :]
                        pass

    return main
