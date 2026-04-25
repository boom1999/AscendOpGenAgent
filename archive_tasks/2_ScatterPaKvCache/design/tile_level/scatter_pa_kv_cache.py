"""Tile-level design for ScatterPaKvCache.

Fills in block-level TODOs with actual TileLang operations.

Per-token processing:
  1. Read slot from slot_mapping (scalar GM read)
  2. Compute block_index and block_offset
  3. Load key[token_idx] into key_ub (GM -> UB, contiguous)
  4. For each kv_slice: T.copy key_slice_ub -> key_cache (UB -> GM, last_dim_k elements)
  5. Load value[token_idx] into val_ub (GM -> UB, contiguous)
  6. For each vv_slice: T.copy val_slice_ub -> value_cache (UB -> GM, last_dim_v elements)

Dynamic indexing pattern (following gather_elements_v2 reference):
  slot = T.cast(slot_mapping[token_idx], "int32")
  Used to compute cache row indices at runtime.
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
            # Allocate UB buffers (outside loop for reuse)
            key_ub = T.alloc_ub((key_flat_dim,), key_dtype)
            key_slice_ub = T.alloc_ub((last_dim_k,), key_dtype)
            val_ub = T.alloc_ub((val_flat_dim,), val_dtype)
            val_slice_ub = T.alloc_ub((last_dim_v,), val_dtype)

            for localIdx in T.serial(tokensPerCore):
                token_idx = cid * tokensPerCore + localIdx
                with T.Scope("V"):
                    if token_idx < n_tokens:
                        # 1. Read slot mapping (scalar GM read, dynamic index)
                        slot = T.cast(slot_mapping[token_idx], "int32")
                        block_index = slot // block_size_param
                        block_offset = slot % block_size_param

                        # 2. Load key for this token (GM -> UB, contiguous read)
                        T.copy(key[token_idx, 0], key_ub)

                        # 3. Scatter key slices to cache
                        for s in T.serial(num_kv_slices):
                            cache_row = block_index * num_kv_slices + s
                            # Extract slice from key_ub (UB -> UB)
                            T.copy(key_ub[s * last_dim_k], key_slice_ub)
                            # Write slice to cache (UB -> GM)
                            T.copy(key_slice_ub, key_cache[cache_row, block_offset, 0])

                        # 4. Load value for this token (GM -> UB, contiguous read)
                        T.copy(value[token_idx, 0], val_ub)

                        # 5. Scatter value slices to cache
                        for s in T.serial(num_vv_slices):
                            cache_row = block_index * num_vv_slices + s
                            # Extract slice from val_ub (UB -> UB)
                            T.copy(val_ub[s * last_dim_v], val_slice_ub)
                            # Write slice to cache (UB -> GM)
                            T.copy(val_slice_ub, value_cache[cache_row, block_offset, 0])

    return main
