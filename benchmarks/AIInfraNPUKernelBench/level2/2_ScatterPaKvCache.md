## 功能说明

更新KvCache中指定位置的key和value。

支持多种输入输出场景（PA_NZ、Norm等），根据cacheMode和scatterMode区分。场景一（PA_NZ）为典型场景：
- key: (batch*seq_len, num_head, k_head_size)
- value: (batch*seq_len, num_head, v_head_size)
- keyCache: (num_blocks, num_head*k_head_size//last_dim_k, block_size, last_dim_k)
- valueCache: (num_blocks, num_head*v_head_size//last_dim_v, block_size, last_dim_v)
- slotMapping: (batch*seq_len,)

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | shape |
|---|---|---|---|---|
| key | 输入 | 待更新的key值，当前step多个token的key | FLOAT16、BFLOAT16、INT8、UINT8、INT16、UINT16、INT32、UINT32、FLOAT32、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN | 场景相关 |
| key_cache | 输入/输出 | 需要更新的key cache | 同key | 场景相关 |
| slot_mapping | 输入 | 每个token在cache中的存储偏移 | INT32、INT64 | 场景相关 |
| value | 输入 | 待更新的value值，当前step多个token的value | 同key | 场景相关 |
| value_cache | 输入/输出 | 需要更新的value cache | 同key | 场景相关 |
| compress_lens | 可选输入 | 压缩量 | INT32、INT64 | 场景相关 |
| compress_seq_offset | 可选输入 | 每个batch每个head的压缩起点 | INT32、INT64 | 场景相关 |
| seq_lens | 可选输入 | 每个batch的实际seqLens | INT32、INT64 | 场景相关 |
| cache_mode | 属性 | keyCacheRef和valueCacheRef的内存排布格式 | STRING | "PA_NZ"/"Norm" |
| scatter_mode | 属性 | 更新的key和value的状态 | STRING | "None"/"Nct"/"Alibi"/"Rope"/"Omni" |

Atlas A3/A2 训练/推理系列：key/value数据类型仅支持FLOAT16、BFLOAT16、INT8。

## 约束说明

- 输入shape限制：
  - 除了key和value，输入参数不支持非连续。
  - 当key和value都是3维，前两维shape必须相同。
  - 当key和value都是4维，前三维shape必须相同，且keyCacheRef和valueCacheRef的第三维必须是1。
  - 当cacheMode为"PA_NZ"时，keyCacheRef和valueCacheRef的倒数第二维必须小于UINT16_MAX。
- 输入值域限制：
  - slotMapping取值范围[0, num_blocks*block_size-1]，且内部元素值不重复。
  - key与keyCacheRef的数据类型必须一致，value与valueCacheRef的数据类型必须一致（key和value的数据类型可以不同）。
  - slotMapping、compressLensOptional、compressSeqOffsetOptional、seqLensOptional的数据类型必须一致。

```python
class Model(nn.Module):
    """
    Scatter PA KV Cache: updates KV cache at specified slot positions.

    Scatters key and value tokens into paged attention KV cache blocks
    based on slot_mapping indices.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        key: torch.Tensor,
        key_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        value: torch.Tensor,
        value_cache: torch.Tensor,
        cache_mode: str = "PA_NZ",
    ) -> List[torch.Tensor]:
        """
        Args:
            key: (batch*seq_len, num_head, k_head_size)
            key_cache: (num_blocks, num_head*k_head_size//last_dim, block_size, last_dim)
            slot_mapping: (batch*seq_len,) int32
            value: (batch*seq_len, num_head, v_head_size)
            value_cache: (num_blocks, num_head*v_head_size//last_dim, block_size, last_dim)
            cache_mode: str, "PA_NZ"

        Returns:
            List of [updated_key_cache, updated_value_cache]
        """
        key_cache_out = key_cache.clone()
        value_cache_out = value_cache.clone()

        block_size = key_cache.shape[2]
        num_kv_slices = key_cache.shape[1]
        last_dim_k = key_cache.shape[3]
        num_vv_slices = value_cache.shape[1]
        last_dim_v = value_cache.shape[3]

        for i in range(slot_mapping.shape[0]):
            slot = slot_mapping[i].item()
            if slot < 0:
                continue
            block_index = slot // block_size
            block_offset = slot % block_size

            token_key = key[i].reshape(-1)
            for k in range(num_kv_slices):
                key_cache_out[block_index][k][block_offset][:] = token_key[k * last_dim_k: k * last_dim_k + last_dim_k]

            token_value = value[i].reshape(-1)
            for v in range(num_vv_slices):
                value_cache_out[block_index][v][block_offset][:] = token_value[v * last_dim_v: v * last_dim_v + last_dim_v]

        return [key_cache_out, value_cache_out]
```
