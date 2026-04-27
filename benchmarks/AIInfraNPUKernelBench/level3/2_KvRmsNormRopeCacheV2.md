## 功能说明

- 接口功能：融合了MLA（Multi-head Latent Attention）结构中RMSNorm归一化计算与RoPE（Rotary Position Embedding）位置编码以及更新KVCache的ScatterUpdate操作。

- 计算公式：
  首先对尾轴进行拆分得到kv_a部分和kv_pe部分，分别进行rms_norm和rope计算，然后进行量化(可选)并更新kvcache

  $$
  kv\_a=kv[...,:rms\_size]
  $$

  $$
  kv\_pe=kv[...,rms\_size:]
  $$

  $$
  y\_out=RmsNorm(kv\_a)
  $$

  $$
  k\_embed\_out = Rope(kv\_pe)
  $$

  $$
  k\_cache[index] = Quant(x = y\_out, scale = k\_scale, offset = k\_offset)
  $$

  $$
  ckv\_cache[index] = Quant(x = k\_embed\_out, scale = k\_scale, offset = k\_offset)
  $$

  (1) RmsNorm逻辑如下:

  $$
  \operatorname{RmsNorm}(x_i)=\frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * gamma_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  (2) Rope逻辑如下：

  当rotary_mode=half时：

  $$
  x1, x2 = torch.chunk(x, 2, dim=-1)
  $$

  当rotary_mode=interleave_half时：

  $$
  x1=x[...,::2],x2=x[...,1::2]
  $$

  然后进行拼接与旋转计算：

  $$
  x\_rotate = torch.cat((-x2, x1), dim=-1)
  $$

  $$
  x\_rope = x * cos + x\_rotate * sin
  $$

  (3) Quant逻辑如下：

  当quant_mode=static时：

  $$
  x = x * scale,\ if\ scale\ !=\ None
  $$

  $$
  x = x + offset,\ if\ offset\ !=\ None
  $$

  $$
  y = round(x).clamp(-128,127)
  $$

  当quant_mode=pertile128时：（每128个元素为一组进行动态量化）

  $$
  tile\_size = 128 \\
  x\_rng[i]=[(i-1)*tile\_size: i * tile\_size - 1]
  $$

  $$
  scaleOut[i]=row\_max(abs(x[x\_rng[i]]))/127
  $$

  $$
  yOut=round(\frac{x[x\_rng[i]]}{scaleOut[i]})
  $$

## 参数说明

> Tensor中shape使用的变量说明：
> - batch_size：batch的大小。
> - seq_len：sequence的长度。
> - hidden_size：MLA输入的向量长度，取值仅支持576。
> - rms_size：RMSNorm分支的向量长度，取值仅支持512。
> - rope_size：RoPE分支的向量长度，取值仅支持64。
> - cache_length：Norm模式下有效，表示KVCache支持的最大长度。
> - block_num：PagedAttention模式下有效，表示Block的个数。
> - block_size：PagedAttention模式下有效，表示Block的大小。

| 参数名 | 输入/输出 | 描述 | 数据类型 | shape |
|---|---|---|---|---|
| kv | 输入 | 输入的特征张量，4维BNSD格式 | BFLOAT16、FLOAT16 | [batch_size, 1, seq_len, hidden_size] |
| gamma | 输入 | RMS归一化的缩放参数，1维 | BFLOAT16、FLOAT16 | [rms_size] |
| cos | 输入 | RoPE旋转位置编码的余弦分量，4维 | BFLOAT16、FLOAT16 | [batch_size, 1, seq_len, rope_size] |
| sin | 输入 | RoPE旋转位置编码的正弦分量，4维 | BFLOAT16、FLOAT16 | [batch_size, 1, seq_len, rope_size] |
| index | 输入 | 缓存索引张量，用于定位k_cache和ckv_cache的写入位置 | INT64 | 取决于cache_mode |
| k_cache | 输入/输出 | 用于存储量化/非量化的键向量 | BFLOAT16、FLOAT16、INT8 | 取决于cache_mode |
| ckv_cache | 输入/输出 | 用于存储量化/非量化的压缩后kv向量 | BFLOAT16、FLOAT16、INT8 | 取决于cache_mode |
| k_rope_scale | 可选输入 | k旋转位置编码的量化缩放因子（量化模式下必填） | FLOAT32 | [rope_size] |
| c_kv_scale | 可选输入 | 压缩后kv的量化缩放因子（量化模式下必填） | FLOAT32 | [rms_size] |
| k_rope_offset | 可选输入 | k旋转位置编码量化偏移量（量化模式下必填） | FLOAT32 | [rope_size] |
| c_kv_offset | 可选输入 | 压缩后kv的量化偏移量（量化模式下必填） | FLOAT32 | [rms_size] |
| epsilon | 可选属性 | RMS归一化中的极小值，防止除以零，默认1e-5 | FLOAT | 标量 |
| cache_mode | 可选属性 | 缓存模式，默认'Norm' | STRING | - |
| rotary_mode | 可选属性 | rope旋转模式，默认'interleave_half'，支持'interleave_half'、'half_half' | STRING | - |
| quant_mode | 可选属性 | 量化模式，默认'static'，支持'none'、'static'、'pertile128' | STRING | - |
| is_output_kv | 可选属性 | 是否输出处理后的k_embed_out和y_out（未量化原始值），默认False | BOOL | - |
| k_cache | 输出 | 与输入k_cache一致（in-place更新） | 同输入k_cache | 同输入 |
| ckv_cache | 输出 | 与输入ckv_cache一致（in-place更新） | 同输入ckv_cache | 同输入 |
| k_embed_out | 可选输出 | RoPE处理后的值（is_output_kv=True时） | 同kv | [batch_size, 1, seq_len, 64] |
| y_out | 可选输出 | RMSNorm处理后的值（is_output_kv=True时） | 同kv | [batch_size, 1, seq_len, 512] |

cache_mode支持的模式：

| 枚举值 | 模式名 | 说明 |
|---|---|---|
| Norm | KV-Cache更新模式 | k_cache形状为[batch_size, 1, cache_length, rope_size]，ckv_cache形状为[batch_size, 1, cache_length, rms_size]。index形状为[batch_size, seq_len]。 |
| PA/PA_BNSD | PagedAttention模式 | k_cache形状为[block_num, block_size, 1, rope_size]，ckv_cache形状为[block_num, block_size, 1, rms_size]。index形状为[batch_size*seq_len]。 |
| PA_NZ | FRACTAL_NZ的PA模式 | k_cache形状为[block_num, block_size, 1, rope_size]，ckv_cache形状为[block_num, block_size, 1, rms_size]。非量化下数据排布为[block_num, size//16, block_size, 1, 16]；量化下为[block_num, size//32, block_size, 1, 32]。 |
| PA_BLK_BNSD | 特殊PA模式 | k_cache/ckv_cache同PA_BNSD。index形状为[batch_size*Ceil(seq_len/block_size)]，值表示每个block的起始偏移。 |
| PA_BLK_NZ | FRACTAL_NZ的特殊PA模式 | 同PA_NZ数据排布。index形状为[batch_size*Ceil(seq_len/block_size)]。 |

## 约束说明

- 该接口支持推理场景下使用，支持图模式。
- 量化模式：当k_rope_scale和c_kv_scale非空时，k_cache和ckv_cache的dtype为int8，缓存形状的最后一个维度需要为32（FRACTAL_NZ模式），k_rope_scale和c_kv_scale必须同时非空，k_rope_offset和c_kv_offset必须同时为None或非空。
- 非量化模式：当k_rope_scale和c_kv_scale为空时，k_cache和ckv_cache的dtype为bfloat16或float16。
- 索引映射：所有cache_mode缓存模式下，index的值不可以重复。
  - Norm：index的值表示每个Batch下的偏移。
  - PA/PA_BNSD/PA_NZ：index的值表示全局的偏移。
  - PA_BLK_BNSD/PA_BLK_NZ：index的值表示每个页的全局偏移；该场景假设cache更新是连续的，不支持非连续更新的cache。
- Shape关联规则：
  - Norm：cache_length >= seq_len。
  - 非Norm模式（PagedAttention相关）：要求block_num >= Ceil(seq_len/block_size) * batch_size。

```python
class Model(nn.Module):
    """KV RmsNorm + interleave RoPE + Cache scatter V2."""

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        kv: torch.Tensor,
        gamma: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        index: torch.Tensor,
        k_cache: torch.Tensor,
        ckv_cache: torch.Tensor,
        k_scale: torch.Tensor = None,
        v_scale: torch.Tensor = None,
        k_offset: torch.Tensor = None,
        v_offset: torch.Tensor = None,
        vOptional: torch.Tensor = None,
        eps: float = 1e-5,
        cacheMode: str = "Norm",
        isOutputKv: bool = False,
    ) -> List[torch.Tensor]:
        """
        KV RmsNorm + interleave RoPE + cache scatter.

        (1) interleaveRope on kv[..., Dv:]:
            x1 = x[...,::2], x2 = x[...,1::2]
            x_part1 = cat(x1, x2), x_part2 = cat(-x2, x1)
            y = x_part1 * cos + x_part2 * sin

        (2) rmsNorm on kv[..., :Dv]:
            y = (x / sqrt(mean(x^2) + eps)) * gamma

        All bf16/fp16 inputs cast to fp32 for computation.

        Returns:
            List of [k_cache, ckv_cache] (+ optional kv outputs)
        """
        # Handle optional tensors with shape [0]
        if k_scale is not None and k_scale.numel() == 0:
            k_scale = None
        if v_scale is not None and v_scale.numel() == 0:
            v_scale = None
        if k_offset is not None and k_offset.numel() == 0:
            k_offset = None
        if v_offset is not None and v_offset.numel() == 0:
            v_offset = None

        kv_dtype = kv.dtype
        kv_f = kv.float()
        gamma_f = gamma.float()
        cos_f = cos.float()
        sin_f = sin.float()

        kv_shape = kv_f.shape
        Bkv = kv_shape[0]
        Nkv = kv_shape[1]
        Skv = kv_shape[2]
        Dkv = kv_shape[3]

        if vOptional is None:
            method_mode = 0
        else:
            method_mode = 1
            vOptional_f = vOptional.float()

        if method_mode == 0:
            v_dim = gamma_f.shape[-1]
            k_dim = cos_f.shape[-1]
        else:
            v_dim = vOptional.shape[3]
            k_dim = Dkv

        # Transpose to (B, S, N, D)
        kv_bsnd = kv_f.permute(0, 2, 1, 3)
        cos_bsnd = cos_f.permute(0, 2, 1, 3)
        sin_bsnd = sin_f.permute(0, 2, 1, 3)

        if method_mode == 0:
            rms_in = kv_bsnd[..., :v_dim]
            rope_in_raw = kv_bsnd[..., v_dim:]

            # RmsNorm
            v = rms_in / torch.sqrt(torch.mean(rms_in ** 2, dim=-1, keepdim=True) + eps)
            v = v * gamma_f
            v_out = v.permute(0, 2, 1, 3)
            if v_scale is not None:
                v = v * v_scale.float()
            if v_offset is not None:
                v = v + v_offset.float()
            if v_scale is not None:
                v = torch.round(v).clamp(-128, 127)

            # Interleave RoPE
            k = rope_in_raw.reshape(Bkv, Skv, Nkv, k_dim // 2, 2).transpose(-1, -2).reshape(Bkv, Skv, Nkv, k_dim)
            k1 = k[..., :k.shape[-1] // 2]
            k2 = k[..., k.shape[-1] // 2:]
            rotate_half_k = torch.cat((-k2, k1), dim=-1)
            k_embed = (k * cos_bsnd) + (rotate_half_k * sin_bsnd)
            k_embed_out = k_embed.permute(0, 2, 1, 3)
            if k_scale is not None:
                k_embed = k_embed * k_scale.float()
            if k_offset is not None:
                k_embed = k_embed + k_offset.float()
            if k_scale is not None:
                k_embed = torch.round(k_embed).clamp(-128, 127)
        else:
            rms_in = kv_bsnd
            v_in = vOptional_f.permute(0, 2, 1, 3)

            # RmsNorm
            v = rms_in / torch.sqrt(torch.mean(rms_in ** 2, dim=-1, keepdim=True) + eps)
            v = v * gamma_f

            # RoPE
            rope_dim = cos_bsnd.shape[-1]
            rope_in = v[..., :rope_dim]
            k = rope_in.reshape(Bkv, Skv, Nkv, rope_dim // 2, 2).transpose(-1, -2).reshape(Bkv, Skv, Nkv, rope_dim)
            k1 = k[..., :k.shape[-1] // 2]
            k2 = k[..., k.shape[-1] // 2:]
            rotate_half_k = torch.cat((-k2, k1), dim=-1)
            k_embed = (k * cos_bsnd) + (rotate_half_k * sin_bsnd)
            kv_out = torch.cat([k_embed, v[..., rope_dim:]], dim=-1)
            k_embed_out = kv_out.permute(0, 2, 1, 3).to(kv_dtype)
            if k_scale is not None:
                kv_out = kv_out * k_scale.float()
            if k_offset is not None:
                kv_out = kv_out + k_offset.float()
            if k_scale is not None:
                kv_out = torch.round(kv_out).clamp(-128, 127)
            k_embed = kv_out

            v_out = v_in.permute(0, 2, 1, 3).to(kv_dtype)
            if v_scale is not None:
                v_in = v_in * v_scale.float()
            if v_offset is not None:
                v_in = v_in + v_offset.float()
            if v_scale is not None:
                v_in = torch.round(v_in).clamp(-128, 127)
            v = v_in

        k_embed_out = k_embed_out.to(kv_dtype)
        v_out = v_out.to(kv_dtype)

        # Scatter into caches (simplified: just return the computed outputs)
        if isOutputKv:
            return [k_cache, ckv_cache, k_embed_out, v_out]
        else:
            return [k_cache, ckv_cache]
```
