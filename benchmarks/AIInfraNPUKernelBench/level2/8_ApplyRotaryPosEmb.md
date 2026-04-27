## 功能说明

执行旋转位置编码计算，推理网络为了提升性能，将query和key两路算子融合成一路。计算结果执行原地更新。

## 计算公式

（1）rotaryMode为"half"：

$$
query\_q1 = query[..., : query.shape[-1] // 2], \quad query\_q2 = query[..., query.shape[-1] // 2 :]
$$

$$
query\_rotate = torch.cat((-query\_q2, query\_q1), dim=-1)
$$

$$
key\_k1 = key[..., : key.shape[-1] // 2], \quad key\_k2 = key[..., key.shape[-1] // 2 :]
$$

$$
key\_rotate = torch.cat((-key\_k2, key\_k1), dim=-1)
$$

$$
q\_embed = (query * cos) + query\_rotate * sin
$$

$$
k\_embed = (key * cos) + key\_rotate * sin
$$

（2）rotaryMode为"quarter"：将query/key分为4等分，旋转方式类似half但作用于1/4和3/4区间。

（3）rotaryMode为"interleave"：对query/key的奇偶位交叉旋转。

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | shape |
|---|---|---|---|---|
| query | 输入/输出 | 输入query张量和输出q_embed，4维张量 | BFLOAT16、FLOAT16、FLOAT32 | 4维，layout相关 |
| key | 输入/输出 | 输入key张量和输出k_embed，4维张量 | BFLOAT16、FLOAT16、FLOAT32 | 4维，layout相关 |
| cos | 输入 | 位置编码cos张量，4维 | BFLOAT16、FLOAT16、FLOAT32 | 4维，layout相关 |
| sin | 输入 | 位置编码sin张量，4维 | BFLOAT16、FLOAT16、FLOAT32 | 4维，layout相关 |
| layout | 属性 | 输入张量的布局格式 | STRING | "BSND"、"SBND"、"BNSD"、"TND" |
| rotary_mode | 属性 | 旋转模式 | STRING | "half"、"interleave"、"quarter" |

## 约束说明

- Atlas A2/A3 训练/推理系列：
  - 输入张量query、key、cos、sin支持4维和3维的shape，layout支持1-BSND和4-TND。
  - 4个输入shape的前2维（BSND格式）或第一维（TND格式）和最后一维必须相等。
  - cos和sin的shape倒数第2维（N维）必须等于1。
  - 输入shape最后一维必须等于128或64。
  - 输入张量query、key、cos、sin的dtype必须相同。
  - rotary_mode只支持"half"。
  - 不支持空tensor场景。
- Ascend 950PR/950DT：
  - query与key除N维度外其他维度必须相同；cos与sin shape必须相同。
  - rotary_mode为"half"/"interleave"时最后一维必须被2整除；"quarter"时必须被4整除。
  - D维度小于等于1024。

```python
class Model(nn.Module):
    """
    Fused Apply Rotary Position Embedding for query and key.

    Applies half-rotation RoPE to both query and key tensors in a single pass.
    Supports partial rotary dimension and bf16/fp16 inputs (cast to fp32 internally).
    """

    def __init__(self):
        super(Model, self).__init__()

    def _single_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        if dtype == torch.bfloat16 or dtype == torch.float16:
            x = x.float()
            cos = cos.float()
            sin = sin.float()

        d_rotary = cos.shape[-1]
        if x.shape[-1] != d_rotary:
            x_front = x[..., :d_rotary]
            x_back = x[..., d_rotary:]
            split = d_rotary // 2
            x1 = x_front[..., :split]
            x2 = x_front[..., split:]
            x_new = torch.cat((-x2, x1), dim=-1)
            res_front = x_front * cos + x_new * sin
            res = torch.cat((res_front, x_back), dim=-1)
        else:
            x1, x2 = torch.chunk(x, 2, dim=-1)
            x_new = torch.cat((-x2, x1), dim=-1)
            res = x * cos + x_new * sin

        return res.to(dtype)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor,
                layout: str = "BSND") -> List[torch.Tensor]:
        """
        Apply rotary position embedding to query and key.

        Args:
            query: Query tensor.
            key: Key tensor.
            cos: Cosine frequency tensor.
            sin: Sine frequency tensor.
            layout: "BSND" or "TND".

        Returns:
            List of [q_embed, k_embed].
        """
        q_res = self._single_rope(query, cos, sin)
        k_res = self._single_rope(key, cos, sin)
        return [q_res, k_res]
```
