## 功能说明

mhc_post基于一系列计算对mHC架构中上一层输出 $h_{t}^{out}$ 进行Post Mapping，对上一层的输入 $x_j$ 进行ResMapping，然后对二者进行残差连接，得到下一层的输入 $x_{l+1}$。

## 计算公式

$$
x_{l+1} = (H_{l}^{res})^{T} \times x_l + h_{l}^{out} \otimes H_{t}^{post}
$$

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | shape |
|---|---|---|---|---|
| x | 输入 | mHC层的输入数据，支持非连续Tensor | float16、bfloat16 | (B,S,n,D) 或 (T,n,D) |
| h_res | 输入 | mHC的h_res变换矩阵（sinkhorn双随机矩阵），支持非连续Tensor | float32 | (B,S,n,n) 或 (T,n,n) |
| h_out | 输入 | Atten/MLP层的输出，支持非连续Tensor | float16、bfloat16 | (B,S,D) 或 (T,D) |
| h_post | 输入 | mHC的h_post变换矩阵，支持非连续Tensor | float32 | (B,S,n) 或 (T,n) |
| output | 输出 | mHC层的输出数据，作为下一层的输入，支持非连续Tensor | float16、bfloat16 | (B,S,n,D) 或 (T,n,D) |

维度含义：B（Batch Size）、S（Sequence Length）、n（Head Num）、D（Head Dim）；$T=B*S$。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持图模式。
- 确定性计算：默认确定性实现，相同输入多次调用结果一致。
- 输入Tensor x、h_res、h_out、h_post 不能为空，且必须为Device侧Tensor。
- h_res 分布满足双随机矩阵。
- 规格约束：

| 规格项 | 规格 | 说明 |
|---|---|---|
| T或B*S | 1~512k | B*S 或T支持1~512k范围 |
| n | 4、6、8 | n值目前支持4, 6, 8 |
| D | 384~24k | D支持384~24k范围 |

- 典型值：T或B*S: 1024/2048/4096, n: 4（推荐）, D: 2560/5120（推荐）。

```python
class Model(nn.Module):
    """
    Manifold Constrained Hyper Connection Post:
    x_{l+1} = (H_res)^T * x_l + h_out (x) H_post
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        x: torch.Tensor,
        h_res: torch.Tensor,
        h_out: torch.Tensor,
        h_post: torch.Tensor,
    ) -> torch.Tensor:
        """
        mHC Post forward.

        y = h_post.unsqueeze(-1) * h_out.unsqueeze(-2)
            + sum(h_res.unsqueeze(-1) * x.unsqueeze(-2), dim=-3)

        All bf16/fp16 inputs are cast to float32 for computation;
        result is cast back to original dtype.

        Args:
            x:      (..., n, D), bf16/fp16
            h_res:  (..., n, n), float32
            h_out:  (..., D), bf16/fp16
            h_post: (..., n), float32

        Returns:
            y: (..., n, D)
        """
        orig_dtype = x.dtype
        x_f = x.float()
        h_out_f = h_out.float()

        y = h_post.unsqueeze(-1) * h_out_f.unsqueeze(-2) + torch.sum(
            h_res.unsqueeze(-1) * x_f.unsqueeze(-2), dim=-3
        )
        return y.to(orig_dtype)
```
