## 功能说明

mhc_post基于一系列计算对mHC架构中上一层输出 $h_{t}^{out}$ 进行Post Mapping，对上一层的输入 $x_j$ 进行ResMapping，然后对二者进行残差连接，得到下一层的输入 $x_{l+1}$。该算子实现前述过程的反向功能。

## 计算公式

$$
grad\_x = H_{l}^{res} \times grad\_output\\
grad\_h\_res = x_{l} \times {grad\_output}^{T}
$$

$$
grad\_h\_out=({grad\_output} * (H_{l}^{post}.unsqueeze(-1))).sum(dim=-2)\\
grad\_h\_post=({grad\_output} * (h_{l}^{out}.unsqueeze(-2))).sum(dim=-1)
$$

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | shape |
|---|---|---|---|---|
| grad_output | 输入 | mHC层的输出梯度，支持非连续Tensor | float16、bfloat16 | (B,S,n,D) 或 (T,n,D) |
| x | 输入 | mHC层的输入数据，支持非连续Tensor | float16、bfloat16 | (B,S,n,D) 或 (T,n,D) |
| h_res | 输入 | mHC的h_res变换矩阵（sinkhorn双随机矩阵），支持非连续Tensor | float32 | (B,S,n,n) 或 (T,n,n) |
| h_out | 输入 | Atten/MLP层的输出，支持非连续Tensor | float16、bfloat16 | (B,S,D) 或 (T,D) |
| h_post | 输入 | mHC的h_post变换矩阵，支持非连续Tensor | float32 | (B,S,n) 或 (T,n) |
| grad_x | 输出 | 输入数据x的梯度，支持非连续Tensor | float16、bfloat16 | (B,S,n,D) 或 (T,n,D) |
| grad_h_res | 输出 | h_res变换矩阵的梯度，支持非连续Tensor | float32 | (B,S,n,n) 或 (T,n,n) |
| grad_h_out | 输出 | Atten/MLP层输出的梯度，支持非连续Tensor | float16、bfloat16 | (B,S,D) 或 (T,D) |
| grad_h_post | 输出 | h_post变换矩阵的梯度，支持非连续Tensor | float32 | (B,S,n) 或 (T,n) |

维度含义：B（Batch Size）、S（Sequence Length）、n（Head Num）、D（Head Dim）；$T=B*S$。

## 约束说明

- 确定性计算：默认确定性实现，相同输入多次调用结果一致。
- 输入Tensor grad_output、x、h_res、h_out、h_post 不能为空，且必须为Device侧Tensor。
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
    mHC Post backward: computes gradients for ManifoldConstrainedHyperConnectionPost.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        grad_output: torch.Tensor,
        x: torch.Tensor,
        h_res: torch.Tensor,
        h_out: torch.Tensor,
        h_post: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Functional implementation of mHC Post backward.

        Computes four gradients via autograd of the forward:
          y = h_post.unsqueeze(-1) * h_out.unsqueeze(-2)
              + sum(h_res.unsqueeze(-1) * x.unsqueeze(-2), dim=-3)

        All bf16/fp16 inputs are cast to float32 for computation;
        grad_x and grad_h_out are cast back to original dtype.

        Args:
            grad_output: (..., n, D), bf16/fp16
            x:           (..., n, D), bf16/fp16
            h_res:       (..., n, n), float32
            h_out:       (..., D), bf16/fp16
            h_post:      (..., n), float32

        Returns:
            List of [grad_x, grad_h_res, grad_h_out, grad_h_post]
        """
        orig_dtype = grad_output.dtype

        grad_output_f = grad_output.float()
        x_f = x.float()
        h_out_f = h_out.float()

        # grad_x = h_res @ grad_output
        grad_x = torch.matmul(h_res, grad_output_f).to(orig_dtype)

        # grad_h_res = x @ grad_output^T
        grad_h_res = torch.matmul(x_f, grad_output_f.transpose(-1, -2))

        # grad_h_out = sum(grad_output * h_post.unsqueeze(-1), dim=-2)
        grad_h_out = torch.sum(grad_output_f * h_post.unsqueeze(-1), dim=-2).to(orig_dtype)

        # grad_h_post = sum(grad_output * h_out.unsqueeze(-2), dim=-1)
        grad_h_post = torch.sum(grad_output_f * h_out_f.unsqueeze(-2), dim=-1)

        return [grad_x, grad_h_res, grad_h_out, grad_h_post]
```
