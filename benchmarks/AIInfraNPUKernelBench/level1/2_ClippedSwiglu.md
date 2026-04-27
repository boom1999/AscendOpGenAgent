## 功能说明

带截断的Swish门控线性单元激活函数，实现x的SwiGlu计算。本算子相较于SwiGlu算子，新增了部分输入参数：groupIndex、alpha、limit、bias、interleaved，用于支持GPT-OSS模型使用的变体SwiGlu以及MoE模型使用的分组场景。

## 计算公式

对给定的输入张量 x，其维度为[a,b,c,d,e,f,g…]，算子ClippedSwiglu对其进行以下计算：

1. 将x基于输入参数dim进行合轴，合轴后维度为[pre,cut,after]。其中cut轴为合轴之后需要切分为两个张量的轴，切分方式分为前后切分或者奇偶切分；pre，after可以等于1。例如当dim为3，合轴后x的维度为[a*b*c,d,e*f*g*…]。此外，由于after轴的元素为连续存放，且计算操作为逐元素的，因此将cut轴与after轴合并，得到x的维度为[pre,cut]。

2. 根据输入参数 group_index, 对 x 的pre轴进行过滤处理：

$$
sum = \text{Sum}(group\_index)
$$

$$
x = x[ : sum, : ]
$$

其中sum表示group_index的所有元素之和。当不输入 group_index 时，跳过该步骤。

3. 根据输入参数 interleaved，对 x 进行切分：

当 interleaved 为 true 时，表示奇偶切分：

$$
A = x[ : , : : 2]
$$

$$
B = x[ : , 1 : : 2]
$$

当 interleaved 为 false 时，表示前后切分：

$$
h = x.shape[1] // 2
$$

$$
A = x[ : , : h]
$$

$$
B = x[ : , h : ]
$$

4. 根据输入参数 alpha、limit、bias 进行变体SwiGlu计算：

$$
A = A.clamp(min=None, max=limit)
$$

$$
B = B.clamp(min=-limit, max=limit)
$$

$$
y\_glu = A * sigmoid(alpha * A)
$$

$$
y = y\_glu * (B + bias)
$$

5. 重塑输出张量y的维度数量与合轴前的x的维度数量一致，dim轴上的大小为x的一半，其他维度与x相同。

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | shape |
|---|---|---|---|---|
| x | 输入 | 主输入，dim对应轴必须为偶数，维度必须大于0 | FLOAT32、FLOAT16、BFLOAT16 | 任意，dim轴为偶数 |
| group_index | 可选输入 | 分组过滤索引，1维 | INT64 | (G,) |
| dim | 可选属性 | 合轴/切分维度，范围[-x.dim(), x.dim()-1]，默认-1 | INT64 | 标量 |
| alpha | 可选属性 | sigmoid缩放因子，默认1.702 | FLOAT | 标量 |
| limit | 可选属性 | clamp门限值，默认7.0 | FLOAT | 标量 |
| bias | 可选属性 | 偏差参数，默认1.0 | FLOAT | 标量 |
| interleaved | 可选属性 | true=奇偶切分，false=前后切分，默认true | BOOL | 标量 |
| y | 输出 | dim轴大小为x的一半，其他维度与x一致 | 同x | 同x，dim轴减半 |

## 约束说明

无

```python
class Model(nn.Module):
    """
    ClippedSwiglu: Clipped Swish-Gated Linear Unit activation.
    Supports interleaved/non-interleaved split, group_index filtering,
    and parameterized alpha/limit/bias.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        x: torch.Tensor,
        group_index: Optional[torch.Tensor],
        dim: int,
        alpha: float,
        limit: float,
        bias: float,
        interleaved: bool,
    ) -> torch.Tensor:
        """
        Clipped SwiGLU computation.

        1. Merge dims around `dim` into [pre, cut*after] = [pre, cut]
        2. Filter by group_index sum if provided
        3. Split into A, B (interleaved or halved)
        4. A = clamp(A, max=limit); B = clamp(B, -limit, limit)
        5. y = A * sigmoid(alpha * A) * (B + bias)
        6. Reshape output to match input dims with dim halved

        Args:
            x:            input tensor, bf16/fp16/fp32
            group_index:  (G,) int64 or None
            dim:          int, dimension to split
            alpha:        float
            limit:        float
            bias:         float
            interleaved:  bool

        Returns:
            output tensor with dim halved at `dim`
        """
        input_x = x
        orig_dtype = input_x.dtype

        # Merge dims around `dim` into [pre, cut]
        before_shape = input_x.shape[:dim]
        before_total = 1
        for s in before_shape:
            before_total *= s
        after_shape = input_x.shape[dim:]
        after_total = 1
        for s in after_shape:
            after_total *= s
        x_2d = input_x.reshape(before_total, after_total)

        if orig_dtype != torch.float32:
            x_2d = x_2d.to(torch.float32)

        # Group filtering
        if group_index is not None:
            group_sum = min(int(torch.sum(group_index).item()), x_2d.shape[0])
        else:
            group_sum = x_2d.shape[0]
        x_tensor = x_2d[:group_sum]

        # Split
        if interleaved:
            x_glu = x_tensor[..., ::2]
            x_linear = x_tensor[..., 1::2]
        else:
            out = torch.chunk(x_tensor, 2, dim=-1)
            x_glu = out[0]
            x_linear = out[1]

        # Clamp and compute
        x_glu = x_glu.clamp(min=None, max=limit)
        x_linear = x_linear.clamp(min=-limit, max=limit)
        sigmoid_part = torch.sigmoid(alpha * x_glu)
        result = x_glu * sigmoid_part * (x_linear + bias)
        result = result.to(orig_dtype)

        # Build output with zeros for filtered rows
        y = torch.zeros((x_2d.shape[0], x_2d.shape[1] // 2), dtype=orig_dtype)
        y[:group_sum] = result

        # Reshape to original dims with dim halved
        shape = list(input_x.shape)
        shape[dim] = shape[dim] // 2
        return y.reshape(shape)
```
