## 功能说明

- 算子功能：MoE计算中，对输入x做Sigmoid或者SoftMax计算，对计算结果分组进行排序，最后根据分组排序的结果选取前k个专家。
- 计算公式：

  对输入做Sigmoid或者SoftMax：

  $$
  if normType==1:
      normOut=Sigmoid(x)
  else:
      normOut=SoftMax(x)
  $$

  如果bias不为空：

  $$
  normValue = normOut + bias
  $$

  对计算结果按照groupCount进行分组，每组按照groupSelectMode取max或topk2的sum值对group进行排序，取前kGroup个组：

  $$
  groupOut, groupId = TopK(ReduceSum(TopK(Split(normValue, groupCount), k=2, dim=-1), dim=-1),k=kGroup)
  $$

  根据上一步的groupId获取normValue中对应的元素，将数据再做TopK，得到expertIdxOut的结果：

  $$
  y,expertIdxOut=TopK(normOut[groupId, :],k=k)
  $$

  对y按照输入的routedScalingFactor和eps参数进行计算，得到yOut的结果：

  $$
  yOut = y / (ReduceSum(y, dim=-1)+eps)*routedScalingFactor
  $$

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | shape |
|---|---|---|---|---|
| x | 输入 | 待计算输入 | FLOAT16、BFLOAT16、FLOAT32 | ND |
| bias | 可选输入 | 与输入x进行计算的bias值 | FLOAT16、BFLOAT16、FLOAT32 | ND |
| k | 属性 | topk的k值 | INT64 | 标量 |
| k_group | 属性 | 分组排序后取的group个数 | INT64 | 标量 |
| group_count | 属性 | 分组的总个数 | INT64 | 标量 |
| routed_scaling_factor | 属性 | 计算yOut使用的系数 | DOUBLE | 标量 |
| eps | 属性 | 计算yOut使用的eps系数 | DOUBLE | 标量 |
| group_select_mode | 属性 | 分组排序方式。0=最大值排序，1=topk2的sum值排序 | INT64 | 标量 |
| renorm | 属性 | renorm标记，仅支持0（先norm后topk） | INT64 | 标量 |
| norm_type | 属性 | norm函数类型。0=Softmax，1=Sigmoid | INT64 | 标量 |
| out_flag | 属性 | 是否输出norm操作结果（true/false） | BOOL | - |
| y_out | 输出 | 对x做norm、分组排序topk后计算的结果 | FLOAT16、BFLOAT16、FLOAT32 | ND |
| expert_idx_out | 输出 | 对x做norm、分组排序topk后的索引 | INT32 | ND |
| norm_out | 可选输出 | norm计算的输出结果（out_flag=true时输出） | FLOAT32 | ND |

## 约束说明

- 输入shape限制：
  - x最后一维（即专家数）要求不大于2048。
- 输入值域限制：
  - 要求 1 <= k <= x_shape[-1] / groupCount * kGroup。
  - 要求 1 <= kGroup <= groupCount，并且 kGroup * x_shape[-1] / groupCount 的值要大于等于k。
  - 要求 groupCount > 0，x_shape[-1]能够被groupCount整除且整除后的结果大于groupSelectMode，并且整除的结果按照32个数对齐后乘groupCount的结果不大于2048。
  - renorm仅支持0，表示先进行norm操作，再计算topk。
- 其他限制：
  - groupSelectMode取值0和1：0表示使用最大值对group进行排序，1表示使用topk2的sum值对group进行排序。
  - normType取值0和1：0表示使用Softmax函数，1表示使用Sigmoid函数。
  - outFlag取值true和false：true表示输出normOut，false表示不输出。

```python
class Model(nn.Module):
    """MoE Gating TopK: Sigmoid/SoftMax + group sort + topK expert selection."""

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        x: torch.Tensor,
        bias: torch.Tensor,
        k: int = 8,
        k_group: int = 1,
        group_count: int = 1,
        group_select_mode: int = 0,
        renorm: int = 0,
        norm_type: int = 0,
        out_flag: bool = False,
        routed_scaling_factor: float = 1.0,
        eps: float = 1e-20,
    ) -> List[torch.Tensor]:
        """
        MoE gating with topK expert selection.

        1. Apply Sigmoid (norm_type=1) or SoftMax (norm_type=0) to x
        2. Add bias if provided
        3. Group-wise topK selection
        4. Scale output: y / (sum(y) + eps) * routed_scaling_factor

        Args:
            x: (batch, num_experts) gating logits
            bias: (num_experts,) optional bias
            k: number of experts to select
            k_group: number of groups to select
            group_count: number of expert groups
            group_select_mode: 0=max, 1=topk2_sum
            renorm: whether to renormalize
            norm_type: 0=softmax, 1=sigmoid
            out_flag: whether to output full y2
            routed_scaling_factor: output scaling
            eps: epsilon for normalization
        Returns:
            List of [y, expert_idx, y2_optional]
        """
        ori_dtype = x.dtype
        dtype = x.dtype
        if dtype != torch.float32:
            x = x.to(dtype=torch.float32)
            if bias is not None:
                bias = bias.to(dtype=torch.float32)

        x_np = x.numpy()
        bias_np = bias.numpy() if bias is not None else None

        def softmax_func(x_in, axis=None):
            x_max = x_in.max(axis=axis, keepdims=True)
            x_sub = x_in - x_max
            y = numpy.exp(x_sub)
            x_sum = y.sum(axis=axis, keepdims=True)
            return y / x_sum

        if norm_type == 0:
            x_np = softmax_func(x_np, -1)
        else:
            x_np = 1.0 / (1.0 + numpy.exp(-x_np))

        original_x = x_np.copy()
        if bias_np is not None:
            x_np = x_np + bias_np

        if group_count > 1:
            x_np = x_np.reshape(x_np.shape[0], group_count, -1)
            if group_select_mode == 0:
                group_x = numpy.amax(x_np, axis=-1)
            else:
                group_x = numpy.partition(x_np, -2, axis=-1)[..., -2:].sum(axis=-1)
            indices = numpy.argsort(-group_x, axis=-1, kind='stable')[:, :k_group]
            mask = numpy.ones((x_np.shape[0], group_count), dtype=bool)
            mask[numpy.arange(x_np.shape[0])[:, None], indices] = False
            x_np = numpy.where(mask[..., None], float('-inf'), x_np)
            x_np = x_np.reshape(x_np.shape[0], -1)

        _, indices = torch.sort(torch.from_numpy(x_np), dim=-1, stable=True, descending=True)
        indices = numpy.asarray(indices[:, :k])

        y = numpy.take_along_axis(original_x, indices, axis=1)

        if norm_type == 1 or renorm == 1:
            y /= (numpy.sum(y, axis=-1, keepdims=True) + eps)
        y *= routed_scaling_factor

        y2 = original_x if out_flag else None

        y_tensor = torch.tensor(y, dtype=dtype)
        idx_tensor = torch.from_numpy(indices.astype(numpy.int32))
        results = [y_tensor, idx_tensor]
        if y2 is not None:
            results.append(torch.tensor(y2, dtype=torch.float32))
        else:
            results.append(torch.empty(0))
        return results
```
