## 功能说明

根据索引将更新值写入输入张量的指定位置（原地更新）。

## 计算公式

$$
input[indices[k,0],indices[k,1],:] = update[k,:]
$$

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | shape |
|---|---|---|---|---|
| input | 输入 | 待更新张量，原地更新。必选参数，不能为空Tensor，支持非连续Tensor | FLOAT16、BFLOAT16、FLOAT32、INT64、BOOL、INT8 | (b_n, b_s, D) |
| indices | 输入 | 索引。必选参数，不能为空Tensor，不能出现重复索引，支持非连续Tensor | INT32、INT64 | (T, 2) |
| update | 输入 | 更新值。必选参数，不能为空Tensor，数据类型与input一致，支持非连续Tensor | 同input | (T, D) |

## 约束说明

- 确定性计算：默认确定性实现，相同输入多次调用结果一致。
- 输入Tensor input、indices、update 不能为空，indices不能存在重复索引，且必须为Device侧Tensor。
- 入图约束：支持torch v2.6以上torch.compile+eager+aclgraph和torch.compile+npugraph_ex+aclgraph方式入图。不支持ge入图，不支持aot_eager入图。
- 规格约束：

| 规格项 | 规格 | 说明 |
|---|---|---|
| b_n | 1~16384 | b_n支持1~16384范围 |
| b_s | 1~1024 | b_s支持1~1024范围 |
| D | 1~256 | D支持1~256范围 |
| T | 1~262144 | T支持1~262144范围 |

- 典型值：b_n: 2048/4096/8192/10240, b_s: 128, D: 1/128/256, T: 1/240/2k/4k/256k。

```python
class Model(nn.Module):
    """
    ScatterNdUpdate: input[indices[k,0], indices[k,1], :] = update[k, :]
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        input: torch.Tensor,
        indices: torch.Tensor,
        update: torch.Tensor,
    ) -> torch.Tensor:
        """
        ScatterNdUpdate Golden implementation.
        Pure move operator: input[indices[k,0], indices[k,1], :] = update[k, :]

        Args:
            input:   (D0, D1, D2), bf16/fp16/fp32
            indices: (K, 2), int32
            update:  (K, D2), bf16/fp16/fp32

        Returns:
            output tensor with same shape as input
        """
        output = input.clone()
        for k in range(indices.shape[0]):
            idx0 = indices[k, 0].item()
            idx1 = indices[k, 1].item()
            output[idx0, idx1, :] = update[k, :]
        return output
```
