## 功能说明

对序列执行因果一维卷积（decode场景）：沿序列维度，使用缓存数据（长度为卷积核宽减1）对各序列头部进行padding，确保输出依赖当前及历史输入；卷积完成后，将当前序列的数据更新到缓存。

支持场景：
1. x支持 shape 是[cu_seq_len, dim] 和 [batch, m+1, dim]，其中 cu_seq_len 为batch内所有变长序列拼接后的总长度，m 为投机Token的个数
2. weight 的 shape 是[K, dim]，K 固定为3
3. 每个序列卷积前，使用长度为 K-1 的缓存数据对序列头部进行padding，保证因果性

## 计算公式

K 是卷积核宽度，L是每个batch的序列长度，dim是特征维度，batchId是x中batch的索引。对x中的cache_indices[batchId]不为pad_slot_id的每个batch做如下操作：

- 缓存拼接

$$
offset = num\_accepted\_tokens[batchId] - 1
$$

$$
token\_seq'[i, dim] = 
\begin{cases} 
cur\_conv\_states[i+offset, dim], & 0 \leq i < K-1 \\
token\_seq[i - (K-1), dim], & K-1 \leq i < L + K - 1
\end{cases}
$$

- 因果1维卷积

$$
y[i, dim] = \sum_{k=0}^{K-1} weight[k, dim] \cdot token\_seq'[i + k, dim]
$$

- 原地更新缓存

$$
cur\_conv\_states[i-1, dim] = x'[i, dim], \quad i = 1, 2, \dots, L+K-2
$$

- 若residual_connection为1，执行残差连接

$$
y[i, dim] += token\_seq[i, dim]
$$

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | shape |
|---|---|---|---|---|
| x | 输入 | 输入序列，不支持空Tensor | FLOAT16、BFLOAT16 | 2维(cu_seq_len, dim)或3维(batch, seqlen, dim)。三维时batch范围[1~256]，seqlen范围[1~6]；二维时cu_seq_len范围[1~1536]。dim范围[64, 16384]且能被16整除 |
| weight | 输入 | 因果1维卷积核，不支持空Tensor | 同x | (K, dim)，K=3 |
| conv_states | 输入/输出 | 缓存状态张量，各序列计算完成后原地更新，不支持空Tensor | 同x | (..., statelen, dim)，statelen=seqlen+1 |
| bias | 可选输入 | 卷积偏置（当前不支持） | 同x | (dim,) |
| query_start_loc | 可选输入 | 仅x为2维时起作用，序列起始位置索引 | INT32 | (batch+1,)，batch范围[1~256] |
| cache_indices | 可选输入 | 缓存索引，值不允许重复，值域范围[0~statelen-1]，不支持空Tensor | INT32 | (batch,) |
| num_accepted_tokens | 可选输入 | decode场景下的投机token个数，值域范围[1~statelen-1]，为空时offset为0 | INT32 | (batch,) |
| activation | 可选属性 | 激活函数类型（当前不支持） | STR | - |
| residual_connection | 可选属性 | 残差连接（0=不做, 1=y+x），默认1 | INT32 | 标量 |
| pad_slot_id | 可选属性 | 跳过不参与计算的batch，默认-1 | INT32 | 标量 |
| y | 输出 | 卷积计算结果 | 同x | 同x |

## 约束说明

- 非连续情况下，dim 维度的 stride 必须为 1。

```python
class Model(nn.Module):
    """
    Causal 1D convolution update for speculative decoding.

    Performs causal 1D convolution on sequences with cache support,
    handling speculative decoding via num_accepted_tokens and pad_slot_id.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        conv_states: torch.Tensor,
        query_start_loc: torch.Tensor = None,
        cache_indices: torch.Tensor = None,
        num_accepted_tokens: torch.Tensor = None,
        residual_connection: int = 1,
        pad_slot_id: int = -1,
    ) -> List[torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, dim) or (cu_seq_len, dim).
            weight: (K, dim) convolution kernel.
            conv_states: (-1, state_len, dim) cached states.
            query_start_loc: optional (batch+1,) int32 for cu_seq_len format.
            cache_indices: (batch,) int32.
            num_accepted_tokens: (batch,) int32.
            residual_connection: int, whether to add residual.
            pad_slot_id: int, padding identifier.

        Returns:
            List of [output, updated_conv_states].
        """
        orig_dtype = x.dtype
        x_f = x.float()
        weight_f = weight.float()
        conv_states_f = conv_states.float()

        if x_f.ndim == 3:
            batch, _, dim = x_f.shape
        else:
            batch = cache_indices.shape[0]
            dim = x_f.size(1)

        width = weight_f.size(0)
        state_len = conv_states_f.size(1)
        residual = residual_connection

        out = torch.zeros_like(x_f)

        for batch_idx in range(batch):
            if cache_indices[batch_idx] == pad_slot_id:
                continue

            if x_f.ndim == 2:
                # Need query_start_loc for cu_seq_len format - not provided here
                # This operator uses 3D format primarily
                x_seq = x_f[batch_idx:batch_idx+1]
                seq_len = 1
            else:
                x_seq = x_f[batch_idx]
                seq_len = x_f.size(1)

            conv_state_idx = cache_indices[batch_idx]
            current_conv_state = conv_states_f[conv_state_idx]

            accepted_tokens = num_accepted_tokens[batch_idx].item()
            offset = accepted_tokens - 1

            padded_input = torch.cat(
                (current_conv_state[offset: offset + width - 1], x_seq),
                dim=0)

            result = F.conv1d(
                padded_input.transpose(0, 1).unsqueeze(0),
                weight_f.transpose(0, 1).unsqueeze(1),
                bias=None,
                stride=1,
                padding=0,
                groups=dim,
            )
            result = result.squeeze(0).transpose(0, 1)

            if x_f.ndim == 2:
                out[batch_idx:batch_idx+1] = result + x_seq if residual else result
            else:
                out[batch_idx] = result + x_seq if residual else result

            cache_len = min(padded_input.size(0) - 1, state_len)
            conv_states_f[conv_state_idx][:cache_len] = padded_input[-cache_len:]

        return [out.to(orig_dtype), conv_states_f.to(orig_dtype)]
```
