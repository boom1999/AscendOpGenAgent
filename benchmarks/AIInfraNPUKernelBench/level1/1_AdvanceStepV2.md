## 功能说明

vLLM是一个高性能的LLM推理和服务框架，专注于优化大规模语言模型的推理效率。它的核心特点包括PageAttention和高效内存管理。advance_step算子的主要作用是推进推理步骤，即在每个生成步骤中更新模型的状态并生成新的inputTokens、inputPositions、seqLens和slotMapping，为vLLM的推理提升效率。

## 计算公式

$$
blockTablesStride = blockTables.stride(0)
$$

$$
inputTokens[blockIdx] = sampledTokenIds[blockIdx]
$$

$$
inputPositions[blockIdx] = seqLens[blockIdx]
$$

$$
seqLens[blockIdx] = seqLens[blockIdx] + 1
$$

$$
slotMapping[blockIdx] = (blockTables[blockIdx] + blockTablesStride * blockIdx) * blockSize + (seqLens[blockIdx] \% blockSize)
$$

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | shape |
|---|---|---|---|---|
| input_tokens | 输入/输出 | 用于更新vLLM模型中的token值。不支持空Tensor，取值为正整数。 | INT64 | (numSeqs, 1+specNum) |
| sampled_token_ids | 输入 | 储存tokenID。不支持空Tensor，取值为正整数。 | INT64 | (numSeqs, 1+specNum) |
| input_positions | 输入/输出 | 记录token的index。不支持空Tensor，取值为正整数。 | INT64 | (numSeqs,) |
| seq_lens | 输入/输出 | 记录不同blockIdx下seq的长度。不支持空Tensor，取值为正整数。 | INT64 | (numSeqs,) |
| slot_mapping | 输入/输出 | 将token在序列中的位置映射到物理位置。不支持空Tensor，取值为正整数。 | INT64 | (numSeqs,) |
| block_tables | 输入 | 记录不同blockIdx下block的大小。不支持空Tensor，第二维大于max(seqLens)/blockSize。 | INT64 | (numSeqs, maxBlocks) |
| spec_token | 输入 | 记录当前投机模型的token的index。不支持空Tensor，取值为正整数。 | INT64 | (numSeqs, specNum) |
| accepted_num | 输入 | 记录每个request接受的投机数量。不支持空Tensor，取值为正整数。 | INT64 | (numSeqs,) |
| num_seqs | 输入 | 输入的seq数量，与seqLens长度一致。numSeqs >= numQueries。 | INT64 | 标量 |
| num_queries | 输入 | 输入的Query数量，与sampledTokenIds第一维长度一致。 | INT64 | 标量 |
| block_size | 输入 | 每个block的大小。取值为正整数。 | INT64 | 标量 |

```python
class Model(nn.Module):
    """
    AdvanceStepV2: advances vLLM inference step by updating input_tokens,
    input_positions, seq_lens, and slot_mapping.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        input_tokens: torch.Tensor,
        sampled_tokens: torch.Tensor,
        input_positions: torch.Tensor,
        seq_lens: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
        spec_tokens: torch.Tensor,
        accepted_num: torch.Tensor,
        num_seqs: int,
        num_queries: int,
        block_size: int,
    ) -> List[torch.Tensor]:
        """
        Advance step for vLLM speculative decoding.

        Args:
            input_tokens:   (num_reqs * (spec_num+1),), int64
            sampled_tokens: (num_reqs, spec_num+1), int64
            input_positions:(num_reqs * (spec_num+1),), int64
            seq_lens:       (num_reqs * (spec_num+1),), int64
            slot_mapping:   (num_reqs * (spec_num+1),), int64
            block_table:    (num_reqs, max_blocks), int64
            spec_tokens:    (num_reqs, spec_num), int64
            accepted_num:   (num_reqs,), int64
            num_seqs:       int
            num_queries:    int
            block_size:     int

        Returns:
            List of [out_input_tokens, out_input_positions, out_seq_lens, out_slot_mapping]
        """
        num_reqs = num_seqs
        token_each_reqs = 1 + spec_tokens.shape[1]

        out_input_positions = input_positions + torch.repeat_interleave(accepted_num, token_each_reqs) + 1
        out_seq_lens = out_input_positions + 1

        index = torch.argmin(
            torch.cat([sampled_tokens, torch.full((num_reqs, 1), -1, device=sampled_tokens.device)], dim=1),
            dim=1
        ) - 1
        last_tokens = sampled_tokens[torch.arange(num_reqs), index]

        out_input_tokens = input_tokens.clone()
        if token_each_reqs == 1:
            out_input_tokens[:num_reqs] = last_tokens.to(dtype=input_tokens.dtype)
        else:
            input_tokens_2d = out_input_tokens.view(-1, token_each_reqs)
            input_tokens_2d[:num_reqs, 0] = last_tokens
            input_tokens_2d[:num_reqs, 1:] = spec_tokens

        req_indices = torch.repeat_interleave(torch.arange(num_reqs), token_each_reqs, dim=0)
        max_num_blocks_per_req = block_table.shape[1]
        block_table_indices = (
            req_indices * max_num_blocks_per_req +
            out_input_positions // block_size
        )
        block_numbers = block_table.flatten()[block_table_indices]
        block_offsets = out_input_positions % block_size
        out_slot_mapping = block_numbers * block_size + block_offsets

        return [out_input_tokens, out_input_positions, out_seq_lens, out_slot_mapping]
```
