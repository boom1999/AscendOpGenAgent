## 功能说明

- 算子功能：MoE的routing计算，根据MoeGatingTopK的计算结果做routing处理，支持不量化和动态量化模式。相比V2接口增加了动态量化功能（支持输出expandedX的int8动态量化输出）、增加参数activeExpertRangeOptional（支持筛选有效范围内的expertId）。

- 计算公式：  

  1.对输入expertIdx做排序，得出排序后的结果sortedExpertIdx和对应的序号sortedRowIdx：

    $$
    sortedExpertIdx, sortedRowIdx=keyValueSort(expertIdx,rowIdx)
    $$

  2.以sortedRowIdx做位置映射得出expandedRowIdxOut：

    $$
    expandedRowIdxOut[sortedRowIdx[i]]=i
    $$

  3.在drop模式下，对sortedExpertIdx的每个专家统计直方图结果，得出expertTokensCountOrCumsumOutOptional：

    $$
    expertTokensCountOrCumsumOutOptional[i]=Histogram(sortedExpertIdx)
    $$

  4.计算quant结果：
    - 动态quant：
        - 若不输入scale：
            $$
            dynamicQuantScaleOutOptional = row\_max(abs(x)) / 127
            $$

            $$
            quantResult = round(x / dynamicQuantScaleOutOptional)
            $$
        - 若输入scale:
            $$
            dynamicQuantScaleOutOptional = row\_max(abs(x * scaleOptional)) / 127
            $$

            $$
            quantResult = round(x / dynamicQuantScaleOutOptional)
            $$
  
  5.对quantResult取前NUM\_ROWS个sortedRowIdx的对应位置的值，得出expandedXOut：

    $$
    expandedXOut[i]=quantResult[sortedRowIdx[i]\%NUM\_ROWS]
    $$

  6.expandedRowIdxOut的有效元素数量availableIdxNum计算方式为，expertIdx中activeExpertRangeOptional范围内的元素的个数
    $$
    availableIdxNum = |\{x\in expertIdx| expert\_start \le x<expert\_end \ \}|
    $$

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | shape |
|---|---|---|---|---|
| x | 输入 | MOE的输入token特征 | FLOAT32、FLOAT16、BFLOAT16、INT8、HIFLOAT8 | ND |
| expert_idx | 输入 | 每一行特征对应的K个处理专家，元素专家id不能超过专家数 | INT32 | ND |
| scale | 可选输入 | 用于计算quant结果的参数 | FLOAT32 | ND |
| offset | 可选输入 | 用于计算quant结果的偏移值（非量化和动态quant场景下不输入） | FLOAT32 | ND |
| active_num | 属性 | 总的最大处理row数，输出expandedXOut只有这么多行有效 | INT | - |
| expert_capacity | 属性 | 每个专家能够处理的tokens数，取值>=0 | INT | - |
| expert_num | 属性 | 专家数，expertTokensNumType为key_value模式时取值[0,5120]，其它模式[0,10240] | INT | - |
| drop_pad_mode | 属性 | 是否为DropPad场景。0=Dropless，1=DropPad | INT | - |
| expert_tokens_num_type | 属性 | 0=cumsum模式，1=count模式，2=key_value模式 | INT | - |
| expert_tokens_num_flag | 属性 | 是否输出expertTokensCountOrCumsumOut（true/false） | BOOL | - |
| quant_mode | 属性 | -1=不量化，0=静态quant，1=动态quant，2/3=MXFP8量化，6/7/8=HIF8量化 | INT | - |
| active_expert_range | 可选属性 | [expertStart, expertEnd]，活跃expert范围，左闭右开 | ListInt | - |
| row_idx_type | 属性 | expandedRowIdxOut的索引类型。0=gather，1=scatter | INT | - |
| expanded_x_out | 输出 | 根据expertIdx扩展过的特征 | 非量化同x；量化时INT8/FLOAT8_E5M2/FLOAT8_E4M3FN/HIFLOAT8 | ND |
| expanded_row_idx_out | 输出 | expandedXOut和x的索引映射关系 | INT32 | ND |
| expert_tokens_count_or_cumsum_out | 输出 | expert对应的处理token总数或key-value对 | INT64 | ND |
| expanded_scale_out | 输出 | 量化计算过程中的scale中间值 | FLOAT32、FLOAT8_E8M0 | ND |

## 约束说明

- 输入值域限制：
  - activeNum当前未使用，校验需等于NUM_ROWS*K。
  - expertCapacity当前未使用，仅校验非空。
  - dropPadMode当前只支持0，代表Dropless场景。
  - expertTokensNumType当前只支持1和2，分别代表count模式和key_value模式。
  - expertTokensNumFlag只支持true，代表输出expertTokensCountOrCumsumOut。
  - quantMode：
    - Atlas A2/A3 训练/推理系列：支持1、-1（动态量化和不量化）。
    - Ascend 950PR/950DT：支持-1、1、2、3、6、7、8。
  - HIFLOAT8输入仅quantMode=6时支持。
- 性能模板：
  - Atlas A2/A3系列支持两种性能模板。
  - 低时延性能模板条件：x shape=(1, 7168)，expertIdx shape=(1, 8)，scale shape=(256, 7168)，x为BFLOAT16，quantMode=1，expertTokensNumType=2，expertNum=256，activeExpertRange=[0, 256]。
  - 大batch性能模板条件：NUM_ROWS范围[384, 8192]，K=8，expertNum=256，expertEnd-expertStart<=32，quantMode=-1，rowIdxType=1，expertTokensNumType=1。

```python
class Model(nn.Module):
    """MoE init routing V3: routing + optional quantization."""

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        x: torch.Tensor,
        expert_idx: torch.Tensor,
        scale: torch.Tensor = None,
        offset: torch.Tensor = None,
        active_num: int = 0,
        expert_capacity: int = 0,
        expert_num: int = 0,
        drop_pad_mode: int = 0,
        expert_tokens_num_type: int = 1,
        expert_tokens_num_flag: bool = True,
        quant_mode: int = -1,
        active_expert_range: list = None,
        row_idx_type: int = 0,
    ) -> List[torch.Tensor]:
        """
        MoE routing: sort expert_idx, compute expanded_row_idx, optional quant.

        Args:
            x: (num_rows, h) input hidden states
            expert_idx: (num_rows, k) expert assignments
            scale: optional scale tensor
            offset: optional offset tensor
            active_num: max active tokens
            expert_capacity: capacity per expert (drop_pad_mode=1)
            expert_num: total number of experts
            drop_pad_mode: 0=no drop, 1=drop+pad
            expert_tokens_num_type: histogram type
            expert_tokens_num_flag: whether to output histogram
            quant_mode: -1=none, 0=static, 1=dynamic
            active_expert_range: (start, end) range
            row_idx_type: 0 or 1
        Returns:
            List of [expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale]
        """
        if active_expert_range is None:
            active_expert_range = [0, expert_num]

        def to_np(t):
            if t is None:
                return None
            if isinstance(t, torch.Tensor):
                if t.dtype in (torch.bfloat16, torch.float16):
                    return t.float().numpy()
                return t.numpy()
            return t

        x_np = to_np(x)
        expert_idx_np = to_np(expert_idx)
        scale_np = to_np(scale)
        offset_np = to_np(offset)

        expert_start = active_expert_range[0] if drop_pad_mode == 0 else 0
        expert_end = active_expert_range[1] if drop_pad_mode == 0 else expert_num
        num_rows = x_np.shape[0]
        h = x_np.shape[1]
        k = expert_idx_np.shape[-1]
        expert_idx_in = expert_idx_np.copy().reshape(-1)
        actual_expert_total_num = int(np.sum((expert_idx_in >= expert_start) & (expert_idx_in < expert_end)))

        expert_idx_in[(expert_idx_in < expert_start)] = np.int32(np.iinfo(np.int32).max)
        sorted_expert_indices = np.argsort(expert_idx_in, axis=-1, kind="stable")
        sorted_expert_idx = expert_idx_in[sorted_expert_indices]

        if row_idx_type == 1:
            expanded_row_idx = sorted_expert_indices[:actual_expert_total_num]
        else:
            expanded_row_idx = np.ones(num_rows * k, dtype=np.int32) * -1
            tmp_indices = np.arange(actual_expert_total_num)
            expanded_row_idx[sorted_expert_indices[:actual_expert_total_num]] = tmp_indices

        if not expert_tokens_num_flag:
            expert_tokens_count = None
        else:
            if drop_pad_mode == 0:
                if expert_tokens_num_type == 1:
                    expert_tokens_count = np.bincount(sorted_expert_idx[:actual_expert_total_num] - expert_start)
                    expert_tokens_count = np.concatenate([expert_tokens_count, np.zeros((expert_end - expert_start) - len(expert_tokens_count)).astype(np.int64)])
                elif expert_tokens_num_type == 0:
                    expert_tokens_count = np.bincount(sorted_expert_idx[:actual_expert_total_num] - expert_start)
                    expert_tokens_count = np.concatenate([expert_tokens_count, np.zeros((expert_end - expert_start) - len(expert_tokens_count)).astype(np.int64)])
                    expert_tokens_count = np.cumsum(expert_tokens_count)
                elif expert_tokens_num_type == 2:
                    expert_id, counts = np.unique(sorted_expert_idx[:actual_expert_total_num], return_counts=True)
                    expert_tokens_count = np.column_stack((expert_id, counts))
                    if expert_tokens_count.shape[0] < expert_num:
                        expert_tokens_count = np.concatenate((expert_tokens_count, [[0, 0]]), axis=0)
            else:
                expert_tokens_count = np.bincount(sorted_expert_idx[:actual_expert_total_num] - expert_start)
                expert_tokens_count = np.concatenate([expert_tokens_count, np.zeros((expert_end - expert_start) - len(expert_tokens_count)).astype(np.int64)])
            expert_tokens_count = expert_tokens_count.astype(np.int64)

        if drop_pad_mode == 0:
            if active_num == 0:
                active_num = actual_expert_total_num
            else:
                active_num = min(active_num, actual_expert_total_num)
            expanded_scale = None
            expanded_x = x_np[sorted_expert_indices[:active_num] // k, :]
            if scale_np is not None and quant_mode == -1:
                expanded_scale = scale_np[sorted_expert_indices[:active_num] // k]
        else:
            # drop_pad_mode == 1
            def adapter_capacity(sorted_row_idx, sorted_expert_idx_l, capacity):
                count = 0
                last = sorted_expert_idx_l[0]
                for i, val in enumerate(sorted_expert_idx_l):
                    if last != val:
                        count = 1
                        last = val
                    else:
                        count += 1
                        if count > capacity:
                            sorted_expert_idx_l[i] = -1
                            sorted_row_idx[i] = -1

            adapter_capacity(sorted_expert_indices, sorted_expert_idx, expert_capacity)
            sort_row_tmp = np.full((expert_num * expert_capacity), -1, dtype=int)
            offset_tmp = 0
            lastExpertId = 0
            for i, val in enumerate(sorted_expert_indices):
                if val != -1:
                    if lastExpertId != sorted_expert_idx[i]:
                        offset_tmp = 0
                        lastExpertId = sorted_expert_idx[i]
                    sort_row_tmp[sorted_expert_idx[i] * expert_capacity + offset_tmp] = sorted_expert_indices[i]
                    offset_tmp += 1

            expanded_row_idx = np.full(sorted_expert_indices.shape, -1)
            for i, val in enumerate(sort_row_tmp):
                if val != -1:
                    expanded_row_idx[val] = i

            expanded_x_mask = np.full((expert_num * expert_capacity, h), 1, dtype=int)
            expanded_x = np.full((expert_num * expert_capacity, h), 0, dtype=x_np.dtype)
            for i, val in enumerate(sort_row_tmp):
                if val != -1:
                    expanded_x[i] = x_np[val // k]
                    expanded_x_mask[i] = np.full((h,), 0, dtype=int)

        if quant_mode == -1:
            if scale_np is not None and drop_pad_mode == 1:
                expanded_scale = np.full((expert_num * expert_capacity,), 0, dtype=scale_np.dtype)
                for i, val in enumerate(sort_row_tmp):
                    if val != -1:
                        expanded_scale[i] = scale_np[val // k]
            if scale_np is None:
                expanded_scale = None
        elif quant_mode == 0:
            expanded_scale = None
            expanded_x_fp16 = expanded_x.astype(np.float16)
            scale_val = scale_np.astype(np.float16)
            offset_val = offset_np.astype(np.float16)
            scale_rst = expanded_x_fp16 * scale_val[0]
            add_offset = scale_rst + offset_val[0]
            round_data = np.rint(add_offset)
            round_data = np.clip(round_data, -128, 127)
            expanded_x = round_data.astype(np.int8)
        elif quant_mode == 1:
            x_final = expanded_x.astype(np.float32)
            if scale_np is None:
                x_abs = np.abs(x_final)
                x_max = np.max(x_abs, axis=-1, keepdims=True)
                expanded_scale = x_max / 127
                expanded_x = x_final / expanded_scale
                expanded_x = np.round(expanded_x).astype(np.int8)
            else:
                if scale_np.shape[0] == 1:
                    x_final = x_final * scale_np
                else:
                    if drop_pad_mode == 0:
                        x_final = x_final * scale_np[sorted_expert_idx[:active_num] - expert_start]
                    else:
                        for i, val in enumerate(sort_row_tmp):
                            if val != -1:
                                x_final[i] = x_final[i] * scale_np[i // expert_capacity]
                x_abs = np.abs(x_final)
                x_max = np.max(x_abs, axis=-1, keepdims=True)
                expanded_scale = x_max / 127
                expanded_x = x_final / expanded_scale
                expanded_x = np.round(expanded_x).astype(np.int8)

        if drop_pad_mode == 1:
            expanded_x = np.ma.array(expanded_x, mask=expanded_x_mask).filled(0)
            expanded_x = expanded_x.reshape(expert_num, expert_capacity, h)

        results = [torch.from_numpy(expanded_x), torch.from_numpy(expanded_row_idx.astype(np.int32))]
        if expert_tokens_count is not None:
            results.append(torch.from_numpy(expert_tokens_count))
        else:
            results.append(torch.empty(0, dtype=torch.int64))
        if expanded_scale is not None:
            results.append(torch.from_numpy(np.asarray(expanded_scale).astype(np.float32)))
        else:
            results.append(torch.empty(0, dtype=torch.float32))
        return results
```
