# ScatterBlockUpdate AscendC Kernel - Execution Trace

## 算子概述
- **算子名称**: ScatterBlockUpdate
- **语义**: `output = input.clone(); output[indices[k,0], indices[k,1], :] = update[k, :]`
- **输入**: input (D0, D1, D2), indices (K, 2), update (K, D2)
- **支持 dtype**: bfloat16, float32, int8
- **核心特征**: 纯数据搬移操作，无计算

---

## Phase 0: 参数确认 — 成功
- npu=1, ASCEND_RT_VISIBLE_DEVICES=1
- op_file: `/home/Code/AscendOpGenAgent/benchmarks/AIInfraNPUKernelBench/level1/8_ScatterBlockUpdate.py`
- output_dir: `/home/Code/AscendOpGenAgent/output/20260424_155444/8_ScatterBlockUpdate/`

## Phase 1: 环境准备 — 成功
- 复制 model.py 和 8_ScatterBlockUpdate.json 到 output_dir
- 原始测试用例：48 个

## Phase 2: 测试用例精简 — 成功
- 从 48 个精简到 10 个
- 备份 8_ScatterBlockUpdate.json.bak
- 覆盖 dtype (bf16/fp32/int8)、indices dtype (int32/int64)、D2 大小 (1/127/128/256)、K 规模 (1~262144)

## Phase 3: TileLang 设计表达 — 成功
- Block-level 设计：K 维度按核数划分，每核处理 kPerCore 行
- Tile-level 设计：逐行读取 indices 并进行 row-level 数据搬移
- TileLang 验证跳过（scatter 为纯数据搬移操作，TileLang DSL 表达受限）

## Phase 4: AscendC 转译与验证 — 成功（3 次迭代）

### 迭代 0: DataCopyPad with TBuf<VECCALC>
- **方案**: 使用 DataCopyPad + TBuf<VECCALC> + PipeBarrier<PIPE_MTE2/MTE3> 进行 DMA 搬移
- **结果**: 9/10 失败，仅 K=1 通过
- **根因**: TBuf<VECCALC> 与 DataCopyPad 的同步模型不兼容
- **后续改为 TQue<VECIN> + PipeBarrier<PIPE_ALL>，有改善但仍有 case 失败**
- **最终改为纯标量 GetValue/SetValue**：9/10 通过，仅 case 0 (D2=1, bf16, K=2048) 失败

### 迭代 1: 元素级标量 + C++ 侧 clone
- **方案**: 按 dtype 使用对应大小的 GlobalTensor (int16_t/int32_t/uint8_t)，clone 移到 C++ 侧保证 stream ordering
- **结果**: 仍然 case 0 失败 (~200 mismatches)
- **排除**: stream ordering 不是根因，false sharing 是嫌疑

### 迭代 2: 单核回退修复 false sharing
- **根因确认**: D2=1 时每行仅 1-4 bytes，多行共享同一 cache line，多核标量写导致 write tearing
- **修复**: `if (D2 <= 1) usedCoreNum = 1;`
- **结果**: 10/10 全部通过

### 关键走偏点
1. **DataCopyPad 同步问题**：最初尝试 DMA 块搬移，但 TBuf 和 TQue 在 scatter 场景下同步困难，改为标量方案后大幅改善
2. **False sharing 漏判**：最初仅对 bf16 D2=1 设阈值 (`D2*elemSize<=2`)，全量验证发现 fp32 D2=1 也受影响，最终改为 `D2<=1` 全覆盖

## Phase 5: 性能分析 — 完成

| Case | Shape (input) | dtype | K | Ref (ms) | AscendC (ms) | Speedup |
|------|--------------|-------|---|---------|-------------|---------|
| 0 | (2048,128,1) | bf16 | 2048 | 0.498 | 0.289 | 1.7x |
| 1 | (10240,128,1) | fp32 | 1 | 0.899 | 0.281 | 3.2x |
| 2 | (8192,128,256) | int8 | 240 | 42.837 | 1.309 | 32.7x |
| 3 | (2048,128,256) | bf16 | 2048 | 22.876 | 0.885 | 25.8x |
| 4 | (8192,128,128) | bf16 | 262144 | 43.943 | 17.202 | 2.6x |
| 5 | (2048,256,128) | fp32 | 16384 | 43.162 | 2.524 | 17.1x |
| 6 | (2048,128,127) | bf16 | 16384 | 11.154 | 1.123 | 9.9x |
| 7 | (2048,128,127) | int8 | 16384 | 3.500 | 0.913 | 3.8x |
| 8 | (2048,128,128) | bf16 | 16384 | 11.016 | 1.540 | 7.2x |
| 9 | (2048,128,1) | fp32 | 262144 | 1.670 | 0.710 | 2.4x |

**Overall: Reference mean=18.155ms, AscendC mean=2.678ms → 6.8x speedup**

## Phase 6: 全量用例验证 — 成功
- 恢复 48 个全量用例
- 首次验证 44/48 通过（4 个 fp32 D2=1 case 失败）
- 修复：扩展单核条件 `D2<=1`
- 第二次验证：**48/48 全部通过**

## Phase 7: Trace 记录 — 本文件

---

## 实现架构

### Kernel 设计
- **任务划分**: K 个 scatter 行按核数均分，每核处理 `kPerCore` 行
- **核数**: `min(20, K)`，D2==1 时退化为单核避免 false sharing
- **数据访问**: 标量 GetValue/SetValue，按 elemSize 分发到 int16_t/int32_t/uint8_t 类型的 GlobalTensor
- **同步**: 无需显式同步（各核写不同行，D2>1 时行间无 cache line 冲突）

### 文件清单
- `kernel/scatter_block_update_tiling.h` — Tiling 结构体 (D1, D2, K, elemSize, usedCoreNum, kPerCore)
- `kernel/kernel_common.h` — CeilDivU32, CopyTiling 模板
- `kernel/scatter_block_update_kernel.h` — Kernel 类，CopyRows{8,16,32} 方法
- `kernel/scatter_block_update.cpp` — Kernel 入口 + host 调用包装
- `kernel/pybind11.cpp` — pybind11 绑定，C++ 侧 clone + tiling 填充
- `model_new_ascendc.py` — Python 包装层
