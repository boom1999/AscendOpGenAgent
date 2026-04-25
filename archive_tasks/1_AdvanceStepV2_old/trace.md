# AdvanceStepV2 AscendC Operator Trace

## 算子信息

| 项目 | 值 |
|------|-----|
| 算子名称 | AdvanceStepV2 |
| 来源 | vLLM speculative decoding advance step |
| 数据类型 | int64 |
| 输入 | input_tokens, sampled_tokens, input_positions, seq_lens, slot_mapping, block_table, spec_tokens, accepted_num, num_seqs, num_queries, block_size |
| 输出 | out_input_tokens, out_input_positions, out_seq_lens, out_slot_mapping |
| 硬件 | Ascend 910B3 (40 Vector Cores, UB=192KB, 1800MHz) |
| NPU | npu:0 |

## 执行概览

| 阶段 | 状态 | 迭代次数 | 备注 |
|------|------|----------|------|
| Phase 0: 参数确认 | PASS | - | npu=0, op_file=1_AdvanceStepV2.py |
| Phase 1: 环境准备 | PASS | - | 文件复制完成 |
| Phase 2: 用例精简 | PASS | - | 300 → 10 cases |
| Phase 3: TileLang 设计表达 | PASS | 2 | 首轮退化(Type3)，修复后通过退化检测；TileLang 功能验证因 int64 框架限制未做强制要求 |
| Phase 4: AscendC 转译与验证 | PASS | 1 | 首轮即通过（10/10 cases matched） |
| Phase 5: 性能分析 | PASS | - | AscendC 2.39x 加速 |
| Phase 6: 全量用例验证 | PASS | 1 | 300/300 cases matched |
| Phase 7: Trace 记录 | PASS | - | 本文件 |

## Phase 0: 参数确认

- `npu`: 0
- `op_file`: `/home/Code/AscendOpGenAgent/benchmarks/AIInfraNPUKernelBench/level1/1_AdvanceStepV2.py`
- `output_dir`: `/home/Code/AscendOpGenAgent/output/1_AdvanceStepV2/`
- 环境变量: `ASCEND_RT_VISIBLE_DEVICES=0`

## Phase 1: 环境准备

- 复制 `1_AdvanceStepV2.py` → `output/1_AdvanceStepV2/model.py`
- 复制 `1_AdvanceStepV2.json` → `output/1_AdvanceStepV2/1_AdvanceStepV2.json`
- 修改 `model.py` 的 `get_input_groups()` 从 generator 改为 list comprehension（验证脚本要求返回 list）

## Phase 2: 用例精简

- 原始用例: 300 个
- 精简后: 10 个
- 覆盖: num_seqs=1~8739, token_each_reqs=2~122, 不同 sampled_cols
- 备份: `1_AdvanceStepV2.json.bak`

## Phase 3: TileLang 设计表达

### 迭代 0 (tl_iteration=0)

**退化检测**: Type3 - forward() 中使用了 `torch.cat` 和 `torch.argmin`
**修复**: 将 argmin 逻辑移入 TileLang kernel（扫描 sampled_tokens 寻找首个负值）

### 迭代 1 (tl_iteration=1)

**退化检测**: PASS
**功能验证**: TileLang 对 int64 间接索引支持有限，设计文件保留作为 AscendC 转译参考

**产出**:
- `design/block_level/advance_step_v2.py` — 并行策略设计
- `design/tile_level/advance_step_v2.py` — TileLang tile-level kernel 设计
- `model_new_tilelang.py` — TileLang wrapper（通过退化检测）

## Phase 4: AscendC 转译与验证

### 迭代 0 (ac_iteration=0)

**核心设计决策**:
1. **int64 标量处理**: 使用 `GlobalTensor<int64_t>` 的 `GetValue()/SetValue()` 标量访问，因 NPU 对 int64 向量指令支持有限
2. **多核并行**: `usedCoreNum = min(numReqs, 40)`，每核处理 `reqsPerCore` 个请求
3. **argmin 实现**: 顺序扫描 sampled_tokens 行，找到首个负值位置
4. **Tiling**: 简单 struct 传参（numReqs, tokenEachReqs, sampledCols, maxNumBlocks, blockSize, usedCoreNum, reqsPerCore）

**编译过程中的修复**:
1. kernel 入口函数参数改为 `GM_ADDR` 类型 + 独立 `_do` wrapper 函数
2. 移除 pybind11.cpp 中对 `kernel_common.h` 的包含（`__aicore__` 属性不可用于 host 编译）
3. 修复 TensorAddr 的 `const_cast` 写法

**验证结果**: 10/10 cases matched (首轮通过)

**产出**:
- `kernel/advance_step_v2_tiling.h` — Tiling 结构体
- `kernel/kernel_common.h` — 通用工具函数
- `kernel/advance_step_v2_kernel.h` — 核心 kernel 类
- `kernel/advance_step_v2.cpp` — 设备入口 + host launcher
- `kernel/pybind11.cpp` — Python 绑定
- `model_new_ascendc.py` — AscendC wrapper

## Phase 5: 性能分析

| 实现 | Mean(ms) | Median(ms) | Min(ms) | Max(ms) | Std(ms) |
|------|----------|------------|---------|---------|---------|
| Reference (torch_npu) | 0.647 | 0.658 | 0.565 | 0.889 | 0.054 |
| AscendC | 0.271 | 0.272 | 0.238 | 0.309 | 0.012 |

**加速比: 2.39x**

说明: Reference 使用 `torch_npu.npu_advance_step_flashattn` 内置算子。AscendC 自定义 kernel 通过标量 GM 访问和多核并行实现了显著加速，且方差更小（0.012 vs 0.054）。

## Phase 6: 全量用例验证

- 恢复 `1_AdvanceStepV2.json.bak` → `1_AdvanceStepV2.json` (300 cases)
- 设置 `AIINFRABENCH_FULL_CASES=1` 环境变量
- **验证结果**: 300/300 cases matched (PASS)
- 无需修复，首轮全量通过

## 走偏点分析

1. **TileLang Phase 退化 (Type3)**: 初始 `model_new_tilelang.py` 中错误使用了 `torch.cat` 和 `torch.argmin`，应将这些操作融入 kernel。修复方向正确，一轮修复后通过。
2. **AscendC 编译问题**: 主要是 kernel 入口函数签名格式和 host/device 代码分离问题，通过参考 archive_tasks 中的 gather_elements_v2 实现解决。
3. **全量验证环境变量**: 初次全量验证时未设置 `AIINFRABENCH_FULL_CASES=1`，导致仍只跑 10 个 smoke 用例。后设置环境变量后正确运行 300 个用例。

## 文件清单

```
output/1_AdvanceStepV2/
├── model.py                              # 算子描述（reference）
├── 1_AdvanceStepV2.json                  # 测试用例（300 cases，已恢复全量）
├── 1_AdvanceStepV2.json.bak              # 全量用例备份
├── design/
│   ├── block_level/
│   │   └── advance_step_v2.py            # Block-level 并行设计
│   └── tile_level/
│       └── advance_step_v2.py            # TileLang tile-level kernel 设计
├── kernel/
│   ├── advance_step_v2_tiling.h          # Tiling 结构体
│   ├── kernel_common.h                   # 通用工具函数
│   ├── advance_step_v2_kernel.h          # 核心 kernel 类
│   ├── advance_step_v2.cpp               # 设备入口 + host launcher
│   ├── pybind11.cpp                      # Python 绑定
│   └── build/                            # 编译产物
│       └── _advance_step_v2_ext.cpython-311-aarch64-linux-gnu.so
├── model_new_tilelang.py                 # TileLang wrapper
├── model_new_ascendc.py                  # AscendC wrapper
└── trace.md                              # 本文件
```
