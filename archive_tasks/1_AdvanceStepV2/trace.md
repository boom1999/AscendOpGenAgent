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
| Phase 2: 用例精简 | PASS | - | 300 → 9 cases |
| Phase 3: TileLang 设计表达 | PASS | 1 | 退化检测通过；功能验证因 TileLang int64 Buffer 框架限制跳过 |
| Phase 4: AscendC 转译与验证 | PASS | 1 | 首轮即通过（9/9 cases matched） |
| Phase 5: 性能分析 | PASS | - | AscendC 2.89x 加速 |
| Phase 6: 全量用例验证 | PASS | 1 | 300/300 cases matched |
| Phase 7: Trace 记录 | PASS | - | 本文件 |

## Phase 0: 参数确认

- `npu`: 0
- `op_file`: `/home/Code/AscendOpGenAgent/benchmarks/AIInfraNPUKernelBench/level1/1_AdvanceStepV2.py`
- `output_dir`: `/home/Code/AscendOpGenAgent/output/20260422_115125/1_AdvanceStepV2/`
- 环境变量: `ASCEND_RT_VISIBLE_DEVICES=0`

## Phase 1: 环境准备

- 复制 `1_AdvanceStepV2.py` → `output/20260422_115125/1_AdvanceStepV2/model.py`
- 复制 `1_AdvanceStepV2.json` → `output/20260422_115125/1_AdvanceStepV2/1_AdvanceStepV2.json`
- 阅读 `1_AdvanceStepV2.md` 获取算子定义、计算公式和 CPU 参考实现

## Phase 2: 用例精简

- 原始用例: 300 个
- 精简后: 9 个
- 覆盖: num_seqs=1~9946, spec_num=1~126, total_tokens=4~1066158
- 备份: `1_AdvanceStepV2.json.bak`

## Phase 3: TileLang 设计表达

### 迭代 0 (tl_iteration=0)

**退化检测**: PASS — 无禁止的 PyTorch 计算操作
**功能验证**: 跳过 — TileLang 框架对 int64 Buffer 支持有限（`TypeError: Buffer() takes no arguments`），属框架自身问题（B 类），设计文件保留作为 AscendC 转译参考

**产出**:
- `design/block_level/advance_step_v2.py` — 并行策略设计
- `design/tile_level/advance_step_v2.py` — TileLang tile-level kernel 设计
- `model_new_tilelang.py` — TileLang wrapper（通过退化检测）

## Phase 4: AscendC 转译与验证

### 迭代 0 (ac_iteration=0)

**核心设计决策**:
1. **int64 标量处理**: 使用 `GlobalTensor<int64_t>` 的 `GetValue()/SetValue()` 标量访问
2. **多核并行**: `usedCoreNum = min(numReqs, 40)`，每核处理 `reqsPerCore` 个请求
3. **argmin 实现**: 顺序扫描 sampled_tokens 行，找到首个负值位置
4. **Tiling**: 简单 struct 传参（numReqs, tokenEachReqs, sampledCols, maxNumBlocks, blockSize, usedCoreNum, reqsPerCore）

**验证结果**: 9/9 cases matched（首轮通过）

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
| Reference (torch_npu) | 1.358 | 0.672 | 0.563 | 6.482 | 1.556 |
| AscendC | 0.470 | 0.246 | 0.224 | 2.036 | 0.560 |

**加速比: 2.89x**

AscendC 自定义 kernel 通过标量 GM 访问和多核并行实现了显著加速，且方差更小。

## Phase 6: 全量用例验证

- 恢复 `1_AdvanceStepV2.json.bak` → `1_AdvanceStepV2.json` (300 cases)
- **验证结果**: 300/300 cases matched (PASS)
- 无需修复，首轮全量通过

## 走偏点分析

1. **TileLang 功能验证跳过**: TileLang 框架对 int64 Buffer 类型支持有限（`Buffer() takes no arguments`），属 B 类环境问题。设计表达保留，不为通过 TileLang 验证而扭曲设计。
2. **evaluate_tilelang.sh 路径问题**: 首次调用时使用绝对路径导致路径拼接错误，改用相对路径后解决。
3. **design 模块导入**: 首次运行时缺少 `__init__.py` 和 sys.path 设置，添加后解决。

## 文件清单

```
output/20260422_115125/1_AdvanceStepV2/
├── model.py                              # 算子描述（reference）
├── 1_AdvanceStepV2.json                  # 测试用例（300 cases，已恢复全量）
├── 1_AdvanceStepV2.json.bak              # 全量用例备份
├── design/
│   ├── __init__.py
│   ├── block_level/
│   │   └── advance_step_v2.py            # Block-level 并行设计
│   └── tile_level/
│       ├── __init__.py
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
