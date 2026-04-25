# Trace: 6_CausalConv1dFn AscendC Kernel

## 基本信息

| 项目 | 值 |
|------|-----|
| 算子 | CausalConv1dFn |
| NPU | 1 |
| 输出目录 | output/20260423_181735/6_CausalConv1dFn/ |
| 最终状态 | **成功** |

---

## Phase 0: 参数确认
- **状态**: 成功
- npu=1, op_file=/home/Code/AscendOpGenAgent/benchmarks/AIInfraNPUKernelBench/level2/6_CausalConv1dFn.py
- output_dir=/home/Code/AscendOpGenAgent/output/20260423_181735/6_CausalConv1dFn/

## Phase 1: 环境准备
- **状态**: 成功
- 复制 model.py 和 6_CausalConv1dFn.json (60 cases) 到输出目录

## Phase 2: 测试用例精简
- **状态**: 成功
- 原始 60 cases → 精简至 10 cases
- 覆盖: bf16/fp16 dtype, dim=[64,512,768,4096,6144,16384], batch=[1,4,8,60,128,256], residualConnection=[0,1], ISM=[0,2]/[1,2]
- .json.bak 已备份

## Phase 3: TileLang 设计表达
- **状态**: 设计完成，TileLang 验证跳过
- 完成了 block-level 和 tile-level 设计
- TileLang kernel 设计: 8 UB buffers (w0/w1/w2/prev2/prev1/curr/out/tmp), 滑动窗口 K=3
- **TileLang 验证跳过原因**: 编译时报错 `the 1st parameter maybe need a type '__ubuf__ half *'`，硬件不支持 bf16 上的 vmul/vadd 操作。属 B 类（框架/硬件限制），保留设计表达并继续进入 Phase 4。

## Phase 4: AscendC 转译与验证
- **状态**: 成功 (ac_iteration=1, 共迭代 2 次)
- **迭代历史**:
  - **Iteration 0**: 
    - 转译 TileLang 设计为 AscendC kernel
    - 编译成功
    - 退化检测通过
    - 功能验证失败: `RuntimeError: Expected all tensors to be on the same device. Expected NPU tensor`
    - 分析: A 类错误 — model_new_ascendc.py 中 conv_states scatter 操作混合了 NPU 和 CPU tensor
    - 修复: 将 scatter 操作统一在 CPU 上执行 (`conv_states.cpu().to(orig_dtype).clone()`, `cache_indices.cpu().long()`)
    - 重新验证仍失败: 输出全为零（100% mismatch）
    - 进一步分析: kernel 内部 pipe barrier 使用了特定 pipe (PIPE_V/PIPE_MTE2/PIPE_MTE3) 导致同步不足
  - **Iteration 1**:
    - 参考 archive_tasks/2_ClippedSwiglu 和 6_ManifoldConstrainedHyperConnectionPost 的 barrier 模式
    - 将所有 `pipe_barrier(PIPE_V/MTE2/MTE3)` 替换为 `PipeBarrier<PIPE_ALL>()`
    - 重新编译并验证: **10/10 cases 全部通过**

### 走偏点分析
1. **初始 pipe barrier 过于精细**: 使用了 PIPE_V/PIPE_MTE2/PIPE_MTE3 等特定管道同步，但 TBuf<VECCALC> buffer 在跨管道操作（如 vector 计算后 MTE3 写 GM）时需要 PIPE_ALL 全同步
2. **NPU/CPU tensor 混合**: 验证框架会将输入移到 NPU，model_new_ascendc.py 中 clone 和 index 操作需要确保设备一致

## Phase 5: 性能分析
- **状态**: 完成

| 实现 | Mean (ms) | Median (ms) | Min (ms) | Max (ms) |
|------|-----------|-------------|----------|----------|
| Reference | 4.66 | 2.05 | 0.41 | 19.20 |
| AscendC | 15.79 | 8.34 | 0.77 | 43.48 |

- AscendC 约 3.4x 慢于 reference（整体），主要因素:
  - 保守的 PIPE_ALL 全同步
  - 逐 token 串行处理序列位置
  - fp16/bf16 → fp32 类型转换增加数据搬运开销

## Phase 6: 全量用例验证
- **状态**: 成功
- 恢复 .json.bak → .json (60 cases)
- **60/60 cases 全部通过**

## Phase 7: Trace 记录
- **状态**: 本文件

---

## 产物清单

| 文件 | 说明 |
|------|------|
| model.py | 参考实现（只读） |
| 6_CausalConv1dFn.json | 测试用例（已恢复全量 60 cases） |
| 6_CausalConv1dFn.json.bak | 原始备份 |
| design/block_level/ | TileLang block-level 设计 |
| design/tile_level/ | TileLang tile-level 设计 |
| model_new_tilelang.py | TileLang 实现（验证因框架限制跳过） |
| model_new_ascendc.py | AscendC Python wrapper |
| kernel/causal_conv1d_kernel.h | AscendC kernel 主实现 |
| kernel/causal_conv1d.cpp | kernel 入口 + launch 函数 |
| kernel/pybind11.cpp | pybind11 binding |
| kernel/causal_conv1d_tiling.h | Tiling 结构体 |
| kernel/kernel_common.h | 工具函数 (CopyTiling) |
| kernel/CMakeLists.txt | 构建脚本 |
| trace.md | 本 trace 文件 |
