# ApplyRotaryPosEmb (8_ApplyRotaryPosEmb) - Trace

## 基本信息

- **算子**: ApplyRotaryPosEmb (半旋转 Rotary Position Embedding)
- **NPU**: 3
- **输出目录**: `output/20260423_181735/8_ApplyRotaryPosEmb/`
- **日期**: 2026-04-23

## 算子说明

对 query 和 key 张量应用旋转位置编码。支持 BSND 和 TND 两种 layout。
核心计算: `split = D_rot // 2`, `out[:split] = x1*cos1 - x2*sin1`, `out[split:D_rot] = x2*cos2 + x1*sin2`, `out[D_rot:D] = x[D_rot:D]` (passthrough)。

---

## Phase 0: 参数确认 — 成功

- op_file: `/home/Code/AscendOpGenAgent/benchmarks/AIInfraNPUKernelBench/level2/8_ApplyRotaryPosEmb.py`
- output_dir: `output/20260423_181735/8_ApplyRotaryPosEmb/`
- npu: 3
- ASCEND_RT_VISIBLE_DEVICES=3

## Phase 1: 环境准备 — 成功

- 复制 model.py 和 8_ApplyRotaryPosEmb.json (20 cases) 到输出目录

## Phase 2: 测试用例精简 — 成功

- 原始: 20 cases → 精简: 10 cases
- 覆盖: bf16/fp16/fp32, BSND/TND, D=64/128, D_rot=64/128, partial rotary (D_rot < D)
- 备份: 8_ApplyRotaryPosEmb.json.bak

## Phase 3: TileLang 设计表达 — 成功

### Block-Level 设计
- Persistent kernel: 20 cores, blockM=64 rows/block, vec_num=2 sub-blocks
- 每行独立处理 RoPE: full-row copy + rotation overwrite

### Tile-Level 设计
- 逐行处理: 加载 x, cos, sin 的 split 段 → cast 到 fp32 → 乘加减计算 → cast 回 → 存储
- 关键发现: TileLang 的列偏移 GM 读取 (`x[row, D_rot:D]`) 不可靠, 改用 full-row copy + overwrite

### TileLang 验证
- 退化检测: 通过
- 功能验证: 10/10 全部通过
- 迭代次数: 3 (tl_iteration=0: 初始实现有 passthrough 问题 47% mismatch; tl_iteration=1: 修复列偏移 GM 读取问题; tl_iteration=2: 修复 UB slice 问题)

## Phase 4: AscendC 转译与验证 — 成功 (2 次迭代)

### 首次转译 (ac_iteration=0)
- 从 TileLang tile-level 转译为 AscendC kernel
- 使用 KERNEL_TYPE_MIX_AIC_1_2, DataCopyPad, fine-grained barriers
- **问题 1**: `half_t` 编译错误 → 改为 `half`
- **问题 2**: 所有 dtype 放在同一 .cpp 文件导致 `RegisterAscendBinary mix ret 107000` → 拆分为 3 个独立 .cpp 文件
- **问题 3**: 修复注册错误后, 计算结果全错 (98% mismatch, NaN)

### 第二次迭代 (ac_iteration=1)
- 参考 archive_tasks/ 中的工作 kernel 模式 (MhcPostGrad, ClippedSwiglu)
- **根本原因**: KERNEL_TYPE_MIX_AIC_1_2 模式下的 sub-block 索引和 fine-grained pipe barriers 有问题
- **修复**:
  1. `KERNEL_TYPE_MIX_AIC_1_2` → `KERNEL_TYPE_AIV_ONLY`
  2. `DataCopyPad` → `DataCopy`
  3. Fine-grained barriers (`PIPE_MTE2`, `PIPE_MTE3`, `PIPE_V`) → `PipeBarrier<PIPE_ALL>()`
  4. 移除 sub-block splitting (`GetSubBlockNum`, `GetSubBlockIdx`)
  5. 简单 `GetBlockIdx()` 核索引
  6. Cache `Get<>()` 调用到 Init() 中
- 退化检测: 通过
- 功能验证: 10/10 全部通过

### 走偏点分析
1. **KERNEL_TYPE_MIX_AIC_1_2 不适合此 kernel**: 纯 vector 计算的 element-wise kernel 应使用 KERNEL_TYPE_AIV_ONLY, 无需 AIC core
2. **Fine-grained barriers 容易出错**: 使用 PIPE_ALL 更安全, 虽然性能较低但保证正确性
3. **DataCopyPad vs DataCopy**: 当数据大小对齐时, 直接使用 DataCopy 更简单可靠
4. **单文件多 kernel 入口**: 每个 `__global__ __aicore__` 函数必须在独立 .cpp 文件中

## Phase 5: 性能分析

| 实现 | Mean(ms) | Median(ms) | Min(ms) | Max(ms) |
|------|----------|------------|---------|---------|
| reference | 0.621 | 0.544 | 0.324 | 1.312 |
| ascendc | 1.386 | 1.442 | 0.772 | 2.607 |

- 加速比: ~0.45x (AscendC 慢于 reference)
- 原因: 逐行处理 + PIPE_ALL 保守同步 + full-row copy 双写带宽开销
- 优化方向: 批量行处理、fine-grained barriers (需谨慎)、条件跳过 full-row copy (D_rot==D 时)

## Phase 6: 全量用例验证 — 成功

- 恢复 .json.bak (20 cases)
- 全量验证: 20/20 全部 matched
- 无需修复

## 产出文件

```
output/20260423_181735/8_ApplyRotaryPosEmb/
├── model.py                          # 参考实现
├── 8_ApplyRotaryPosEmb.json          # 全量测试用例 (20 cases)
├── 8_ApplyRotaryPosEmb.json.bak      # 备份
├── design/
│   ├── block_level/                  # Block-level TileLang 设计
│   └── tile_level/                   # Tile-level TileLang 设计
├── kernel/
│   ├── apply_rotary_pos_emb_kernel.h # AscendC kernel 模板类
│   ├── apply_rotary_pos_emb_tiling.h # Tiling 结构体
│   ├── kernel_common.h               # CopyTiling 工具
│   ├── vector_tile.h                 # DataCopyPad 工具 (已不使用)
│   ├── apply_rotary_pos_emb.cpp      # fp32 入口
│   ├── apply_rotary_pos_emb_fp16.cpp # fp16 入口
│   ├── apply_rotary_pos_emb_bf16.cpp # bf16 入口
│   ├── pybind11.cpp                  # Python 绑定
│   └── build/                        # 编译输出
├── model_new_tilelang.py             # TileLang 优化实现
├── model_new_ascendc.py              # AscendC 优化实现
└── trace.md                          # 本文件
```

## 总结

- **正确性**: 全量 20/20 cases 通过
- **TileLang 验证**: 10/10 通过
- **AscendC 迭代**: 2 次 (首次: 注册/计算错误; 第二次: 切换 AIV_ONLY 模式修复)
- **性能**: 当前慢于 reference ~2.2x, 可通过优化 barrier 和批处理改善
