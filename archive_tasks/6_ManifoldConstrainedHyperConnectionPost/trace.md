# Trace: ManifoldConstrainedHyperConnectionPost

- 时间: 2026-04-22
- 算子: 6_ManifoldConstrainedHyperConnectionPost
- 最终结果: PASS (tilelang) | PASS (ascendc)

## 阶段零: Case 精简

- 结果: 通过
- 原始 case 数: 300
- 精简后 case 数: 10
- 备注: 覆盖 fp16/bf16、3D/4D shape、n=4/6/8、小/大 D、不同 range

## 阶段一: TileLang

- 结果: 通过
- evaluate_tilelang.sh 执行次数: 2
- 关键错误信息:
  - 第 1 次: `ValueError: too many values to unpack (expected 1)` — `(y_pad,) = kernel(...)` 解包失败
- Agent 行为记录:
  - 第 1 轮: 读取 BlockLevelDesign.md、TileLangAscendProgrammingGuide.md 和 MhcPostGrad archive 参考实现。设计 block-level（persistent-kernel over batch, D-tiled, pure Vector）和 tile-level（完整 kernel 实现）。生成 model_new_tilelang.py。退化检测通过。evaluate_tilelang.sh 失败：`ValueError: too many values to unpack (expected 1)`
  - 第 2 轮: 修改 model_new_tilelang.py 中 kernel 返回值解包方式，从 `(y_pad,) = kernel(...)` 改为 `result = kernel(...); y_pad = result[0] if isinstance(result, (tuple, list)) else result`。evaluate_tilelang.sh 通过，10/10 case matched
- 走偏点: 首次使用 tuple 解包 `(y_pad,)` 不兼容 tilelang 单输出返回行为（直接返回 tensor 而非 tuple）。参考 MhcPostGrad 有 4 个输出所以用 tuple 解包没问题，但本算子只有 1 个输出。

## 阶段二: AscendC

- 结果: 通过
- evaluate_ascendc.sh 执行次数: 1
- 关键错误信息: 无
- Agent 行为记录:
  - 第 1 轮: 读取 TileLang-AscendC-API-Mapping.md、dsl2Ascendc.md、AscendCVerification.md 和 MhcPostGrad archive kernel 实现。按映射关系转译：T.copy → DataCopy, T.tile.mul(scalar) → Muls, T.tile.add → Add, T.tile.fill → Duplicate, T.tile.cast → Cast。生成 kernel 文件（mhc_post_tiling.h, kernel_common.h, mhc_post_kernel.h, mhc_post_fp16.cpp, mhc_post_bf16.cpp, pybind11.cpp）和 model_new_ascendc.py。退化检测通过。evaluate_ascendc.sh 编译+验证一次通过，10/10 case matched
- 走偏点: 无

## 阶段三: 性能分析

- 结果: 完成

| 实现 | 平均耗时(ms) | Median(ms) |
|------|-------------|------------|
| Reference | 1.755 | 0.348 |
| TileLang | 2.075 | 1.363 |
| AscendC | 2.090 | 1.135 |

- 大 batch 场景 (case 0/1, B=1024): AscendC 6.4ms vs Reference 7.0ms，加速约 8%
- 小 batch 场景: AscendC 因 kernel launch 开销略慢于 reference

## 阶段四: 全量用例验证

- 结果: 通过
- 全量 case 数: 300
- 通过 case 数: 300/300
- 修复次数: 0

## 评测输出摘要

```
========================================================================
AscendC Verification Report
========================================================================
Status    : PASS
Tolerance : atol=0.01, rtol=0.01
case[0] ~ case[299]: all matched
------------------------------------------------------------------------
Result: pass
```

## 违规路径记录

- 外部 web 检索: 否
- 直接整体复制参考实现: 否（参考 MhcPostGrad archive 的代码结构和模式，但算法逻辑为 forward pass 独立实现）
- PyTorch / torch_npu 语义回退: 否
