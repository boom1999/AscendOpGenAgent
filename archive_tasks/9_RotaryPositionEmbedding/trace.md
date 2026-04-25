# Trace: RotaryPositionEmbedding

- 时间: 2026-04-23
- 算子: 9_RotaryPositionEmbedding
- 最终结果: PASS (tilelang) | PASS (ascendc)

## 阶段零: Case 精简

- 结果: 通过
- 原始 case 数: 16
- 精简后 case 数: 10
- 备注: 保留了 fp32/fp16/bf16 三种 dtype 覆盖，mode 0 和 mode 1 两种模式覆盖，D=64 和 D=128 两种维度覆盖，以及极端大 shape（如 [2,32,2048,128]）和极端小 shape（如 [2,4,8,128]）

## 阶段一: TileLang

- 结果: 通过
- evaluate_tilelang.sh 执行次数: 1
- 关键错误信息: 无
- Agent 行为记录:
  - 第 1 轮: 阅读算子 .md 文档理解 RoPE 算法（mode 0 半旋转、mode 1 交错），参考 archive_tasks/8_ApplyRotaryPosEmb/ 获取类似 RoPE 算子的 TileLang/AscendC 实现模式。设计 block-level（persistent kernel, 20 cores, block_M=64, AIV_ONLY）和 tile-level（加载 x/cos/sin 的两半，fp16/bf16 cast 到 fp32 计算，旋转公式 res1=x1*cos1-x2*sin1, res2=x2*cos2+x1*sin2，cast 回原类型）。host 端通过 deinterleave/reinterleave 将 mode 1 转换为 mode 0 模式，避免重复 kernel。evaluate_tilelang.sh 一次通过，10/10 cases 全部匹配。
- 走偏点: 无

## 阶段二: AscendC

- 结果: 通过
- evaluate_ascendc.sh 执行次数: 1
- 关键错误信息: 无
- Agent 行为记录:
  - 第 1 轮: 阅读 TileLang-AscendC-API-Mapping.md，将 tile-level 设计转译为 AscendC kernel。实现了 RotaryPosEmbKernel 模板类，支持 float/half/bfloat16_t 三种数据类型。kernel 结构：Init() 设置 GM 张量和 TBuf<VECCALC> UB 缓冲区；Process() persistent kernel 循环处理 block 和行；ProcessRow() 加载两半数据、cast 到 fp32、计算旋转、cast 回、存储。使用 PipeBarrier<PIPE_ALL>() 保守同步。pybind11.cpp 提供 Python 绑定，model_new_ascendc.py host 端与 TileLang 版本结构一致。evaluate_ascendc.sh 一次通过，10/10 cases 全部匹配。
- 走偏点: 无

## 阶段三: 性能分析

- 结果: AscendC 实现约为参考实现的 0.47x 速度
- performance-analyzer 执行详情:
  - 测试的实现: reference, ascendc
  - 各实现平均耗时: reference: 0.797ms, ascendc: 1.702ms
  - 加速比: ascendc vs reference: 0.47x
- 分析: 参考实现 torch_npu.npu_rotary_mul 是华为高度优化的融合 NPU 算子，AscendC 自定义实现存在性能差距主要来自：(1) host 端张量操作（broadcast、flatten、deinterleave）开销；(2) 逐行处理模式未充分利用流水线并行

## 阶段四: 全量验证

- 结果: 通过
- 全量 case 数: 16
- 通过 case 数: 16/16
- 备注: 恢复 .json.bak 后一次全部通过

## 评测输出摘要

```
Status    : PASS
case[0] through case[15]: all matched
Result: pass
```

## 违规路径记录

- 外部 web 检索: 否
- 直接整体复制参考实现: 否
- PyTorch / torch_npu 语义回退: 否
- 说明: model_new_ascendc.py 中仅使用张量创建（empty_like）和变换（view/permute/reshape/contiguous/expand），核心计算完全由 AscendC kernel 完成
