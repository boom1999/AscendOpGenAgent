# Trace: MhcPostGrad

- 时间: 2026-04-14
- 算子: MhcPostGrad
- 最终结果: PASS (tilelang) | PASS (ascendc)

## 阶段零: Case 精简

- 结果: 通过
- 原始 case 数: 10
- 精简后 case 数: 9（移除 case 1，fp16 4D (1,2048,4,2560)，与 case 0 和 2 冗余）
- 备注: 无异常

## 阶段一: TileLang

- 结果: 通过
- evaluate_tilelang.sh 执行次数: 1
- 关键错误信息: 无（一次通过）
- Agent 行为记录:
  - 第 1 轮: 设计并实现 TileLang kernel，采用纯 Vector kernel（n 太小不适合 Cube），D 按 block_D 分片，n_pad 对齐到 8 的倍数保证 32B DMA。Phase 1 计算 grad_h_out, grad_h_post, grad_x（D-tiled），Phase 2 单独计算 grad_h_res（row-by-row full D pass）。一次通过评测。
- 走偏点: 无

## 阶段二: AscendC

- 结果: 通过
- evaluate_ascendc.sh 执行次数: 3
- 关键错误信息: 前两次失败的根本原因是跨流水线冲突（cross-pipeline hazards）。使用 PipeBarrier<PIPE_MTE2>/PIPE_V/PIPE_MTE3 时，TBuf<VECCALC> 缓冲区上 MTE2（GM读）、Vector（计算）、MTE3（GM写）可能并行执行，导致数据在 MTE2 传输完成前就被读取。
- Agent 行为记录:
  - 第 1 轮: 完成 AscendC kernel 转译和 model_new_ascendc.py 编写，首次验证出现精度不匹配
  - 第 2 轮: 调试发现 per-pipe PipeBarrier 不够，尝试部分修改同步逻辑，验证仍失败
  - 第 3 轮: 统一使用 PipeBarrier<PIPE_ALL>() 替换所有 per-pipe barrier，验证通过
- 走偏点: 初始实现使用了 per-pipe PipeBarrier 模式，但在 TBuf-only（无 TQue 流水管理）场景下不够安全。这是一个常见的 AscendC 新手陷阱。

## 阶段三: 性能分析

- 结果: 已执行
- 性能数据:
  - Reference: Mean 0.892ms, Median 0.412ms
  - TileLang: Mean 4.656ms, Median 0.995ms
  - AscendC: Mean 4.347ms, Median 0.696ms
- 备注: 小 case 上 AscendC 与 reference 接近，大 case 因 PIPE_ALL 全同步开销较大

## 阶段四: 全量用例验证

- 结果: 通过
- 恢复 model.py.bak 后运行 evaluate_ascendc.sh，全部 10 个 case 通过（包括之前精简掉的 case 1）

## 评测输出摘要

最后一次 evaluate_ascendc.sh 输出：

```
case[0]: output[0]: matched
case[1]: output[1]: matched
case[2]: output[2]: matched
case[3]: output[3]: matched
case[4]: output[4]: matched
case[5]: output[5]: matched
case[6]: output[6]: matched
case[7]: output[7]: matched
case[8]: output[8]: matched
case[9]: output[9]: matched
Result: pass
```
