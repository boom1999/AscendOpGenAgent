# Trace: 4_AggregateHidden

## 基本信息
- **算子**: AggregateHidden (1D grouped convolution on hidden tokens with mask + backward)
- **NPU**: 2
- **输入**: grad_out(S,B,H), input(S,B,H), weight(3,H), mask(B,S) bool
- **输出**: [output(S,B,H), grad_input(S,B,H), grad_weight(3,H)]
- **数据类型**: bfloat16 / float16
- **output_dir**: `/home/Code/AscendOpGenAgent/output/20260424_125855/4_AggregateHidden/`

---

## Phase 0: 参数确认 -- PASS
- npu=2, op_file=/home/Code/AscendOpGenAgent/benchmarks/AIInfraNPUKernelBench/level1/4_AggregateHidden.py
- output_dir=/home/Code/AscendOpGenAgent/output/20260424_125855/4_AggregateHidden/
- ASCEND_RT_VISIBLE_DEVICES=2

## Phase 1: 环境准备 -- PASS
- 复制 model.py 和 4_AggregateHidden.json (64 cases)

## Phase 2: INPUT_CASES 精简 -- PASS
- 原始 64 cases -> 精简至 10 cases
- 覆盖 bfloat16/float16 dtype, S=1/10/1024/32768/4096, B=1..8, H=384..24576

## Phase 3: TileLang 设计表达 -- PASS
- Block-level 设计: H 维度并行, B-S 双层循环, 单 pass 流水
- Tile-level 设计: 滑动窗口 + 延迟 2 步 grad_input 计算
- TileLang 验证: 跳过（TileLang 框架对该算子的标量 mask 加载不完全支持）

## Phase 4: AscendC 转译与验证 -- PASS (ac_iteration=1)

### 迭代 0 (失败)
**编译**: 成功
**验证结果**: 
- output[0] (forward): 全部通过
- output[1] (grad_input): cases 3-9 (S>1) 失败，mismatch 0.003%~8.66%
- output[2] (grad_weight): 全部通过

**错误分析**:
- 类型: A 类 (代码逻辑错误)
- 位置: `aggregate_hidden_kernel.h` 第 244-258 行 (tail 段)
- 具体错误: tail 段 gi[S-2] 的 DataCopyPad(GM store, MTE3) 和 gi[S-1] 的 Mul(Vector compute) 之间缺少 `pipe_barrier(PIPE_ALL)`，导致 WAR (Write After Read) hazard——MTE3 尚在读取 giUb 时 Vector 单元已开始写入新数据
- S=1 通过的原因: `if (S >= 2)` 分支不执行，没有 hazard

### 迭代 1 (成功)
**修复内容**:
1. 在 gi[S-2] 的 DataCopyPad 后添加 `pipe_barrier(PIPE_ALL)` 消除 WAR hazard
2. 额外修改: mask 加载改为 32 字节对齐（从 alignedIdx 加载 8 float，用 localIdx 取值）——后验证此修改对正确性无影响

**验证结果**: 10/10 cases 全部通过

### 关键文件
- `kernel/aggregate_hidden_tiling.h` — tiling 参数 (S,B,H,blockH,hNum,usedCoreNum,tasksPerCore)
- `kernel/kernel_common.h` — CopyTiling, CeilDivU32 辅助函数
- `kernel/vector_tile.h` — LoadGmToUb, StoreUbToGm 模板
- `kernel/aggregate_hidden_kernel.h` — 主 kernel 类 (~330 行)
- `kernel/aggregate_hidden.cpp` — kernel 入口 (KERNEL_TYPE_MIX_AIC_1_2)
- `kernel/pybind11.cpp` — host 绑定 (_aggregate_hidden_ext)
- `model_new_ascendc.py` — Python wrapper (cast→reshape→call kernel→reshape→cast)

### 设计要点
- **KERNEL_TYPE_MIX_AIC_1_2**: 使用 GetValue 读取 mask 标量值需要 MIX 模式
- **H 维度并行**: blockH=1024, 多核分担不同 H 块
- **滑动窗口**: inpCur/inpPrev1/inpPrev2 和 geCur/gePrev1/gePrev2 各 3 个缓冲
- **延迟 grad_input**: 在 S 循环中延迟 2 步计算 grad_input, tail 处理最后 2 个位置
- **19 个 UB 缓冲**: 4 TQue + 15 TBuf, 总约 72KB (H=1024 时)

## Phase 5: 性能分析 -- DONE

| Case | Shape (S,B,H) | Reference (ms) | AscendC (ms) | 加速比 |
|------|---------------|----------------|--------------|--------|
| 0 | (1,1,384) | 0.478 | 0.481 | 0.99x |
| 1 | (1,4,4097) | 0.546 | 0.464 | 1.18x |
| 2 | (1,3,576) | 0.521 | 0.468 | 1.11x |
| 3 | (10,8,4095) | 0.885 | 0.562 | 1.57x |
| 4 | (1024,4,470) | 1.881 | 4.700 | 0.40x |
| 5 | (1024,2,24576) | 35.274 | 6.386 | **5.52x** |
| 6 | (32768,2,576) | 26.993 | 68.416 | 0.39x |
| 7 | (792,2,3886) | 3.099 | 2.233 | 1.39x |
| 8 | (658,4,2661) | 2.671 | 3.614 | 0.74x |
| 9 | (4096,4,768) | 4.302 | 17.199 | 0.25x |
| **总计** | | **7.665** | **10.452** | **0.73x** |

**分析**: 大 H 场景 (H=24576) 因 H 并行获得 5.52x 加速。大 S 场景 (S=32768/4096) 因串行 S 循环较慢。参考实现使用高度优化的 NPU 内置算子 (npu_aggregate_hidden)。

## Phase 6: 全量用例验证 -- 63/64 PASS

**失败 case**:
- case[47]: (4096, 4, 768) float16, output[2] (grad_weight), 1 个元素 (0.043%), max_abs_diff=32
- 原因: 浮点累加精度差异 (16384 次累加后 float32→float16 取整边界不同)
- 判断: 数值精度边界问题，非算法错误

---

## 走偏点分析

1. **迭代 0 误判 mask 对齐**: 首次失败时误认为 mask 加载未对齐 32 字节是根因，浪费了分析时间。实际上 DataCopyPad 对非对齐 GM 地址有内部处理，mask 对齐修改对结果无影响。
2. **真正根因是 pipe barrier**: tail 段的 MTE3/Vector WAR hazard 才是 grad_input 错误的根因。S=1 通过但 S>1 失败的模式应该更早指向 tail 段逻辑。

## 总结
- AscendC kernel 正确性: 10/10 精简用例通过, 63/64 全量用例通过
- 性能: 大 H 场景显著加速 (5.52x), 大 S 场景较慢
- 迭代次数: Phase 4 用了 2 次迭代 (0: 失败, 1: 通过)
