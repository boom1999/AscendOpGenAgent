# Performance Summary

**Operator**: /home/Code/AscendOpGenAgent/output/20260424_125855/3_AttentionUpdate

**Task Dir**: /home/Code/AscendOpGenAgent/output/20260424_125855/3_AttentionUpdate

**Device**: npu (parallel)

**Warmup**: 5, **Repeat**: 10, **Seed**: 0

**Timestamp**: 20260424_164026

---

## Overall Statistics

| Impl | Status | Mean(ms) | Median(ms) | Min(ms) | Max(ms) | Std(ms) | Timeout |
|------|--------|----------|------------|---------|---------|---------|----------|
| reference | OK | 5.431 | 1.203 | 0.271 | 55.686 | 7.673 | - |
| tilelang | ERROR | - | - | - | - | - | - |
| ascendc | OK | 2.129 | 1.973 | 0.221 | 46.298 | 1.431 | - |

**AscendC vs Reference Speedup**: 2.551x

---

**Files**:

- `summary.csv`: per-case detailed results
- `summary.md`: this file
- `groups/`: per-NPU group CSVs
