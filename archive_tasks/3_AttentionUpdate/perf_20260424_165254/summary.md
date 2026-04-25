# Performance Summary

**Operator**: /home/Code/AscendOpGenAgent/output/20260424_125855/3_AttentionUpdate

**Task Dir**: /home/Code/AscendOpGenAgent/output/20260424_125855/3_AttentionUpdate

**Device**: npu (parallel)

**Warmup**: 5, **Repeat**: 10, **Seed**: 0

**Timestamp**: 20260424_165254

---

## Overall Statistics

| Impl | Status | Mean(ms) | Median(ms) | Min(ms) | Max(ms) | Std(ms) | Timeout |
|------|--------|----------|------------|---------|---------|---------|----------|
| reference | OK | 5.246 | 1.229 | 0.273 | 50.853 | 7.363 | - |
| tilelang | ERROR | - | - | - | - | - | - |
| ascendc | OK | 2.051 | 1.896 | 0.222 | 19.666 | 1.279 | - |

**AscendC vs Reference Speedup**: 2.558x

---

**Files**:

- `summary.csv`: per-case detailed results
- `summary.md`: this file
- `groups/`: per-NPU group CSVs
