# Performance Summary

**Operator**: /home/Code/AscendOpGenAgent/archive_tasks/MhcPostGrad

**Task Dir**: /home/Code/AscendOpGenAgent/archive_tasks/MhcPostGrad

**Device**: npu (parallel)

**Warmup**: 5, **Repeat**: 10, **Seed**: 0

**Timestamp**: 20260423_150154

---

## Overall Statistics

| Impl | Status | Mean(ms) | Median(ms) | Min(ms) | Max(ms) | Std(ms) | Timeout |
|------|--------|----------|------------|---------|---------|---------|----------|
| reference | OK | 0.606 | 0.445 | 0.344 | 17.134 | 0.554 | - |
| tilelang | OK | 7.060 | 4.711 | 0.610 | 64.910 | 6.737 | - |
| ascendc | OK | 7.204 | 4.594 | 0.360 | 40.441 | 7.324 | - |

**AscendC vs Reference Speedup**: 0.084x

**TileLang vs Reference Speedup**: 0.086x

---

**Files**:

- `summary.csv`: per-case detailed results
- `summary.md`: this file
- `groups/`: per-NPU group CSVs
