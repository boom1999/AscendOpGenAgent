# Performance Summary

**Total Cases**: 300

**NPU Groups**: 8

---

## Overall Statistics

| Impl | Cases | Mean(ms) | Median(ms) | Min(ms) | Max(ms) | Std(ms) |
|------|-------|----------|------------|---------|---------|----------|
| reference | 300 | 10.628 | 0.683 | 0.223 | 163.939 | 28.081 |
| tilelang | 300 | 27.639 | 2.242 | 0.509 | 455.651 | 80.899 |
| ascendc | 300 | 32.552 | 2.299 | 0.279 | 540.512 | 96.445 |

**AscendC vs Reference Speedup**: 0.327x

**TileLang vs Reference Speedup**: 0.385x

---

**Files**:

- `summary.csv`: per-case detailed results with speedup
- `groups/`: 8 NPU group CSVs
