## 2026-02-02 - Vectorized Searchsorted vs Iterative Loop
**Learning:** Replacing iterative index tracking loops with `np.searchsorted` for mapping flat indices to range-based groups (puzzles) yielded a ~25x speedup in batch collation on CPU. Python loops in data loaders are significant bottlenecks.
**Action:** Look for other data loading loops where item-to-group mapping is done iteratively and vectorize them using numpy or torch operations.
