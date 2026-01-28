## 2024-05-23 - [Vectorized Puzzle Index Lookup]
**Learning:** Python loops for per-example logic in data loaders are bottlenecks. Replacing `for i in range` with `np.searchsorted` on the whole range yields massive speedups (observed ~7.7x in synthetic microbenchmark).
**Action:** Always look for loops over batch dimensions in data loading and replace with vectorized NumPy operations where possible.
