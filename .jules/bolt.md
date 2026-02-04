## 2024-05-22 - Vectorized Dataset Indexing
**Learning:** Python loops over batch items are surprisingly expensive even for simple operations. `np.searchsorted` is extremely fast for finding ranges. Replacing a per-item loop with full vectorization yielded ~16x speedup in `_iter_test`.
**Action:** When mapping indices to segments (like puzzles/groups), always use `np.searchsorted` on the full index array instead of looping.
