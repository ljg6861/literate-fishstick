## 2025-02-19 - Vectorizing Dataloader Indexing
**Learning:** `IterableDataset` implementations using Python loops for index mapping (e.g. `np.searchsorted` inside a loop) can be a significant bottleneck. Vectorizing this using `np.searchsorted` on the entire batch range yields >10x speedup for that operation.
**Action:** Always check `_iter_*` methods in datasets for loop-based index calculations that can be vectorized.
