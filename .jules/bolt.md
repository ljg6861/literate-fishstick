## 2025-02-18 - [Vectorization of Dataset Indexing]
**Learning:** Python loops in data loading/dataset iteration (like `_iter_test`) can be significant bottlenecks even if the operations inside are simple. Replacing sequential interval searching with `np.searchsorted` on the full index array provided a 13x speedup.
**Action:** Always check `_iter_*` or `__getitem__` methods in Datasets for loop-based logic that can be vectorized with numpy, especially for index mapping.
