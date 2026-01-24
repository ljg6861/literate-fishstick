## 2024-10-18 - [Vectorization of Dataset Iteration]
**Learning:** Python loops for batch processing in data loaders are a significant bottleneck, even for simple logic. Vectorized numpy operations (like `np.searchsorted`) are vastly faster (14x in this case) and cleaner.
**Action:** Always look for loops in data loading (`__iter__`, `__getitem__`, `collate_fn`) that can be replaced with numpy operations.
