## 2024-05-22 - Vectorized Interval Mapping
**Learning:** Manual stateful loops for mapping indices to intervals (like puzzle IDs) are significantly slower than `np.searchsorted` in data loading pipelines.
**Action:** Look for `while` loops inside `for` loops used for interval lookups and replace with `np.searchsorted` to improve data loading throughput.
