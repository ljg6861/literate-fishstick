## 2025-05-23 - [Partial Vectorization Anti-Pattern]
**Learning:** Found a "mixed" approach where `np.searchsorted` was used only for initialization, followed by a slow Python loop. This defeated the purpose of using numpy.
**Action:** Always check if the *entire* loop can be vectorized when seeing numpy operations mixed with Python loops.
