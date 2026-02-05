## 2025-10-26 - Einops Overhead in Hot Loops
**Learning:** Using `einops.rearrange` in the innermost loop of the Attention mechanism added ~17% overhead compared to native `torch.transpose` on CPU. While `einops` is readable, its string parsing and dispatch overhead becomes significant when called thousands of times per second.
**Action:** Prefer native PyTorch operations (`transpose`, `permute`, `view`) in critical hot paths (like Attention layers), especially for recursive models where these layers are called repeatedly.
