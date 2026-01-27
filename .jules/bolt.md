## 2024-05-23 - Tensor Hoisting in Nested TRM Loops
**Learning:** PyTorch eager mode does not automatically hoist loop-invariant tensor additions. In `TinyRecursiveReasoningModel_ACTV1_Inner`, `z_H + input_embeddings` was recomputed `H * L` times, even though it's constant for `L` iterations.
**Action:** Always inspect nested loops in custom RNN/Recursive architectures for invariant tensor operations and hoist them manually.
