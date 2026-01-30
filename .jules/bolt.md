## 2024-05-22 - [Recursive Model Inner Loop Hoisting]
**Learning:** The `TinyRecursiveReasoningModel_ACTV1` uses a nested loop (`H` cycles * `L` cycles). The `input_embeddings` are constant, and `z_H` is constant within the inner `L` loop. Hoisting `z_H + input_embeddings` out of the inner loop avoids `(L-1) * H` large tensor additions.
**Action:** When working with custom recurrent/recursive architectures, carefully check nested loops for loop-invariant calculations that involve large tensors, especially in PyTorch where element-wise operations have overhead.
