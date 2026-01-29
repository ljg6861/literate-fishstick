## 2024-05-22 - Hoisting Invariant Calculations in Recursive Loops
**Learning:** In the `TinyRecursiveReasoningModel_ACTV1` architecture, the nested loop structure (`H` outer cycles, `L` inner cycles) can hide redundant calculations. specifically, inputs to the inner `L` loop that depend only on `z_H` (which is constant during the inner loop) were being recomputed in every inner iteration.
**Action:** When working with recursive models, carefully check nested loops for operations that depend only on the outer loop's state and hoist them.
