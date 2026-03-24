# NEU HPC for AI

Weekly GPU programming assignments — from naive CUDA kernels to production-grade AI operator implementations.

## Progress

| Week | Topic | Key Result |
|------|-------|-----------|
| [Week 1](week1/) | CUDA Basics | Vector addition, first GPU kernel |
| [Week 2](week2/) | GEMM | 5 transpose/beta tests PASS — 5.15 TFLOPS |
| [Week 3](week3/) | Tiled GEMM | Shared memory tiling — 6.80 TFLOPS (4.25× speedup) |
| [Week 4](week4/) | FlashAttention-2 | Algorithm 1 in C (sequential) + CUDA (parallel) |
| [Week 5](week5/) | CuTe / Layout Algebra | FlashAttention reimplemented with NVIDIA CuTe library |
| [Week 7](week7/) | MoE / Communication | DeepSeekV3 MoE operator in pure C |

## Stack

- **GPU**: NVIDIA H100 (via [Modal](https://modal.com))
- **Languages**: CUDA C, C, Python
- **Libraries**: CUTLASS/CuTe (Week 5), PyTorch (test generation)

## How to Run

Each week has a `run_weekN.py` Modal script:

```bash
# Example: run Week 7
modal run week7/run_week7.py
```

Requires a Modal account and `modal` CLI installed (`pip install modal`).
