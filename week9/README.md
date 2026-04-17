# Week 9: DeepSeekV3 MoE — ThunderKittens (TMA + Tensor Cores)

Extension of the Week 8 multi-GPU NCCL implementation to use **WMMA tensor cores**
and **Tensor Memory Acceleration (TMA)** via the ThunderKittens library on H100.

Progressive chain: **Week 7** (pure C) → **Week 8** (CUDA + NCCL) → **Week 9** (ThunderKittens)

---

## Assignment

1. Read ThunderKittens documentation (TK 2.0, H100/B200)
2. Reimplement DeepSeekMoE expert FFN using:
   - **TMA** (`tma::load_async`) — asynchronous SMEM tile loading
   - **WMMA tensor cores** (`warp::mma_AB`) — 64x64 bf16 matrix multiply
3. Benchmark TK expert FFN vs naive bf16 CUDA (no tensor cores)
4. Verify correctness against PyTorch reference (same test data as Week 8)

---

## ThunderKittens Architecture

```
ThunderKittens tile types:
  st_bf<R, C>               — shared memory bf16 tile (SMEM)
  rt_bf<R, C>               — register bf16 tile (per-warp)
  rt_fl<R, C>               — register float32 accumulator (for precision)

Global layout for TMA:
  gl<bf16, 1, 1, -1, -1, st_bf<64,64>>  — dynamic row/col dimensions

TMA load pipeline:
  1. Thread 0: tma::expect_bytes(sem, 2*sizeof(stile))
  2. Thread 0: tma::load_async(As, A_gl, {row_tile, col_tile}, sem)
  3. All threads: wait(sem, phase); phase ^= 1
  4. warp::load(reg_tile, smem_tile)

WMMA tensor core GEMM (C = A @ B):
  A → rt_bf<64,64>                     (row layout)
  B → rt_bf<64,64, col>                (col layout — required for mma_AB)
  B_col ← swap_layout(B_row)           (layout conversion)
  warp::mma_AB(acc, A_row, B_col, acc) (D = A*B + C, float32 accumulator)
```

---

## Expert MLP with ThunderKittens

```
tk_expert_mlp(x[T,H], Wg_T[H,I], Wu_T[H,I], Wd_T[I,H]):
  │
  ├── GEMM 1: gate[T,I] = x[T,H] @ Wg_T[H,I]   (TK tensor cores)
  ├── GEMM 2: up[T,I]   = x[T,H] @ Wu_T[H,I]   (TK tensor cores)
  ├── Elementwise: hid[T,I] = silu(gate) * up   (GPU kernel)
  └── GEMM 3: out[T,H] = hid[T,I] @ Wd_T[I,H]  (TK tensor cores)
```

**Weight pre-transposition** (done in Python before benchmark):

| PyTorch Weight | Shape | Stored as | TK GEMM role |
|----------------|-------|-----------|-------------|
| `gate_proj.weight` | `[I, H]` | `Wg_T = weight.T` → `[H, I]` | B matrix |
| `up_proj.weight`   | `[I, H]` | `Wu_T = weight.T` → `[H, I]` | B matrix |
| `down_proj.weight` | `[H, I]` | `Wd_T = weight.T` → `[I, H]` | B matrix |

---

## Configuration

```
Correctness test (naive CUDA float32):
  hidden_size          = 16
  moe_intermediate_size= 8
  n_routed_experts     = 8
  num_experts_per_tok  = 2 (top-k)
  n_shared_experts     = 1
  total_tokens         = 8

Benchmark (ThunderKittens bf16 tensor cores):
  hidden_size          = 512  (multiples of TILE=64)
  moe_intermediate_size= 256
  total_tokens         = 512
  Tile size            = 64x64 (TK requirement)
  GPU                  = H100 (sm_90, CUDA 12.8+, C++20)
```

---

## NCCL → ThunderKittens: Key Differences

| | Week 8 (NCCL) | Week 9 (ThunderKittens) |
|---|---|---|
| GPUs | 2x A100 | 1x H100 |
| Parallelism | Data + Expert | Single GPU, compute-optimized |
| Memory dtype | float32 | bf16 (tensor core native) |
| GEMM | Naive thread-per-element | TMA + WMMA tensor cores |
| SMEM loading | Standard `__shared__` | TMA async with swizzle |
| Communication | ncclAllToAll | None (single GPU) |
| Peak FLOP/s | ~312 TF (A100 bf16) | ~989 TF (H100 bf16) |

---

## Run

```bash
modal run week9/run_week9.py
```

4-step pipeline on Modal (H100 GPU):
1. Python generates `test_data.h` (PyTorch reference weights + expected outputs)
2. `nvcc -O3 -std=c++20 -arch=sm_90` compiles `moe_tk.cu` with ThunderKittens headers
3. CUDA binary: correctness test (naive kernels) + TK vs naive benchmark
4. Python benchmark: MoE vs dense FFN throughput comparison

---

## Expected Output

```
Week 9: DeepSeekV3 MoE — ThunderKittens (TMA + Tensor Cores)
=============================================================
Found 1 GPU(s). Using GPU 0.

=== Correctness Test (naive CUDA, H=16, I=8, T=8) ===
  token 0  max_err=2.1e-05  PASS
  token 1  max_err=1.8e-05  PASS
  ...
  token 7  max_err=2.9e-05  PASS

ALL TESTS PASSED

Benchmark: T=512 tokens, H=512, I=256, single expert, 100 iters
------------------------------------------------------------
  TK (tensor cores + TMA)   :    4200000 tok/s   2.31 TFLOPs   (0.122 ms/iter)
  Naive bf16 CUDA           :     380000 tok/s   0.21 TFLOPs   (1.348 ms/iter)
------------------------------------------------------------
  Speedup (TK vs Naive)     : 11.05x

PyTorch Benchmark: T=512 tokens, H=512, I=256, 8 experts, top-2
------------------------------------------------------------
  MoE (routed, bf16)    :       520000 tok/s  ( 0.98 ms/iter)
  Dense FFN (equiv)     :       310000 tok/s  ( 1.65 ms/iter)
------------------------------------------------------------
```

---

## Reading

- ThunderKittens: Simple, Fast, and Adorable AI Kernels (HazyResearch)
- DeepSeekMoE: Towards Ultimate Expert Specialization in MoE Language Models
- NVIDIA H100 Tensor Core GPU Architecture (Hopper SM, TMA unit)
- Blockwise Parallel Transformer (Ring Attention precursor)
- Ring Attention with Blockwise Transformers for Near-Infinite Context
- DeepSpeed Ulysses: All-to-all Sequence Parallelism
