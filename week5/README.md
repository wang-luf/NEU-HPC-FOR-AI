# Week 5: FlashAttention with CuTe

Reimplementation of FlashAttention Algorithm 1 using NVIDIA's **CuTe** library
(part of CUTLASS), replacing raw pointer arithmetic with typed tensor abstractions.

## What Changed from Week 4

| Week 4 (raw CUDA) | Week 5 (CuTe) |
|---|---|
| `Q[(row+r)*d + k]` | `gQi(r, k)` via `local_tile` |
| `Qi_s[r*D_MAX + k]` | `sQ(r, k)` via `make_tensor` |
| manual `for k` load loop | `copy(gQi(r,_), sQ(r,_))` |
| manual `for k` write loop | `copy(sO(r,_), gOi(r,_))` |

## CuTe Concepts Used

| API | Purpose |
|-----|---------|
| `make_tensor(ptr, layout)` | Wrap a pointer into a typed Tensor view |
| `make_layout(shape, stride)` | Define `(Shape, Stride)` coordinate map |
| `make_gmem_ptr` / `make_smem_ptr` | Tag pointer for memory-space dispatch |
| `local_tile(tensor, tile, coord)` | Extract a tile without manual offset arithmetic |
| `cute::copy(src, dst)` | Generic element-wise copy (dispatches to vectorised LDG/STG) |
| `tensor(i, _)` | Row-slice returning a 1-D view |

## Algorithm

FlashAttention-2 Algorithm 1 (forward pass, single head):
- **Line 4**: Load Qi tile via `copy`
- **Line 5**: Init Oi=0, mi=−∞, li=0
- **Lines 7–10**: For each Kj/Vj tile: Sij = QK^T/√d → online softmax update → accumulate Oi
- **Line 12**: Oi /= li (final normalization)
- **Lines 14–15**: Write Oi back via `copy`

## Run

```bash
modal run week5/run_week5.py
```

Image build clones CUTLASS headers, compiles with:
```
nvcc -O3 --std=c++17 -I/opt/cutlass/include -o flash_attn_cute flash_attn_cute.cu -lm
```

## Expected Output

```
Result : PASS   (max error < 1e-4 vs CPU naive attention)
```
