```markdown
# Week 3: Tiled GPU Matrix Multiplication (Shared Memory)

## Overview
This project heavily optimizes the GPU matrix multiplication using **Tiling** and **Shared Memory**.

## Key Optimizations
- Threads in a block collaboratively load a "tile" of data from Global Memory into the ultra-fast `__shared__` memory.
- `__syncthreads()` is used as a barrier to ensure the tile is fully loaded before computation begins.
- This drastically reduces Global Memory traffic, overcoming the memory bandwidth bottleneck seen in Week 2.
- **Result:** Reaches high computational performance (~3.0+ TFLOPS on an H100 GPU).

## How to Run
This code is executed on an **NVIDIA H100 GPU** using Modal.
```bash
modal run run_week3.py