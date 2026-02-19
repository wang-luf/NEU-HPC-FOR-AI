```markdown
# Week 2: Naive GPU Matrix Multiplication (GEMM)

## Overview
This project moves the matrix multiplication to the GPU using CUDA. It is a **Naive** implementation where each thread calculates exactly one element of the output matrix by reading directly from Global Memory.

## Performance Limitation
Because threads constantly access the slow Global Memory, this approach is highly **memory-bound** and limited by the global memory bandwidth.

## How to Run
This code is executed on an **NVIDIA H100 GPU** using Modal.
```bash
modal run run_week2.py