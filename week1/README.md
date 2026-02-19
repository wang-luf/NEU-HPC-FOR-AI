# Week 1: CPU Matrix Multiplication with Pthreads

## Overview
This project implements matrix multiplication in C using **Pthreads** to demonstrate CPU parallelism. The workload is distributed among multiple threads by dividing the rows of the result matrix.

## How to Run
1. Compile the code:
   ```bash
   gcc -o matrix_mult matrix_mult.c -lpthread