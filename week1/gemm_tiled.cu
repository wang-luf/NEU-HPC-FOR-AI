#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void matrixMulTiled(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    
    float sum = 0.0f;

    for (int m = 0; m < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        if (Row < M && m * TILE_WIDTH + tx < K)
            ds_A[ty][tx] = A[Row * K + (m * TILE_WIDTH + tx)];
        else
            ds_A[ty][tx] = 0.0f;

        if (Col < N && m * TILE_WIDTH + ty < K)
            ds_B[ty][tx] = B[(m * TILE_WIDTH + ty) * N + Col];
        else
            ds_B[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            sum += ds_A[ty][k] * ds_B[k][tx];

        __syncthreads();
    }

    if (Row < M && Col < N)
        C[Row * N + Col] = sum;
}

int main() {
    int M = 4096, N = 4096, K = 4096;
    printf("Matrix Size: %d x %d x %d\n", M, N, K);

    size_t size = M * N * sizeof(float);
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    for(int i=0; i<M*N; i++) { h_A[i] = 1.0f; h_B[i] = 0.01f; }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size); cudaMalloc(&d_B, size); cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    printf("Launching Tiled Kernel on H100...\n");
    cudaEventRecord(start);
    matrixMulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Execution Time: %.3f ms\n", milliseconds);
    
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double tflops = (flops / (milliseconds / 1000.0)) / 1e12;
    printf("Performance: %.3f TFLOPS\n", tflops);

    return 0;
}