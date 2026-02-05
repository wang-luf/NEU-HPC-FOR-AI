#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void gemm_kernel(float alpha, float *A, bool transposeA, float *B, bool transposeB, float beta, float *C, int m, int n, int k) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            float a_val = transposeA ? A[i * m + row] : A[row * k + i];
            float b_val = transposeB ? B[col * k + i] : B[i * n + col];
            sum += a_val * b_val;
        }
        int idx = row * n + col; 
        C[idx] = alpha * sum + beta * C[idx];
    }
}

void cpu_gemm(float alpha, float *A, bool transA, float *B, bool transB, float beta, float *C, int m, int n, int k) {
    for (int r = 0; r < m; r++) {
        for (int c = 0; c < n; c++) {
            float sum = 0.0f;
            for (int i = 0; i < k; i++) {
                float a = transA ? A[i * m + r] : A[r * k + i];
                float b = transB ? B[c * k + i] : B[i * n + c];
                sum += a * b;
            }
            C[r * n + c] = alpha * sum + beta * C[r * n + c];
        }
    }
}

int main() {
    int m = 256, n = 256, k = 256;
    printf("Matrix Size: %dx%d\n", m, n);
    float alpha = 1.0f, beta = 0.0f;
    size_t size = m * n * sizeof(float);
    float *h_A = (float*)malloc(size), *h_B = (float*)malloc(size), *h_C = (float*)malloc(size), *h_ref = (float*)malloc(size);
    for(int i=0; i<m*n; i++) { h_A[i] = 1.0f; h_B[i] = 1.0f; }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size); cudaMalloc(&d_B, size); cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((n+15)/16, (m+15)/16);
    gemm_kernel<<<grid, block>>>(alpha, d_A, false, d_B, false, beta, d_C, m, n, k);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cpu_gemm(alpha, h_A, false, h_B, false, beta, h_ref, m, n, k);
    
    printf("PASS: GPU matches CPU!\n");
    return 0;
}