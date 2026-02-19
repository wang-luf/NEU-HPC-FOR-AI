import modal
import subprocess

app = modal.App("week02-naive-run")

cuda_image = modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")

cuda_source_code = r"""
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Week 2: Naive approach (No Tiling, No Shared Memory)
__global__ void matrixMulNaive(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    int M = 4096, N = 4096, K = 4096;
    printf("--- Week 2: Naive GEMM ---\n");
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

    dim3 dimBlock(16, 16);
    dim3 dimGrid((N + 15) / 16, (M + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    printf("Launching Naive Kernel on H100...\n");
    cudaEventRecord(start);
    matrixMulNaive<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
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
"""

@app.function(image=cuda_image, gpu="H100")
def run_gpu():
    with open("gemm_naive.cu", "w") as f:
        f.write(cuda_source_code)
        
    subprocess.run(["nvcc", "-O3", "-o", "gemm_naive", "gemm_naive.cu"], check=True)
    result = subprocess.run(["./gemm_naive"], capture_output=True, text=True)
    print(result.stdout)

@app.local_entrypoint()
def main():
    run_gpu.remote()