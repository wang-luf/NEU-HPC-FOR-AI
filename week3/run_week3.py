import modal
import subprocess

app = modal.App("week03-tiled-gemm")

cuda_image = modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")

cuda_source_code = r"""
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void sgemm_tiled(const float *A, const float *B, float *C,
                            int M, int N, int K,
                            float alpha, float beta,
                            bool transA, bool transB) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float tmp = 0.0f;
    
    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int m = 0; m < numTiles; ++m) {
        if (row < M && (m * TILE_WIDTH + tx) < K) {
            if (!transA) {
                ds_A[ty][tx] = A[row * K + (m * TILE_WIDTH + tx)];
            } else {
                ds_A[ty][tx] = A[(m * TILE_WIDTH + tx) * M + row];
            }
        } else {
            ds_A[ty][tx] = 0.0f;
        }

        if ((m * TILE_WIDTH + ty) < K && col < N) {
            if (!transB) {
                ds_B[ty][tx] = B[(m * TILE_WIDTH + ty) * N + col];
            } else {
                ds_B[ty][tx] = B[col * K + (m * TILE_WIDTH + ty)];
            }
        } else {
            ds_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            tmp += ds_A[ty][k] * ds_B[k][tx];
        }
        
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
    }
}

void cpu_gemm(const float *A, const float *B, float *C,
              int M, int N, int K,
              float alpha, float beta,
              bool transA, bool transB) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int i = 0; i < K; i++) {
                float a = transA ? A[i * M + row] : A[row * K + i];
                float b = transB ? B[col * K + i] : B[i * N + col];
                sum += a * b;
            }
            C[row * N + col] = alpha * sum + beta * C[row * N + col];
        }
    }
}

int main() {
    printf("Week 3: Tiled GEMM with Shared Memory\n");
    printf("========================================\n\n");
    
    printf("PART 1: Correctness Verification (1024x1024x1024)\n");
    printf("--------------------------------------------------\n");
    
    int M_test = 1024, N_test = 1024, K_test = 1024;
    
    size_t size_A_test = M_test * K_test * sizeof(float);
    size_t size_B_test = K_test * N_test * sizeof(float);
    size_t size_C_test = M_test * N_test * sizeof(float);
    
    float *h_A = (float*)malloc(size_A_test);
    float *h_B = (float*)malloc(size_B_test);
    float *h_C = (float*)malloc(size_C_test);
    float *h_C_ref = (float*)malloc(size_C_test);
    
    srand(42);
    for (int i = 0; i < M_test*K_test; i++) h_A[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K_test*N_test; i++) h_B[i] = (float)(rand() % 100) / 100.0f;
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A_test);
    cudaMalloc(&d_B, size_B_test);
    cudaMalloc(&d_C, size_C_test);
    
    cudaMemcpy(d_A, h_A, size_A_test, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B_test, cudaMemcpyHostToDevice);
    
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N_test + TILE_WIDTH - 1) / TILE_WIDTH, 
                 (M_test + TILE_WIDTH - 1) / TILE_WIDTH);
    
    printf("Test 1: C = A*B\n");
    float alpha = 1.0f, beta = 0.0f;
    bool transA = false, transB = false;
    
    for(int i=0; i<M_test*N_test; i++) { h_C[i] = 0.0f; h_C_ref[i] = 0.0f; }
    cudaMemcpy(d_C, h_C, size_C_test, cudaMemcpyHostToDevice);
    
    sgemm_tiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M_test, N_test, K_test, alpha, beta, transA, transB);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, size_C_test, cudaMemcpyDeviceToHost);
    
    printf("  Running CPU reference...\n");
    cpu_gemm(h_A, h_B, h_C_ref, M_test, N_test, K_test, alpha, beta, transA, transB);
    
    int errors = 0;
    float max_error = 0.0f;
    for (int i = 0; i < M_test*N_test; i++) {
        float diff = fabs(h_C[i] - h_C_ref[i]);
        if (diff > max_error) max_error = diff;
        if (diff > 1e-2) {
            errors++;
            if (errors <= 3) {
                printf("  Error at %d: GPU=%.4f, CPU=%.4f\n", i, h_C[i], h_C_ref[i]);
            }
        }
    }
    printf("  Result: %s (%d/%d errors, max_error=%.6f)\n\n", 
           errors == 0 ? "PASS" : "FAIL", errors, M_test*N_test, max_error);
    
    printf("Test 2: C = A*B + C (beta=1.0)\n");
    alpha = 1.0f; beta = 1.0f;
    
    for(int i=0; i<M_test*N_test; i++) { h_C[i] = 0.5f; h_C_ref[i] = 0.5f; }
    cudaMemcpy(d_C, h_C, size_C_test, cudaMemcpyHostToDevice);
    
    sgemm_tiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M_test, N_test, K_test, alpha, beta, transA, transB);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, size_C_test, cudaMemcpyDeviceToHost);
    cpu_gemm(h_A, h_B, h_C_ref, M_test, N_test, K_test, alpha, beta, transA, transB);
    
    errors = 0;
    max_error = 0.0f;
    for (int i = 0; i < M_test*N_test; i++) {
        float diff = fabs(h_C[i] - h_C_ref[i]);
        if (diff > max_error) max_error = diff;
        if (diff > 1e-2) errors++;
    }
    printf("  Result: %s (%d errors, max_error=%.6f)\n", 
           errors == 0 ? "PASS" : "FAIL", errors, max_error);
    
    printf("\nCorrectness verification complete.\n\n");
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    
    printf("PART 2: Performance Benchmark (4096x4096x4096)\n");
    printf("--------------------------------------------------\n");
    
    int M = 4096, N = 4096, K = 4096;
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C = (float*)malloc(size_C);
    
    for (int i = 0; i < M*K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < K*N; i++) h_B[i] = 0.01f;
    for (int i = 0; i < M*N; i++) h_C[i] = 0.0f;
    
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);
    
    dim3 dimBlock_large(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid_large((N + TILE_WIDTH - 1) / TILE_WIDTH, 
                       (M + TILE_WIDTH - 1) / TILE_WIDTH);
    
    alpha = 1.0f; beta = 0.0f;
    transA = false; transB = false;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    printf("Warming up...\n");
    for (int i = 0; i < 3; i++) {
        sgemm_tiled<<<dimGrid_large, dimBlock_large>>>(d_A, d_B, d_C, M, N, K, alpha, beta, transA, transB);
    }
    cudaDeviceSynchronize();
    
    printf("Running benchmark...\n");
    cudaEventRecord(start);
    const int ITERS = 10;
    for (int i = 0; i < ITERS; i++) {
        sgemm_tiled<<<dimGrid_large, dimBlock_large>>>(d_A, d_B, d_C, M, N, K, alpha, beta, transA, transB);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float ms_per_iter = ms / ITERS;
    
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double tflops = (flops / (ms_per_iter / 1000.0)) / 1e12;
    
    printf("\nBenchmark Results:\n");
    printf("  Matrix Size: %dx%dx%d\n", M, N, K);
    printf("  Average Time: %.3f ms\n", ms_per_iter);
    printf("  Performance: %.3f TFLOPS\n", tflops);
    
    if (tflops > 3.0) {
        printf("\nAchieved target performance (>3.0 TFLOPS).\n");
        printf("Shared memory optimization effective.\n");
    }
    
    printf("\nComparison:\n");
    printf("  Week 2 Naive: ~1.6 TFLOPS\n");
    printf("  Week 3 Tiled: %.2f TFLOPS\n", tflops);
    printf("  Speedup: %.2fx\n", tflops / 1.6);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    
    printf("\nWeek 3 complete.\n");
    
    return 0;
}
"""

@app.function(image=cuda_image, gpu="H100", timeout=600)
def run_gpu():
    import subprocess
    
    print("Writing CUDA source...")
    with open("gemm_tiled.cu", "w") as f:
        f.write(cuda_source_code)
    
    print("Compiling...")
    result = subprocess.run(
        ["nvcc", "-O3", "-o", "gemm_tiled", "gemm_tiled.cu"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("Compilation failed:")
        print(result.stderr)
        return
    
    print("Compilation successful.\n")
    print("Running on H100...\n")
    
    result = subprocess.run(["./gemm_tiled"], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)

@app.local_entrypoint()
def main():
    run_gpu.remote()