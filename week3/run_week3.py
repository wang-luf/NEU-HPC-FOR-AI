import modal
import subprocess

app = modal.App("week03-tiled-gemm-fast")

cuda_image = modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")

cuda_source_code = r"""
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

// Week 3: Tiled GEMM with Shared Memory
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
        // Load A tile with transpose handling
        if (row < M && (m * TILE_WIDTH + tx) < K) {
            if (!transA) {
                ds_A[ty][tx] = A[row * K + (m * TILE_WIDTH + tx)];
            } else {
                ds_A[ty][tx] = A[(m * TILE_WIDTH + tx) * M + row];
            }
        } else {
            ds_A[ty][tx] = 0.0f;
        }

        // Load B tile with transpose handling
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

        // Compute from shared memory
        for (int k = 0; k < TILE_WIDTH; ++k) {
            tmp += ds_A[ty][k] * ds_B[k][tx];
        }
        
        __syncthreads();
    }

    // GEMM formula with in-place update
    if (row < M && col < N) {
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
    }
}

// Quick CPU verification (only check small portion to avoid timeout)
void verify_sample(const float *h_A, const float *h_B, const float *h_C_gpu,
                   int M, int N, int K, float alpha, float beta,
                   bool transA, bool transB) {
    printf("  Verifying sample of results (first 100 elements)...\n");
    
    int errors = 0;
    int checked = 0;
    
    // Only verify first 100 elements to save time
    for (int idx = 0; idx < 100 && idx < M*N; idx++) {
        int row = idx / N;
        int col = idx % N;
        
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            float a = transA ? h_A[i * M + row] : h_A[row * K + i];
            float b = transB ? h_B[col * K + i] : h_B[i * N + col];
            sum += a * b;
        }
        float expected = alpha * sum + beta * 0.0f;  // Assume C_old was 0
        
        if (fabs(h_C_gpu[idx] - expected) > 1e-2) {
            errors++;
            if (errors <= 3) {
                printf("    Mismatch at [%d]: GPU=%.4f, Expected=%.4f\n", 
                       idx, h_C_gpu[idx], expected);
            }
        }
        checked++;
    }
    
    printf("  Checked: %d elements, Errors: %d\n", checked, errors);
    printf("  Sample Verification: %s\n", errors == 0 ? "✓ PASS" : "⚠ FAIL");
}

int main() {
    printf("=== Week 3: Tiled GEMM with Shared Memory ===\n");
    printf("Optimization: Reduce global memory access with tiling\n\n");
    
    // Use 4096 for large benchmark, but skip full CPU verification
    int M = 4096, N = 4096, K = 4096;
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    
    srand(42);
    for (int i = 0; i < M*K; i++) h_A[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K*N; i++) h_B[i] = (float)(rand() % 100) / 100.0f;
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    
    printf("Configuration:\n");
    printf("  Grid: (%d, %d)\n", dimGrid.x, dimGrid.y);
    printf("  Block: (%d, %d)\n", dimBlock.x, dimBlock.y);
    printf("  Tile Size: %dx%d\n", TILE_WIDTH, TILE_WIDTH);
    printf("  Total Threads: %d\n\n", dimGrid.x * dimGrid.y * TILE_WIDTH * TILE_WIDTH);
    
    // ========== Test 1: C = A*B ==========
    printf("--- Test 1: C = A*B (no transpose) ---\n");
    float alpha = 1.0f, beta = 0.0f;
    bool transA = false, transB = false;
    
    for(int i=0; i<M*N; i++) h_C[i] = 0.0f;
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);
    
    sgemm_tiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K, alpha, beta, transA, transB);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    verify_sample(h_A, h_B, h_C, M, N, K, alpha, beta, transA, transB);
    printf("\n");
    
    // ========== Test 2: C = A*B + C (beta=1) ==========
    printf("--- Test 2: C = A*B + C (beta=1.0, accumulate) ---\n");
    alpha = 1.0f; beta = 1.0f;
    
    for(int i=0; i<M*N; i++) h_C[i] = 0.5f;
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);
    
    sgemm_tiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K, alpha, beta, transA, transB);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    
    // Manual check for beta scaling
    printf("  Checking beta scaling (C should have A*B + 0.5)...\n");
    float expected_min = 0.5f;  // At minimum, should have initial C value
    int count_above_min = 0;
    for (int i = 0; i < 100; i++) {
        if (h_C[i] > expected_min) count_above_min++;
    }
    printf("  Elements > 0.5: %d/100 (should be >95)\n", count_above_min);
    printf("  Beta scaling: %s\n\n", count_above_min > 95 ? "✓ Working" : "⚠ Check");
    
    // ========== Performance Benchmark ==========
    printf("--- Performance Benchmark (C = A*B) ---\n");
    alpha = 1.0f; beta = 0.0f;
    transA = false; transB = false;
    
    for(int i=0; i<M*N; i++) h_C[i] = 0.0f;
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    // Warmup
    printf("  Warming up GPU...\n");
    for (int i = 0; i < 3; i++) {
        sgemm_tiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K, alpha, beta, transA, transB);
    }
    cudaDeviceSynchronize();
    
    printf("  Running benchmark (10 iterations)...\n");
    cudaEventRecord(start);
    const int ITERS = 10;
    for (int i = 0; i < ITERS; i++) {
        sgemm_tiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K, alpha, beta, transA, transB);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float ms_per_iter = ms / ITERS;
    
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double tflops = (flops / (ms_per_iter / 1000.0)) / 1e12;
    
    printf("\n  Results:\n");
    printf("  Matrix: %dx%dx%d\n", M, N, K);
    printf("  Avg Time: %.3f ms\n", ms_per_iter);
    printf("  Performance: %.3f TFLOPS\n", tflops);
    
    if (tflops > 3.0) {
        printf("\n🎉 SUCCESS! Achieved 3+ TFLOPS with shared memory!\n");
        printf("   This demonstrates effective memory optimization.\n");
    }
    
    printf("\n=== Week 3 Complete ===\n");
    
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    
    return 0;
}
"""

@app.function(image=cuda_image, gpu="H100", timeout=300)  # 减少timeout到5分钟
def run_gpu():
    import subprocess
    
    print("📝 Writing Week 3 Tiled GEMM (optimized for speed)...")
    with open("gemm_tiled.cu", "w") as f:
        f.write(cuda_source_code)
    
    print("🔨 Compiling with nvcc -O3...")
    result = subprocess.run(
        ["nvcc", "-O3", "-o", "gemm_tiled", "gemm_tiled.cu"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(" Compilation failed!")
        print(result.stderr)
        return
    
    print(" Compilation successful!\n")
    print(" Running on H100...\n")
    
    result = subprocess.run(["./gemm_tiled"], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)

@app.local_entrypoint()
def main():
    print("="*60)
    print("Week 3: Tiled GEMM with Shared Memory")
    print("="*60)
    run_gpu.remote()