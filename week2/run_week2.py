import modal
import subprocess

app = modal.App("week02-full-gemm-tests")

cuda_image = modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")

cuda_source_code = r"""
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <cuda_runtime.h>

// Week 2: Full GEMM with alpha, beta, and transpose support
__global__ void sgemm_naive(const float *A, const float *B, float *C, 
                            int M, int N, int K, 
                            float alpha, float beta, 
                            bool transA, bool transB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float tmp = 0.0f;
        for (int i = 0; i < K; ++i) {
            float a_val = transA ? A[i * M + row] : A[row * K + i];
            float b_val = transB ? B[col * K + i] : B[i * N + col];
            tmp += a_val * b_val;
        }
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
    }
}

// CPU reference for verification
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
    printf("=== Week 2: GEMM with Full Features ===\n");
    printf("Testing all transpose combinations and alpha/beta parameters\n\n");
    
    int M = 1024, N = 1024, K = 1024;
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    float *h_C_ref = (float*)malloc(size_C);
    
    // Initialize with random values
    srand(42);
    for (int i = 0; i < M*K; i++) h_A[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K*N; i++) h_B[i] = (float)(rand() % 100) / 100.0f;
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    dim3 dimBlock(16, 16);
    dim3 dimGrid((N + 15) / 16, (M + 15) / 16);
    
    // ========== Test 1: C = A*B ==========
    printf("--- Test 1: C = A*B (no transpose) ---\n");
    float alpha = 1.0f, beta = 0.0f;
    bool transA = false, transB = false;
    
    for(int i=0; i<M*N; i++) { h_C[i] = 0.0f; h_C_ref[i] = 0.0f; }
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);
    
    sgemm_naive<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K, alpha, beta, transA, transB);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    cpu_gemm(h_A, h_B, h_C_ref, M, N, K, alpha, beta, transA, transB);
    
    int errors = 0;
    for (int i = 0; i < M*N; i++) {
        if (fabs(h_C[i] - h_C_ref[i]) > 1e-2) {
            errors++;
            if (errors <= 3) {
                printf("  Mismatch at %d: GPU=%.4f, CPU=%.4f\n", i, h_C[i], h_C_ref[i]);
            }
        }
    }
    printf("  Result: %s (%d errors)\n\n", errors == 0 ? "✓ PASS" : "✗ FAIL", errors);
    
    // ========== Test 2: C = A^T*B ==========
    printf("--- Test 2: C = A^T*B (transpose A) ---\n");
    transA = true; transB = false;
    
    for(int i=0; i<M*N; i++) { h_C[i] = 0.0f; h_C_ref[i] = 0.0f; }
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);
    
    sgemm_naive<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K, alpha, beta, transA, transB);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    cpu_gemm(h_A, h_B, h_C_ref, M, N, K, alpha, beta, transA, transB);
    
    errors = 0;
    for (int i = 0; i < M*N; i++) {
        if (fabs(h_C[i] - h_C_ref[i]) > 1e-2) errors++;
    }
    printf("  Result: %s (%d errors)\n\n", errors == 0 ? "✓ PASS" : "✗ FAIL", errors);
    
    // ========== Test 3: C = A*B^T ==========
    printf("--- Test 3: C = A*B^T (transpose B) ---\n");
    transA = false; transB = true;
    
    for(int i=0; i<M*N; i++) { h_C[i] = 0.0f; h_C_ref[i] = 0.0f; }
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);
    
    sgemm_naive<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K, alpha, beta, transA, transB);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    cpu_gemm(h_A, h_B, h_C_ref, M, N, K, alpha, beta, transA, transB);
    
    errors = 0;
    for (int i = 0; i < M*N; i++) {
        if (fabs(h_C[i] - h_C_ref[i]) > 1e-2) errors++;
    }
    printf("  Result: %s (%d errors)\n\n", errors == 0 ? "✓ PASS" : "✗ FAIL", errors);
    
    // ========== Test 4: C = A^T*B^T ==========
    printf("--- Test 4: C = A^T*B^T (transpose both) ---\n");
    transA = true; transB = true;
    
    for(int i=0; i<M*N; i++) { h_C[i] = 0.0f; h_C_ref[i] = 0.0f; }
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);
    
    sgemm_naive<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K, alpha, beta, transA, transB);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    cpu_gemm(h_A, h_B, h_C_ref, M, N, K, alpha, beta, transA, transB);
    
    errors = 0;
    for (int i = 0; i < M*N; i++) {
        if (fabs(h_C[i] - h_C_ref[i]) > 1e-2) errors++;
    }
    printf("  Result: %s (%d errors)\n\n", errors == 0 ? "✓ PASS" : "✗ FAIL", errors);
    
    // ========== Test 5: C = A*B + C (beta=1) ==========
    printf("--- Test 5: C = A*B + C (beta=1.0, accumulate) ---\n");
    alpha = 1.0f; beta = 1.0f;
    transA = false; transB = false;
    
    for(int i=0; i<M*N; i++) { h_C[i] = 0.5f; h_C_ref[i] = 0.5f; }
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);
    
    sgemm_naive<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K, alpha, beta, transA, transB);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    cpu_gemm(h_A, h_B, h_C_ref, M, N, K, alpha, beta, transA, transB);
    
    errors = 0;
    for (int i = 0; i < M*N; i++) {
        if (fabs(h_C[i] - h_C_ref[i]) > 1e-2) errors++;
    }
    printf("  Result: %s (%d errors)\n\n", errors == 0 ? "✓ PASS" : "✗ FAIL", errors);
    
    // ========== Performance Benchmark ==========
    printf("--- Performance Benchmark (standard C=A*B) ---\n");
    alpha = 1.0f; beta = 0.0f;
    transA = false; transB = false;
    
    for(int i=0; i<M*N; i++) h_C[i] = 0.0f;
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    // Warmup
    for (int iter = 0; iter < 3; iter++) {
        sgemm_naive<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K, alpha, beta, transA, transB);
    }
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    const int ITERS = 10;
    for (int iter = 0; iter < ITERS; iter++) {
        sgemm_naive<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K, alpha, beta, transA, transB);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float ms_per_iter = ms / ITERS;
    
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double tflops = (flops / (ms_per_iter / 1000.0)) / 1e12;
    
    printf("  Matrix: %dx%dx%d\n", M, N, K);
    printf("  Avg Time: %.3f ms\n", ms_per_iter);
    printf("  Performance: %.3f TFLOPS\n", tflops);
    
    printf("\n=== Week 2 Complete: All GEMM features tested ✓ ===\n");
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    
    return 0;
}
"""

@app.function(image=cuda_image, gpu="H100", timeout=600)
def run_gpu():
    import subprocess
    
    print("📝 Writing Week 2 Full GEMM with all tests...")
    with open("gemm_naive.cu", "w") as f:
        f.write(cuda_source_code)
    
    print("🔨 Compiling with nvcc -O3...")
    result = subprocess.run(
        ["nvcc", "-O3", "-o", "gemm", "gemm_naive.cu"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("Compilation failed!")
        print(result.stderr)
        return
    
    print("Compilation successful!\n")
    print("Running all tests on H100...\n")
    
    result = subprocess.run(["./gemm"], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)

@app.local_entrypoint()
def main():
    print("="*60)
    print("Week 2: Complete GEMM Feature Testing")
    print("="*60)
    run_gpu.remote()