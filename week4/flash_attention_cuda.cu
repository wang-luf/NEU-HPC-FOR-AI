// Week 4 - Task 2: FlashAttention-2 Algorithm 1 in Parallelized CUDA
// Implements Algorithm 1 from "FlashAttention-2: Faster Attention with Better
// Parallelism and Work Partitioning" (Section 3.1), forward pass.
//
// Parallelization strategy (matches paper spec):
//   Grid : (Tr,) thread blocks  – one block per Q row-chunk
//   Block: (Br,) threads        – one thread per row inside the Q chunk
//
//   Each threadblock:
//     - Loads its one Qi block into shared memory (line 4)
//     - Owns and accumulates its Oi block in shared memory
//     - Iterates through ALL Kj, Vj blocks in shared memory (inner loop)
//     - Writes Oi, Li back to global memory (lines 14-15)
//
// Compile: nvcc -O3 -o flash_attention_cuda flash_attention_cuda.cu
// Run:     ./flash_attention_cuda

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>

// Compile-time tile dimensions (can be tuned; must satisfy shared-mem budget)
#define BR     32    // row tile size  (= block size in threads)
#define BC     32    // col tile size
#define D_MAX  64    // max head dimension supported

// Shared memory layout (per block, all floats):
//   Qi_s   [BR][D_MAX]   — Q tile, loaded once
//   Oi_s   [BR][D_MAX]   — O tile, accumulated
//   Kj_s   [BC][D_MAX]   — K tile, refreshed each inner iter
//   Vj_s   [BC][D_MAX]   — V tile, refreshed each inner iter
//   Sij_s  [BR][BC]      — attention scores
//   Pij_s  [BR][BC]      — softmax numerators
//   mi_s   [BR]          — running row-max
//   li_s   [BR]          — running row-sum (denominator)
// Total: (2*BR + 2*BC)*D_MAX + 2*BR*BC + 2*BR  floats
//      = (2*32+2*32)*64 + 2*32*32 + 2*32 = 8192 + 2048 + 64 = 10304 floats
//      = 41216 bytes (~40 KB, fits in 48 KB H100 shared memory)

__global__ void flash_attention_kernel(
    const float * __restrict__ Q,
    const float * __restrict__ K,
    const float * __restrict__ V,
    float * __restrict__ O,
    float * __restrict__ L,
    int N, int d, int Bc, float scale)
{
    int i         = blockIdx.x;          // which Q block (outer loop index)
    int r         = threadIdx.x;         // row within Q block
    int Br        = blockDim.x;
    int row_start = i * Br;
    int Tc        = (N + Bc - 1) / Bc;
    int actual_Br = min(Br, N - row_start);

    if (r >= actual_Br) return;

    // ---- Shared memory pointers ----
    extern __shared__ float smem[];
    float *Qi_s  = smem;
    float *Oi_s  = Qi_s  + BR * D_MAX;
    float *Kj_s  = Oi_s  + BR * D_MAX;
    float *Vj_s  = Kj_s  + BC * D_MAX;
    float *Sij_s = Vj_s  + BC * D_MAX;
    float *Pij_s = Sij_s + BR * BC;
    float *mi_s  = Pij_s + BR * BC;
    float *li_s  = mi_s  + BR;

    // Line 4: Load Qi row for this thread from HBM
    for (int k = 0; k < d; k++)
        Qi_s[r * D_MAX + k] = Q[(row_start + r) * d + k];

    // Line 5: Initialize Oi = 0, mi = -inf, li = 0
    for (int k = 0; k < d; k++) Oi_s[r * D_MAX + k] = 0.0f;
    mi_s[r] = -FLT_MAX;
    li_s[r] = 0.0f;
    __syncthreads();   // all threads ready before inner loop

    // Lines 6-11: inner loop over K/V blocks
    for (int j = 0; j < Tc; j++) {
        int col_start = j * Bc;
        int actual_Bc = min(Bc, N - col_start);

        // Line 7: Collaborative load of Kj, Vj
        // Thread r loads rows r, r+Br, r+2*Br, ... of Kj/Vj
        for (int row = r; row < actual_Bc; row += Br) {
            for (int k = 0; k < d; k++) {
                Kj_s[row * D_MAX + k] = K[(col_start + row) * d + k];
                Vj_s[row * D_MAX + k] = V[(col_start + row) * d + k];
            }
        }
        __syncthreads();   // Kj/Vj fully loaded before use

        // Line 8: Sij[r, c] = Qi[r, :] . Kj[c, :] / sqrt(d)
        for (int c = 0; c < actual_Bc; c++) {
            float dot = 0.0f;
            for (int k = 0; k < d; k++)
                dot += Qi_s[r * D_MAX + k] * Kj_s[c * D_MAX + k];
            Sij_s[r * BC + c] = dot * scale;
        }

        // Line 9a: m_ij = max(mi, rowmax(Sij[r, :]))
        float mij = mi_s[r];
        for (int c = 0; c < actual_Bc; c++)
            if (Sij_s[r * BC + c] > mij) mij = Sij_s[r * BC + c];

        // Line 9b: Pij = exp(Sij - m_ij)
        for (int c = 0; c < actual_Bc; c++)
            Pij_s[r * BC + c] = expf(Sij_s[r * BC + c] - mij);

        // Line 9c: l_ij = exp(mi - m_ij) * li + rowsum(Pij)
        float row_sum = 0.0f;
        for (int c = 0; c < actual_Bc; c++) row_sum += Pij_s[r * BC + c];
        float lij = expf(mi_s[r] - mij) * li_s[r] + row_sum;

        // Line 10: Oi = diag(exp(mi - m_ij))^-1 * Oi + Pij * Vj
        //   Per row r: scale old Oi by exp(mi - mij), add Pij[r,:] * Vj
        float rescale = expf(mi_s[r] - mij);
        for (int k = 0; k < d; k++) {
            float pv = 0.0f;
            for (int c = 0; c < actual_Bc; c++)
                pv += Pij_s[r * BC + c] * Vj_s[c * D_MAX + k];
            Oi_s[r * D_MAX + k] = rescale * Oi_s[r * D_MAX + k] + pv;
        }

        // Update running statistics for this row
        mi_s[r] = mij;
        li_s[r] = lij;
        __syncthreads();   // done with Kj/Vj — safe to overwrite next iter
    } // end inner loop (line 11)

    // Line 12: Oi = diag(li)^-1 * Oi  ->  Oi[r, :] /= li[r]
    for (int k = 0; k < d; k++)
        Oi_s[r * D_MAX + k] /= li_s[r];

    // Line 13: Li = mi + log(li)
    L[row_start + r] = mi_s[r] + logf(li_s[r]);

    // Lines 14-15: Write Oi block to HBM
    for (int k = 0; k < d; k++)
        O[(row_start + r) * d + k] = Oi_s[r * D_MAX + k];
}

// ---------------------------------------------------------------------------
// CPU naive attention for correctness verification
// ---------------------------------------------------------------------------
void naive_attention_cpu(const float *Q, const float *K, const float *V,
                         float *O, int N, int d) {
    float scale = 1.0f / sqrtf((float)d);
    float *S = (float *)malloc(N * N * sizeof(float));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float dot = 0.0f;
            for (int k = 0; k < d; k++) dot += Q[i*d+k] * K[j*d+k];
            S[i*N+j] = dot * scale;
        }
        float max_val = -FLT_MAX;
        for (int j = 0; j < N; j++) if (S[i*N+j] > max_val) max_val = S[i*N+j];
        float sum = 0.0f;
        for (int j = 0; j < N; j++) { S[i*N+j] = expf(S[i*N+j]-max_val); sum += S[i*N+j]; }
        for (int j = 0; j < N; j++) S[i*N+j] /= sum;
    }
    for (int i = 0; i < N; i++)
        for (int k = 0; k < d; k++) {
            float sum = 0.0f;
            for (int j = 0; j < N; j++) sum += S[i*N+j] * V[j*d+k];
            O[i*d+k] = sum;
        }
    free(S);
}

int main() {
    int N = 256, d = 64;
    int Br = BR, Bc = BC;
    float scale = 1.0f / sqrtf((float)d);

    printf("Week 4 - Task 2: FlashAttention-2 CUDA\n");
    printf("========================================\n\n");
    printf("Parameters: N=%d, d=%d, Br=%d, Bc=%d\n", N, d, Br, Bc);

    int Tr = (N + Br - 1) / Br;
    int Tc = (N + Bc - 1) / Bc;
    printf("Tr=%d blocks (outer), Tc=%d blocks (inner)\n", Tr, Tc);

    // Shared memory per block
    size_t smem_size = ((size_t)(2*BR + 2*BC) * D_MAX + 2*BR*BC + 2*BR) * sizeof(float);
    printf("Shared memory per block: %zu bytes (%.1f KB)\n\n",
           smem_size, smem_size / 1024.0f);

    size_t qkv_sz = (size_t)N * d * sizeof(float);
    float *h_Q     = (float *)malloc(qkv_sz);
    float *h_K     = (float *)malloc(qkv_sz);
    float *h_V     = (float *)malloc(qkv_sz);
    float *h_O     = (float *)malloc(qkv_sz);
    float *h_O_ref = (float *)malloc(qkv_sz);
    float *h_L     = (float *)malloc(N * sizeof(float));

    srand(42);
    for (int i = 0; i < N * d; i++) {
        h_Q[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_K[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_V[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    cudaMalloc(&d_Q, qkv_sz);
    cudaMalloc(&d_K, qkv_sz);
    cudaMalloc(&d_V, qkv_sz);
    cudaMalloc(&d_O, qkv_sz);
    cudaMalloc(&d_L, N * sizeof(float));

    cudaMemcpy(d_Q, h_Q, qkv_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, qkv_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, qkv_sz, cudaMemcpyHostToDevice);

    dim3 grid(Tr);
    dim3 block(Br);

    // ---- Correctness check (N=256) ----
    printf("--- Correctness Check (N=%d) ---\n", N);
    flash_attention_kernel<<<grid, block, smem_size>>>(
        d_Q, d_K, d_V, d_O, d_L, N, d, Bc, scale);
    cudaDeviceSynchronize();
    cudaMemcpy(h_O, d_O, qkv_sz, cudaMemcpyDeviceToHost);

    printf("Running CPU reference (naive attention)...\n");
    naive_attention_cpu(h_Q, h_K, h_V, h_O_ref, N, d);

    float max_err = 0.0f;
    int errors = 0;
    for (int i = 0; i < N * d; i++) {
        float err = fabsf(h_O[i] - h_O_ref[i]);
        if (err > max_err) max_err = err;
        if (err > 1e-4f) errors++;
    }
    printf("Max absolute error : %.2e\n", max_err);
    printf("Elements > 1e-4 err: %d / %d\n", errors, N * d);
    printf("Result             : %s\n\n", (max_err < 1e-4f) ? "PASS" : "FAIL");

    // ---- Performance benchmark (larger N=1024) ----
    int N_bench = 1024;
    int Tr_bench = (N_bench + Br - 1) / Br;
    size_t qkv_sz_bench = (size_t)N_bench * d * sizeof(float);

    float *h_Q2 = (float *)malloc(qkv_sz_bench);
    float *h_K2 = (float *)malloc(qkv_sz_bench);
    float *h_V2 = (float *)malloc(qkv_sz_bench);
    for (int i = 0; i < N_bench * d; i++) {
        h_Q2[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_K2[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_V2[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    float *d_Q2, *d_K2, *d_V2, *d_O2, *d_L2;
    cudaMalloc(&d_Q2, qkv_sz_bench);
    cudaMalloc(&d_K2, qkv_sz_bench);
    cudaMalloc(&d_V2, qkv_sz_bench);
    cudaMalloc(&d_O2, qkv_sz_bench);
    cudaMalloc(&d_L2, N_bench * sizeof(float));
    cudaMemcpy(d_Q2, h_Q2, qkv_sz_bench, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K2, h_K2, qkv_sz_bench, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V2, h_V2, qkv_sz_bench, cudaMemcpyHostToDevice);

    dim3 grid_bench(Tr_bench);
    float scale_bench = 1.0f / sqrtf((float)d);

    // Warmup
    for (int w = 0; w < 3; w++)
        flash_attention_kernel<<<grid_bench, block, smem_size>>>(
            d_Q2, d_K2, d_V2, d_O2, d_L2, N_bench, d, Bc, scale_bench);
    cudaDeviceSynchronize();

    printf("--- Performance Benchmark (N=%d, d=%d) ---\n", N_bench, d);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int ITERS = 100;
    cudaEventRecord(start);
    for (int it = 0; it < ITERS; it++)
        flash_attention_kernel<<<grid_bench, block, smem_size>>>(
            d_Q2, d_K2, d_V2, d_O2, d_L2, N_bench, d, Bc, scale_bench);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    float ms_per_iter = ms / ITERS;

    // FLOPs: 2*N*N*d (QK^T matmul) + 2*N*N*d (PV matmul) = 4*N^2*d
    double flops = 4.0 * (double)N_bench * (double)N_bench * (double)d;
    double gflops = flops / (ms_per_iter * 1e-3) / 1e9;

    printf("Grid: %d blocks, Block: %d threads\n", Tr_bench, Br);
    printf("Average kernel time : %.4f ms\n", ms_per_iter);
    printf("Approx throughput   : %.1f GFLOPS\n\n", gflops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_Q);  cudaFree(d_K);  cudaFree(d_V);  cudaFree(d_O);  cudaFree(d_L);
    cudaFree(d_Q2); cudaFree(d_K2); cudaFree(d_V2); cudaFree(d_O2); cudaFree(d_L2);
    free(h_Q); free(h_K); free(h_V); free(h_O); free(h_O_ref); free(h_L);
    free(h_Q2); free(h_K2); free(h_V2);

    printf("Week 4 CUDA complete.\n");
    return 0;
}
