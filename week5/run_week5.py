import modal

app = modal.App("week05-flash-attention-cute")

# Install CUTLASS (header-only) alongside the CUDA dev image
cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .run_commands(
        "apt-get update -qq && apt-get install -y -qq git",
        "git clone --depth 1 https://github.com/NVIDIA/cutlass.git /opt/cutlass",
    )
)

# ============================================================
# Week 5: FlashAttention Algorithm 1 re-implemented with CuTe
# ============================================================
#
# CuTe features used:
#   make_tensor       – create typed Tensor views over gmem / smem
#   make_gmem_ptr     – tag pointer as global-memory (enables async copy dispatch)
#   make_smem_ptr     – tag pointer as shared-memory
#   make_layout       – (Shape, Stride) pair describing the index space
#   local_tile        – extract a compile-time-sized tile from a Tensor
#   cute::copy        – generic element-wise copy (dispatches to vectorised LDG)
#   tensor(i, j)      – coordinate-based element access
#   tensor(i, _)      – row-slice returning a 1-D view
#
# Algorithm: FlashAttention-2 Algorithm 1 (forward pass, single head)
#   Line  4: Load Qi
#   Line  5: Init Oi=0, mi=-inf, li=0
#   Lines 7-10: for each Kj/Vj tile:
#               compute Sij, online softmax update, accumulate Oi
#   Line 12: Oi /= li
#   Line 13: Li = mi + log(li)
#   Lines 14-15: write Oi back to HBM
# ============================================================

cuda_source_code = r"""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>

// CuTe headers (CUTLASS must be on the include path: -I/opt/cutlass/include)
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>

using namespace cute;

// ── Compile-time tile / head dimensions ─────────────────────────────────────
static constexpr int kBR = 32;   // Q row-tile  (= threads per block)
static constexpr int kBC = 32;   // K/V col-tile
static constexpr int kD  = 64;   // head dimension (fixed)

// Shared-memory budget (floats):
//   sQ[kBR,kD] + sO[kBR,kD] + sK[kBC,kD] + sV[kBC,kD]  = 4*32*64 = 8192
//   sS[kBR,kBC]                                           =   32*32 = 1024
//   mi[kBR]    + li[kBR]                                  =    2*32 =   64
//   Total = 9280 floats = 37 120 bytes  (<< 48 KB H100 smem per block)
static constexpr int kSmemFloats = 4*kBR*kD + kBR*kBC + 2*kBR;

// ── FlashAttention kernel (CuTe version) ────────────────────────────────────
__global__ void flash_attention_cute(
    const float* __restrict__ Q_ptr,
    const float* __restrict__ K_ptr,
    const float* __restrict__ V_ptr,
    float*       __restrict__ O_ptr,
    float*       __restrict__ L_ptr,
    int N, float scale)
{
    int block_i   = blockIdx.x;
    int r         = threadIdx.x;              // row within Q tile
    int row_start = block_i * kBR;
    int actual_Br = min(kBR, N - row_start);
    if (r >= actual_Br) return;               // guard last block

    // ── Shared memory raw pointers ───────────────────────────────────────────
    extern __shared__ float smem[];
    float* Qi_raw  = smem;
    float* Oi_raw  = Qi_raw  + kBR * kD;
    float* Kj_raw  = Oi_raw  + kBR * kD;
    float* Vj_raw  = Kj_raw  + kBC * kD;
    float* Sij_raw = Vj_raw  + kBC * kD;
    float* mi_raw  = Sij_raw + kBR * kBC;
    float* li_raw  = mi_raw  + kBR;

    // ── CuTe Tensors: shared memory ─────────────────────────────────────────
    // Each tensor wraps a raw pointer with a (Shape, Stride) Layout.
    // Row-major layout: stride = (num_cols, 1).
    auto sQ = make_tensor(make_smem_ptr(Qi_raw),
                  make_layout(make_shape (Int<kBR>{}, Int<kD>{}),
                              make_stride(Int<kD>{},  Int<1>{})));
    auto sO = make_tensor(make_smem_ptr(Oi_raw),
                  make_layout(make_shape (Int<kBR>{}, Int<kD>{}),
                              make_stride(Int<kD>{},  Int<1>{})));
    auto sK = make_tensor(make_smem_ptr(Kj_raw),
                  make_layout(make_shape (Int<kBC>{}, Int<kD>{}),
                              make_stride(Int<kD>{},  Int<1>{})));
    auto sV = make_tensor(make_smem_ptr(Vj_raw),
                  make_layout(make_shape (Int<kBC>{}, Int<kD>{}),
                              make_stride(Int<kD>{},  Int<1>{})));
    // Attention score tile: [kBR, kBC]
    auto sS = make_tensor(make_smem_ptr(Sij_raw),
                  make_layout(make_shape (Int<kBR>{}, Int<kBC>{}),
                              make_stride(Int<kBC>{}, Int<1>{})));

    // ── CuTe Tensors: global memory ──────────────────────────────────────────
    // Shape (N, kD), row-major.  N is a runtime value; kD is compile-time.
    auto gQ = make_tensor(make_gmem_ptr(Q_ptr),
                  make_layout(make_shape (N, Int<kD>{}),
                              make_stride(Int<kD>{}, Int<1>{})));
    auto gK = make_tensor(make_gmem_ptr(K_ptr),
                  make_layout(make_shape (N, Int<kD>{}),
                              make_stride(Int<kD>{}, Int<1>{})));
    auto gV = make_tensor(make_gmem_ptr(V_ptr),
                  make_layout(make_shape (N, Int<kD>{}),
                              make_stride(Int<kD>{}, Int<1>{})));
    auto gO = make_tensor(make_gmem_ptr(O_ptr),
                  make_layout(make_shape (N, Int<kD>{}),
                              make_stride(Int<kD>{}, Int<1>{})));

    // local_tile(tensor, tile_shape, coord) returns the tile at coord when
    // tensor is logically divided into tiles of tile_shape.
    // gQi = Q[block_i*kBR : (block_i+1)*kBR, 0:kD]
    auto gQi = local_tile(gQ,
                  make_shape(Int<kBR>{}, Int<kD>{}),
                  make_coord(block_i, 0));

    // ── Line 4: Load row r of Qi from gmem → smem ────────────────────────────
    // gQi(r, _) is a 1-D view of the r-th row (size kD).
    // cute::copy dispatches to the fastest available load instruction.
    copy(gQi(r, _), sQ(r, _));

    // ── Line 5: Init Oi=0, mi=-inf, li=0 ────────────────────────────────────
    #pragma unroll
    for (int k = 0; k < kD; k++) sO(r, k) = 0.0f;
    mi_raw[r] = -FLT_MAX;
    li_raw[r] = 0.0f;
    __syncthreads();

    // ── Lines 6-11: Iterate over all K/V tiles ───────────────────────────────
    int Tc = (N + kBC - 1) / kBC;
    for (int j = 0; j < Tc; j++) {
        int actual_Bc = min(kBC, N - j * kBC);

        // local_tile gives the j-th K and V tile
        auto gKj = local_tile(gK,
                      make_shape(Int<kBC>{}, Int<kD>{}),
                      make_coord(j, 0));
        auto gVj = local_tile(gV,
                      make_shape(Int<kBC>{}, Int<kD>{}),
                      make_coord(j, 0));

        // ── Line 7: Collaborative gmem→smem load of Kj and Vj ────────────────
        // All kBR threads cooperate; each handles rows r, r+kBR, r+2*kBR, ...
        for (int row = r; row < actual_Bc; row += kBR) {
            copy(gKj(row, _), sK(row, _));
            copy(gVj(row, _), sV(row, _));
        }
        __syncthreads();   // Kj, Vj fully in smem before any use

        // ── Line 8: Sij[r,c] = dot(sQ[r,:], sK[c,:]) * scale ────────────────
        for (int c = 0; c < actual_Bc; c++) {
            float dot = 0.0f;
            #pragma unroll
            for (int k = 0; k < kD; k++)
                dot += sQ(r, k) * sK(c, k);   // CuTe element access
            sS(r, c) = dot * scale;
        }

        // ── Line 9a: mij = max(mi_old, rowmax(Sij[r,:])) ─────────────────────
        float mij = mi_raw[r];
        for (int c = 0; c < actual_Bc; c++)
            mij = fmaxf(mij, sS(r, c));

        // ── Line 9b: Pij = exp(Sij - mij)  (stored back into sS) ─────────────
        for (int c = 0; c < actual_Bc; c++)
            sS(r, c) = expf(sS(r, c) - mij);

        // ── Line 9c: lij = exp(mi_old - mij)*li_old + rowsum(Pij) ────────────
        float row_sum = 0.0f;
        for (int c = 0; c < actual_Bc; c++) row_sum += sS(r, c);
        float lij = expf(mi_raw[r] - mij) * li_raw[r] + row_sum;

        // ── Line 10: Oi = exp(mi_old-mij)*Oi + Pij*Vj ───────────────────────
        float rescale = expf(mi_raw[r] - mij);
        #pragma unroll
        for (int k = 0; k < kD; k++) {
            float pv = 0.0f;
            for (int c = 0; c < actual_Bc; c++)
                pv += sS(r, c) * sV(c, k);    // CuTe element access
            sO(r, k) = rescale * sO(r, k) + pv;
        }

        mi_raw[r] = mij;
        li_raw[r] = lij;
        __syncthreads();   // done with Kj/Vj smem region before next iteration
    }

    // ── Line 12: Oi = diag(li)^-1 * Oi  →  Oi[r,:] /= li[r] ───────────────
    #pragma unroll
    for (int k = 0; k < kD; k++) sO(r, k) /= li_raw[r];

    // ── Line 13: Li = mi + log(li)  (logsumexp for each row) ────────────────
    L_ptr[row_start + r] = mi_raw[r] + logf(li_raw[r]);

    // ── Lines 14-15: Write Oi back to global memory ──────────────────────────
    auto gOi = local_tile(gO,
                  make_shape(Int<kBR>{}, Int<kD>{}),
                  make_coord(block_i, 0));
    copy(sO(r, _), gOi(r, _));   // smem → gmem, element-wise via cute::copy
}

// ── CPU naive attention (reference) ─────────────────────────────────────────
void naive_attention_cpu(const float* Q, const float* K, const float* V,
                         float* O, int N, int d) {
    float scale = 1.0f / sqrtf((float)d);
    float* S = (float*)malloc((size_t)N * N * sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float dot = 0.0f;
            for (int k = 0; k < d; k++) dot += Q[i*d+k] * K[j*d+k];
            S[i*N+j] = dot * scale;
        }
        float mx = -FLT_MAX;
        for (int j = 0; j < N; j++) if (S[i*N+j] > mx) mx = S[i*N+j];
        float sm = 0.0f;
        for (int j = 0; j < N; j++) { S[i*N+j] = expf(S[i*N+j]-mx); sm += S[i*N+j]; }
        for (int j = 0; j < N; j++) S[i*N+j] /= sm;
    }
    for (int i = 0; i < N; i++)
        for (int k = 0; k < d; k++) {
            float s = 0.0f;
            for (int j = 0; j < N; j++) s += S[i*N+j] * V[j*d+k];
            O[i*d+k] = s;
        }
    free(S);
}

// ── main ─────────────────────────────────────────────────────────────────────
int main() {
    printf("Week 5: FlashAttention Algorithm 1 with CuTe\n");
    printf("=============================================\n\n");

    int N = 256, d = kD;
    float scale = 1.0f / sqrtf((float)d);
    int Tr = (N + kBR - 1) / kBR;
    size_t smem_bytes = (size_t)kSmemFloats * sizeof(float);

    printf("Parameters: N=%d, d=%d, Br=%d, Bc=%d\n", N, d, kBR, kBC);
    printf("Shared memory per block: %zu bytes (%.1f KB)\n\n",
           smem_bytes, smem_bytes / 1024.0f);

    size_t qkv_sz = (size_t)N * d * sizeof(float);
    float *h_Q     = (float*)malloc(qkv_sz);
    float *h_K     = (float*)malloc(qkv_sz);
    float *h_V     = (float*)malloc(qkv_sz);
    float *h_O     = (float*)malloc(qkv_sz);
    float *h_O_ref = (float*)malloc(qkv_sz);
    float *h_L     = (float*)malloc(N * sizeof(float));

    srand(42);
    for (int i = 0; i < N*d; i++) {
        h_Q[i] = ((float)rand()/RAND_MAX)*2.0f-1.0f;
        h_K[i] = ((float)rand()/RAND_MAX)*2.0f-1.0f;
        h_V[i] = ((float)rand()/RAND_MAX)*2.0f-1.0f;
    }

    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    cudaMalloc(&d_Q, qkv_sz); cudaMalloc(&d_K, qkv_sz);
    cudaMalloc(&d_V, qkv_sz); cudaMalloc(&d_O, qkv_sz);
    cudaMalloc(&d_L, N * sizeof(float));
    cudaMemcpy(d_Q, h_Q, qkv_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, qkv_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, qkv_sz, cudaMemcpyHostToDevice);

    // ── Correctness check ─────────────────────────────────────────────────────
    printf("--- Correctness Check (N=%d) ---\n", N);
    flash_attention_cute<<<dim3(Tr), dim3(kBR), smem_bytes>>>(
        d_Q, d_K, d_V, d_O, d_L, N, scale);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaMemcpy(h_O, d_O, qkv_sz, cudaMemcpyDeviceToHost);

    printf("Running CPU reference (naive)...\n");
    naive_attention_cpu(h_Q, h_K, h_V, h_O_ref, N, d);

    float max_err = 0.0f; int errors = 0;
    for (int i = 0; i < N*d; i++) {
        float e = fabsf(h_O[i] - h_O_ref[i]);
        if (e > max_err) max_err = e;
        if (e > 1e-4f) errors++;
    }
    printf("Max absolute error : %.2e\n", max_err);
    printf("Elements > 1e-4 err: %d / %d\n", errors, N*d);
    printf("Result             : %s\n\n", (max_err < 1e-4f) ? "PASS" : "FAIL");

    // ── Performance benchmark ─────────────────────────────────────────────────
    int N2 = 1024, Tr2 = (N2 + kBR - 1) / kBR;
    size_t sz2 = (size_t)N2 * d * sizeof(float);
    float *h_Q2=(float*)malloc(sz2), *h_K2=(float*)malloc(sz2), *h_V2=(float*)malloc(sz2);
    for (int i = 0; i < N2*d; i++) {
        h_Q2[i] = ((float)rand()/RAND_MAX)*2.0f-1.0f;
        h_K2[i] = ((float)rand()/RAND_MAX)*2.0f-1.0f;
        h_V2[i] = ((float)rand()/RAND_MAX)*2.0f-1.0f;
    }
    float *d_Q2, *d_K2, *d_V2, *d_O2, *d_L2;
    cudaMalloc(&d_Q2, sz2); cudaMalloc(&d_K2, sz2); cudaMalloc(&d_V2, sz2);
    cudaMalloc(&d_O2, sz2); cudaMalloc(&d_L2, N2*sizeof(float));
    cudaMemcpy(d_Q2, h_Q2, sz2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K2, h_K2, sz2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V2, h_V2, sz2, cudaMemcpyHostToDevice);

    float scale2 = 1.0f / sqrtf((float)d);
    // Warmup
    for (int w = 0; w < 3; w++)
        flash_attention_cute<<<dim3(Tr2), dim3(kBR), smem_bytes>>>(
            d_Q2, d_K2, d_V2, d_O2, d_L2, N2, scale2);
    cudaDeviceSynchronize();

    printf("--- Performance Benchmark (N=%d, d=%d) ---\n", N2, d);
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    const int ITERS = 100;
    cudaEventRecord(start);
    for (int it = 0; it < ITERS; it++)
        flash_attention_cute<<<dim3(Tr2), dim3(kBR), smem_bytes>>>(
            d_Q2, d_K2, d_V2, d_O2, d_L2, N2, scale2);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    float ms_iter = ms / ITERS;
    // FLOPs: QK^T (2*N^2*d) + PV (2*N^2*d) = 4*N^2*d
    double flops  = 4.0 * (double)N2 * (double)N2 * (double)d;
    double gflops = flops / (ms_iter * 1e-3) / 1e9;

    printf("Grid: %d blocks  Block: %d threads\n", Tr2, kBR);
    printf("Average kernel time : %.4f ms\n", ms_iter);
    printf("Approx throughput   : %.1f GFLOPS\n\n", gflops);

    // Compare vs Week 4 CUDA
    printf("Comparison:\n");
    printf("  Week 4 (raw CUDA pointers): baseline\n");
    printf("  Week 5 (CuTe Tensors)     : %.1f GFLOPS\n", gflops);
    printf("  (same algorithm, CuTe abstractions replace raw ptr arithmetic)\n");

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); cudaFree(d_L);
    cudaFree(d_Q2); cudaFree(d_K2); cudaFree(d_V2); cudaFree(d_O2); cudaFree(d_L2);
    free(h_Q); free(h_K); free(h_V); free(h_O); free(h_O_ref); free(h_L);
    free(h_Q2); free(h_K2); free(h_V2);

    printf("Week 5 CuTe complete.\n");
    return 0;
}
"""


@app.function(image=cuda_image, gpu="H100", timeout=600)
def run_gpu():
    import subprocess

    print("=" * 60)
    print("Week 5: FlashAttention with CuTe")
    print("=" * 60)

    # Write CUDA source
    with open("flash_attn_cute.cu", "w") as f:
        f.write(cuda_source_code)

    print("Compiling with CuTe (CUTLASS headers)...")
    result = subprocess.run(
        [
            "nvcc", "-O3", "--std=c++17",
            "-I/opt/cutlass/include",
            "-o", "flash_attn_cute",
            "flash_attn_cute.cu",
            "-lm",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Compilation FAILED:")
        print(result.stderr)
        return

    print("Compilation successful.\n")
    print("Running on H100...\n")

    result = subprocess.run(["./flash_attn_cute"], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Stderr:", result.stderr)


@app.local_entrypoint()
def main():
    run_gpu.remote()
