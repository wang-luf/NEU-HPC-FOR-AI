import modal
import subprocess

app = modal.App("week04-flash-attention")

cuda_image = modal.Image.from_registry(
    "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
)

# ============================================================
# Task 1: Sequential C — FlashAttention-2 Algorithm 1
# ============================================================
c_source_code = r"""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>

void naive_attention(const float *Q, const float *K, const float *V,
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

void flash_attention(const float *Q, const float *K, const float *V,
                     float *O, float *L,
                     int N, int d, int Br, int Bc) {
    float scale = 1.0f / sqrtf((float)d);
    int Tr = (N + Br - 1) / Br;
    int Tc = (N + Bc - 1) / Bc;

    float *Qi  = (float *)malloc(Br * d  * sizeof(float));
    float *Kj  = (float *)malloc(Bc * d  * sizeof(float));
    float *Vj  = (float *)malloc(Bc * d  * sizeof(float));
    float *Oi  = (float *)malloc(Br * d  * sizeof(float));
    float *Sij = (float *)malloc(Br * Bc * sizeof(float));
    float *Pij = (float *)malloc(Br * Bc * sizeof(float));
    float *mi  = (float *)malloc(Br * sizeof(float));
    float *li  = (float *)malloc(Br * sizeof(float));
    float *mij = (float *)malloc(Br * sizeof(float));
    float *lij = (float *)malloc(Br * sizeof(float));

    for (int i = 0; i < Tr; i++) {
        int row_start = i * Br;
        int actual_Br = (row_start + Br <= N) ? Br : (N - row_start);

        /* Line 4: Load Qi */
        for (int r = 0; r < actual_Br; r++)
            for (int k = 0; k < d; k++)
                Qi[r*d+k] = Q[(row_start+r)*d+k];

        /* Line 5: Init Oi=0, mi=-inf, li=0 */
        for (int r = 0; r < actual_Br; r++) {
            for (int k = 0; k < d; k++) Oi[r*d+k] = 0.0f;
            mi[r] = -FLT_MAX;
            li[r] = 0.0f;
        }

        for (int j = 0; j < Tc; j++) {
            int col_start = j * Bc;
            int actual_Bc = (col_start + Bc <= N) ? Bc : (N - col_start);

            /* Line 7: Load Kj, Vj */
            for (int r = 0; r < actual_Bc; r++)
                for (int k = 0; k < d; k++) {
                    Kj[r*d+k] = K[(col_start+r)*d+k];
                    Vj[r*d+k] = V[(col_start+r)*d+k];
                }

            /* Line 8: Sij = Qi * Kj^T / sqrt(d) */
            for (int r = 0; r < actual_Br; r++)
                for (int c = 0; c < actual_Bc; c++) {
                    float dot = 0.0f;
                    for (int k = 0; k < d; k++) dot += Qi[r*d+k] * Kj[c*d+k];
                    Sij[r*Bc+c] = dot * scale;
                }

            /* Line 9: mij = max(mi, rowmax(Sij)) */
            for (int r = 0; r < actual_Br; r++) {
                float row_max = mi[r];
                for (int c = 0; c < actual_Bc; c++)
                    if (Sij[r*Bc+c] > row_max) row_max = Sij[r*Bc+c];
                mij[r] = row_max;
            }

            /* Line 9: Pij = exp(Sij - mij) */
            for (int r = 0; r < actual_Br; r++)
                for (int c = 0; c < actual_Bc; c++)
                    Pij[r*Bc+c] = expf(Sij[r*Bc+c] - mij[r]);

            /* Line 9: lij = exp(mi - mij)*li + rowsum(Pij) */
            for (int r = 0; r < actual_Br; r++) {
                float row_sum = 0.0f;
                for (int c = 0; c < actual_Bc; c++) row_sum += Pij[r*Bc+c];
                lij[r] = expf(mi[r] - mij[r]) * li[r] + row_sum;
            }

            /* Line 10: Oi = diag(exp(mi-mij))^-1 * Oi + Pij*Vj */
            for (int r = 0; r < actual_Br; r++) {
                float rescale = expf(mi[r] - mij[r]);
                for (int k = 0; k < d; k++) {
                    float pv = 0.0f;
                    for (int c = 0; c < actual_Bc; c++)
                        pv += Pij[r*Bc+c] * Vj[c*d+k];
                    Oi[r*d+k] = rescale * Oi[r*d+k] + pv;
                }
            }

            for (int r = 0; r < actual_Br; r++) { mi[r] = mij[r]; li[r] = lij[r]; }
        }

        /* Line 12: Oi /= li */
        for (int r = 0; r < actual_Br; r++)
            for (int k = 0; k < d; k++) Oi[r*d+k] /= li[r];

        /* Line 13: Li = mi + log(li) */
        for (int r = 0; r < actual_Br; r++)
            L[row_start+r] = mi[r] + logf(li[r]);

        /* Lines 14-15: Write Oi back */
        for (int r = 0; r < actual_Br; r++)
            for (int k = 0; k < d; k++)
                O[(row_start+r)*d+k] = Oi[r*d+k];
    }

    free(Qi); free(Kj); free(Vj); free(Oi);
    free(Sij); free(Pij); free(mi); free(li); free(mij); free(lij);
}

int main() {
    printf("Week 4 - Task 1: FlashAttention-2 Sequential C\n");
    printf("================================================\n\n");

    int N = 128, d = 64, Br = 16, Bc = 16;
    printf("N=%d, d=%d, Br=%d, Bc=%d\n", N, d, Br, Bc);
    printf("Tr=%d, Tc=%d\n\n", (N+Br-1)/Br, (N+Bc-1)/Bc);

    size_t sz = (size_t)N * d * sizeof(float);
    float *Q = (float*)malloc(sz), *K = (float*)malloc(sz), *V = (float*)malloc(sz);
    float *O_flash = (float*)malloc(sz), *O_naive = (float*)malloc(sz);
    float *L = (float*)malloc(N * sizeof(float));

    srand(42);
    for (int i = 0; i < N*d; i++) {
        Q[i] = ((float)rand()/RAND_MAX)*2.0f-1.0f;
        K[i] = ((float)rand()/RAND_MAX)*2.0f-1.0f;
        V[i] = ((float)rand()/RAND_MAX)*2.0f-1.0f;
    }

    printf("Running FlashAttention-2 Algorithm 1 (C)...\n");
    flash_attention(Q, K, V, O_flash, L, N, d, Br, Bc);

    printf("Running naive attention (reference)...\n");
    naive_attention(Q, K, V, O_naive, N, d);

    float max_err = 0.0f; int errors = 0;
    for (int i = 0; i < N*d; i++) {
        float err = fabsf(O_flash[i] - O_naive[i]);
        if (err > max_err) max_err = err;
        if (err > 1e-4f) errors++;
    }
    printf("\nMax absolute error : %.2e\n", max_err);
    printf("Elements > 1e-4 err: %d / %d\n", errors, N*d);
    printf("Result             : %s\n\n", (max_err < 1e-4f) ? "PASS" : "FAIL");
    printf("Logsumexp L[0..3]: %.4f  %.4f  %.4f  %.4f\n",L[0],L[1],L[2],L[3]);

    free(Q); free(K); free(V); free(O_flash); free(O_naive); free(L);
    return 0;
}
"""

# ============================================================
# Task 2: Parallelized CUDA — FlashAttention-2 Algorithm 1
# ============================================================
cuda_source_code = r"""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>

#define BR     32
#define BC     32
#define D_MAX  64

__global__ void flash_attention_kernel(
    const float * __restrict__ Q,
    const float * __restrict__ K,
    const float * __restrict__ V,
    float * __restrict__ O,
    float * __restrict__ L,
    int N, int d, int Bc, float scale)
{
    int i         = blockIdx.x;
    int r         = threadIdx.x;
    int Br        = blockDim.x;
    int row_start = i * Br;
    int Tc        = (N + Bc - 1) / Bc;
    int actual_Br = min(Br, N - row_start);

    if (r >= actual_Br) return;

    extern __shared__ float smem[];
    float *Qi_s  = smem;
    float *Oi_s  = Qi_s  + BR * D_MAX;
    float *Kj_s  = Oi_s  + BR * D_MAX;
    float *Vj_s  = Kj_s  + BC * D_MAX;
    float *Sij_s = Vj_s  + BC * D_MAX;
    float *Pij_s = Sij_s + BR * BC;
    float *mi_s  = Pij_s + BR * BC;
    float *li_s  = mi_s  + BR;

    /* Line 4: Load Qi */
    for (int k = 0; k < d; k++)
        Qi_s[r * D_MAX + k] = Q[(row_start + r) * d + k];

    /* Line 5: Init Oi=0, mi=-inf, li=0 */
    for (int k = 0; k < d; k++) Oi_s[r * D_MAX + k] = 0.0f;
    mi_s[r] = -FLT_MAX;
    li_s[r] = 0.0f;
    __syncthreads();

    for (int j = 0; j < Tc; j++) {
        int col_start = j * Bc;
        int actual_Bc = min(Bc, N - col_start);

        /* Line 7: Collaborative load of Kj, Vj */
        for (int row = r; row < actual_Bc; row += Br)
            for (int k = 0; k < d; k++) {
                Kj_s[row * D_MAX + k] = K[(col_start + row) * d + k];
                Vj_s[row * D_MAX + k] = V[(col_start + row) * d + k];
            }
        __syncthreads();

        /* Line 8: Sij = Qi * Kj^T / sqrt(d) */
        for (int c = 0; c < actual_Bc; c++) {
            float dot = 0.0f;
            for (int k = 0; k < d; k++)
                dot += Qi_s[r * D_MAX + k] * Kj_s[c * D_MAX + k];
            Sij_s[r * BC + c] = dot * scale;
        }

        /* Line 9a: mij = max(mi, rowmax(Sij[r,:])) */
        float mij = mi_s[r];
        for (int c = 0; c < actual_Bc; c++)
            if (Sij_s[r * BC + c] > mij) mij = Sij_s[r * BC + c];

        /* Line 9b: Pij = exp(Sij - mij) */
        for (int c = 0; c < actual_Bc; c++)
            Pij_s[r * BC + c] = expf(Sij_s[r * BC + c] - mij);

        /* Line 9c: lij = exp(mi - mij)*li + rowsum(Pij) */
        float row_sum = 0.0f;
        for (int c = 0; c < actual_Bc; c++) row_sum += Pij_s[r * BC + c];
        float lij = expf(mi_s[r] - mij) * li_s[r] + row_sum;

        /* Line 10: Oi = diag(exp(mi-mij))^-1 * Oi + Pij*Vj */
        float rescale = expf(mi_s[r] - mij);
        for (int k = 0; k < d; k++) {
            float pv = 0.0f;
            for (int c = 0; c < actual_Bc; c++)
                pv += Pij_s[r * BC + c] * Vj_s[c * D_MAX + k];
            Oi_s[r * D_MAX + k] = rescale * Oi_s[r * D_MAX + k] + pv;
        }

        mi_s[r] = mij;
        li_s[r] = lij;
        __syncthreads();
    }

    /* Line 12: Oi /= li */
    for (int k = 0; k < d; k++) Oi_s[r * D_MAX + k] /= li_s[r];

    /* Line 13: Li = mi + log(li) */
    L[row_start + r] = mi_s[r] + logf(li_s[r]);

    /* Lines 14-15: Write Oi to HBM */
    for (int k = 0; k < d; k++)
        O[(row_start + r) * d + k] = Oi_s[r * D_MAX + k];
}

void naive_attention_cpu(const float *Q, const float *K, const float *V,
                         float *O, int N, int d) {
    float scale = 1.0f / sqrtf((float)d);
    float *S = (float *)malloc(N * N * sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float dot = 0.0f;
            for (int k = 0; k < d; k++) dot += Q[i*d+k]*K[j*d+k];
            S[i*N+j] = dot*scale;
        }
        float mv = -FLT_MAX;
        for (int j=0;j<N;j++) if(S[i*N+j]>mv) mv=S[i*N+j];
        float sm=0; for(int j=0;j<N;j++){S[i*N+j]=expf(S[i*N+j]-mv);sm+=S[i*N+j];}
        for(int j=0;j<N;j++) S[i*N+j]/=sm;
    }
    for(int i=0;i<N;i++) for(int k=0;k<d;k++){
        float s=0; for(int j=0;j<N;j++) s+=S[i*N+j]*V[j*d+k]; O[i*d+k]=s;
    }
    free(S);
}

int main() {
    int N = 256, d = 64, Br = BR, Bc = BC;
    float scale = 1.0f / sqrtf((float)d);

    printf("Week 4 - Task 2: FlashAttention-2 CUDA\n");
    printf("========================================\n\n");
    printf("N=%d, d=%d, Br=%d, Bc=%d\n", N, d, Br, Bc);

    int Tr = (N + Br - 1) / Br;
    int Tc = (N + Bc - 1) / Bc;
    printf("Tr=%d, Tc=%d\n", Tr, Tc);

    size_t smem_size = ((size_t)(2*BR+2*BC)*D_MAX + 2*BR*BC + 2*BR)*sizeof(float);
    printf("Shared memory per block: %zu bytes (%.1f KB)\n\n",
           smem_size, smem_size/1024.0f);

    size_t qkv_sz = (size_t)N*d*sizeof(float);
    float *h_Q=(float*)malloc(qkv_sz), *h_K=(float*)malloc(qkv_sz);
    float *h_V=(float*)malloc(qkv_sz), *h_O=(float*)malloc(qkv_sz);
    float *h_O_ref=(float*)malloc(qkv_sz), *h_L=(float*)malloc(N*sizeof(float));

    srand(42);
    for(int i=0;i<N*d;i++){
        h_Q[i]=((float)rand()/RAND_MAX)*2.0f-1.0f;
        h_K[i]=((float)rand()/RAND_MAX)*2.0f-1.0f;
        h_V[i]=((float)rand()/RAND_MAX)*2.0f-1.0f;
    }

    float *d_Q,*d_K,*d_V,*d_O,*d_L;
    cudaMalloc(&d_Q,qkv_sz); cudaMalloc(&d_K,qkv_sz); cudaMalloc(&d_V,qkv_sz);
    cudaMalloc(&d_O,qkv_sz); cudaMalloc(&d_L,N*sizeof(float));
    cudaMemcpy(d_Q,h_Q,qkv_sz,cudaMemcpyHostToDevice);
    cudaMemcpy(d_K,h_K,qkv_sz,cudaMemcpyHostToDevice);
    cudaMemcpy(d_V,h_V,qkv_sz,cudaMemcpyHostToDevice);

    printf("--- Correctness Check (N=%d) ---\n", N);
    flash_attention_kernel<<<dim3(Tr),dim3(Br),smem_size>>>(
        d_Q,d_K,d_V,d_O,d_L,N,d,Bc,scale);
    cudaDeviceSynchronize();
    cudaMemcpy(h_O,d_O,qkv_sz,cudaMemcpyDeviceToHost);

    printf("Running CPU reference...\n");
    naive_attention_cpu(h_Q,h_K,h_V,h_O_ref,N,d);

    float max_err=0.0f; int errors=0;
    for(int i=0;i<N*d;i++){
        float err=fabsf(h_O[i]-h_O_ref[i]);
        if(err>max_err) max_err=err;
        if(err>1e-4f) errors++;
    }
    printf("Max absolute error : %.2e\n", max_err);
    printf("Elements > 1e-4 err: %d / %d\n", errors, N*d);
    printf("Result             : %s\n\n",(max_err<1e-4f)?"PASS":"FAIL");

    /* ---- Performance benchmark ---- */
    int N2=1024, Tr2=(N2+Br-1)/Br;
    size_t sz2=(size_t)N2*d*sizeof(float);
    float *h_Q2=(float*)malloc(sz2),*h_K2=(float*)malloc(sz2),*h_V2=(float*)malloc(sz2);
    for(int i=0;i<N2*d;i++){
        h_Q2[i]=((float)rand()/RAND_MAX)*2.0f-1.0f;
        h_K2[i]=((float)rand()/RAND_MAX)*2.0f-1.0f;
        h_V2[i]=((float)rand()/RAND_MAX)*2.0f-1.0f;
    }
    float *d_Q2,*d_K2,*d_V2,*d_O2,*d_L2;
    cudaMalloc(&d_Q2,sz2);cudaMalloc(&d_K2,sz2);cudaMalloc(&d_V2,sz2);
    cudaMalloc(&d_O2,sz2);cudaMalloc(&d_L2,N2*sizeof(float));
    cudaMemcpy(d_Q2,h_Q2,sz2,cudaMemcpyHostToDevice);
    cudaMemcpy(d_K2,h_K2,sz2,cudaMemcpyHostToDevice);
    cudaMemcpy(d_V2,h_V2,sz2,cudaMemcpyHostToDevice);

    float scale2=1.0f/sqrtf((float)d);
    for(int w=0;w<3;w++)
        flash_attention_kernel<<<dim3(Tr2),dim3(Br),smem_size>>>(
            d_Q2,d_K2,d_V2,d_O2,d_L2,N2,d,Bc,scale2);
    cudaDeviceSynchronize();

    printf("--- Performance Benchmark (N=%d, d=%d) ---\n", N2, d);
    cudaEvent_t start,stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    const int ITERS=100;
    cudaEventRecord(start);
    for(int it=0;it<ITERS;it++)
        flash_attention_kernel<<<dim3(Tr2),dim3(Br),smem_size>>>(
            d_Q2,d_K2,d_V2,d_O2,d_L2,N2,d,Bc,scale2);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms=0.0f;
    cudaEventElapsedTime(&ms,start,stop);
    float ms_iter=ms/ITERS;
    double flops=4.0*(double)N2*(double)N2*(double)d;
    double gflops=flops/(ms_iter*1e-3)/1e9;

    printf("Grid: %d blocks  Block: %d threads\n", Tr2, Br);
    printf("Average kernel time : %.4f ms\n", ms_iter);
    printf("Approx throughput   : %.1f GFLOPS\n\n", gflops);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_Q);cudaFree(d_K);cudaFree(d_V);cudaFree(d_O);cudaFree(d_L);
    cudaFree(d_Q2);cudaFree(d_K2);cudaFree(d_V2);cudaFree(d_O2);cudaFree(d_L2);
    free(h_Q);free(h_K);free(h_V);free(h_O);free(h_O_ref);free(h_L);
    free(h_Q2);free(h_K2);free(h_V2);

    printf("Week 4 CUDA complete.\n");
    return 0;
}
"""


@app.function(image=cuda_image, gpu="H100", timeout=600)
def run_gpu():
    import subprocess

    # ---- Task 1: Sequential C ----
    print("=" * 60)
    print("TASK 1: Sequential C (FlashAttention-2 Algorithm 1)")
    print("=" * 60)
    with open("flash_attention_c.c", "w") as f:
        f.write(c_source_code)

    result = subprocess.run(
        ["gcc", "-O3", "-o", "flash_attention_c", "flash_attention_c.c", "-lm"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("C compilation failed:")
        print(result.stderr)
    else:
        print("C compilation successful.\n")
        result = subprocess.run(["./flash_attention_c"], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Stderr:", result.stderr)

    # ---- Task 2: CUDA ----
    print("=" * 60)
    print("TASK 2: Parallelized CUDA (FlashAttention-2 Algorithm 1)")
    print("=" * 60)
    with open("flash_attention_cuda.cu", "w") as f:
        f.write(cuda_source_code)

    result = subprocess.run(
        ["nvcc", "-O3", "-o", "flash_attention_cuda", "flash_attention_cuda.cu"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("CUDA compilation failed:")
        print(result.stderr)
    else:
        print("CUDA compilation successful.\n")
        result = subprocess.run(["./flash_attention_cuda"], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Stderr:", result.stderr)


@app.local_entrypoint()
def main():
    run_gpu.remote()
