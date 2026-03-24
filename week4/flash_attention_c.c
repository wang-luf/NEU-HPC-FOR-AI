// Week 4 - Task 1: FlashAttention-2 Algorithm 1 in Sequential C
// Implements Algorithm 1 from "FlashAttention-2: Faster Attention with Better
// Parallelism and Work Partitioning" (Section 3.1), forward pass only.
//
// Compile: gcc -O3 -o flash_attention_c flash_attention_c.c -lm
// Run:     ./flash_attention_c

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Naive attention (reference): O = softmax(Q*K^T / sqrt(d)) * V
// ---------------------------------------------------------------------------
void naive_attention(const float *Q, const float *K, const float *V,
                     float *O, int N, int d) {
    float scale = 1.0f / sqrtf((float)d);
    float *S = (float *)malloc(N * N * sizeof(float));

    // S = Q * K^T / sqrt(d)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float dot = 0.0f;
            for (int k = 0; k < d; k++)
                dot += Q[i * d + k] * K[j * d + k];
            S[i * N + j] = dot * scale;
        }
    }

    // row-wise softmax
    for (int i = 0; i < N; i++) {
        float max_val = -FLT_MAX;
        for (int j = 0; j < N; j++)
            if (S[i * N + j] > max_val) max_val = S[i * N + j];
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            S[i * N + j] = expf(S[i * N + j] - max_val);
            sum += S[i * N + j];
        }
        for (int j = 0; j < N; j++)
            S[i * N + j] /= sum;
    }

    // O = S * V
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < d; k++) {
            float sum = 0.0f;
            for (int j = 0; j < N; j++)
                sum += S[i * N + j] * V[j * d + k];
            O[i * d + k] = sum;
        }
    }

    free(S);
}

// ---------------------------------------------------------------------------
// FlashAttention-2 Algorithm 1 forward pass
//
// Inputs:  Q, K, V  in R^(N x d)  (row-major, "HBM")
// Outputs: O        in R^(N x d)
//          L        in R^N         (logsumexp for each row)
// Params:  Br = row block size, Bc = col block size
//
// Each iteration of the outer loop simulates one threadblock loading Qi into
// "on-chip SRAM" and iterating through all K/V tiles.
// ---------------------------------------------------------------------------
void flash_attention(const float *Q, const float *K, const float *V,
                     float *O, float *L,
                     int N, int d, int Br, int Bc) {
    float scale = 1.0f / sqrtf((float)d);

    int Tr = (N + Br - 1) / Br;   // number of Q  blocks (outer loop iterations)
    int Tc = (N + Bc - 1) / Bc;   // number of KV blocks (inner loop iterations)

    // "On-chip SRAM" buffers (Algorithm 1, line comments reference paper)
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

    // Line 3: for 1 <= i <= Tr
    for (int i = 0; i < Tr; i++) {
        int row_start = i * Br;
        int actual_Br = (row_start + Br <= N) ? Br : (N - row_start);

        // Line 4: Load Qi block from HBM to SRAM
        for (int r = 0; r < actual_Br; r++)
            for (int k = 0; k < d; k++)
                Qi[r * d + k] = Q[(row_start + r) * d + k];

        // Line 5: Initialize Oi = 0, li = 0, mi = -inf  (on chip)
        for (int r = 0; r < actual_Br; r++) {
            for (int k = 0; k < d; k++) Oi[r * d + k] = 0.0f;
            mi[r] = -FLT_MAX;
            li[r] = 0.0f;
        }

        // Line 6: for 1 <= j <= Tc
        for (int j = 0; j < Tc; j++) {
            int col_start = j * Bc;
            int actual_Bc = (col_start + Bc <= N) ? Bc : (N - col_start);

            // Line 7: Load Kj, Vj from HBM to SRAM
            for (int r = 0; r < actual_Bc; r++)
                for (int k = 0; k < d; k++) {
                    Kj[r * d + k] = K[(col_start + r) * d + k];
                    Vj[r * d + k] = V[(col_start + r) * d + k];
                }

            // Line 8: Sij = Qi * Kj^T / sqrt(d)  [shape Br x Bc]
            for (int r = 0; r < actual_Br; r++)
                for (int c = 0; c < actual_Bc; c++) {
                    float dot = 0.0f;
                    for (int k = 0; k < d; k++)
                        dot += Qi[r * d + k] * Kj[c * d + k];
                    Sij[r * Bc + c] = dot * scale;
                }

            // Line 9: m_ij = max(mi, rowmax(Sij))
            for (int r = 0; r < actual_Br; r++) {
                float row_max = mi[r];
                for (int c = 0; c < actual_Bc; c++)
                    if (Sij[r * Bc + c] > row_max) row_max = Sij[r * Bc + c];
                mij[r] = row_max;
            }

            // Line 9: Pij = exp(Sij - m_ij)  [pointwise, shape Br x Bc]
            for (int r = 0; r < actual_Br; r++)
                for (int c = 0; c < actual_Bc; c++)
                    Pij[r * Bc + c] = expf(Sij[r * Bc + c] - mij[r]);

            // Line 9: l_ij = exp(mi - m_ij) * li + rowsum(Pij)
            for (int r = 0; r < actual_Br; r++) {
                float row_sum = 0.0f;
                for (int c = 0; c < actual_Bc; c++) row_sum += Pij[r * Bc + c];
                lij[r] = expf(mi[r] - mij[r]) * li[r] + row_sum;
            }

            // Line 10: Oi = diag(exp(mi - m_ij))^-1 * Oi + Pij * Vj
            //   Per row r: Oi[r] = Oi[r] * exp(mi[r] - mij[r]) + Pij[r,:] * Vj
            for (int r = 0; r < actual_Br; r++) {
                float rescale = expf(mi[r] - mij[r]);
                for (int k = 0; k < d; k++) {
                    float pv = 0.0f;
                    for (int c = 0; c < actual_Bc; c++)
                        pv += Pij[r * Bc + c] * Vj[c * d + k];
                    Oi[r * d + k] = rescale * Oi[r * d + k] + pv;
                }
            }

            // Update running stats
            for (int r = 0; r < actual_Br; r++) {
                mi[r] = mij[r];
                li[r] = lij[r];
            }
        } // end inner loop (line 11)

        // Line 12: Oi = diag(li)^-1 * Oi  ->  Oi[r,:] /= li[r]
        for (int r = 0; r < actual_Br; r++)
            for (int k = 0; k < d; k++)
                Oi[r * d + k] /= li[r];

        // Line 13: Li = mi + log(li)
        for (int r = 0; r < actual_Br; r++)
            L[row_start + r] = mi[r] + logf(li[r]);

        // Lines 14-15: Write Oi, Li back to HBM
        for (int r = 0; r < actual_Br; r++)
            for (int k = 0; k < d; k++)
                O[(row_start + r) * d + k] = Oi[r * d + k];
    } // end outer loop (line 16)

    free(Qi); free(Kj); free(Vj); free(Oi);
    free(Sij); free(Pij); free(mi); free(li); free(mij); free(lij);
}

int main() {
    printf("Week 4 - Task 1: FlashAttention-2 Sequential C\n");
    printf("================================================\n\n");

    int N = 128, d = 64, Br = 16, Bc = 16;
    printf("Parameters: N=%d, d=%d, Br=%d, Bc=%d\n", N, d, Br, Bc);
    printf("Tr=%d blocks (outer), Tc=%d blocks (inner)\n\n",
           (N + Br - 1) / Br, (N + Bc - 1) / Bc);

    size_t sz = (size_t)N * d * sizeof(float);
    float *Q       = (float *)malloc(sz);
    float *K       = (float *)malloc(sz);
    float *V       = (float *)malloc(sz);
    float *O_flash = (float *)malloc(sz);
    float *O_naive = (float *)malloc(sz);
    float *L       = (float *)malloc(N * sizeof(float));

    srand(42);
    for (int i = 0; i < N * d; i++) {
        Q[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        K[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        V[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    printf("Running FlashAttention-2 Algorithm 1...\n");
    flash_attention(Q, K, V, O_flash, L, N, d, Br, Bc);

    printf("Running naive attention (reference)...\n");
    naive_attention(Q, K, V, O_naive, N, d);

    // Verify
    float max_err = 0.0f;
    int errors = 0;
    for (int i = 0; i < N * d; i++) {
        float err = fabsf(O_flash[i] - O_naive[i]);
        if (err > max_err) max_err = err;
        if (err > 1e-4f) errors++;
    }

    printf("\nVerification Results:\n");
    printf("  Max absolute error : %.2e\n", max_err);
    printf("  Elements > 1e-4 err: %d / %d\n", errors, N * d);
    printf("  Result             : %s\n\n", (max_err < 1e-4f) ? "PASS" : "FAIL");

    // Show first few logsumexp values
    printf("Logsumexp L[0..3]: %.4f  %.4f  %.4f  %.4f\n\n",
           L[0], L[1], L[2], L[3]);

    free(Q); free(K); free(V); free(O_flash); free(O_naive); free(L);
    return 0;
}
