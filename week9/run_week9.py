import modal
import os

app = modal.App("week09-deepseek-moe-thunderkittens")

# CUDA 12.8 devel image required by ThunderKittens (C++20, sm_90, CUDA 12.8+)
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .run_commands(
        "apt-get update -qq && apt-get install -y -qq gcc g++ make git",
        "pip install torch --index-url https://download.pytorch.org/whl/cu128",
        "pip install numpy",
        # Clone ThunderKittens — header-only, no build step needed
        "git clone --depth=1 https://github.com/HazyResearch/ThunderKittens /opt/ThunderKittens",
    )
)

# ============================================================
# PART 1: PyTorch reference — generates test_data.h
#
# Config (same as Week 8 for fair comparison):
#   hidden_size=16, intermediate_size=8
#   n_routed_experts=8, top_k=2, n_shared_experts=1, T_TOK=8
#
# Note: TK tile size is 64x64, so TK kernels are only used for
# the large benchmark (H=512, I=256, T=512).
# Correctness tests use naive CUDA kernels (any H, I size).
# ============================================================
test_gen_script = """
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(0)

H      = 16   # hidden_size
I      = 8    # moe_intermediate_size (per routed expert)
N_EXP  = 8    # n_routed_experts
TOP_K  = 2    # num_experts_per_tok
N_SH   = 1    # n_shared_experts
SI     = I * N_SH   # shared intermediate = 8
T_TOK  = 8    # total tokens

class DeepseekV3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn    = nn.SiLU()
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class DeepseekV3TopkRouter(nn.Module):
    def __init__(self, hidden_size, n_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.weight = nn.Parameter(torch.empty(n_experts, hidden_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.e_score_correction_bias = nn.Parameter(torch.zeros(n_experts))
    def forward(self, x_flat):
        logits = F.linear(x_flat, self.weight, None)
        scores = logits.sigmoid()
        topk_weight, topk_idx = torch.topk(
            scores + self.e_score_correction_bias,
            self.top_k, dim=-1, sorted=False
        )
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
        return topk_idx, topk_weight

torch.manual_seed(42)
router     = DeepseekV3TopkRouter(H, N_EXP, TOP_K)
experts    = nn.ModuleList([DeepseekV3MLP(H, I) for _ in range(N_EXP)])
shared_exp = DeepseekV3MLP(H, SI)

torch.manual_seed(99)
x = torch.randn(T_TOK, H)

with torch.no_grad():
    exp_mlp_outs = [experts[e](x[0:1]).squeeze(0) for e in range(N_EXP)]
    topk_idx, topk_weight = router(x)
    moe_outs = []
    for t in range(T_TOK):
        xt  = x[t].unsqueeze(0)
        out = torch.zeros(H)
        for ki in range(TOP_K):
            eidx = topk_idx[t, ki].item()
            w    = topk_weight[t, ki].item()
            out += w * experts[eidx](xt).squeeze(0)
        out += shared_exp(xt).squeeze(0)
        moe_outs.append(out)

def arr_c(arr, name):
    vals = ", ".join(f"{v:.8f}f" for v in arr.detach().numpy().flatten())
    return f"static const float {name}[] = {{{vals}}};"

def int_arr_c(arr, name):
    vals = ", ".join(str(int(v)) for v in arr.detach().numpy().flatten())
    return f"static const int {name}[] = {{{vals}}};"

lines = [
    "// Auto-generated test data — DeepSeekV3 MoE (Week 9, ThunderKittens)",
    f"#define HIDDEN_SZ   {H}",
    f"#define I_SZ        {I}",
    f"#define SI_SZ       {SI}",
    f"#define N_EXP       {N_EXP}",
    f"#define TOP_K_      {TOP_K}",
    f"#define T_TOK       {T_TOK}",
    "",
    arr_c(router.weight,                  "ROUTER_W"),
    arr_c(router.e_score_correction_bias, "ROUTER_BIAS"),
]
for e in range(N_EXP):
    lines += [
        arr_c(experts[e].gate_proj.weight, f"E{e}_GW"),
        arr_c(experts[e].up_proj.weight,   f"E{e}_UW"),
        arr_c(experts[e].down_proj.weight, f"E{e}_DW"),
    ]
lines += [
    arr_c(shared_exp.gate_proj.weight, "SH_GW"),
    arr_c(shared_exp.up_proj.weight,   "SH_UW"),
    arr_c(shared_exp.down_proj.weight, "SH_DW"),
    arr_c(x, "INPUT"),
]
for e in range(N_EXP):
    lines.append(arr_c(exp_mlp_outs[e], f"EXP_OUT_{e}"))
lines.append(int_arr_c(topk_idx,   "ROUTER_IDX"))
lines.append(arr_c(topk_weight,    "ROUTER_WGT"))
for t in range(T_TOK):
    lines.append(arr_c(moe_outs[t], f"MOE_OUT_{t}"))

print("\\n".join(lines))
"""

# ============================================================
# PART 2: CUDA + ThunderKittens implementation
#
# Three sections:
#   A. Naive CUDA kernels  — correctness test, H=16, I=8 (float32)
#   B. TK GEMM kernel      — warp::mma_AB tensor cores, bf16, 64x64 tiles
#   C. Benchmark           — TK vs naive bf16 FFN (H=512, I=256, T=512)
#
# ThunderKittens (TK) architecture on H100:
#   - st_bf<R,C>  : shared memory bf16 tile (SMEM)
#   - rt_bf<R,C>  : register bf16 tile
#   - rt_fl<R,C>  : register float32 accumulator
#   - gl<...>     : global memory layout descriptor for TMA
#   - gl<...>     : global memory layout descriptor (tile-addressed)
#   - warp::mma_AB : WMMA tensor core matmul
#
# Weight layout for TK GEMM C = A @ B:
#   PyTorch Linear weight = [out, in]; we need [K, N]
#   → pre-transpose in Python: Wg_T[H,I], Wu_T[H,I], Wd_T[I,H]
# ============================================================
cuda_source = r"""
/*
 * Week 9: DeepSeekV3 MoE — ThunderKittens (WMMA Tensor Cores)
 *
 * Section A: Naive CUDA float32 kernels (correctness, H=16, I=8)
 * Section B: TK bf16 GEMM kernel (warp::mma_AB tensor cores, 64x64 tiles)
 * Section C: Benchmark — TK vs naive bf16 FFN on H=512, I=256, T=512
 *
 * Build:
 *   nvcc -O3 -std=c++20 -arch=sm_90 \
 *        -I/opt/ThunderKittens/include \
 *        moe_tk.cu -o moe_tk
 */

/* ── Section B requires ThunderKittens ────────────────────────────────────── */
#include "kittens.cuh"
using namespace kittens;

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "test_data.h"

/* ── Error-check macros ──────────────────────────────────────────────────── */
#define CUDA_CHECK(call) do {                                              \
    cudaError_t _e = (call);                                              \
    if (_e != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error %s:%d  %s\n",                        \
                __FILE__, __LINE__, cudaGetErrorString(_e));              \
        exit(1);                                                          \
    }                                                                     \
} while(0)

/* ── Derived constants ───────────────────────────────────────────────────── */
#define H_SZ  HIDDEN_SZ
#define I_DIM I_SZ
#define NE    N_EXP
#define K_TOP TOP_K_
#define T     T_TOK

/* ══════════════════════════════════════════════════════════════════════════
 * SECTION A: Naive CUDA kernels for correctness test (single GPU, float32)
 * ══════════════════════════════════════════════════════════════════════════ */

__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + expf(-x));
}
__device__ __forceinline__ float sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/* Router: sigmoid(logits) + bias → topk → normalize */
__global__ void router_kernel(
    const float *d_x,    /* [T, H] */
    const float *d_rw,   /* [NE, H] */
    const float *d_rb,   /* [NE] */
    int         *d_idx,  /* [T, K] */
    float       *d_wt,   /* [T, K] */
    int t_loc, int h, int ne, int k_top)
{
    int tok = blockIdx.x;
    if (tok >= t_loc) return;

    __shared__ float scores[N_EXP];
    const float *x = d_x + tok * h;

    for (int e = threadIdx.x; e < ne; e += blockDim.x) {
        float dot = 0.0f;
        for (int j = 0; j < h; j++) dot += d_rw[e * h + j] * x[j];
        scores[e] = sigmoid_f(dot) + d_rb[e];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        bool  used[N_EXP] = {};
        int   sel[TOP_K_];
        float wt [TOP_K_];
        for (int ki = 0; ki < k_top; ki++) {
            float best = -1e30f; int bi = -1;
            for (int e = 0; e < ne; e++) {
                if (!used[e] && scores[e] > best) { best = scores[e]; bi = e; }
            }
            sel[ki] = bi; wt[ki] = scores[bi]; used[bi] = true;
        }
        float wsum = 0.0f;
        for (int ki = 0; ki < k_top; ki++) wsum += wt[ki];
        for (int ki = 0; ki < k_top; ki++) {
            d_idx[tok * k_top + ki] = sel[ki];
            d_wt [tok * k_top + ki] = wt[ki] / wsum;
        }
    }
}

/* Expert MLP: out = down(silu(gate(x)) * up(x)) — float32 naive */
__global__ void expert_mlp_kernel(
    const float *d_x,   /* [1, H] */
    const float *d_wg,  /* [I, H] gate weight */
    const float *d_wu,  /* [I, H] up weight */
    const float *d_wd,  /* [H, I] down weight */
    float       *d_out, /* [1, H] */
    int h, int i_dim)
{
    /* one block, threads cover intermediate dimension */
    extern __shared__ float smem[];   /* gate[I] | up[I] | hid[I] */
    float *s_gate = smem;
    float *s_up   = smem + i_dim;
    float *s_hid  = smem + 2 * i_dim;

    /* gate and up projections */
    for (int i = threadIdx.x; i < i_dim; i += blockDim.x) {
        float g = 0.0f, u = 0.0f;
        for (int j = 0; j < h; j++) {
            g += d_wg[i * h + j] * d_x[j];
            u += d_wu[i * h + j] * d_x[j];
        }
        s_gate[i] = g; s_up[i] = u;
        s_hid[i]  = silu_f(g) * u;
    }
    __syncthreads();

    /* down projection */
    for (int j = threadIdx.x; j < h; j += blockDim.x) {
        float val = 0.0f;
        for (int i = 0; i < i_dim; i++) val += d_wd[j * i_dim + i] * s_hid[i];
        d_out[j] = val;
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 * SECTION B: ThunderKittens GEMM kernel — TMA + WMMA tensor cores
 *
 * Computes: C[M,N] = A[M,K] @ B[K,N]  (all bf16, 64x64 tiles)
 *
 * One warp (32 threads) per block.
 * TMA loads 64x64 bf16 tiles asynchronously into SMEM.
 * warp::mma_AB uses WMMA for tensor-core multiply-accumulate.
 *
 * B must be in column layout for mma_AB; swap_layout converts row→col.
 * ══════════════════════════════════════════════════════════════════════════ */

static constexpr int TILE = 64;
using stile  = st_bf<TILE, TILE>;
using gltype = gl<bf16, 1, 1, -1, -1, stile>;

/*
 * TK GEMM kernel — uses warp::load/store with gl global layout descriptors.
 *
 * This avoids TMA (which requires hardware descriptor setup outside the kernel)
 * and instead uses ThunderKittens' warp-level global-memory load/store with
 * the gl<> layout type, which handles the tile-coordinate addressing.
 *
 * warp::load(reg_tile, gl, {row_tile, col_tile})  — global → registers
 * warp::mma_AB(acc, A_row, B_col, acc)            — WMMA tensor core GEMM
 * copy(bf16_tile, fl_tile)                         — fp32 → bf16 downcast
 * warp::store(gl, reg_tile, {row_tile, col_tile}) — registers → global
 */
/*
 * Output uses float32 global layout to directly receive the rt_fl accumulator.
 * A separate fp32→bf16 kernel converts the benchmark output.
 *
 * The st_fl<TILE,TILE> shared tile type is needed for warp::store with gl<float>.
 */
using stile_fl  = st_fl<TILE, TILE>;
using gltype_fl = gl<float, 1, 1, -1, -1, stile_fl>;

__global__ __launch_bounds__(32, 4)
void tk_gemm_kernel(
    const __grid_constant__ gltype    A,      /* bf16 input */
    const __grid_constant__ gltype    B,      /* bf16 input */
                            gltype_fl C,      /* float32 output (no __grid_constant__ — mutable) */
    int K_tiles)
{
    int m = blockIdx.x, n = blockIdx.y;

    /* Float32 accumulator — avoids precision loss across K tiles */
    rt_fl<TILE, TILE> acc;
    warp::zero(acc);

    for (int k = 0; k < K_tiles; k++) {
        /* Load bf16 tiles from global layout directly into registers */
        rt_bf<TILE, TILE> Ar, Br;
        rt_bf<TILE, TILE, ducks::rt_layout::col> Bc;

        warp::load(Ar, A, {m, k});   /* global bf16 → registers (row layout)   */
        warp::load(Br, B, {k, n});   /* global bf16 → registers (row layout)   */
        warp::swap_layout(Bc, Br);   /* row → col layout for mma_AB B operand  */

        /* Tensor-core multiply-accumulate: acc += Ar @ Bc (fp32 accumulator) */
        warp::mma_AB(acc, Ar, Bc, acc);
    }

    /* Store fp32 accumulator directly to global float32 layout */
    warp::store(C, acc, {m, n});
}

/* ── TK Expert MLP: 3 GEMMs + silu_mul (bf16, large dims) ─────────────── */

/* Elementwise: out[i] = silu(gate[i]) * up[i]  (bf16) */
__global__ void silu_mul_bf16(
    const __nv_bfloat16 *gate,
    const __nv_bfloat16 *up,
          __nv_bfloat16 *out, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float g = __bfloat162float(gate[i]);
    float u = __bfloat162float(up[i]);
    out[i] = __float2bfloat16((g / (1.0f + expf(-g))) * u);
}

/* Float32 → bf16 element conversion (post-GEMM, TK outputs fp32) */
__global__ void fp32_to_bf16(const float *src, __nv_bfloat16 *dst, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = __float2bfloat16(src[i]);
}

/*
 * Launch TK GEMM: C[M,N] = A[M,K] @ B[K,N], all dims multiples of TILE=64.
 *
 * The kernel accumulates in fp32 and writes to a temporary float32 buffer.
 * A follow-up kernel converts float32 → bf16 for the final output.
 */
static void launch_tk_gemm(
    __nv_bfloat16 *A_ptr, int M, int K,
    __nv_bfloat16 *B_ptr,          int N,
    __nv_bfloat16 *C_ptr,
    cudaStream_t stream)
{
    /* Temporary float32 output buffer */
    float *C_fl = nullptr;
    CUDA_CHECK(cudaMalloc(&C_fl, (size_t)M * N * sizeof(float)));

    gltype    A_gl(A_ptr, nullptr, nullptr, M, K);
    gltype    B_gl(B_ptr, nullptr, nullptr, K, N);
    gltype_fl C_fl_gl(C_fl, nullptr, nullptr, M, N);

    dim3 grid(M / TILE, N / TILE);
    tk_gemm_kernel<<<grid, 32, 0, stream>>>(A_gl, B_gl, C_fl_gl, K / TILE);
    CUDA_CHECK(cudaGetLastError());

    /* Convert fp32 output → bf16 */
    fp32_to_bf16<<<(M*N+255)/256, 256, 0, stream>>>(C_fl, C_ptr, M * N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(C_fl));
}

/*
 * TK Expert MLP (bf16, tensor cores):
 *
 * Weights are pre-transposed so TK GEMM C=A@B works directly:
 *   Wg_T[H, I]   W_gate.T  →  gate = x[T,H] @ Wg_T[H,I] = [T,I]
 *   Wu_T[H, I]   W_up.T    →  up   = x[T,H] @ Wu_T[H,I] = [T,I]
 *   Wd_T[I, H]   W_down.T  →  out  = hid[T,I] @ Wd_T[I,H] = [T,H]
 */
static void tk_expert_mlp(
    __nv_bfloat16 *d_x,    /* [T, H] input */
    __nv_bfloat16 *d_wgT,  /* [H, I] gate weight (pre-transposed) */
    __nv_bfloat16 *d_wuT,  /* [H, I] up weight   (pre-transposed) */
    __nv_bfloat16 *d_wdT,  /* [I, H] down weight (pre-transposed) */
    __nv_bfloat16 *d_gate, /* [T, I] scratch */
    __nv_bfloat16 *d_up,   /* [T, I] scratch */
    __nv_bfloat16 *d_hid,  /* [T, I] scratch */
    __nv_bfloat16 *d_out,  /* [T, H] output */
    int T_dim, int H_dim, int I_dim,
    cudaStream_t stream)
{
    /* GEMM 1: gate[T,I] = x[T,H] @ Wg_T[H,I] */
    launch_tk_gemm(d_x, T_dim, H_dim, d_wgT, I_dim, d_gate, stream);
    /* GEMM 2: up[T,I]   = x[T,H] @ Wu_T[H,I] */
    launch_tk_gemm(d_x, T_dim, H_dim, d_wuT, I_dim, d_up,   stream);
    /* Elementwise: hid[T,I] = silu(gate) * up */
    silu_mul_bf16<<<(T_dim*I_dim+255)/256, 256, 0, stream>>>(
        d_gate, d_up, d_hid, T_dim * I_dim);
    CUDA_CHECK(cudaGetLastError());
    /* GEMM 3: out[T,H] = hid[T,I] @ Wd_T[I,H] */
    launch_tk_gemm(d_hid, T_dim, I_dim, d_wdT, H_dim, d_out, stream);
}

/* ══════════════════════════════════════════════════════════════════════════
 * SECTION C: Naive bf16 FFN for benchmark comparison
 *
 * Same FFN structure (gate, up, silu_mul, down) but uses plain CUDA threads
 * without tensor cores or TMA — exposes the speedup from TK.
 * ══════════════════════════════════════════════════════════════════════════ */

/* Naive bf16 GEMM: C[M,N] = A[M,K] @ B[K,N] — one thread per output element */
__global__ void naive_gemm_bf16(
    const __nv_bfloat16 *A, const __nv_bfloat16 *B,
          __nv_bfloat16 *C,
    int M, int K, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; k++)
        sum += __bfloat162float(A[row*K + k]) * __bfloat162float(B[k*N + col]);
    C[row*N + col] = __float2bfloat16(sum);
}

static void naive_ffn_bf16(
    __nv_bfloat16 *d_x,
    __nv_bfloat16 *d_wgT, __nv_bfloat16 *d_wuT, __nv_bfloat16 *d_wdT,
    __nv_bfloat16 *d_gate, __nv_bfloat16 *d_up, __nv_bfloat16 *d_hid,
    __nv_bfloat16 *d_out,
    int T_dim, int H_dim, int I_dim, cudaStream_t stream)
{
    dim3 blk(16, 16);
    dim3 gTI((T_dim+15)/16, (I_dim+15)/16);
    dim3 gTH((T_dim+15)/16, (H_dim+15)/16);

    naive_gemm_bf16<<<gTI, blk, 0, stream>>>(d_x, d_wgT, d_gate, T_dim, H_dim, I_dim);
    naive_gemm_bf16<<<gTI, blk, 0, stream>>>(d_x, d_wuT, d_up,   T_dim, H_dim, I_dim);
    silu_mul_bf16<<<(T_dim*I_dim+255)/256, 256, 0, stream>>>(
        d_gate, d_up, d_hid, T_dim*I_dim);
    naive_gemm_bf16<<<gTH, blk, 0, stream>>>(d_hid, d_wdT, d_out, T_dim, I_dim, H_dim);
}

/* ══════════════════════════════════════════════════════════════════════════
 * CORRECTNESS TEST: single-GPU float32, H=16, I=8, T=8
 * (mirrors Week 7/8 logic; validates MoE routing + expert FFN)
 * ══════════════════════════════════════════════════════════════════════════ */

static void run_correctness_test(void) {
    printf("\n=== Correctness Test (naive CUDA, H=%d, I=%d, T=%d) ===\n",
           H_SZ, I_DIM, T);

    /* ── allocate device memory ── */
    float *d_x, *d_rw, *d_rb;
    int   *d_idx;
    float *d_wt, *d_exp_out;
    float *e_wg[N_EXP], *e_wu[N_EXP], *e_wd[N_EXP];
    float *d_sh_wg, *d_sh_wu, *d_sh_wd, *d_sh_out;
    float *d_moe_out;

    CUDA_CHECK(cudaMalloc(&d_x,      T * H_SZ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rw,     NE * H_SZ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rb,     NE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_idx,    T * K_TOP * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_wt,     T * K_TOP * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_exp_out,H_SZ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_moe_out,T * H_SZ * sizeof(float)));
    for (int e = 0; e < NE; e++) {
        CUDA_CHECK(cudaMalloc(&e_wg[e], I_DIM * H_SZ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&e_wu[e], I_DIM * H_SZ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&e_wd[e], H_SZ * I_DIM * sizeof(float)));
    }
    CUDA_CHECK(cudaMalloc(&d_sh_wg, I_DIM * H_SZ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sh_wu, I_DIM * H_SZ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sh_wd, H_SZ * I_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sh_out,H_SZ * sizeof(float)));

    /* ── copy constants ── */
    CUDA_CHECK(cudaMemcpy(d_x,  INPUT,       T*H_SZ*sizeof(float),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rw, ROUTER_W,    NE*H_SZ*sizeof(float),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rb, ROUTER_BIAS, NE*sizeof(float),        cudaMemcpyHostToDevice));

    const float *ew_g[] = {E0_GW,E1_GW,E2_GW,E3_GW,E4_GW,E5_GW,E6_GW,E7_GW};
    const float *ew_u[] = {E0_UW,E1_UW,E2_UW,E3_UW,E4_UW,E5_UW,E6_UW,E7_UW};
    const float *ew_d[] = {E0_DW,E1_DW,E2_DW,E3_DW,E4_DW,E5_DW,E6_DW,E7_DW};
    for (int e = 0; e < NE; e++) {
        CUDA_CHECK(cudaMemcpy(e_wg[e], ew_g[e], I_DIM*H_SZ*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(e_wu[e], ew_u[e], I_DIM*H_SZ*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(e_wd[e], ew_d[e], H_SZ*I_DIM*sizeof(float), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_sh_wg, SH_GW, I_DIM*H_SZ*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sh_wu, SH_UW, I_DIM*H_SZ*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sh_wd, SH_DW, H_SZ*I_DIM*sizeof(float), cudaMemcpyHostToDevice));

    /* ── router ── */
    router_kernel<<<T, 32, 0, 0>>>(
        d_x, d_rw, d_rb, d_idx, d_wt, T, H_SZ, NE, K_TOP);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* ── per-token MoE forward ── */
    int   smem = 3 * I_DIM * sizeof(float);
    float h_moe_out[T][H_SZ];

    for (int tok = 0; tok < T; tok++) {
        /* read routing for this token */
        int   h_idx[K_TOP];
        float h_wt [K_TOP];
        CUDA_CHECK(cudaMemcpy(h_idx, d_idx + tok*K_TOP, K_TOP*sizeof(int),   cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_wt,  d_wt  + tok*K_TOP, K_TOP*sizeof(float), cudaMemcpyDeviceToHost));

        float acc[H_SZ] = {};

        /* run each selected expert */
        for (int ki = 0; ki < K_TOP; ki++) {
            int eid = h_idx[ki];
            expert_mlp_kernel<<<1, 32, smem>>>(
                d_x + tok*H_SZ, e_wg[eid], e_wu[eid], e_wd[eid],
                d_exp_out, H_SZ, I_DIM);
            CUDA_CHECK(cudaDeviceSynchronize());
            float h_eout[H_SZ];
            CUDA_CHECK(cudaMemcpy(h_eout, d_exp_out, H_SZ*sizeof(float), cudaMemcpyDeviceToHost));
            for (int j = 0; j < H_SZ; j++) acc[j] += h_wt[ki] * h_eout[j];
        }

        /* shared expert */
        expert_mlp_kernel<<<1, 32, smem>>>(
            d_x + tok*H_SZ, d_sh_wg, d_sh_wu, d_sh_wd,
            d_sh_out, H_SZ, I_DIM);
        CUDA_CHECK(cudaDeviceSynchronize());
        float h_sh[H_SZ];
        CUDA_CHECK(cudaMemcpy(h_sh, d_sh_out, H_SZ*sizeof(float), cudaMemcpyDeviceToHost));
        for (int j = 0; j < H_SZ; j++) acc[j] += h_sh[j];

        for (int j = 0; j < H_SZ; j++) h_moe_out[tok][j] = acc[j];
    }

    /* ── verify against PyTorch reference ── */
    const float *refs[] = {
        MOE_OUT_0, MOE_OUT_1, MOE_OUT_2, MOE_OUT_3,
        MOE_OUT_4, MOE_OUT_5, MOE_OUT_6, MOE_OUT_7
    };
    int all_pass = 1;
    for (int tok = 0; tok < T; tok++) {
        float max_err = 0.0f;
        for (int j = 0; j < H_SZ; j++) {
            float err = fabsf(h_moe_out[tok][j] - refs[tok][j]);
            if (err > max_err) max_err = err;
        }
        const char *status = (max_err < 1e-3f) ? "PASS" : "FAIL";
        if (max_err >= 1e-3f) all_pass = 0;
        printf("  token %d  max_err=%.2e  %s\n", tok, max_err, status);
    }
    printf("\n%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    /* cleanup */
    cudaFree(d_x); cudaFree(d_rw); cudaFree(d_rb);
    cudaFree(d_idx); cudaFree(d_wt); cudaFree(d_exp_out); cudaFree(d_moe_out);
    for (int e = 0; e < NE; e++) { cudaFree(e_wg[e]); cudaFree(e_wu[e]); cudaFree(e_wd[e]); }
    cudaFree(d_sh_wg); cudaFree(d_sh_wu); cudaFree(d_sh_wd); cudaFree(d_sh_out);
}

/* ══════════════════════════════════════════════════════════════════════════
 * BENCHMARK: TK bf16 tensor cores vs naive bf16 CUDA
 *
 * Config: T=512, H=512, I=256  (all multiples of TILE=64)
 * Simulates single-expert FFN for N_ITER iterations.
 * Reports throughput (tokens/sec) and TFLOPs.
 * ══════════════════════════════════════════════════════════════════════════ */

static void run_benchmark(void) {
    const int T_B  = 512;   /* tokens */
    const int H_B  = 512;   /* hidden_size */
    const int I_B  = 256;   /* intermediate_size */
    const int N_IT = 100;   /* iterations for timing */

    printf("\nBenchmark: T=%d tokens, H=%d, I=%d, single expert, %d iters\n",
           T_B, H_B, I_B, N_IT);
    printf("------------------------------------------------------------\n");

    /* FLOPs per FFN forward pass (3 GEMMs, ignoring silu_mul): */
    /* gate: T*H*I*2, up: T*H*I*2, down: T*I*H*2 */
    double flops = 3.0 * 2.0 * T_B * H_B * I_B;

    size_t xsz   = (size_t)T_B * H_B * sizeof(__nv_bfloat16);
    size_t wgsz  = (size_t)H_B * I_B * sizeof(__nv_bfloat16);  /* [H,I] transposed */
    size_t wdsz  = (size_t)I_B * H_B * sizeof(__nv_bfloat16);  /* [I,H] transposed */
    size_t gatesz= (size_t)T_B * I_B * sizeof(__nv_bfloat16);
    size_t outsz = (size_t)T_B * H_B * sizeof(__nv_bfloat16);

    __nv_bfloat16 *d_x, *d_wgT, *d_wuT, *d_wdT;
    __nv_bfloat16 *d_gate, *d_up, *d_hid, *d_out;

    CUDA_CHECK(cudaMalloc(&d_x,    xsz));
    CUDA_CHECK(cudaMalloc(&d_wgT,  wgsz));
    CUDA_CHECK(cudaMalloc(&d_wuT,  wgsz));
    CUDA_CHECK(cudaMalloc(&d_wdT,  wdsz));
    CUDA_CHECK(cudaMalloc(&d_gate, gatesz));
    CUDA_CHECK(cudaMalloc(&d_up,   gatesz));
    CUDA_CHECK(cudaMalloc(&d_hid,  gatesz));
    CUDA_CHECK(cudaMalloc(&d_out,  outsz));

    /* Fill with random bf16 values */
    {
        float *h_tmp = (float*)malloc(xsz * 2 / sizeof(__nv_bfloat16) * sizeof(float));
        srand(42);
        auto randf16 = [&](size_t n, __nv_bfloat16 *dst) {
            for (size_t i = 0; i < n; i++) {
                float v = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
                dst[i]  = __float2bfloat16(v);
            }
        };
        /* use host-side random fill then memcpy */
        __nv_bfloat16 *h_x    = (__nv_bfloat16*)malloc(xsz);
        __nv_bfloat16 *h_wgT  = (__nv_bfloat16*)malloc(wgsz);
        __nv_bfloat16 *h_wuT  = (__nv_bfloat16*)malloc(wgsz);
        __nv_bfloat16 *h_wdT  = (__nv_bfloat16*)malloc(wdsz);
        randf16(T_B*H_B, h_x); randf16(H_B*I_B, h_wgT);
        randf16(H_B*I_B, h_wuT); randf16(I_B*H_B, h_wdT);
        CUDA_CHECK(cudaMemcpy(d_x,   h_x,   xsz,  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_wgT, h_wgT, wgsz, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_wuT, h_wuT, wgsz, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_wdT, h_wdT, wdsz, cudaMemcpyHostToDevice));
        free(h_x); free(h_wgT); free(h_wuT); free(h_wdT); free(h_tmp);
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    /* ── Benchmark TK (tensor cores + TMA) ─────────────────────────────── */
    /* warm-up */
    for (int i = 0; i < 5; i++)
        tk_expert_mlp(d_x, d_wgT, d_wuT, d_wdT, d_gate, d_up, d_hid, d_out,
                      T_B, H_B, I_B, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaEventRecord(t0, stream));
    for (int i = 0; i < N_IT; i++)
        tk_expert_mlp(d_x, d_wgT, d_wuT, d_wdT, d_gate, d_up, d_hid, d_out,
                      T_B, H_B, I_B, stream);
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float ms_tk = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_tk, t0, t1));
    ms_tk /= N_IT;

    double tflops_tk   = flops / (ms_tk * 1e-3) / 1e12;
    double tokps_tk    = T_B / (ms_tk * 1e-3);

    /* ── Benchmark Naive bf16 CUDA ──────────────────────────────────────── */
    /* warm-up */
    for (int i = 0; i < 5; i++)
        naive_ffn_bf16(d_x, d_wgT, d_wuT, d_wdT, d_gate, d_up, d_hid, d_out,
                       T_B, H_B, I_B, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaEventRecord(t0, stream));
    for (int i = 0; i < N_IT; i++)
        naive_ffn_bf16(d_x, d_wgT, d_wuT, d_wdT, d_gate, d_up, d_hid, d_out,
                       T_B, H_B, I_B, stream);
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float ms_naive = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_naive, t0, t1));
    ms_naive /= N_IT;

    double tflops_naive = flops / (ms_naive * 1e-3) / 1e12;
    double tokps_naive  = T_B / (ms_naive * 1e-3);

    /* ── Print results ──────────────────────────────────────────────────── */
    printf("  TK (tensor cores + TMA)   : %10.0f tok/s   %.2f TFLOPs   (%.3f ms/iter)\n",
           tokps_tk,    tflops_tk,    ms_tk);
    printf("  Naive bf16 CUDA           : %10.0f tok/s   %.2f TFLOPs   (%.3f ms/iter)\n",
           tokps_naive, tflops_naive, ms_naive);
    printf("------------------------------------------------------------\n");
    printf("  Speedup (TK vs Naive)     : %.2fx\n", ms_naive / ms_tk);

    cudaFree(d_x); cudaFree(d_wgT); cudaFree(d_wuT); cudaFree(d_wdT);
    cudaFree(d_gate); cudaFree(d_up); cudaFree(d_hid); cudaFree(d_out);
    cudaStreamDestroy(stream);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}

/* ══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ══════════════════════════════════════════════════════════════════════════ */
int main(void) {
    printf("Week 9: DeepSeekV3 MoE — ThunderKittens (TMA + Tensor Cores)\n");
    printf("=============================================================\n");

    int n_dev = 0;
    cudaGetDeviceCount(&n_dev);
    printf("Found %d GPU(s). Using GPU 0.\n", n_dev);
    cudaSetDevice(0);

    run_correctness_test();
    run_benchmark();
    return 0;
}
"""

# ============================================================
# PART 3: Python benchmark wrapper — MoE vs Dense FFN (tokens/sec)
# Runs after the CUDA binary to measure Python-side PyTorch throughput.
# ============================================================
bench_script = """
import torch
import torch.nn as nn
import time

torch.manual_seed(0)

# Benchmark config
T  = 512    # tokens
H  = 512    # hidden_size
I  = 256    # intermediate_size
N_EXP = 8  # routed experts
TOP_K = 2  # experts per token
ITERS = 200

device = torch.device("cuda")

# ─── MoE forward (simulated data+expert parallelism, single GPU) ───────────
class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(H, I, bias=False, dtype=torch.bfloat16)
        self.up   = nn.Linear(H, I, bias=False, dtype=torch.bfloat16)
        self.down = nn.Linear(I, H, bias=False, dtype=torch.bfloat16)
    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))

experts = nn.ModuleList([Expert() for _ in range(N_EXP)]).to(device)
router  = nn.Linear(H, N_EXP, bias=False, dtype=torch.bfloat16).to(device)
shared  = Expert().to(device)

# ─── Dense FFN (equivalent FLOPs) ─────────────────────────────────────────
# Dense FFN uses I*TOP_K intermediate dim to match MoE FLOPs
I_dense = I * TOP_K
dense_ffn = nn.Sequential(
    nn.Linear(H, I_dense*2, bias=False, dtype=torch.bfloat16),
    nn.SiLU(),
    nn.Linear(I_dense*2, H, bias=False, dtype=torch.bfloat16),
).to(device)

x = torch.randn(T, H, dtype=torch.bfloat16, device=device)

def moe_forward(x):
    logits = router(x).sigmoid()
    topk_w, topk_idx = logits.topk(TOP_K, dim=-1)
    topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
    out = torch.zeros_like(x)
    for ki in range(TOP_K):
        for eid in range(N_EXP):
            mask = (topk_idx[:, ki] == eid)
            if mask.any():
                out[mask] += topk_w[mask, ki:ki+1] * experts[eid](x[mask])
    out += shared(x)
    return out

# warm-up
for _ in range(10): moe_forward(x); torch.cuda.synchronize()
for _ in range(10): dense_ffn(x);  torch.cuda.synchronize()

# ─── Time MoE ──────────────────────────────────────────────────────────────
t0 = time.perf_counter()
for _ in range(ITERS):
    _ = moe_forward(x)
    torch.cuda.synchronize()
t1 = time.perf_counter()
ms_moe = (t1 - t0) / ITERS * 1000
tokps_moe = T / (ms_moe / 1000)

# ─── Time Dense ────────────────────────────────────────────────────────────
t0 = time.perf_counter()
for _ in range(ITERS):
    _ = dense_ffn(x)
    torch.cuda.synchronize()
t1 = time.perf_counter()
ms_dense = (t1 - t0) / ITERS * 1000
tokps_dense = T / (ms_dense / 1000)

print()
print(f"PyTorch Benchmark: T={T} tokens, H={H}, I={I}, {N_EXP} experts, top-{TOP_K}")
print("------------------------------------------------------------")
print(f"  MoE (routed, bf16)    : {tokps_moe:>12.0f} tok/s  ({ms_moe:6.2f} ms/iter)")
print(f"  Dense FFN (equiv)     : {tokps_dense:>12.0f} tok/s  ({ms_dense:6.2f} ms/iter)")
print("------------------------------------------------------------")
"""


@app.function(
    image=image,
    gpu="H100",
    timeout=600,
)
def run_week9():
    import subprocess
    import tempfile
    import os

    print("Week 9: DeepSeekV3 MoE — ThunderKittens (TMA + Tensor Cores)")
    print("=============================================================")

    # Step 1: Generate test_data.h via PyTorch reference
    print("\n[Step 1] Generating test data from PyTorch reference...")
    result = subprocess.run(
        ["python3", "-c", test_gen_script],
        capture_output=True, text=True, check=True
    )
    test_data_h = result.stdout

    # Step 1b: Diagnostic — show TK include structure and key functions
    diag = subprocess.run(
        ["find", "/opt/ThunderKittens/include", "-name", "*.cuh", "-type", "f"],
        capture_output=True, text=True
    )
    all_headers = sorted(diag.stdout.splitlines())
    print("[Diagnostic] ThunderKittens headers:")
    for f in all_headers[:30]:
        print(" ", f)
    if len(all_headers) > 30:
        print(f"  ... and {len(all_headers)-30} more")

    # Grep for copy-related warp functions
    copy_grep = subprocess.run(
        ["grep", "-r", "void copy", "/opt/ThunderKittens/include/", "--include=*.cuh", "-l"],
        capture_output=True, text=True
    )
    print("[Diagnostic] Files defining 'void copy':", copy_grep.stdout.strip())

    # Step 2: Write source files to tmpdir and compile
    print("\n[Step 2] Compiling moe_tk.cu with nvcc + ThunderKittens...")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write test_data.h
        with open(os.path.join(tmpdir, "test_data.h"), "w") as f:
            f.write(test_data_h)

        # Write CUDA source
        cu_path = os.path.join(tmpdir, "moe_tk.cu")
        with open(cu_path, "w") as f:
            f.write(cuda_source)

        # Compile with nvcc targeting sm_90 (H100), C++20, ThunderKittens headers
        bin_path = os.path.join(tmpdir, "moe_tk")
        compile_cmd = [
            "nvcc", "-O3",
            "-std=c++20",
            "-arch=sm_90",
            "-I/opt/ThunderKittens/include",
            f"-I{tmpdir}",
            cu_path,
            "-o", bin_path,
            "--expt-relaxed-constexpr",   # required by TK template metaprogramming
            "--expt-extended-lambda",     # required by TK device lambdas in maps.cuh
            "-Xcompiler", "-fPIC",
        ]
        print("  $", " ".join(compile_cmd))
        comp = subprocess.run(compile_cmd, capture_output=True, text=True)
        if comp.returncode != 0:
            print("COMPILATION FAILED:")
            print(comp.stderr[-4000:])
            raise RuntimeError("nvcc compilation failed")
        print("  Compilation successful.")

        # Step 3: Run correctness + benchmark
        print("\n[Step 3] Running correctness tests and benchmark...")
        run = subprocess.run(
            [bin_path],
            capture_output=True, text=True,
            cwd=tmpdir
        )
        print(run.stdout)
        if run.returncode != 0:
            print("STDERR:", run.stderr[-2000:])
            raise RuntimeError(f"moe_tk exited with code {run.returncode}")

    # Step 4: PyTorch throughput benchmark
    print("\n[Step 4] PyTorch MoE vs Dense FFN throughput benchmark...")
    subprocess.run(["python3", "-c", bench_script], check=True)

    print("\n=== Week 9 complete ===")


@app.local_entrypoint()
def main():
    run_week9.remote()
