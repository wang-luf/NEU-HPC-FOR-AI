import modal

app = modal.App("week08-deepseek-moe-multigpu")

# GPU image: CUDA 12.4 devel (includes nvcc, NCCL headers/libs) + PyTorch
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .run_commands(
        # NCCL is bundled in the CUDA devel image under /usr/lib/x86_64-linux-gnu
        # and headers under /usr/include. Verify and install PyTorch.
        "apt-get update -qq && apt-get install -y -qq gcc g++ make",
        "pip install torch --index-url https://download.pytorch.org/whl/cu121",
        "pip install numpy",
    )
)

# ============================================================
# PART 1: PyTorch reference — generates test_data.h
#
# Config:
#   hidden_size=16, intermediate_size=8
#   n_routed_experts=8  (4 per GPU with 2 GPUs)
#   top_k=2, n_shared_experts=1, T_TOK=8
#
# Expert parallelism: GPU 0 owns experts {0,1,2,3}
#                     GPU 1 owns experts {4,5,6,7}
# Data parallelism  : tokens split evenly across GPUs
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
    "// Auto-generated test data — DeepSeekV3 MoE (Week 8, multi-GPU)",
    f"#define HIDDEN_SZ   {H}",
    f"#define I_SZ        {I}",
    f"#define SI_SZ       {SI}",
    f"#define N_EXP       {N_EXP}",
    f"#define TOP_K_      {TOP_K}",
    f"#define T_TOK       {T_TOK}",
    f"#define N_GPU       2",
    f"#define EXP_PER_GPU {N_EXP // 2}",
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
# PART 2: Multi-GPU CUDA + NCCL implementation
#
# Buffer layout for AllToAll correctness:
#   d_send / d_recv : [N_GPU, EXP_PER_GPU, SLOTS_PER_EXP, H]
#
#   Dimension breakdown:
#     dim-0 (N_GPU)        = destination GPU index
#     dim-1 (EXP_PER_GPU)  = local expert index on that destination GPU
#     dim-2 (SLOTS_PER_EXP)= token slots per expert (worst case = T_TOK)
#     dim-3 (H)            = hidden vector
#
#   By encoding expert identity in the buffer position (dim-1),
#   the receiving GPU knows exactly which expert processes each slot
#   without any extra metadata transmission.
#
# Forward pass per GPU (rank r):
#   1. router_kernel      : sigmoid logits → topk → normalize
#   2. scatter_kernel     : pack tokens into d_send[dst][lexp][slot]
#   3. ncclAllToAll #1    : dispatch tokens to expert-owning GPUs
#   4. expert_mlp_kernel  : each local expert processes its recv slots
#   5. ncclAllToAll #2    : return results to token-owning GPUs
#   6. gather_kernel      : weighted accumulate from result buffer
#   7. expert_mlp_kernel  : shared expert (always active)
#   8. add shared output  : out += shared_expert(x)
# ============================================================
cuda_source = r"""
/*
 * Week 8: DeepSeekV3 MoE — Multi-GPU CUDA + NCCL
 *
 * Data parallelism  : tokens partitioned across GPUs
 * Expert parallelism: routed experts partitioned across GPUs
 *
 * AllToAll send/recv buffer layout: [N_GPU][EXP_PER_GPU][SLOTS][H]
 *   - Expert identity is encoded in the buffer index (dim 1)
 *   - Receiving GPU knows which expert to apply to each slot
 *   - No extra metadata transmission needed
 *
 * Build:
 *   nvcc -O2 -arch=sm_80 moe_multigpu.cu -lnccl -o moe_multigpu
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/wait.h>

#include <cuda_runtime.h>
#include <nccl.h>

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

#define NCCL_CHECK(call) do {                                             \
    ncclResult_t _r = (call);                                             \
    if (_r != ncclSuccess) {                                              \
        fprintf(stderr, "NCCL error %s:%d  %s\n",                        \
                __FILE__, __LINE__, ncclGetErrorString(_r));              \
        exit(1);                                                          \
    }                                                                     \
} while(0)

/* ── Derived constants ───────────────────────────────────────────────────── */
#define H     HIDDEN_SZ
#define I     I_SZ
#define NE    N_EXP
#define K     TOP_K_
#define T     T_TOK
#define NG    N_GPU
#define EPG   EXP_PER_GPU

/* Tokens per GPU (input data parallelism) */
#define T_LOC  (T / NG)

/* Send/recv buffer: worst-case slots per (dst_gpu, local_expert) pair.
 * Each of T_LOC local tokens can select at most K experts, all potentially
 * on the same GPU and the same expert → bound = K * T_LOC.            */
#define SLOTS  (K * T_LOC)

/* Total floats per GPU-to-GPU slab: EPG experts × SLOTS tokens × H dims */
#define SLAB_F (EPG * SLOTS * H)

/* ── Activations ─────────────────────────────────────────────────────────── */
__device__ __forceinline__ float silu_d(float x) {
    return x / (1.0f + expf(-x));
}
__device__ __forceinline__ float sigmoid_d(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/* ════════════════════════════════════════════════════════════════════════════
 * KERNEL 1: router_kernel
 *
 * One block per token; threads cooperate on the NE dot-products.
 * After __syncthreads(), thread 0 runs serial topk (NE ≤ 8 is tiny).
 *
 * Output: d_tidx[tok, ki] = global expert index (0..NE-1)
 *         d_twt [tok, ki] = normalized weight
 * ════════════════════════════════════════════════════════════════════════════ */
__global__ void router_kernel(
    const float *d_x,      /* [T_LOC, H] */
    const float *d_rw,     /* [NE,    H] router weight */
    const float *d_rb,     /* [NE]       router bias   */
    int         *d_tidx,   /* [T_LOC, K] */
    float       *d_twt,    /* [T_LOC, K] */
    int t_loc)
{
    int tok = blockIdx.x;
    if (tok >= t_loc) return;

    const float *x = d_x + tok * H;
    __shared__ float scores[N_EXP];   /* sigmoid(logit) + bias per expert */

    /* Parallel dot products: each thread handles one or more experts */
    for (int e = threadIdx.x; e < NE; e += blockDim.x) {
        float dot = 0.0f;
        for (int j = 0; j < H; j++)
            dot += d_rw[e * H + j] * x[j];
        scores[e] = sigmoid_d(dot) + d_rb[e];
    }
    __syncthreads();

    /* Thread 0: serial topk + normalize (NE is small) */
    if (threadIdx.x == 0) {
        bool  used[N_EXP] = {};
        int   sel[TOP_K_];
        float wt [TOP_K_];

        for (int ki = 0; ki < K; ki++) {
            int best = -1; float bv = -1e30f;
            for (int e = 0; e < NE; e++)
                if (!used[e] && scores[e] > bv) { bv = scores[e]; best = e; }
            sel[ki] = best;  wt[ki] = scores[best];  used[best] = true;
        }
        float s = 0.0f;
        for (int ki = 0; ki < K; ki++) s += wt[ki];
        for (int ki = 0; ki < K; ki++) {
            d_tidx[tok * K + ki] = sel[ki];
            d_twt [tok * K + ki] = wt[ki] / s;
        }
    }
}

/* ════════════════════════════════════════════════════════════════════════════
 * KERNEL 2: expert_mlp_kernel
 *
 * Batched SiLU-gated FFN.  One block per token slot.
 * Shared memory holds intermediate activations [I].
 *
 * Works for both routed experts (intermediate=I) and shared expert (SI==I
 * when N_SH==1, which is our config).
 * ════════════════════════════════════════════════════════════════════════════ */
__global__ void expert_mlp_kernel(
    const float *d_in,    /* [n_tok, H]  input  */
    const float *gw,      /* [I,     H]  gate_proj weight */
    const float *uw,      /* [I,     H]  up_proj weight   */
    const float *dw,      /* [H,     I]  down_proj weight */
    float       *d_out,   /* [n_tok, H]  output */
    int n_tok,
    int inter_sz)         /* intermediate dimension (I or SI) */
{
    int tok = blockIdx.x;
    if (tok >= n_tok) return;

    const float *x = d_in  + tok * H;
    float       *o = d_out + tok * H;

    /* Use dynamic shared memory to support variable inter_sz */
    extern __shared__ float smem[];
    float *gate_buf = smem;              /* [inter_sz] */
    float *up_buf   = smem + inter_sz;   /* [inter_sz] */
    float *hidden   = smem + 2*inter_sz; /* [inter_sz] */

    /* gate_proj and up_proj — parallel over threads */
    for (int i = threadIdx.x; i < inter_sz; i += blockDim.x) {
        float g = 0.0f, u = 0.0f;
        for (int j = 0; j < H; j++) {
            g += gw[i * H + j] * x[j];
            u += uw[i * H + j] * x[j];
        }
        gate_buf[i] = g;
        up_buf  [i] = u;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < inter_sz; i += blockDim.x)
        hidden[i] = silu_d(gate_buf[i]) * up_buf[i];
    __syncthreads();

    /* down_proj */
    for (int j = threadIdx.x; j < H; j += blockDim.x) {
        float s = 0.0f;
        for (int i = 0; i < inter_sz; i++)
            s += dw[j * inter_sz + i] * hidden[i];
        o[j] = s;
    }
}

/* ════════════════════════════════════════════════════════════════════════════
 * KERNEL 3: scatter_kernel
 *
 * Packs local tokens into the AllToAll send buffer.
 *
 * Send buffer layout: d_send[dst_gpu * SLAB_F + lexp * SLOTS * H + slot * H]
 *   dim-0: destination GPU
 *   dim-1: local expert index on destination GPU
 *   dim-2: slot (token position within that expert's queue)
 *
 * Expert identity is encoded in the buffer position → receiver knows
 * exactly which expert processes each slot without extra metadata.
 *
 * Also writes dispatch info for gather phase:
 *   d_dinfo[tok * K + ki] = {gpu_dst, lexp, slot, weight}
 * ════════════════════════════════════════════════════════════════════════════ */
struct DispatchInfo {
    int   gpu_dst;   /* destination GPU */
    int   lexp;      /* local expert index on dst GPU */
    int   slot;      /* slot within that expert's buffer */
    float weight;    /* router weight for this (token, expert) pair */
};

__global__ void scatter_kernel(
    const float  *d_x,       /* [T_LOC, H]       local input tokens   */
    const int    *d_tidx,    /* [T_LOC, K]       global expert indices */
    const float  *d_twt,     /* [T_LOC, K]       router weights        */
    float        *d_send,    /* [NG, EPG, SLOTS, H] send buffer        */
    int          *send_cnt,  /* [NG * EPG]       per-(gpu,exp) counter  */
    DispatchInfo *d_dinfo,   /* [T_LOC * K]      dispatch metadata      */
    int t_loc, int epg)
{
    /* One block per (token, k) pair */
    int tok = blockIdx.x;
    int ki  = blockIdx.y;
    if (tok >= t_loc || ki >= K) return;

    int   gexp    = d_tidx[tok * K + ki];
    int   gpu_dst = gexp / epg;
    int   lexp    = gexp % epg;
    float w       = d_twt [tok * K + ki];

    /* CRITICAL: only ONE thread may increment the counter.
     * All 64 threads in the block handle the SAME (tok,ki) pair.
     * Thread 0 atomically claims one slot; result is broadcast via smem. */
    __shared__ int shared_slot;
    if (threadIdx.x == 0) {
        shared_slot = atomicAdd(&send_cnt[gpu_dst * epg + lexp], 1);

        /* Record dispatch info — only needed once per (tok,ki) */
        DispatchInfo info;
        info.gpu_dst = gpu_dst;
        info.lexp    = lexp;
        info.slot    = shared_slot;
        info.weight  = w;
        d_dinfo[tok * K + ki] = info;
    }
    __syncthreads();   /* all threads see shared_slot */

    int slot = shared_slot;

    /* Write token into d_send[gpu_dst][lexp][slot][:] */
    float       *dst = d_send + ((size_t)gpu_dst * EPG + lexp) * SLOTS * H
                              + (size_t)slot * H;
    const float *src = d_x + tok * H;
    for (int j = threadIdx.x; j < H; j += blockDim.x)
        dst[j] = src[j];
}

/* ════════════════════════════════════════════════════════════════════════════
 * KERNEL 4: gather_kernel
 *
 * After AllToAll #2, d_res_recv[src_gpu][lexp][slot][:] holds the expert
 * output for the token this GPU originally sent to (src_gpu, lexp, slot).
 * We accumulate: d_out[tok] += weight * expert_result
 * ════════════════════════════════════════════════════════════════════════════ */
__global__ void gather_kernel(
    const float        *d_res_recv,  /* [NG, EPG, SLOTS, H] received results */
    const DispatchInfo *d_dinfo,     /* [T_LOC * K]                           */
    float              *d_out,       /* [T_LOC, H]                            */
    int t_loc)
{
    int tok = blockIdx.x;
    int ki  = blockIdx.y;
    if (tok >= t_loc || ki >= K) return;

    const DispatchInfo info = d_dinfo[tok * K + ki];

    const float *src = d_res_recv
        + ((size_t)info.gpu_dst * EPG + info.lexp) * SLOTS * H
        + (size_t)info.slot * H;
    float *dst = d_out + tok * H;

    for (int j = threadIdx.x; j < H; j += blockDim.x)
        atomicAdd(&dst[j], info.weight * src[j]);
}

/* ════════════════════════════════════════════════════════════════════════════
 * Host: alloc_expert_weights — copy local expert weights from CPU → GPU
 * ════════════════════════════════════════════════════════════════════════════ */
static void alloc_expert_weights(
    int rank, int epg,
    float **d_gw, float **d_uw, float **d_dw)
{
    const float *all_gw[N_EXP] = {E0_GW,E1_GW,E2_GW,E3_GW,E4_GW,E5_GW,E6_GW,E7_GW};
    const float *all_uw[N_EXP] = {E0_UW,E1_UW,E2_UW,E3_UW,E4_UW,E5_UW,E6_UW,E7_UW};
    const float *all_dw[N_EXP] = {E0_DW,E1_DW,E2_DW,E3_DW,E4_DW,E5_DW,E6_DW,E7_DW};

    int base = rank * epg;
    for (int le = 0; le < epg; le++) {
        int ge = base + le;
        CUDA_CHECK(cudaMalloc(&d_gw[le], I * H * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_uw[le], I * H * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dw[le], H * I * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_gw[le], all_gw[ge], I*H*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_uw[le], all_uw[ge], I*H*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dw[le], all_dw[ge], H*I*sizeof(float), cudaMemcpyHostToDevice));
    }
}

/* ════════════════════════════════════════════════════════════════════════════
 * per_gpu_worker — full MoE forward pass for one GPU (rank)
 * ════════════════════════════════════════════════════════════════════════════ */
static int per_gpu_worker(int rank, ncclComm_t comm, cudaStream_t stream)
{
    /* cudaSetDevice was already called by the child before invoking us */
    const int t_loc = T / NG;
    const int epg   = EPG;

    /* ── 1. Upload local tokens ────────────────────────────────────────── */
    float *d_x;
    CUDA_CHECK(cudaMalloc(&d_x, t_loc * H * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x,
        INPUT + rank * t_loc * H,
        t_loc * H * sizeof(float),
        cudaMemcpyHostToDevice));

    /* ── 2. Upload router weights ──────────────────────────────────────── */
    float *d_rw, *d_rb;
    CUDA_CHECK(cudaMalloc(&d_rw, NE * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rb, NE     * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_rw, ROUTER_W,    NE*H*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rb, ROUTER_BIAS, NE  *sizeof(float), cudaMemcpyHostToDevice));

    /* ── 3. Upload local expert weights ────────────────────────────────── */
    float *d_gw[EXP_PER_GPU], *d_uw[EXP_PER_GPU], *d_dw[EXP_PER_GPU];
    alloc_expert_weights(rank, epg, d_gw, d_uw, d_dw);

    /* ── 4. Upload shared expert weights ───────────────────────────────── */
    float *d_sh_gw, *d_sh_uw, *d_sh_dw;
    CUDA_CHECK(cudaMalloc(&d_sh_gw, SI_SZ * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sh_uw, SI_SZ * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sh_dw, H * SI_SZ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_sh_gw, SH_GW, SI_SZ*H*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sh_uw, SH_UW, SI_SZ*H*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sh_dw, SH_DW, H*SI_SZ*sizeof(float), cudaMemcpyHostToDevice));

    /* ── 5. ROUTER KERNEL ──────────────────────────────────────────────── */
    int   *d_tidx;  float *d_twt;
    CUDA_CHECK(cudaMalloc(&d_tidx, t_loc * K * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_twt,  t_loc * K * sizeof(float)));

    router_kernel<<<t_loc, 64, 0, stream>>>(d_x, d_rw, d_rb, d_tidx, d_twt, t_loc);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* ── 6. SCATTER: pack tokens into AllToAll send buffer ─────────────
     *
     * Buffer layout: d_send[NG][EPG][SLOTS][H]
     *   NG=2 destination GPUs, EPG=4 local experts on each, SLOTS slots
     *
     * By encoding expert identity in the buffer index (dim 1), the
     * receiving GPU knows exactly which expert handles each slot.
     */
    size_t abuf = (size_t)NG * EPG * SLOTS * H * sizeof(float);

    float        *d_send, *d_recv;
    int          *d_send_cnt;
    DispatchInfo *d_dinfo;

    CUDA_CHECK(cudaMalloc(&d_send,     abuf));
    CUDA_CHECK(cudaMalloc(&d_recv,     abuf));
    CUDA_CHECK(cudaMemset(d_send, 0,   abuf));
    CUDA_CHECK(cudaMalloc(&d_send_cnt, NG * EPG * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_send_cnt,  0, NG * EPG * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dinfo,    t_loc * K * sizeof(DispatchInfo)));

    dim3 sg(t_loc, K);
    scatter_kernel<<<sg, 64, 0, stream>>>(
        d_x, d_tidx, d_twt, d_send, d_send_cnt, d_dinfo, t_loc, epg);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* ── 7. NCCL AllToAll #1 — dispatch tokens to expert-owning GPUs ───
     *
     * Each GPU sends a contiguous "slab" of SLAB_F floats to every peer.
     * Slab for GPU g:  d_send[g * SLAB_F .. (g+1)*SLAB_F - 1]
     *
     * After AllToAll:
     *   d_recv[g][le][slot][:]  = tokens from GPU g that need our expert le
     *   The (g, le) index tells us exactly which local expert to run.
     */
    NCCL_CHECK(ncclGroupStart());
    for (int g = 0; g < NG; g++) {
        NCCL_CHECK(ncclSend(d_send + (size_t)g * SLAB_F,
                            SLAB_F, ncclFloat, g, comm, stream));
        NCCL_CHECK(ncclRecv(d_recv + (size_t)g * SLAB_F,
                            SLAB_F, ncclFloat, g, comm, stream));
    }
    NCCL_CHECK(ncclGroupEnd());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* ── 8. LOCAL EXPERT COMPUTE ────────────────────────────────────────
     *
     * For each local expert le and each source GPU g:
     *   process all SLOTS token slots from d_recv[g][le]
     *   → d_expert_out[g][le]
     *
     * Buffer positions map 1-to-1, so result[g][le][slot] corresponds
     * exactly to recv[g][le][slot]. Sender's gather_kernel uses stored
     * dispatch info (gpu_dst=this_rank, lexp=le, slot) to find its result.
     */
    float *d_expert_out;
    CUDA_CHECK(cudaMalloc(&d_expert_out, abuf));
    CUDA_CHECK(cudaMemset(d_expert_out, 0, abuf));

    size_t smem = 3 * I_SZ * sizeof(float);   /* gate + up + hidden */
    for (int le = 0; le < epg; le++) {
        for (int g = 0; g < NG; g++) {
            float *tok_in  = d_recv       + ((size_t)g * EPG + le) * SLOTS * H;
            float *tok_out = d_expert_out + ((size_t)g * EPG + le) * SLOTS * H;
            expert_mlp_kernel<<<SLOTS, 64, smem, stream>>>(
                tok_in, d_gw[le], d_uw[le], d_dw[le], tok_out, SLOTS, I_SZ);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* ── 9. NCCL AllToAll #2 — return expert results to source GPUs ────
     *
     * Mirror of AllToAll #1: each GPU sends expert outputs back.
     * d_res_recv[g][le][slot][:] = result of token we sent to (g, le, slot).
     */
    float *d_res_recv;
    CUDA_CHECK(cudaMalloc(&d_res_recv, abuf));

    NCCL_CHECK(ncclGroupStart());
    for (int g = 0; g < NG; g++) {
        NCCL_CHECK(ncclSend(d_expert_out + (size_t)g * SLAB_F,
                            SLAB_F, ncclFloat, g, comm, stream));
        NCCL_CHECK(ncclRecv(d_res_recv   + (size_t)g * SLAB_F,
                            SLAB_F, ncclFloat, g, comm, stream));
    }
    NCCL_CHECK(ncclGroupEnd());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* ── 10. GATHER: weighted accumulate into output ────────────────────
     *
     * d_dinfo[tok*K+ki] records (gpu_dst, lexp, slot, weight) from scatter.
     * gather_kernel looks up the matching slot in d_res_recv and accumulates.
     */
    float *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, t_loc * H * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out, 0, t_loc * H * sizeof(float)));

    gather_kernel<<<sg, 64, 0, stream>>>(d_res_recv, d_dinfo, d_out, t_loc);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* ── 11. SHARED EXPERT (always active) ─────────────────────────────
     *
     * Run shared expert on all local tokens; add directly to d_out.
     * Shared expert uses SI_SZ intermediate size (= I_SZ when N_SH=1).
     */
    float *d_shared_out;
    CUDA_CHECK(cudaMalloc(&d_shared_out, t_loc * H * sizeof(float)));

    size_t sh_smem = 3 * SI_SZ * sizeof(float);
    expert_mlp_kernel<<<t_loc, 64, sh_smem, stream>>>(
        d_x, d_sh_gw, d_sh_uw, d_sh_dw, d_shared_out, t_loc, SI_SZ);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* out += shared_expert(x) — copy both buffers to host and add.
     * In production this stays on GPU; here we memcpy for clarity. */
    int total = t_loc * H;
    float h_out   [T_LOC * H];
    float h_shared[T_LOC * H];
    CUDA_CHECK(cudaMemcpy(h_out,    d_out,         total * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_shared, d_shared_out,  total * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < total; i++)
        h_out[i] += h_shared[i];

    /* ── 12. (Training note) AllReduce for data-parallel gradient sync ──
     *
     * In training, after backward pass each GPU holds partial gradients.
     * ncclAllReduce averages them so every GPU has the full gradient:
     *
     *   ncclAllReduce(grad_buf, grad_buf, n, ncclFloat, ncclSum, comm, stream);
     *   // then divide by NG
     *
     * In inference (here) we skip this step and just verify outputs.
     */

    /* ── 13. Verify against PyTorch reference ────────────────────────── */
    const float *moe_refs[T_TOK] = {
        MOE_OUT_0, MOE_OUT_1, MOE_OUT_2, MOE_OUT_3,
        MOE_OUT_4, MOE_OUT_5, MOE_OUT_6, MOE_OUT_7,
    };

    int all_pass = 1;
    for (int lt = 0; lt < t_loc; lt++) {
        int gt = rank * t_loc + lt;
        const float *ref = moe_refs[gt];
        const float *got = h_out + lt * H;

        float max_err = 0.0f;
        for (int j = 0; j < H; j++) {
            float e = fabsf(got[j] - ref[j]);
            if (e > max_err) max_err = e;
        }
        int pass = (max_err < 1e-3f);
        printf("  GPU %d  token %d (local %d)  max_err=%.3e  %s\n",
               rank, gt, lt, max_err, pass ? "PASS" : "FAIL");
        if (!pass) all_pass = 0;
    }

    /* ── Cleanup ─────────────────────────────────────────────────────── */
    cudaFree(d_x);   cudaFree(d_rw);  cudaFree(d_rb);
    for (int le = 0; le < epg; le++) {
        cudaFree(d_gw[le]); cudaFree(d_uw[le]); cudaFree(d_dw[le]);
    }
    cudaFree(d_sh_gw); cudaFree(d_sh_uw); cudaFree(d_sh_dw);
    cudaFree(d_tidx);  cudaFree(d_twt);
    cudaFree(d_send);  cudaFree(d_recv);
    cudaFree(d_send_cnt); cudaFree(d_dinfo);
    cudaFree(d_expert_out); cudaFree(d_res_recv);
    cudaFree(d_out);   cudaFree(d_shared_out);

    return all_pass;
}

/* ════════════════════════════════════════════════════════════════════════════
 * main — fork one process per GPU, share NCCL unique ID via /tmp file
 * ════════════════════════════════════════════════════════════════════════════ */
int main(void)
{
    printf("Week 8: DeepSeekV3 MoE — Multi-GPU CUDA + NCCL\n");
    printf("================================================\n");

    /* IMPORTANT: do NOT call any CUDA runtime API here before fork().
     * cudaGetDeviceCount() initialises the CUDA driver context in the
     * parent process; forking after that causes "initialization error"
     * in the children because CUDA contexts are not fork-safe.
     *
     * Instead:
     *   - Parent only calls ncclGetUniqueId() (CPU-only, no CUDA context)
     *     and writes it + n_devs placeholder to /tmp.
     *   - Each child initialises its own CUDA context after fork.         */

    /* Generate NCCL unique ID — CPU-only, safe before fork */
    ncclUniqueId nccl_id;
    NCCL_CHECK(ncclGetUniqueId(&nccl_id));
    {
        FILE *fp = fopen("/tmp/nccl_uid_w8", "wb");
        fwrite(&nccl_id, sizeof(nccl_id), 1, fp);
        fclose(fp);
    }

    /* Fork one child per logical rank */
    pid_t pids[N_GPU];
    for (int r = 0; r < NG; r++) {
        pids[r] = fork();
        if (pids[r] == 0) {
            /* ── Child: first CUDA call is here, safe after fork ─────── */
            int n_devs = 0;
            cudaGetDeviceCount(&n_devs);   /* initialise CUDA in this child */

            /* Map logical rank to physical device (supports 1-GPU loopback) */
            int phys = (n_devs > 0) ? (r % n_devs) : 0;
            CUDA_CHECK(cudaSetDevice(phys));

            if (r == 0) {
                printf("Found %d physical GPU(s). Logical ranks: %d.\n\n",
                       n_devs, NG);
                if (n_devs < NG)
                    printf("NOTE: sharing GPU 0 (loopback mode for functional test).\n\n");
                fflush(stdout);
            }

            /* Read shared NCCL unique ID and initialise communicator */
            ncclUniqueId uid;
            {
                FILE *f = fopen("/tmp/nccl_uid_w8", "rb");
                fread(&uid, sizeof(uid), 1, f);
                fclose(f);
            }
            ncclComm_t comm;
            NCCL_CHECK(ncclCommInitRank(&comm, NG, uid, r));

            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreate(&stream));

            /* CUDA timing events */
            cudaEvent_t ev0, ev1;
            CUDA_CHECK(cudaEventCreate(&ev0));
            CUDA_CHECK(cudaEventCreate(&ev1));

            printf("[Rank %d → GPU %d] Starting MoE forward pass...\n", r, phys);
            fflush(stdout);

            CUDA_CHECK(cudaEventRecord(ev0, stream));
            int ok = per_gpu_worker(r, comm, stream);
            CUDA_CHECK(cudaEventRecord(ev1, stream));
            CUDA_CHECK(cudaEventSynchronize(ev1));

            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
            printf("[Rank %d → GPU %d] Elapsed: %.3f ms   %s\n",
                   r, phys, ms, ok ? "ALL PASS" : "SOME FAIL");
            fflush(stdout);

            ncclCommDestroy(comm);
            exit(ok ? 0 : 1);
        }
    }

    /* Parent: wait for all children */
    int all_ok = 1;
    for (int r = 0; r < NG; r++) {
        int status;
        waitpid(pids[r], &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
            all_ok = 0;
    }

    printf("\n%s\n", all_ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_ok ? 0 : 1;
}
"""

# ============================================================
# PART 3: Performance benchmark — MoE vs Dense Transformer FFN
#
# Compares tokens/sec at larger scale (H=512, I=256, T=1024).
# MoE benefits: only TOP_K experts active per token vs full dense FFN.
# ============================================================
bench_script = """
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Benchmark device: {device}")

H     = 512    # hidden_size
I     = 256    # per-expert intermediate
N_EXP = 8
TOP_K = 2
SI    = I      # shared expert (N_SH=1)
T     = 1024   # tokens per batch
ITERS = 50

class SiLUGatedMLP(nn.Module):
    def __init__(self, hidden, inter):
        super().__init__()
        self.gate = nn.Linear(hidden, inter, bias=False)
        self.up   = nn.Linear(hidden, inter, bias=False)
        self.down = nn.Linear(inter,  hidden, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class MoELayer(nn.Module):
    '''Simulates data+expert parallelism in a single process for benchmarking.'''
    def __init__(self):
        super().__init__()
        self.router_w = nn.Parameter(torch.randn(N_EXP, H))
        self.router_b = nn.Parameter(torch.zeros(N_EXP))
        self.experts  = nn.ModuleList([SiLUGatedMLP(H, I) for _ in range(N_EXP)])
        self.shared   = SiLUGatedMLP(H, SI)

    def forward(self, x):
        # Router: sigmoid + topk + normalize
        scores = F.linear(x, self.router_w).sigmoid() + self.router_b
        topk_w, topk_i = torch.topk(scores, TOP_K, dim=-1)
        topk_w = topk_w / topk_w.sum(-1, keepdim=True)

        # Expert dispatch + compute (token grouping simulates AllToAll)
        out = torch.zeros_like(x)
        for e in range(N_EXP):
            mask  = (topk_i == e).any(dim=-1)
            if not mask.any(): continue
            w_e = topk_w[(topk_i == e)].unsqueeze(-1)
            out[mask] += w_e * self.experts[e](x[mask])

        # Shared expert (always active)
        out += self.shared(x)
        return out

class DenseFFN(nn.Module):
    '''Dense FFN with active FLOPs = TOP_K routed + 1 shared expert.'''
    def __init__(self):
        super().__init__()
        equiv_I = I * (TOP_K + 1)
        self.gate = nn.Linear(H, equiv_I, bias=False)
        self.up   = nn.Linear(H, equiv_I, bias=False)
        self.down = nn.Linear(equiv_I, H, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

moe   = MoELayer().to(device).eval()
dense = DenseFFN() .to(device).eval()
x     = torch.randn(T, H, device=device)

def bench(model, label):
    with torch.no_grad():
        for _ in range(5): model(x)      # warmup
    if device.type == 'cuda': torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(ITERS): model(x)
    if device.type == 'cuda': torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    tps = T * ITERS / elapsed
    ms  = elapsed * 1000 / ITERS
    print(f"  {label:<38s}: {tps:>9.0f} tok/s   ({ms:.2f} ms/iter)")

print(f"\\nConfig: T={T} tokens, H={H}, I_per_expert={I}, "
      f"{N_EXP} routed experts, top-{TOP_K}, 1 shared expert")
print("-" * 65)
bench(moe,   "MoE  (data-parallel + expert-parallel sim)")
bench(dense, "Dense FFN  (equiv active FLOPs)")
print("-" * 65)
print("MoE advantage: only 2/{} experts active per token,".format(N_EXP))
print("  amortized via token grouping (AllToAll in real multi-GPU).")
"""


@app.function(
    image=image,
    gpu="A100:2",    # Request 2 × A100 for data-parallel + expert-parallel demo
    timeout=900,
)
def run_multigpu():
    import subprocess, os

    print("=" * 60)
    print("Week 8: DeepSeekV3 MoE — Multi-GPU CUDA + NCCL")
    print("=" * 60)

    # ── Step 1: Generate test data ─────────────────────────────────────
    print("\n[1/4] Generating test data from PyTorch reference...\n")
    with open("generate_tests.py", "w") as f:
        f.write(test_gen_script)

    r = subprocess.run(["python3", "generate_tests.py"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("Test generation FAILED:\n", r.stderr)
        return

    with open("test_data.h", "w") as f:
        f.write(r.stdout)
    print(f"test_data.h written ({len(r.stdout)} bytes).")

    # ── Step 2: Compile ────────────────────────────────────────────────
    print("\n[2/4] Compiling CUDA+NCCL (nvcc -O2 -arch=sm_80)...")
    with open("moe_multigpu.cu", "w") as f:
        f.write(cuda_source)

    # Locate NCCL headers and libraries (path differs by CUDA image version)
    nccl_inc, nccl_lib = "/usr/include", "/usr/lib/x86_64-linux-gnu"
    for inc_candidate in ["/usr/include", "/usr/local/cuda/include",
                          "/usr/local/nccl/include"]:
        if os.path.exists(os.path.join(inc_candidate, "nccl.h")):
            nccl_inc = inc_candidate
            break
    for lib_candidate in ["/usr/lib/x86_64-linux-gnu",
                          "/usr/local/cuda/lib64",
                          "/usr/local/nccl/lib"]:
        import glob as _glob
        if _glob.glob(os.path.join(lib_candidate, "libnccl*")):
            nccl_lib = lib_candidate
            break

    print(f"  NCCL headers: {nccl_inc}/nccl.h  exists={os.path.exists(nccl_inc+'/nccl.h')}")
    print(f"  NCCL libs   : {nccl_lib}")

    compile_cmd = [
        "nvcc", "-O2", "-arch=sm_80",
        f"-I{nccl_inc}",
        f"-L{nccl_lib}",
        "moe_multigpu.cu", "-lnccl", "-o", "moe_multigpu",
    ]
    r = subprocess.run(compile_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("Compilation FAILED:")
        print(r.stderr[:4000])
        print("\n--- Generated moe_multigpu.cu (first 100 lines) ---")
        for i, line in enumerate(cuda_source.split('\n')[:100], 1):
            print(f"{i:4d}  {line}")
        return
    print("Compilation successful.")

    # ── Step 3: Run multi-GPU verification ────────────────────────────
    print("\n[3/4] Running multi-GPU MoE forward pass (2 GPUs)...\n")
    r = subprocess.run(["./moe_multigpu"],
                       capture_output=True, text=True, timeout=120)
    print(r.stdout)
    if r.stderr:
        print("Stderr:", r.stderr[:1000])

    # ── Step 4: Performance benchmark ─────────────────────────────────
    print("\n[4/4] Performance benchmark: MoE vs Dense Transformer FFN\n")
    with open("bench.py", "w") as f:
        f.write(bench_script)

    r = subprocess.run(["python3", "bench.py"],
                       capture_output=True, text=True, timeout=120)
    print(r.stdout)
    if r.stderr and r.returncode != 0:
        print("Benchmark stderr:", r.stderr[:500])


@app.local_entrypoint()
def main():
    run_multigpu.remote()
