import modal

app = modal.App("week07-deepseek-moe")

# CPU-only image: no GPU needed (pure C implementation)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .run_commands(
        "pip install torch --index-url https://download.pytorch.org/whl/cpu",
        "pip install numpy",
    )
    .apt_install("gcc")
)

# ============================================================
# PART 1: Python test case generator
#
# Implements the exact HuggingFace DeepseekV3 MoE blocks:
#   - DeepseekV3MLP  : SiLU-gated FFN (gate_proj, up_proj, down_proj)
#   - DeepseekV3TopkRouter : sigmoid scores + topk + normalize
#   - Full MoE forward: routes tokens + shared expert
#
# Uses tiny dimensions to keep test cases manageable:
#   hidden_size=16, intermediate_size=8, n_experts=4, top_k=2
#
# Outputs a C header file (test_data.h) with embedded arrays
# ============================================================
test_gen_script = """
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(0)

# ── Small config ─────────────────────────────────────────────────────────────
H     = 16   # hidden_size
I     = 8    # moe_intermediate_size (per routed expert)
N_EXP = 4    # n_routed_experts
TOP_K = 2    # num_experts_per_tok
N_SH  = 1    # n_shared_experts
SI    = I * N_SH   # shared_intermediate_size = 8
T_TOK = 6    # total tokens (batch=2, seq=3)

# ── Reference blocks (exact HuggingFace DeepSeekV3 implementation) ───────────

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
    '''Router: sigmoid scoring + top-k selection + weight normalization.'''
    def __init__(self, hidden_size, n_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.weight = nn.Parameter(torch.empty(n_experts, hidden_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.e_score_correction_bias = nn.Parameter(torch.zeros(n_experts))
    def forward(self, x_flat):
        # x_flat: [T, H] — already flattened tokens
        logits = F.linear(x_flat, self.weight, None)          # [T, N_EXP]
        scores = logits.sigmoid()                              # [T, N_EXP]
        topk_weight, topk_idx = torch.topk(
            scores + self.e_score_correction_bias,
            self.top_k, dim=-1, sorted=False
        )
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
        return topk_idx, topk_weight

# ── Initialize with fixed weight seed ────────────────────────────────────────
torch.manual_seed(42)
router       = DeepseekV3TopkRouter(H, N_EXP, TOP_K)
experts      = nn.ModuleList([DeepseekV3MLP(H, I) for _ in range(N_EXP)])
shared_exp   = DeepseekV3MLP(H, SI)

# Fixed input seed
torch.manual_seed(99)
x = torch.randn(T_TOK, H)  # [T, H] flat tokens

# ── Run reference forward passes ─────────────────────────────────────────────
with torch.no_grad():
    # Test 1: Each routed expert MLP on token 0
    exp_mlp_outs = [experts[e](x[0:1]).squeeze(0) for e in range(N_EXP)]

    # Test 2: Router on all tokens
    topk_idx, topk_weight = router(x)  # [T, TOP_K] each

    # Test 3: Full MoE on each token
    moe_outs = []
    for t in range(T_TOK):
        xt   = x[t].unsqueeze(0)       # [1, H]
        out  = torch.zeros(H)
        for ki in range(TOP_K):
            eidx = topk_idx[t, ki].item()
            w    = topk_weight[t, ki].item()
            out += w * experts[eidx](xt).squeeze(0)
        out += shared_exp(xt).squeeze(0)   # shared expert always active
        moe_outs.append(out)

# ── Helper: numpy array → C initializer ──────────────────────────────────────
def arr_c(arr, name):
    vals = ", ".join(f"{v:.8f}f" for v in arr.detach().numpy().flatten())
    return f"static const float {name}[] = {{{vals}}};"

def int_arr_c(arr, name):
    vals = ", ".join(str(int(v)) for v in arr.detach().numpy().flatten())
    return f"static const int {name}[] = {{{vals}}};"

# ── Emit C header ─────────────────────────────────────────────────────────────
lines = [
    "// Auto-generated test data — DeepSeekV3 MoE (Week 7)",
    f"#define HIDDEN_SZ {H}",
    f"#define I_SZ    {I}",
    f"#define SI_SZ   {SI}",
    f"#define N_EXP   {N_EXP}",
    f"#define TOP_K_  {TOP_K}",
    f"#define T_TOK   {T_TOK}",
    "",
    "// Router",
    arr_c(router.weight,                    "ROUTER_W"),
    arr_c(router.e_score_correction_bias,   "ROUTER_BIAS"),
]
for e in range(N_EXP):
    lines += [
        arr_c(experts[e].gate_proj.weight,  f"E{e}_GW"),
        arr_c(experts[e].up_proj.weight,    f"E{e}_UW"),
        arr_c(experts[e].down_proj.weight,  f"E{e}_DW"),
    ]
lines += [
    arr_c(shared_exp.gate_proj.weight,  "SH_GW"),
    arr_c(shared_exp.up_proj.weight,    "SH_UW"),
    arr_c(shared_exp.down_proj.weight,  "SH_DW"),
    arr_c(x,                            "INPUT"),
]
for e in range(N_EXP):
    lines.append(arr_c(exp_mlp_outs[e], f"EXP_OUT_{e}"))
lines.append(int_arr_c(topk_idx,    "ROUTER_IDX"))
lines.append(arr_c(topk_weight,     "ROUTER_WGT"))
for t in range(T_TOK):
    lines.append(arr_c(moe_outs[t], f"MOE_OUT_{t}"))

print("\\n".join(lines))
"""

# ============================================================
# PART 2: Pure C implementation of DeepSeekV3 MoE
#
# Blocks:
#   expert_mlp     — SiLU-gated MLP for one token
#   router_forward — sigmoid + topk + normalize for one token
#   moe_forward    — full MoE for one token
# ============================================================
c_source = r"""
/* Week 7: DeepSeekV3 MoE operator — pure C, no parallelism, no CUDA
 *
 * Implements Algorithm from DeepSeekMoE paper:
 *   For each token:
 *     1. Router: logits = x @ W_r^T  → sigmoid → topk → normalize
 *     2. Routed experts: out += weight_k * ExpertMLP_k(x)  for k in top_k
 *     3. Shared expert: out += SharedMLP(x)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "test_data.h"

/* ── Activations ─────────────────────────────────────────────────────────── */
static float silu(float x)    { return x / (1.0f + expf(-x)); }
static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

/* ── Linear: out[o] = sum_j W[o*in_sz + j] * x[j]
 *   W layout: [out_sz, in_sz] row-major  (matches PyTorch nn.Linear.weight) */
static void linear(const float *W, const float *x, float *out,
                   int in_sz, int out_sz) {
    for (int o = 0; o < out_sz; o++) {
        float s = 0.0f;
        for (int j = 0; j < in_sz; j++)
            s += W[o * in_sz + j] * x[j];
        out[o] = s;
    }
}

/* ── Expert MLP (single token) ───────────────────────────────────────────────
 *   out = down_proj( silu(gate_proj(x)) * up_proj(x) )
 *   gate_w, up_w: [I, H]    down_w: [H, I] */
static void expert_mlp(const float *gw, const float *uw, const float *dw,
                        const float *x, float *out, int H, int I) {
    float *gate   = (float*)malloc(I * sizeof(float));
    float *up     = (float*)malloc(I * sizeof(float));
    float *hidden = (float*)malloc(I * sizeof(float));

    linear(gw, x, gate, H, I);
    linear(uw, x, up,   H, I);
    for (int i = 0; i < I; i++)
        hidden[i] = silu(gate[i]) * up[i];
    linear(dw, hidden, out, I, H);

    free(gate); free(up); free(hidden);
}

/* ── TopK: fill idx/val with the k largest entries from scores[n]
 *   O(n*k) selection — adequate for small n_experts */
static void topk(const float *scores, int n, int k, int *idx, float *val) {
    int *used = (int*)calloc(n, sizeof(int));
    for (int ki = 0; ki < k; ki++) {
        int best = -1; float bv = -1e30f;
        for (int i = 0; i < n; i++)
            if (!used[i] && scores[i] > bv) { bv = scores[i]; best = i; }
        idx[ki] = best; val[ki] = bv; used[best] = 1;
    }
    free(used);
}

/* ── Router (single token) ───────────────────────────────────────────────────
 *   W: [N_EXP, H]   bias: [N_EXP]
 *   logits = W @ x  →  scores = sigmoid(logits)
 *   topk_weight, topk_idx = topk(scores + bias, k)
 *   topk_weight /= sum(topk_weight)   (normalize) */
static void router_forward(const float *W, const float *bias,
                            const float *x, int H, int n_exp, int k,
                            int *tidx, float *twt) {
    float *logits = (float*)malloc(n_exp * sizeof(float));
    float *adj    = (float*)malloc(n_exp * sizeof(float));

    linear(W, x, logits, H, n_exp);
    for (int i = 0; i < n_exp; i++)
        adj[i] = sigmoid(logits[i]) + bias[i];

    topk(adj, n_exp, k, tidx, twt);

    float sum = 0.0f;
    for (int i = 0; i < k; i++) sum += twt[i];
    for (int i = 0; i < k; i++) twt[i] /= sum;

    free(logits); free(adj);
}

/* ── Full MoE (single token) ─────────────────────────────────────────────────
 *   out = sum_k( w_k * expert_k(x) )  +  shared_expert(x) */
static void moe_forward(
    const float *rw, const float *rb,
    const float **e_gw, const float **e_uw, const float **e_dw,
    const float *sh_gw, const float *sh_uw, const float *sh_dw,
    const float *x, float *out,
    int H, int I, int SI, int n_exp, int k)
{
    int   *tidx = (int*)  malloc(k * sizeof(int));
    float *twt  = (float*)malloc(k * sizeof(float));
    float *eout = (float*)malloc(H * sizeof(float));

    router_forward(rw, rb, x, H, n_exp, k, tidx, twt);
    memset(out, 0, H * sizeof(float));

    /* Routed experts */
    for (int ki = 0; ki < k; ki++) {
        int eidx = tidx[ki];
        expert_mlp(e_gw[eidx], e_uw[eidx], e_dw[eidx], x, eout, H, I);
        for (int i = 0; i < H; i++)
            out[i] += twt[ki] * eout[i];
    }

    /* Shared expert (always active) */
    expert_mlp(sh_gw, sh_uw, sh_dw, x, eout, H, SI);
    for (int i = 0; i < H; i++)
        out[i] += eout[i];

    free(eout); free(tidx); free(twt);
}

/* ── Verification helper ──────────────────────────────────────────────────── */
static int check(const float *got, const float *ref, int n,
                 const char *name, float tol) {
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float e = fabsf(got[i] - ref[i]);
        if (e > max_err) max_err = e;
    }
    int pass = (max_err < tol);
    printf("  %-45s  max_err=%.2e  %s\n", name, max_err, pass?"PASS":"FAIL");
    return pass;
}

/* ── Sort two parallel arrays (int idx, float val) by idx ascending ───────── */
static void sort_by_idx(int *idx, float *val, int k) {
    for (int a = 0; a < k-1; a++)
        for (int b = a+1; b < k; b++)
            if (idx[a] > idx[b]) {
                int   ti = idx[a]; idx[a] = idx[b]; idx[b] = ti;
                float tv = val[a]; val[a] = val[b]; val[b] = tv;
            }
}

/* ── main ──────────────────────────────────────────────────────────────────── */
int main(void) {
    printf("Week 7: DeepSeekV3 MoE — pure C implementation\n");
    printf("================================================\n\n");

    /* Expert weight pointer arrays */
    const float *eg[N_EXP] = {E0_GW, E1_GW, E2_GW, E3_GW};
    const float *eu[N_EXP] = {E0_UW, E1_UW, E2_UW, E3_UW};
    const float *ed[N_EXP] = {E0_DW, E1_DW, E2_DW, E3_DW};

    int   all_pass = 1;
    float *buf = (float*)malloc(HIDDEN_SZ * sizeof(float));

    /* ── Test 1: Expert MLP blocks ─────────────────────────────────────── */
    printf("Test 1: Expert MLP (token 0 through each routed expert)\n");
    const float *refs1[4] = {EXP_OUT_0, EXP_OUT_1, EXP_OUT_2, EXP_OUT_3};
    for (int e = 0; e < N_EXP; e++) {
        expert_mlp(eg[e], eu[e], ed[e], INPUT, buf, HIDDEN_SZ, I_SZ);
        char name[64]; snprintf(name, sizeof(name), "Expert %d MLP", e);
        all_pass &= check(buf, refs1[e], HIDDEN_SZ, name, 1e-4f);
    }

    /* ── Test 2: Router (topk routing) ────────────────────────────────── */
    printf("\nTest 2: TopK Router (all %d tokens)\n", T_TOK);
    for (int t = 0; t < T_TOK; t++) {
        const float *xt = INPUT + t * HIDDEN_SZ;
        int   c_idx[TOP_K_]; float c_wt[TOP_K_];
        router_forward(ROUTER_W, ROUTER_BIAS, xt, HIDDEN_SZ, N_EXP, TOP_K_,
                       c_idx, c_wt);

        /* Reference values */
        int   py_idx[TOP_K_]; float py_wt[TOP_K_];
        for (int ki = 0; ki < TOP_K_; ki++) {
            py_idx[ki] = ROUTER_IDX[t * TOP_K_ + ki];
            py_wt[ki]  = ROUTER_WGT[t * TOP_K_ + ki];
        }

        /* Sort both by expert index before comparing (topk is unordered) */
        sort_by_idx(c_idx,  c_wt,  TOP_K_);
        sort_by_idx(py_idx, py_wt, TOP_K_);

        float max_wt_err = 0.0f; int idx_ok = 1;
        for (int ki = 0; ki < TOP_K_; ki++) {
            if (c_idx[ki] != py_idx[ki]) idx_ok = 0;
            float e = fabsf(c_wt[ki] - py_wt[ki]);
            if (e > max_wt_err) max_wt_err = e;
        }
        int pass = idx_ok && (max_wt_err < 1e-4f);
        char name[64]; snprintf(name, sizeof(name), "Router token %d", t);
        printf("  %-45s  idx_ok=%d  max_wt_err=%.2e  %s\n",
               name, idx_ok, max_wt_err, pass ? "PASS" : "FAIL");
        all_pass &= pass;
    }

    /* ── Test 3: Full MoE forward ────────────────────────────────────── */
    printf("\nTest 3: Full MoE (all %d tokens)\n", T_TOK);
    const float *moe_refs[6] = {
        MOE_OUT_0, MOE_OUT_1, MOE_OUT_2,
        MOE_OUT_3, MOE_OUT_4, MOE_OUT_5
    };
    for (int t = 0; t < T_TOK; t++) {
        const float *xt = INPUT + t * HIDDEN_SZ;
        moe_forward(ROUTER_W, ROUTER_BIAS,
                    (const float**)eg, (const float**)eu, (const float**)ed,
                    SH_GW, SH_UW, SH_DW,
                    xt, buf, HIDDEN_SZ, I_SZ, SI_SZ, N_EXP, TOP_K_);
        char name[64]; snprintf(name, sizeof(name), "Full MoE token %d", t);
        all_pass &= check(buf, moe_refs[t], HIDDEN_SZ, name, 1e-4f);
    }

    free(buf);
    printf("\n%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
"""


@app.function(image=image, timeout=600)
def run_cpu():
    import subprocess

    print("=" * 60)
    print("Week 7: DeepSeekV3 MoE Operator in Pure C")
    print("=" * 60)

    # ── Step 1: Generate test cases ──────────────────────────────────────
    print("\n[1/3] Generating test cases from PyTorch reference...\n")
    with open("generate_tests.py", "w") as f:
        f.write(test_gen_script)

    result = subprocess.run(
        ["python3", "generate_tests.py"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("Test generation FAILED:")
        print(result.stderr)
        return

    with open("test_data.h", "w") as f:
        f.write(result.stdout)
    print("test_data.h generated successfully.")
    print(f"  Header size: {len(result.stdout)} bytes")

    # ── Step 2: Compile C ────────────────────────────────────────────────
    print("\n[2/3] Compiling C implementation (gcc -O2)...")
    with open("moe.c", "w") as f:
        f.write(c_source)

    result = subprocess.run(
        ["gcc", "-O2", "-o", "moe", "moe.c", "-lm"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("Compilation FAILED:")
        print(result.stderr)
        return
    print("Compilation successful.")

    # ── Step 3: Run and verify ────────────────────────────────────────────
    print("\n[3/3] Running verification...\n")
    result = subprocess.run(["./moe"], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Stderr:", result.stderr)


@app.local_entrypoint()
def main():
    run_cpu.remote()
