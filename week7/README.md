# Week 7: DeepSeekV3 MoE Operator in Pure C

Implementation of the **Mixture-of-Experts (MoE)** operator from DeepSeekV3,
following the DeepSeekMoE paper. Pure sequential C — no CUDA, no parallelism.

## Assignment

1. Read the DeepSeekMoE paper
2. Generate test cases using HuggingFace transformers as reference
3. Implement the MoE operator in pure C
4. Verify C implementation matches PyTorch reference

## DeepSeekV3 MoE Architecture

```
token x
  │
  ├─► Router (sigmoid + topk + normalize)
  │     └─► select top-2 experts from 256
  │
  ├─► Expert MLP × top_k  (routed, weighted)
  │     gate_proj → SiLU × up_proj → down_proj
  │
  └─► Shared Expert MLP × 1  (always active)
        gate_proj → SiLU × up_proj → down_proj
  │
  ▼
output = Σ(weight_k × expert_k(x)) + shared_expert(x)
```

## Blocks Implemented (C)

### `expert_mlp` — SiLU-gated FFN

```c
gate   = W_gate @ x
up     = W_up   @ x
hidden = silu(gate) * up      // elementwise
out    = W_down @ hidden
```

### `router_forward` — TopK routing

```c
logits   = W_router @ x
scores   = sigmoid(logits)
adjusted = scores + e_score_correction_bias   // load balancing
(idx, wt) = topk(adjusted, k=2)
wt /= sum(wt)                                 // normalize
```

### `moe_forward` — Full MoE

```c
router_forward(x) → topk_idx, topk_weight
for each selected expert k:
    out += topk_weight[k] * expert_mlp_k(x)
out += shared_expert_mlp(x)
```

## Test Case Generation

Uses PyTorch with exact HuggingFace `DeepseekV3MLP` / `DeepseekV3TopkRouter` code.
Small dimensions (hidden=16, experts=4, top_k=2) for fast, deterministic tests.
Fixed seeds: weight seed=42, input seed=99.

## Run

```bash
modal run week7/run_week7.py
```

3-step pipeline on Modal (CPU only, no GPU needed):
1. Python generates `test_data.h` with embedded weights + expected outputs
2. `gcc -O2` compiles `moe.c`
3. C binary runs 3 test suites and prints PASS/FAIL

## Reading

- DeepSeekMoE: Towards Ultimate Expert Specialization in MoE Language Models
- NVIDIA HGX H100: 8-GPU single node, NVSwitch, NVLink 900 GB/s
- NCCL Developer Guide: AllReduce, AllGather, ReduceScatter, All-to-All
