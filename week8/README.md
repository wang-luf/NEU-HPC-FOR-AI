# Week 8: DeepSeekV3 MoE — Multi-GPU CUDA + NCCL

Extension of the Week 7 pure-C MoE operator to **multi-GPU** execution using
**CUDA kernels** and **NCCL collective communications**.

Progressive chain: **Week 7** (pure C) → **Week 8** (CUDA + NCCL) → **Week 9** (ThunderKittens)

---

## Assignment

1. Read the NCCL documentation and NVIDIA HGX H100 architecture overview
2. Implement multi-GPU DeepSeekMoE in CUDA using:
   - **Data parallelism** — partition input tokens across GPUs
   - **Expert parallelism** — partition routed experts across GPUs
3. Use `ncclAllToAll` for expert dispatch and result combine
4. Verify generated test cases pass (match PyTorch reference)
5. Benchmark MoE throughput vs dense Transformer FFN

---

## Architecture

```
Tokens [T, H]
    │
    ├── Data parallelism: split T tokens across N_GPU GPUs
    │
    │   Each GPU (rank r):
    │   ┌──────────────────────────────────────────────────┐
    │   │ 1. Router kernel → topk_idx, topk_weight         │
    │   │                    (sigmoid + topk + normalize)   │
    │   │                                                   │
    │   │ 2. Scatter kernel → pack send buffers             │
    │   │    token[t] → send_buf[gpu_dst][slot]             │
    │   │                                                   │
    │   │ 3. ncclAllToAll #1  ── dispatch tokens ──►        │
    │   │    Each GPU receives tokens destined for its      │
    │   │    local experts                                  │
    │   │                                                   │
    │   │ 4. expert_mlp_kernel × EPG local experts          │
    │   │    SiLU-gated FFN: down(silu(gate(x)) * up(x))   │
    │   │                                                   │
    │   │ 5. ncclAllToAll #2  ◄── return results ──         │
    │   │    Expert results returned to token-owning GPUs   │
    │   │                                                   │
    │   │ 6. Gather kernel → weighted accumulate            │
    │   │    out[t] += weight_k * expert_result_k           │
    │   │                                                   │
    │   │ 7. Shared expert kernel (always active)           │
    │   │    out[t] += shared_expert(x[t])                  │
    │   └──────────────────────────────────────────────────┘
    │
    └── Final output [T, H] distributed across GPUs
```

---

## CUDA Kernels

| Kernel | Input | Output | Purpose |
|--------|-------|--------|---------|
| `router_kernel` | `[T_loc, H]` tokens | `[T_loc, K]` idx + weight | Sigmoid logits → topk → normalize |
| `expert_mlp_kernel` | `[n_tok, H]` | `[n_tok, H]` | Batched SiLU-gated FFN |
| `scatter_kernel` | tokens + routing | send buffer `[NG, MAX_RECV, H]` | Pack tokens for AllToAll |
| `gather_kernel` | result buffer + dispatch info | `[T_loc, H]` | Weighted accumulate |
| `shared_expert_add_kernel` | shared output | final output | Add shared expert contribution |

---

## NCCL Communication Pattern

```
AllToAll #1 (dispatch)
  Each GPU sends:  send_buf[g * MAX_RECV * H]  to GPU g
  Each GPU recvs:  recv_buf[g * MAX_RECV * H]  from GPU g

  Purpose: route tokens to the GPU that owns the requested expert

AllToAll #2 (combine)
  Each GPU sends:  expert_out[g * MAX_RECV * H]  back to GPU g
  Each GPU recvs:  result[g * MAX_RECV * H]      from GPU g

  Purpose: return computed expert outputs to the token-owning GPU

AllReduce (training only, shown conceptually)
  Averages gradients across data-parallel GPUs after backward pass
```

This pattern mirrors **DeepSpeed's expert parallelism** and the
dispatching mechanism described in the DeepSeekMoE paper.

---

## Configuration

```
hidden_size          = 16   (test) / 512 (benchmark)
moe_intermediate_size= 8    (test) / 256 (benchmark)
n_routed_experts     = 8    (4 per GPU with 2 GPUs)
num_experts_per_tok  = 2    (top-k routing)
n_shared_experts     = 1    (always active)
total_tokens         = 8    (test) / 1024 (benchmark)
n_gpu                = 2
```

---

## Multi-Process Launch

Uses `fork()` + NCCL unique ID (written to `/tmp/nccl_uid`) to initialize
one communicator per GPU without MPI:

```
main process
  ├── ncclGetUniqueId()  → writes /tmp/nccl_uid
  ├── fork() → child 0 (GPU 0): reads uid, ncclCommInitRank(rank=0)
  └── fork() → child 1 (GPU 1): reads uid, ncclCommInitRank(rank=1)
```

---

## Run

```bash
modal run week8/run_week8.py
```

4-step pipeline on Modal (A100 GPU):
1. Python generates `test_data.h` (PyTorch reference weights + expected outputs)
2. `nvcc -O2 -arch=sm_80` compiles `moe_multigpu.cu` with NCCL
3. Multi-GPU binary runs forward pass on 2 GPUs, verifies each token
4. Python benchmark compares MoE vs dense Transformer throughput

---

## Expected Output

```
Week 8: DeepSeekV3 MoE — Multi-GPU CUDA + NCCL
================================================
Found 2 GPUs.  Using 2.

[GPU 0] Running MoE forward pass...
[GPU 1] Running MoE forward pass...
  GPU 0  token 0 (local 0)  max_err=2.1e-05  PASS
  GPU 0  token 1 (local 1)  max_err=1.8e-05  PASS
  GPU 0  token 2 (local 2)  max_err=3.2e-05  PASS
  GPU 0  token 3 (local 3)  max_err=2.7e-05  PASS
  GPU 1  token 4 (local 0)  max_err=1.9e-05  PASS
  ...
[GPU 0] Forward pass: 0.842 ms  (ALL PASS)
[GPU 1] Forward pass: 0.851 ms  (ALL PASS)

ALL TESTS PASSED

Benchmark: T=1024 tokens, H=512, I=256, 8 experts, top-2
------------------------------------------------------------
  MoE (data+expert parallel sim)  :    148532 tok/s  ( 6.89 ms/iter)
  Dense FFN (equiv FLOPs)         :     72341 tok/s  (14.15 ms/iter)
------------------------------------------------------------
```

---

## Reading

- DeepSeekMoE: Towards Ultimate Expert Specialization in MoE Language Models
- NVIDIA NCCL Developer Guide: AllReduce, AllGather, ReduceScatter, All-to-All
- NVIDIA HGX H100: NVSwitch, NVLink 900 GB/s interconnect
- Demystifying NCCL: In-depth Analysis of GPU Communication Protocols
- DeepSpeed Ulysses: All-to-all sequence parallelism for large-scale LLM training
- Ring Attention: Blockwise attention distribution across devices
