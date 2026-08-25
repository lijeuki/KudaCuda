# KudaCuda Architecture and NVIDIA Support

KudaCuda is a CUDA-first native sidecar for model runtimes that already own the tensors, allocator, and CUDA stream. It keeps the integration boundary small and explicit:

```text
Transformers / PyTorch tensor
          |
          | shape, dtype, layout, device checks
          v
Python sidecar (ctypes, raw data_ptr, caller stream)
          |
          | stable C ABI
          v
KudaCuda runtime (handle, registry, graph lifecycle, errors)
          |
          +--> CUDA Runtime API launches
          +--> CUDA Driver API module registry
          v
sm_86 / sm_89 CUDA kernels
```

The caller remains responsible for tensor ownership, stream ordering, synchronization, and model-level fallback. KudaCuda does not copy model tensors into a second framework-owned buffer.

## NVIDIA component map

| NVIDIA component | KudaCuda use | Status |
|---|---|---|
| CUDA Toolkit / `nvcc` | Compiles `.cu` kernels and architecture-specific device code | `sm_86` build verified on RTX 3050; `sm_89` source path prepared for L40S |
| CUDA Runtime API | Device selection, stream-aware launches, error handling, graph capture/replay, kernel launches | Vector-add, RMSNorm, and Qwen2 GQA smoke paths verified |
| CUDA Driver API / `nvcuda.dll` | Context retention, context selection, module load, function lookup, module unload | Registry implemented; Windows import library is generated from `runtime/nvcuda.def` |
| CUDA streams | Caller-owned stream is passed through the C ABI without synchronization | Verified with a non-default stream |
| CUDA Graphs | Capture, instantiate/replay, and destroy CUDA-only work; PyTorch `CUDAGraph` owns full-model graph memory pools | Vector-add and multi-launch replay verified; Qwen2 static decode captured/replayed with exact parity on RTX 3050 |
| bitsandbytes NF4 storage contract | Python bridge validates packed `uint8` NF4 weights, FP32 block scales, blocksize 64, and FP16 projection biases before native dispatch | Real Qwen2.5-3B projection parity validated; tiled v2/v3 add constant-memory codebook, `uint4` loads, FP32 accumulation, bias epilogue, and v3 activation-tile reuse; model paths remain experimental |
| CUBIN/PTX/fatbin | Architecture-specific kernel artifacts | `sm_86` artifact is published; build with `89` for L40S |
| Ampere Tensor Cores | Planned INT4/INT8 and mixed-precision matrix paths | Not yet implemented |
| Ada Tensor Cores / FP8 | Planned L40S FP8 E4M3 path | Not yet implemented |
| Nsight Systems / Compute | Kernel timeline and counter evidence | Nsight Systems trace shows `qwen2_gqa_decode_kernel`; Nsight Compute remains permission-blocked |

## Public native ABI

The exported ABI in `include/kudacuda.h` exposes:

- handle creation, freeze, destruction, and last-error reporting;
- shape-gated tactic eligibility checks;
- raw-pointer vector-add, RMSNorm, Qwen2 GQA decode, and fused RMSNorm-QKV dispatch;
- an opt-in Qwen2 NF4 fused RMSNorm-QKV ABI with packed weights, block scales, FP16 biases, and separate Q/K/V outputs;
- an opt-in fixed-workspace Qwen2 GQA ABI with a device-resident logical KV-length scalar;
- Qwen2 decode ABI dimensions: batch 1, query length 1, KV length 1..512, 16 query heads, 2 KV heads, head dimension 128, FP16;
- CUDA graph begin/end/replay operations.

The Python bridge validates CUDA residency, contiguity, rank, dtype, and shape before passing `data_ptr()` values and the active stream pointer to native code.

## Current model support boundary

Implemented native tactics:

- `vector_add`, used for pointer, stream, error, and graph validation;
- `rmsnorm`, currently an FP32 pilot with independent reference comparison.
- `qwen2_gqa_decode_sm86_fp16`, an exact decode-only Qwen2.5-3B GQA path on the RTX 3050;
- `qwen2_gqa_decode_sm89_fp16`, source and manifest prepared for the L40S build, pending an L40S runtime run.
- `qwen2_fused_rmsnorm_qkv_sm86_nf4_v1`, a correctness-first NF4 fused RMSNorm-QKV path for the exact Qwen2.5-3B projection shapes; it is active only with the explicit fused flag.
- `qwen2_fused_rmsnorm_qkv_sm86_nf4_v2`, an opt-in tiled NF4 GEMV path with stable FP32 RMSNorm workspace, constant-memory codebook, `uint4` loads, and bias epilogue; its sm86 audit reports 56 registers and zero spills.
- `qwen2_fused_rmsnorm_qkv_sm89_nf4_v2`, compiled for the L40S target; runtime validation is pending on an L40S.
- `qwen2_fused_rmsnorm_qkv_sm86_nf4_v3`, an opt-in two-output-row activation-tiled NF4 GEMV path that stages the normalized 2,048-element activation vector once per block; its sm86 audit reports 92 registers, 8,224 bytes of shared memory, and zero spills. Five model runs preserved exact parity but measured 8.04% below the existing graph median, so v3 is not promoted.
- `qwen2_fused_rmsnorm_qkv_sm89_nf4_v3`, compiled for the L40S target; runtime validation is pending on an L40S.
- `qwen2_gqa_decode_sm86_fp16_workspace_v3`, a fixed-capacity two-stage workspace reduction path validated for KV lengths 1, 7, 64, 128, and 512; it is active only with the explicit workspace flag.

The opt-in graph tactics `qwen2_gqa_decode_sm86_fp16_graph_v1` and `qwen2_gqa_decode_sm89_fp16_graph_v1` use a fixed-capacity cache and a device-resident logical KV-length scalar. The Python `StaticDecodeGraphRunner` captures the full Qwen2 decode forward with PyTorch's graph allocator, while KudaCuda supplies the graph-safe native attention launch. Token sampling stays outside the captured graph.

Experimental attention tactics are also compiled for both targets: `group_tiled_v2` and `vectorized_v2`. They pass independent correctness checks, but their current RTX 3050 timings are slower than the active default, so the manifests keep them opt-in.

The Qwen2 adapter leaves projections, RoPE, cache mutation, output projection, prefill, non-zero masks, BF16, batching, and unsupported KV lengths in Transformers. Gemma still uses framework fallback because its GQA4 contract is not implemented.

Not yet production-ready:

- Gemma GQA attention and KV-cache update;
- a tuned fused residual/RMSNorm-QKV implementation, MLP, or production quantized matmul;
- promoted Tensor-Core INT4/INT8 weight-only kernels; the current NF4 v2/v3 paths are correctness-first tiled GEMV pilots and remain slower than the existing graph tactic on the RTX 3050;
- FP8 kernels for L40S;
- paged KV cache, continuous batching, and a serving scheduler.

Therefore `gqa8_attn` remains `planned`, and the Qwen2 graph, NF4 fused-QKV, and workspace-GQA tactics remain `experimental` pending repeated promotion evidence. Qwen2 native dispatch is reported as active only when the DLL exports the Qwen symbol, the exact contract passes, and at least one native call succeeds; a manifest entry alone never activates it.

## 300 tok/s stretch-target boundary

The current verified whole-model result is 21.209 tok/s on the 4 GiB RTX 3050. The v3 experiment only replaces QKV and reaches 19.503 tok/s median, so it cannot be promoted or extrapolated to 300 tok/s. A credible path to that stretch target must cover the complete decoder workload: all large NF4 projections and MLP GEMVs, fused norm/residual boundaries, a stable full decode graph, and a serving definition that distinguishes single-stream latency from batched aggregate throughput. Paged KV and continuous batching remain future serving layers; offline weight-layout transforms such as SVDQuant/CALDERA-style residual decomposition require separate accuracy and memory experiments.

The first eager operation profile now provides the ranking input: an instrumented decode step measured 61.726 ms of total attention work, 26.334 ms of total MLP work, 8.263 ms of input RMSNorm, and 8.418 ms of post-attention RMSNorm. Within those families, Q/K/V projections accounted for 15.146 ms per step, output projection 7.044 ms, and the MLP gate/up/down projections 10.657/6.868/6.059 ms. These are diagnostic event sums with hook overhead and are not a graph throughput claim; they identify the next native coverage priorities.

## Architecture portability

The host ABI and contract/evidence formats are shared across targets, but device code is compiled separately:

```text
RTX 3050 Laptop / Ampere: nvcc -arch=sm_86
NVIDIA L40S / Ada:        nvcc -arch=sm_89
```

The same host code can target both GPUs. Launch geometry, shared-memory layout, Tensor Core tile shapes, precision paths, workspace capacity, and benchmark thresholds must still be tuned independently. See [architecture-portability.md](architecture-portability.md) for the AMD-to-NVIDIA mapping from the Netra reference.

## NVIDIA references

- [CUDA Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
- [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/driver-api.html)
- [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda/gpus)
- [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/03-1-cuda-programming-model/07-graphs.html)
