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
| CUDA Graphs | Capture, instantiate/replay, and destroy the vector-add path | Verified by `kudacuda_vector_add_smoke.exe` |
| CUBIN/PTX/fatbin | Architecture-specific kernel artifacts | `sm_86` artifact is published; build with `89` for L40S |
| Ampere Tensor Cores | Planned INT4/INT8 and mixed-precision matrix paths | Not yet implemented |
| Ada Tensor Cores / FP8 | Planned L40S FP8 E4M3 path | Not yet implemented |
| Nsight Systems / Compute | Kernel timeline and counter evidence | Nsight Systems trace shows `qwen2_gqa_decode_kernel`; Nsight Compute remains permission-blocked |

## Public native ABI

The exported ABI in `include/kudacuda.h` exposes:

- handle creation, freeze, destruction, and last-error reporting;
- shape-gated tactic eligibility checks;
- raw-pointer vector-add, RMSNorm, and Qwen2 GQA decode dispatch;
- Qwen2 decode ABI dimensions: batch 1, query length 1, KV length 1..512, 16 query heads, 2 KV heads, head dimension 128, FP16;
- CUDA graph begin/end/replay operations.

The Python bridge validates CUDA residency, contiguity, rank, dtype, and shape before passing `data_ptr()` values and the active stream pointer to native code.

## Current model support boundary

Implemented native tactics:

- `vector_add`, used for pointer, stream, error, and graph validation;
- `rmsnorm`, currently an FP32 pilot with independent reference comparison.
- `qwen2_gqa_decode_sm86_fp16`, an exact decode-only Qwen2.5-3B GQA path on the RTX 3050;
- `qwen2_gqa_decode_sm89_fp16`, source and manifest prepared for the L40S build, pending an L40S runtime run.

The Qwen2 adapter leaves projections, RoPE, cache mutation, output projection, prefill, non-zero masks, BF16, batching, and unsupported KV lengths in Transformers. Gemma still uses framework fallback because its GQA4 contract is not implemented.

Not yet implemented:

- Gemma GQA attention and KV-cache update;
- fused residual/RMSNorm, MLP, or quantized matmul;
- INT4/INT8 weight-only kernels;
- FP8 kernels for L40S;
- paged KV cache, continuous batching, and a serving scheduler.

Therefore `gqa8_attn` remains `planned`. Qwen2 native dispatch is reported as active only when the DLL exports the Qwen symbol, the exact contract passes, and at least one native call succeeds; a manifest entry alone never activates it.

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
