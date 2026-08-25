<p align="center">
  <img src="https://github.com/user-attachments/assets/d19f7271-6c3f-4115-9075-e66c4978ab94" alt="KudaCuda Logo" width="220" />
</p>

# KudaCuda

KudaCuda is a CUDA-first sidecar for direct PyTorch device-pointer dispatch. Its serving boundary is intentionally small:

```text
PyTorch / Transformers
        |
        v
Python shape, dtype, and layout guard
        |
        v
Stable C ABI + caller CUDA stream
        |
        v
Target-specific CUDA kernel
```

The native pilot paths are `vector_add`, FP32 `rmsnorm`, decode-only Qwen2 grouped-query attention, an opt-in NF4 fused RMSNorm-QKV projection, an opt-in tiled NF4 GEMV projection, separate opt-in FP16 and BF16 fused NF4 SwiGLU projections, and an opt-in fixed-workspace GQA tactic. Qwen2.5-3B attention is active only for the exact batch-1/query-1/FP16 profile; the BF16 profile currently keeps attention on framework/SDPA while its SwiGLU contract dispatches natively. Prefill, unsupported masks, Gemma/Llama attention, and all other misses use the original Transformers implementation.

The Netra-parity expansion adds a cached CUDA module registry, graph capture/replay, an RMSNorm pilot kernel, independent output comparison, benchmark/profiling entry points, target capability manifests, and a kernel-promotion evidence format. See [architecture-portability.md](docs/architecture-portability.md) and [kernel-lifecycle.md](docs/kernel-lifecycle.md).

For Gemma-specific serving limits, corrected benchmark results, memory sizing, and the prioritized reconfiguration path, see [gemma-serving-reconfiguration.md](docs/gemma-serving-reconfiguration.md).

The complete parity map and implementation checklist are recorded in [architecture-and-nvidia-support.md](docs/architecture-and-nvidia-support.md) and [kernel-lifecycle.md](docs/kernel-lifecycle.md).

## Published documentation

- [Architecture and NVIDIA component support](docs/architecture-and-nvidia-support.md)
- [Linux and Windows getting started](docs/getting-started-linux-windows.md)
- [Benchmark report and scaling results](docs/benchmark-report-2026-08-24.md)
- [Gemma serving reconfiguration](docs/gemma-serving-reconfiguration.md)

## Build

Requirements: CUDA Toolkit with `nvcc`, CMake 3.24 or newer, and a visible NVIDIA GPU for the runtime smoke test. The toolkit and display driver must be compatible; CUDA 13.x requires a driver in the 580+ family. This test is conducted by running driver 610.88 with CUDA UMD 13.3.

On Windows PowerShell:

```powershell
./scripts/build.ps1 -Architecture 86  # RTX 3050 / Ampere
./scripts/build.ps1 -Architecture 89  # L40S / Ada
```

Equivalent CMake commands:

```powershell
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --config Release
```

The Python bridge loads `kudacuda.dll` on Windows and `libkudacuda.so` on Linux. The native launch API does not synchronize; the caller owns stream ordering and synchronization. The Qwen2 bridge passes post-RoPE, post-cache-update Q/K/V pointers directly to the native kernel without staging copies.

For the verified RTX 3050 build in this workspace, use [`artifacts/sm86/kudacuda.dll`](artifacts/sm86/kudacuda.dll). The companion smoke executables are in the same directory. The DLL exports the Qwen2 ABI and passes vector-add, RMSNorm, Qwen2 GQA, and fused RMSNorm-QKV GPU smoke tests.

## Python bridge usage

```python
from pathlib import Path

import sys
sys.path.insert(0, "python")
from kudacuda_sidecar import KudaCudaBridge, load_manifest

root = Path(".")
manifest = root / "manifests" / "rtx3050-sm86.json"
library = root / "artifacts" / "sm86" / "kudacuda.dll"

bridge = KudaCudaBridge(manifest=manifest, lib_path=library, device_id=0)
try:
    # `a`, `b`, and `output` are contiguous CUDA float32 tensors with equal sizes.
    bridge.vector_add(a, b, output, stream_ptr=torch.cuda.current_stream().cuda_stream)
finally:
    bridge.shutdown()
```

Select `manifests/l40s-sm89.json` and the `sm_89` build output on the L40S. The same Qwen2 decode kernel source is compiled for `sm_89`, but an L40S runtime benchmark is still required before claiming production performance there. The `gqa8_attn` tactic remains planned.

## Latest verification and benchmark results

Run date: 2026-08-25. Hardware: NVIDIA GeForce RTX 3050 Laptop GPU, 4,095.5 MiB VRAM, driver 610.88, CUDA UMD 13.3, `nvcc 13.3.73`, Windows/Python 3.13.

| Check | Result | Evidence |
|---|---|---|
| Python contract/tooling suite | 49/49 passed | `python -m pytest -q` |
| Python compilation | Passed | `python -m py_compile` over all tracked Python files |
| Vector-add native smoke | Passed; non-default stream and CUDA Graph replay | `artifacts/sm86/kudacuda_vector_add_smoke.exe` |
| RMSNorm native smoke | Passed against independent FP64 reference | `artifacts/sm86/kudacuda_rmsnorm_smoke.exe` |
| Fused RMSNorm-QKV native smoke | Passed against independent FP32 reference | `artifacts/sm86/kudacuda_rmsnorm_qkv_smoke.exe` |
| NF4 fused RMSNorm-QKV native smoke | Passed with bitsandbytes nibble order and bias-aware ABI | `artifacts/sm86/kudacuda_nf4_fused_qkv_smoke.exe` |
| Tiled NF4 GEMV native smoke | Passed against an independent FP32 reference; constant codebook, `uint4` loads, and bias epilogue | `artifacts/sm86/kudacuda_nf4_tiled_gemv_smoke.exe` |
| Tiled NF4 GEMV v3 native smoke | Passed against an independent FP32 reference; two output rows share a normalized activation tile | `artifacts/sm86/kudacuda_nf4_tiled_gemv_smoke.exe` |
| NF4 fused SwiGLU native smoke | Passed against an independent FP32 RMSNorm/dequantization/SiLU reference; 36 active graph dispatches in model runs | `artifacts/sm86/kudacuda_nf4_swiglu_smoke.exe` |
| BF16 NF4 fused SwiGLU native smoke | Passed against an independent FP32 reference; BF16 input/norm/output ABI; 48 registers and zero ptxas spills | `artifacts/sm86/kudacuda_nf4_swiglu_bf16_smoke.exe` |
| Workspace GQA native smoke | Passed for KV lengths 1, 7, 64, 128, 512 | `artifacts/sm86/kudacuda_qwen2_gqa_workspace_smoke.exe` |
| RMSNorm PyTorch harness | Validated; max absolute error `5.004e-7` | [rmsnorm-sm86-final.json](results/rmsnorm-sm86-final.json) |
| Vector-add PyTorch harness | Validated exactly; 0.0722 ms average GPU event time | [vector-add-sm86-final.json](results/vector-add-sm86-final.json) |
| Qwen2 GQA native correctness | Validated for KV lengths 1, 7, 128, 512; max absolute error `1.22e-4` | [qwen2-gqa-sm86-native-optimized-final.json](results/qwen2-gqa-sm86-native-optimized-final.json) |
| Qwen2 GQA microbenchmark | Native kernel 6.25–10.08× faster than PyTorch eager attention on tested KV lengths | [qwen2-gqa-sm86-microbenchmark-optimized-final.json](results/qwen2-gqa-sm86-microbenchmark-optimized-final.json) |
| Qwen2 Nsight Systems | Native `qwen2_gqa_decode_kernel` visible in GPU kernel summary | [nsys-qwen2-sm86-native.nsys-rep](results/nsys-qwen2-sm86-native.nsys-rep) |
| CUDA Graph multi-launch smoke | Passed; capture/replay matched an independent reference | `artifacts/sm86/kudacuda_cuda_graph_smoke.exe` |
| Qwen2 static decode CUDA Graph | Captured with PyTorch graph pool; exact token/text parity in the verified run | [qwen25-3b-sm86-graph-model-final.json](results/qwen25-3b-sm86-graph-model-final.json) |
| Nsight Systems | Validated; report generated | [profiling-sm86-final.json](results/profiling-sm86-final.json) |
| Nsight Compute | Blocked by `ERR_NVGPUCTRPERM` performance-counter permissions | [profiling-sm86-final.json](results/profiling-sm86-final.json) |
| `sm_89` compilation | Completed with MSVC Build Tools, CUDA 13.3, and CMake/Ninja; runtime pending on an L40S | [sm89-build-final.json](results/sm89-build-final.json) |
| Qwen2 decode operation profile | Eager NF4 profile completed; attention, QKV/output projection, MLP gate/up/down, and both RMSNorm families recorded across 3 decode steps | [throughput ceiling profile](results/qwen25-3b-sm86-throughput-ceiling.json) |
| Nemotron Ollama GPU/CPU check | GPU path previously failed on a 24.9 GiB allocation; earlier CPU run completed, fresh 180-second recheck timed out | [nemotron-ollama-cpu.json](results/nemotron-ollama-cpu.json), [recheck](results/nemotron-ollama-recheck.json) |

### Baseline versus DLL-loaded sidecar

The table uses the baseline and sidecar halves of the same benchmark process, which keeps model loading and prompt conditions paired. The active Qwen2 sidecar row records 576 native attention launches and exact token/text parity.

| Model, 4-bit | Mode | TTFT | ITL | Throughput | Peak VRAM | Output parity |
|---|---|---:|---:|---:|---:|---|
| Gemma 2 2B | Without sidecar | 87.70 ms | 67.86 ms | 14.47 tok/s | 2,263.8 MiB | — |
| Gemma 2 2B | DLL-loaded sidecar | 82.73 ms | 67.77 ms | 14.55 tok/s | 2,263.8 MiB | Exact |
| Qwen2.5 3B, 4-bit | Without sidecar | 207.28 ms | 196.52 ms | 5.071 tok/s | 2,109.7 MiB | — |
| Qwen2.5 3B, 4-bit | Active native sidecar | 217.61 ms | 159.66 ms | 6.125 tok/s | 2,109.5 MiB | Exact; 576 native launches |

Evidence: [active Qwen2 4-bit paired run](results/qwen25-3b-sm86-native-4bit-sidecar-final.json), [Qwen2 native correctness](results/qwen2-gqa-sm86-native-optimized-final.json), [Qwen2 microbenchmark](results/qwen2-gqa-sm86-microbenchmark-optimized-final.json), and [Nsight Systems trace](results/nsys-qwen2-sm86-native.nsys-rep). This paired run improved decode ITL by about 18.7% and throughput by about 20.8%; repeat runs are still needed for a stable production claim. The standalone attention kernel has stronger repeated CUDA-event evidence.

Fresh paired retry on 2026-08-25, using the same Qwen2.5-3B 4-bit model, prompt, 16-token limit, and RTX 3050, produced the following within-process comparison:

| Mode | TTFT | ITL | Throughput | Peak VRAM | Output parity |
|---|---:|---:|---:|---:|---|
| Without KudaCuda | 74.97 ms | 70.95 ms | 14.045 tok/s | 2,109.7 MiB | — |
| Active KudaCuda sidecar | 68.01 ms | 59.34 ms | 16.699 tok/s | 2,109.5 MiB | Exact; 576 native launches |

The fresh sidecar run improved throughput by 18.9%, reduced decode ITL by 16.4%, reduced total generation time by 15.9%, and preserved exact token/text output. Evidence: [fresh paired retry](results/qwen25-3b-sm86-sidecar-retry.json). A separate standalone baseline run measured 13.363 tok/s, showing normal run-to-run variance; the within-process baseline above is the fair comparison.

### Opt-in static decode CUDA Graph

The new `--graph-decode` path captures the complete fixed-shape Qwen2 decode forward with `torch.cuda.CUDAGraph` and its graph memory pool on a dedicated non-default stream. The KudaCuda graph-safe GQA kernel reads logical KV length from a device scalar while using a fixed-capacity cache, so replay changes tensor contents rather than pointers or shapes. Sampling remains outside the graph. A native `begin/end_capture` remains available for CUDA-only callers and smoke tests; it is not wrapped around PyTorch model allocation.

The verified RTX 3050 run used Qwen2.5-3B 4-bit, four generated tokens, and a 512-token KV bucket:

| Mode | TTFT | ITL | Throughput | Peak VRAM | Graph evidence | Output parity |
|---|---:|---:|---:|---:|---|---|
| Without KudaCuda | 74.27 ms | 70.26 ms | 14.033 tok/s | 2,109.7 MiB | — | — |
| Static CUDA Graph sidecar | 73.04 ms | 38.81 ms | 21.110 tok/s | 2,135.6 MiB | 36 native dispatches; 3 replays; 78.59 ms capture | Exact |

This is a one-run, short-decode validation rather than a production promotion. The graph tactic remains experimental until repeated median benchmarks, longer contexts, graph-pool memory sizing, and `sm_89` runtime validation are complete. The benchmark now fails closed as `failed-parity` if a native/graph path produces different tokens or text.

The latest paired graph rerun on the same RTX 3050 reached `23.139 tok/s` versus `15.456 tok/s` without KudaCuda, with exact token/text parity, 36 native dispatches, and 3 graph replays. This is still a short four-token measurement; it is recorded separately from the earlier `21.110 tok/s` run because GPU and allocator variance is material on this 4 GiB laptop.

| Mode | TTFT | ITL | Throughput | Peak VRAM | Output parity |
|---|---:|---:|---:|---:|---|
| Latest paired baseline | 71.69 ms | 62.37 ms | 15.456 tok/s | 2,109.7 MiB | — |
| Latest static graph sidecar | 67.23 ms | 35.21 ms | 23.139 tok/s | 2,135.6 MiB | Exact; 36 dispatches; 3 replays |

Evidence: [latest graph rerun](results/qwen25-3b-sm86-graph-model-current.json).

### NF4 fused QKV and workspace-GQA results

The fused path was tested against the real D-drive Qwen2.5-3B NF4 checkpoint. Projection-level outputs matched bitsandbytes dequantization plus FP32 matmul within FP16 rounding (`max_abs`: Q `0.0078125`, K `0.001953125`, V `0.00048828125`). The four-token graph run produced exact token/text parity and 36 native fused-QKV dispatches during capture, but the correctness-first kernel regressed throughput because it is not yet a tiled GEMV implementation. The latest paired fused run is:

| Mode | TTFT | ITL | Throughput | Peak VRAM | Dispatch evidence | Output parity |
|---|---:|---:|---:|---:|---|---|
| Without KudaCuda | 75.09 ms | 67.03 ms | 14.483 tok/s | 2,109.7 MiB | — | — |
| NF4 fused QKV + static graph | 67.42 ms | 236.30 ms | 5.153 tok/s | 2,135.8 MiB | 36 fused QKV + 36 GQA | Exact |
| Workspace GQA static graph | 71.27 ms | 48.94 ms | 18.340 tok/s | 2,135.7 MiB | 36 workspace GQA | Exact |

Evidence: [NF4 fused QKV run](results/qwen25-3b-sm86-nf4-fused-qkv.json), [five-run NF4 repeat 1](results/qwen25-3b-sm86-nf4-fused-qkv-repeat-1.json), and [workspace GQA run](results/qwen25-3b-sm86-workspace-gqa.json). The five-run NF4 median was 5.126 tok/s versus 15.358 tok/s for the paired baseline, with exact parity in all five runs. The workspace row's paired baseline was 15.543 tok/s. Both tactics remain experimental and opt-in; the existing graph tactic remains the better local path.

#### Tiled NF4 GEMV v2

The v2 path adds a versioned ABI and an opt-in `--tiled-nf4-gemv` hook for the exact Qwen2.5-3B NF4 decode contract. It stages RMSNorm into a stable FP32 workspace, uses a 64-thread one-output-row GEMV with 128-bit `uint4` packed-weight loads, a constant-memory 16-entry NF4 codebook, FP32 accumulation, blocksize-64 scales, and a fused bias epilogue. The sm86 `ptxas` audit reports 56 registers, zero spill stores/loads, 16 bytes of shared memory for GEMV, and 25 registers/zero spills for RMSNorm. The same source builds for sm89; runtime evidence is still required on an L40S.

Five paired four-token graph runs on the RTX 3050 produced:

| Mode | Median TTFT | Median ITL | Median throughput | Median peak VRAM | Output parity |
|---|---:|---:|---:|---:|---|
| Existing graph sidecar | 74.31 ms | 38.88 ms | 21.209 tok/s | 2,135.6 MiB | Exact |
| Tiled NF4 GEMV v2 | 70.85 ms | 43.83 ms | 19.382 tok/s | 2,136.1 MiB | Exact; 36 native v2 dispatches |

The v2 median is 8.61% slower than the existing graph sidecar, so it fails the 5% promotion gate and remains opt-in. It is nevertheless a substantial improvement over v1 fused QKV (~5.1 tok/s under the same graph harness). Evidence: [tiled GEMV aggregate](results/qwen25-3b-sm86-nf4-tiled-gemv-final.json), the five [baseline runs](results/qwen25-3b-sm86-nf4-baseline-repeat-1.json), and the five [tiled runs](results/qwen25-3b-sm86-nf4-tiled-gemv-repeat-1.json).

#### Multi-row activation-tiled NF4 GEMV v3

The v3 experiment applies the next layout idea: one 64-thread block computes two Q rows or two KV rows and loads the normalized 2,048-element activation vector once into shared memory. It keeps the v2 constant-memory NF4 codebook, `uint4` weight loads, blocksize-64 scales, FP32 accumulation, and fused bias epilogue. The ABI, exact shape guards, graph integration, and v2 fallback remain versioned and opt-in.

The implementation passed the independent native smoke test and built for both `sm_86` and `sm_89`. The sm86 `ptxas` audit reported 92 registers, 0 spill stores, 0 spill loads, and 8,224 bytes of shared memory for v3. That is compiler-clean, but the register/shared-memory increase is materially higher than v2's 56 registers and 16 bytes of GEMV shared memory.

Five paired four-token graph runs on the RTX 3050 produced:

| Mode | Median throughput | Median peak VRAM | Output parity | Promotion |
|---|---:|---:|---|---|
| Existing graph sidecar | 21.209 tok/s | 2,135.6 MiB | Exact | Current default |
| Tiled NF4 GEMV v2 | 19.382 tok/s | 2,136.1 MiB | Exact; 36 native v2 dispatches | Not promoted |
| Multi-row NF4 GEMV v3 | 19.503 tok/s | 2,136.1 MiB | Exact; 36 native v3 dispatches | Not promoted |

Relative to the existing graph sidecar, v3 is 8.04% slower. Relative to the paired vanilla halves of its own five runs, it is 30.91% faster, but that is not the promotion comparison because the existing graph already removes the dominant host-launch overhead. The v3 result therefore does not become the main path; `qwen2_gqa_decode_sm86_fp16_graph_v1` remains the default and v3 remains an experimental tactic. The peak-memory increase is only 0.46 MiB. Evidence: [v3 aggregate](results/qwen25-3b-sm86-nf4-gemv-v3-final.json), [v3 repeat 1](results/qwen25-3b-sm86-nf4-gemv-v3-repeat-1.json), [v2 aggregate](results/qwen25-3b-sm86-nf4-tiled-gemv-final.json), and the [native smoke executable](artifacts/sm86/kudacuda_nf4_tiled_gemv_smoke.exe).

The five fresh graph-control runs taken after v3 were strongly clock-state contaminated (median 8.577 tok/s, with the GPU returning to P8/210 MHz between processes), so they are retained as diagnostic evidence rather than mixed into the promotion aggregate. This is why the promotion comparison uses the established five-run graph aggregate and reports the clock-state limitation explicitly.

#### Fused NF4 SwiGLU experiment

The operation profile made MLP work the next high-ROI family, so KudaCuda now has an opt-in Qwen2.5-3B path that wraps the decoder layer boundary and fuses post-attention RMSNorm, NF4 gate/up projections, SiLU, and the gate/up product. The wrapper owns the norm exactly once, forwards the caller CUDA stream, preserves the framework down projection, and falls back to the original decoder layer for prefill or unsupported contracts. This boundary matters because patching `mlp.forward` alone would normalize twice.

The native reference smoke passed, both `sm_86` and `sm_89` compiled, and the sm86 ptxas audit reported 56 registers, 8,264 bytes of shared memory, a 32-byte stack frame, and zero spill stores/loads. The first one-output-row implementation was replaced by a four-output-row activation-tiled implementation; it remained compiler-clean but did not recover the performance gap.

Five four-token graph controls and five fused-SwiGLU graph runs on the RTX 3050 produced:

| Mode | Median throughput | Median peak VRAM | Output parity | Promotion |
|---|---:|---:|---|---|
| Vanilla eager without sidecar | 15.091 tok/s | 2,109.7 MiB | — | Reference |
| Existing graph without fused SwiGLU | 22.333 tok/s | 2,135.6 MiB | Exact | Current default |
| Fused NF4 SwiGLU graph | 13.335 tok/s | 2,136.4 MiB | Exact; 36 native SwiGLU dispatches | Not promoted |

The fused path is 40.29% slower than its fresh graph control and 37.13% below the established 21.209 tok/s graph baseline. This is evidence that a handwritten scalar NF4 projection is behind bitsandbytes' optimized quantized projection path; four-row activation reuse alone does not change that conclusion. The tactic remains available only through `--fused-swiglu`, while `qwen2_gqa_decode_sm86_fp16_graph_v1` remains the main path. Evidence: [SwiGLU aggregate](results/qwen25-3b-sm86-nf4-swiglu-final.json), [first v2 run](results/qwen25-3b-sm86-nf4-swiglu-v2.json), [graph control](results/qwen25-3b-sm86-nf4-swiglu-control-v2.json), and [native smoke](artifacts/sm86/kudacuda_nf4_swiglu_smoke.exe).

#### BF16/NF4 SwiGLU contract

The real D-drive Qwen2.5-3B checkpoint was explicitly loaded with BF16 compute. Its input and post-attention RMSNorm weights are contiguous CUDA BF16 `(1, 2048)` and `(2048,)`; gate/up storage is non-nested NF4 with packed shape `(11272192, 1)`, logical shape `(11008, 2048)`, blocksize 64, float32 absmax, and no projection bias. KudaCuda now exposes this as the separate `qwen2_swiglu_sm86_bf16_nf4_v1` contract and keeps the existing FP16 `qwen2_swiglu_sm86_nf4_v1` contract unchanged.

The BF16 native kernel passed its independent reference smoke, compiled for `sm_86` and `sm_89`, and the sm86 `ptxas` audit reported 48 registers, 8,264 bytes of shared memory, a 32-byte stack frame, and zero spill stores/loads. The graph runner also updates floating graph masks to the current logical KV length so framework BF16/SDPA attention preserves exact behavior during replay.

Five four-token RTX 3050 graph runs passed exact token/text parity and executed 36 native BF16 SwiGLU dispatches per run. The tactic is not promoted: median throughput was 12.305 tok/s versus 17.119 tok/s for the paired graph control without fused SwiGLU (`-28.12%`), while peak VRAM increased by 0.756 MiB. The result shows the ABI and integration are correct, but the scalar handwritten NF4 projection is slower than the framework's optimized quantized path. Use it for BF16 contract validation with `--fused-swiglu-bf16`; the existing graph remains the performance path. Evidence: [BF16 aggregate](results/qwen25-3b-sm86-nf4-swiglu-bf16-final.json), [active repeat 1](results/qwen25-3b-sm86-nf4-swiglu-bf16-active-1.json), [control repeat 1](results/qwen25-3b-sm86-nf4-bf16-control-1.json), and [BF16 smoke](artifacts/sm86/kudacuda_nf4_swiglu_bf16_smoke.exe).

### 300 tok/s execution plan

The 300 tok/s target is a stretch target for this 4 GiB RTX 3050, not a result of the current QKV experiment. At present, v3 changes only the QKV projection and the verified whole-model ceiling is 21.209 tok/s. Reaching 300 tok/s would require a different serving envelope—likely fixed-shape, single-token decode with all large NF4 projections native, full graph capture, and/or aggregate throughput from batching. The remaining work is staged as follows:

1. Profile the 36-layer decode to rank time by operation and separate GPU math from graph, allocator, and clock variance. The current eager diagnostic measured a 109.857 ms median instrumented decode step: attention total 61.726 ms/step, MLP total 26.334 ms/step, input/post RMSNorm 8.263/8.418 ms/step, QKV projections 15.146 ms/step, and output projection 7.044 ms/step. These are diagnostic event sums, not a production throughput claim.
2. Do not expand the scalar handwritten NF4 GEMV family to more projections until it matches the framework's quantized kernels; the SwiGLU result shows that correctness and zero spills are insufficient. The next native family should use a tensor-core-compatible packed layout or an offline dequantized weight format.
3. Fuse RMSNorm, projection, residual, and activation boundaries only after each constituent kernel has independent correctness and `ptxas` evidence.
4. Capture the complete fixed-shape native decode graph with stable workspaces and paged KV indirection; retain recapture/fallback buckets for dynamic shapes.
5. Benchmark single-stream latency and batched aggregate tok/s separately on the RTX 3050, then repeat runtime validation on an L40S `sm_89` system before making a production claim.

Netra's contract/tactic/evidence model remains the right organizing pattern for this work, while SVDQuant and CALDERA are research directions for offline weight-layout experiments rather than changes that should be mixed into the current correctness path. No 300 tok/s claim is made until the full-model benchmark demonstrates it.

### Attention optimization experiments

The production default remains the per-query-head `qwen2_gqa_decode_sm86_fp16` tactic. It measured `0.0657 ms` at KV length 512 in the direct CUDA-event baseline. Two new FP16 variants were compiled, checked against an independent FP32 reference, and kept experimental because their small-GPU launch geometry is not yet better than the default:

| Tactic | KV 512 time | Result |
|---|---:|---|
| Per-query-head baseline | 0.0657 ms | Active default |
| Group-tiled v2 | 0.1713 ms | Correct; experimental, not promoted |
| Vectorized v2 | 0.1140 ms | Correct; experimental, not promoted |

The group-tiled kernel reuses KV tiles across query heads, but its current two-KV-group grid underutilizes the RTX 3050 at long KV lengths. The vectorized kernel preserves query-head parallelism and is closer, but still loses to the existing baseline. See the [group-tiled microbenchmark](results/qwen2-gqa-sm86-group-tiled-microbenchmark.json), [vectorized microbenchmark](results/qwen2-gqa-sm86-vectorized-microbenchmark.json), [group-tiled model run](results/qwen25-3b-sm86-group-tiled-model-final.json), and [vectorized model run](results/qwen25-3b-sm86-vectorized-model.json).

The NF4 fused RMSNorm-QKV, tiled NF4 GEMV, FP16/BF16 fused NF4 SwiGLU, and workspace GQA kernels are validated and integrated as opt-in experimental paths. They are not the default because each measured projection/workspace variant regressed against the existing graph tactic. The static decode graph remains the best verified KudaCuda path; Gemma, batching, paged KV, and BF16 native attention remain future work.

## Validation

```powershell
$env:PYTHONPATH = (Resolve-Path ..).Path
python -m unittest discover -s tests -p 'test_*.py' -v
python -m py_compile python/kudacuda_sidecar.py harness/reference.py harness/compare.py
build-sm86/kudacuda_vector_add_smoke.exe
```

The Python contract suite is runnable without PyTorch or Transformers. The CUDA smoke executable requires the CUDA toolkit, a compatible driver, and a visible GPU.

When CUDA is available, build and run both pilot kernels:

```powershell
./scripts/build.ps1 -Architecture 86
build-sm86/kudacuda_vector_add_smoke.exe
build-sm86/kudacuda_rmsnorm_smoke.exe
```

Optional evidence tooling uses PyTorch only when a CUDA-enabled installation is available:

```powershell
python tools/correctness/run_rmsnorm_check.py `
  --manifest manifests/rtx3050-sm86.json `
  --library artifacts/sm86/kudacuda.dll `
  --output results/rmsnorm-sm86.json
python tools/benchmark/benchmark_vector_add.py `
  --manifest manifests/rtx3050-sm86.json `
  --library artifacts/sm86/kudacuda.dll `
  --output results/vector-add-sm86.json
python tools/profiling/profile_command.py `
  --tool nsys --output results/nsys-sm86 -- <your-command>
```

The tools emit `skipped` evidence when PyTorch, a GPU, or Nsight is unavailable. Planned families remain framework fallbacks; see `results/kernel-status.json` before enabling a tactic.

Qwen2 native checks:

```powershell
python tools/correctness/run_qwen2_gqa_check.py `
  --manifest manifests/rtx3050-sm86.json `
  --library artifacts/sm86/kudacuda.dll `
  --output results/qwen2-gqa-sm86-native.json
python tools/benchmark/benchmark_qwen2_gqa.py `
  --manifest manifests/rtx3050-sm86.json `
  --library artifacts/sm86/kudacuda.dll `
  --output results/qwen2-gqa-sm86-microbenchmark.json
```

## Shoutout
Thank you for netra runtime to make this project approachable https://github.com/NetraRuntime/netra-kernel/tree/main
