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

The native pilot paths are `vector_add`, FP32 `rmsnorm`, decode-only Qwen2 grouped-query attention, an opt-in NF4 fused RMSNorm-QKV projection, an opt-in tiled NF4 GEMV projection, and an opt-in fixed-workspace GQA tactic. Qwen2.5-3B attention is active only for the exact batch-1/query-1/FP16 profile; prefill, unsupported masks, Gemma/Llama attention, and all other misses use the original Transformers implementation.

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
| Python contract/tooling suite | 42/42 passed | `python -m pytest -q` |
| Python compilation | Passed | `python -m py_compile` over all tracked Python files |
| Vector-add native smoke | Passed; non-default stream and CUDA Graph replay | `artifacts/sm86/kudacuda_vector_add_smoke.exe` |
| RMSNorm native smoke | Passed against independent FP64 reference | `artifacts/sm86/kudacuda_rmsnorm_smoke.exe` |
| Fused RMSNorm-QKV native smoke | Passed against independent FP32 reference | `artifacts/sm86/kudacuda_rmsnorm_qkv_smoke.exe` |
| NF4 fused RMSNorm-QKV native smoke | Passed with bitsandbytes nibble order and bias-aware ABI | `artifacts/sm86/kudacuda_nf4_fused_qkv_smoke.exe` |
| Tiled NF4 GEMV native smoke | Passed against an independent FP32 reference; constant codebook, `uint4` loads, and bias epilogue | `artifacts/sm86/kudacuda_nf4_tiled_gemv_smoke.exe` |
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

### Attention optimization experiments

The production default remains the per-query-head `qwen2_gqa_decode_sm86_fp16` tactic. It measured `0.0657 ms` at KV length 512 in the direct CUDA-event baseline. Two new FP16 variants were compiled, checked against an independent FP32 reference, and kept experimental because their small-GPU launch geometry is not yet better than the default:

| Tactic | KV 512 time | Result |
|---|---:|---|
| Per-query-head baseline | 0.0657 ms | Active default |
| Group-tiled v2 | 0.1713 ms | Correct; experimental, not promoted |
| Vectorized v2 | 0.1140 ms | Correct; experimental, not promoted |

The group-tiled kernel reuses KV tiles across query heads, but its current two-KV-group grid underutilizes the RTX 3050 at long KV lengths. The vectorized kernel preserves query-head parallelism and is closer, but still loses to the existing baseline. See the [group-tiled microbenchmark](results/qwen2-gqa-sm86-group-tiled-microbenchmark.json), [vectorized microbenchmark](results/qwen2-gqa-sm86-vectorized-microbenchmark.json), [group-tiled model run](results/qwen25-3b-sm86-group-tiled-model-final.json), and [vectorized model run](results/qwen25-3b-sm86-vectorized-model.json).

The NF4 fused RMSNorm-QKV, tiled NF4 GEMV, and workspace GQA kernels are validated and integrated as opt-in experimental paths. They are not the default because each measured projection/workspace variant regressed against the existing graph tactic. The static decode graph remains the best verified KudaCuda path; none of these experiments yet cover Gemma, BF16, batching, or paged KV.

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
