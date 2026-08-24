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

The native pilot paths are `vector_add`, FP32 `rmsnorm`, and decode-only Qwen2 grouped-query attention. Qwen2.5-3B attention is active only for the exact batch-1/query-1/FP16 profile; prefill, unsupported masks, Gemma/Llama attention, and all other misses use the original Transformers implementation.

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

For the verified RTX 3050 build in this workspace, use [`artifacts/sm86/kudacuda.dll`](artifacts/sm86/kudacuda.dll). The companion smoke executables are in the same directory. The DLL exports the Qwen2 ABI and passes vector-add, RMSNorm, and Qwen2 GQA GPU smoke tests.

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

Run date: 2026-08-24. Hardware: NVIDIA GeForce RTX 3050 Laptop GPU, 4,095.5 MiB VRAM, driver 610.88, CUDA UMD 13.3, `nvcc 13.3.73`, Windows/Python 3.13.

| Check | Result | Evidence |
|---|---|---|
| Python contract/tooling suite | 25/25 passed | `python -m unittest discover -s tests -p 'test_*.py' -v` |
| Python compilation | Passed | `python -m py_compile` over all tracked Python files |
| Vector-add native smoke | Passed; non-default stream and CUDA Graph replay | `artifacts/sm86/kudacuda_vector_add_smoke.exe` |
| RMSNorm native smoke | Passed against independent FP64 reference | `artifacts/sm86/kudacuda_rmsnorm_smoke.exe` |
| RMSNorm PyTorch harness | Validated; max absolute error `5.004e-7` | [rmsnorm-sm86-final.json](results/rmsnorm-sm86-final.json) |
| Vector-add PyTorch harness | Validated exactly; 0.0722 ms average GPU event time | [vector-add-sm86-final.json](results/vector-add-sm86-final.json) |
| Qwen2 GQA native correctness | Validated for KV lengths 1, 7, 128, 512; max absolute error `1.22e-4` | [qwen2-gqa-sm86-native-optimized-final.json](results/qwen2-gqa-sm86-native-optimized-final.json) |
| Qwen2 GQA microbenchmark | Native kernel 6.25–10.08× faster than PyTorch eager attention on tested KV lengths | [qwen2-gqa-sm86-microbenchmark-optimized-final.json](results/qwen2-gqa-sm86-microbenchmark-optimized-final.json) |
| Qwen2 Nsight Systems | Native `qwen2_gqa_decode_kernel` visible in GPU kernel summary | [nsys-qwen2-sm86-native.nsys-rep](results/nsys-qwen2-sm86-native.nsys-rep) |
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
