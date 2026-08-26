<p align="center">
  <img src="https://github.com/user-attachments/assets/d19f7271-6c3f-4115-9075-e66c4978ab94" alt="KudaCuda Logo" width="220" />
</p>

# KudaCuda

CUDA-first ahead-of-time GPU kernels for production-oriented inference on NVIDIA hardware.

KudaCuda turns exact model-operation contracts into versioned CUDA tactics and loadable native sidecar artifacts. It brings a contract-driven workflow to CUDA inference: validate the serving shape, dtype, layout, quantization, launch geometry, workspace, and stream contract ahead of time, then launch only the specialized path at runtime.

The current production-oriented backend targets NVIDIA Ampere `sm_86`, with an `sm_89` build track for Ada Lovelace. The validated local deployment is Qwen2.5-3B 4-bit inference on an RTX 3050 Laptop GPU, with a fixed-shape CUDA Graph sidecar as the current default path.

[Get started](#get-started) · [Explore the architecture](#the-kudacuda-platform) · [Read the contract model](#contracts-and-tactics) · [Integrate Transformers](#python-transformers-integration) · [Contribute](#contributing)

## What KudaCuda delivers

Native pointer dispatch without a model-locked host integration. Python guards intercept supported Transformers boundaries, extract device pointers from live tensors, and call a stable C ABI on the caller-owned CUDA stream.

Predictable serving. Shapes, dtypes, layouts, quantization metadata, launch dimensions, shared-memory requirements, workspaces, and graph-capture constraints are resolved before dispatch.

A minimal hot path. The active Qwen2 decode path passes post-cache-update Q/K/V pointers directly to the native CUDA kernel. No staging tensor or host copy is introduced by the sidecar.

Safe specialization. Exact predicates prevent approximate dispatch. Unsupported shapes, prefill, masks, dtypes, devices, or cache layouts return to the original PyTorch/Transformers implementation.

Evidence-based promotion. Tactics carry an explicit maturity state and are promoted only when correctness, parity, memory, compiler, and matched end-to-end performance evidence agree.

KudaCuda is intentionally not one dynamically generic CUDA kernel. Generality lives in the contract registry, manifests, Python adapters, and kernel families; generated machine code remains schedule-specific.

## Validated local deployment

The current qualification result is a fixed-shape Qwen2.5-3B 4-bit decode run on an NVIDIA GeForce RTX 3050 Laptop GPU.

| Deployment | Result |
|---|---:|
| RTX 3050 Laptop GPU, `sm_86`, 4,095.5 MiB VRAM | Validated |
| Eager Transformers control, 128 generated tokens | 15.090 steady-state tok/s |
| KudaCuda static CUDA Graph sidecar, 128 generated tokens | 51.825 steady-state tok/s |
| Candidate p25 versus control p75 | 51.722 versus 15.194 tok/s |
| Exact token/text parity | 7 / 7 runs |
| Peak VRAM | 2,135.6 MiB with the sidecar |

The graph result is the current verified local serving path. It describes this checkpoint, prompt, fixed KV bucket, model configuration, driver, allocator state, graph mode, and seven-run measurement; it is not a universal performance claim and is not an L40S result.

The promotion evidence is recorded in [the aggregate result](results/qwen25-3b-sm86-nf4-128tok-graph-aggregate.json) and [the Tier-1 benchmark report](docs/benchmark-report-2026-08-25-tier1.md).

### Achievable interactive response on the RTX 3050

A fresh side-by-side Transformers chat run used the same Qwen2.5-3B checkpoint, prompts, and `--max-new-tokens 128`. KudaCuda was confirmed active by `reason=native`, the implemented tactic, and nonzero native dispatch counts.

| Chat turn | Generated | Without sidecar | KudaCuda sidecar | Sidecar improvement |
|---|---:|---:|---:|---:|
| Short greeting | 10 tokens | 5.19 tok/s, 1.926 s | 7.72 tok/s, 1.296 s | 48.8% tok/s |
| 128-token sky explanation | 128 tokens | 11.62 tok/s, 11.019 s | 15.45 tok/s, 8.283 s | 33.0% tok/s |
| Follow-up assistant answer | 128 tokens | Not measured | 15.35 tok/s, 8.340 s | — |

Across the two paired turns, KudaCuda generated 138 tokens in 9.579 seconds versus 12.946 seconds without the sidecar: 14.41 versus 10.66 token/s, or approximately 35.2% higher measured response throughput. The 128-token turns reached 15.35–15.45 tok/s with native attention active. These are interactive measurements on this exact laptop, checkpoint, driver state, prompts, and terminal session; they are an achievable local response range, not a universal hardware guarantee. The one-time model weight-load display was approximately 3 seconds with the sidecar and 5 seconds without it, but startup loading was not controlled as a benchmark variable.

### Direct decode long-context qualification

The interactive Transformers path now has a separate direct-decode KV profile through 2,048 tokens. This removes the accidental 512-token cliff that previously sent a chat turn such as `kv_len=585` to framework attention. The fixed-shape graph and experimental GQA profiles remain capped at 512 until separately tuned.

| Logical KV length | Native GPU time | PyTorch eager GPU time | Native speedup vs eager |
|---:|---:|---:|---:|
| 512 | 0.066 ms | 0.093 ms | 1.40x |
| 585 | 0.075 ms | 0.104 ms | 1.38x |
| 1,024 | 0.128 ms | 0.214 ms | 1.67x |
| 2,048 | 0.447 ms | 0.119 ms | 0.27x |

All seven direct-kernel cases (1, 7, 128, 512, 585, 1,024, and 2,048) passed the independent FP32 reference check. The 2,048-token result is correctness-qualified but not performance-promoted; use the direct profile through the tested 1,024-token range while the long-context tactic is optimized further. Evidence: [correctness](results/qwen2-gqa-sm86-long-context-build.json) and [microbenchmark](results/qwen2-gqa-sm86-long-context-build-benchmark.json).

## The KudaCuda platform

```text
Hugging Face Transformers / PyTorch
                |
                v
      Python sidecar adapter
      exact shape/dtype/layout guards
                |
                v
       Stable KudaCuda C ABI
       raw device pointers + CUDA stream
                |
                v
        Native module registry
        manifest-selected tactic
                |
        +-------+--------+
        |                |
        v                v
  Direct fixed launch   PyTorch CUDA Graph
  caller-owned stream   stable pointers/workspace
        |                |
        +-------+--------+
                v
      Specialized CUDA kernel
                |
        unsupported contract
                v
       Framework fallback
```

### Host integration

The Python bridge is the framework boundary. It monkey-patches only the supported model operation, verifies the complete contract, extracts `.data_ptr()` values, normalizes the CUDA stream handle, and invokes the native library. The native launch API does not synchronize; the caller owns stream ordering and synchronization.

### Native runtime

The runtime loads `kudacuda.dll` on Windows or `libkudacuda.so` on Linux, resolves versioned symbols through the registry, tracks native errors, and exposes direct launches plus CUDA capture helpers. The public C ABI is in [`include/kudacuda.h`](include/kudacuda.h).

### CUDA execution

KudaCuda uses NVIDIA CUDA components at their natural boundaries:

| NVIDIA component | KudaCuda use |
|---|---|
| `nvcc` and `ptxas` | Ahead-of-time compilation and register/spill audits |
| CUDA Runtime API | Kernel launches, events, device memory, and stream operations |
| CUDA streams | Caller-owned ordering and asynchronous pointer dispatch |
| CUDA Graphs | Fixed-shape decode capture and replay |
| CUDA device pointers | Zero-copy tensor boundary between PyTorch and native kernels |
| Shared/constant memory | Experimental GQA tiling and NF4 codebook layouts |
| Ampere `sm_86` | Validated RTX 3050 target |
| Ada `sm_89` | Build target; L40S runtime qualification remains pending |

KudaCuda complements PyTorch and bitsandbytes. It does not replace the framework's quantized projection implementation unless an opt-in native tactic passes the promotion gate.

## Contracts and tactics

A kernel is eligible only when the complete serving contract matches:

- operation and exact dimensions;
- batch, query length, KV length, and cache capacity;
- input/output dtypes and accumulation type;
- tensor, cache, and weight layouts;
- quantization format, block size, scales, nibble order, and bias semantics;
- ABI, launch dimensions, stream behavior, and workspace;
- graph-capture pointer stability and determinism requirements.

When every field matches, the adapter can select the same tactic across compatible model frontends. When a field differs, KudaCuda emits an explicit miss and preserves the framework path. Model names are not kernel identities.

### Tactic maturity

| State | Meaning |
|---|---|
| `planned` | Contract or implementation is recorded but not dispatchable |
| `experimental` | Opt-in path with correctness evidence but no promotion |
| `verified` | Correctness, ABI, and compiler evidence passed for the recorded scope |
| `active` | Default path for the exact contract and evidence scope |
| `rejected` | Retained as diagnostic evidence and never selected by default |

A faster microbenchmark alone does not promote a tactic. The current gate uses a long decode, seven-run statistics, exact parity where applicable, stable memory use, and a candidate p25 greater than the control p75.

## Supported today

| Capability | Current support |
|---|---|
| GPU target | NVIDIA Ampere `sm_86`; Ada `sm_89` compilation track |
| Validated device | RTX 3050 Laptop GPU, 4 GiB class |
| Runtime platforms | Windows DLL and Linux shared-library paths |
| Active native operation | Qwen2 decode-only FP16 grouped-query attention; direct profile supports KV length 1..2,048 |
| Graph mode | Fixed-shape Qwen2 decode graph with stable cache capacity and device-resident logical KV length; current bucket remains 512 |
| Other native pilots | Vector add, FP32 RMSNorm, RMSNorm-QKV reference, Qwen2 workspace/GQA variants |
| Quantization focus | Qwen2.5-3B bitsandbytes NF4 integration; handwritten NF4 projection tactics remain experimental |
| BF16 support | Separate experimental BF16/NF4 SwiGLU ABI and native smoke coverage |
| Framework integration | Transformers/PyTorch Python sidecar with fail-closed fallback |
| Model coverage | Qwen2.5-3B decode path validated; Gemma harness and fallback plumbing available |
| L40S status | `sm_89` artifacts compile; runtime and performance acceptance still require an L40S |

Prefill, unsupported masks, dynamic contracts outside the configured graph bucket, unsupported model families, and unpromoted tactics remain on the framework path. The interactive chat tool prints a KV preflight warning before a request that would exceed its loaded direct profile and supports `/reset` to clear accumulated history.

## Repository layout

The public integration surface is separated from target-specific kernels, runtime code, and evidence:

```text
KudaCuda/
├── include/                    # Stable framework-neutral C ABI
│   └── kudacuda.h
├── runtime/                    # Native runtime and module registry
├── kernels/common/             # Shared contracts and device helpers
├── kernels/sm86/               # Ampere-specific CUDA kernels
├── kernels/sm89/               # Ada-specific CUDA kernels
├── python/                     # Sidecar, graph plan, and static decode helpers
├── integrations/transformers/  # Transformers integration notes and adapters
├── manifests/                  # Device and tactic contract manifests
├── tools/                      # Build, benchmark, correctness, and profiling tools
├── tests/                      # Python contract tests and CUDA smoke tests
├── artifacts/                  # Locally built DLLs and smoke executables
├── results/                    # Machine-readable benchmark and validation evidence
└── docs/                       # Architecture, deployment, and lifecycle guides
```

KudaCuda currently uses CMake and `nvcc` as its AOT build layer. The manifests and native registry provide contract selection; this repository does not yet claim to provide Netra's separate model compiler, engine-directory format, or installable SDK distribution.

## Get started

### Requirements

CPU-only contract tests require Python 3.10 or newer. CUDA builds require:

- NVIDIA CUDA Toolkit with `nvcc`;
- CMake 3.24 or newer;
- a compatible display driver;
- a visible NVIDIA GPU for runtime smoke tests.

The validated local environment used CUDA 13.3, driver 610.88, Python 3.13, and PyTorch `2.11.0+cu128`.

### Clone and build on Windows

```powershell
git clone https://github.com/lijeuki/KudaCuda.git
cd KudaCuda

# RTX 3050 / Ampere
./scripts/build.ps1 -Architecture 86

# L40S / Ada build target
./scripts/build.ps1 -Architecture 89
```

The script automatically loads the VS 2022 developer environment and uses
Ninja because this CUDA 13.3 installation is not registered with CMake's
Visual Studio CUDA toolset discovery. Equivalent commands from a developer
shell are:

```powershell
cmake -S . -B build-sm86-ninja -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_CUDA_COMPILER="$env:CUDA_PATH\bin\nvcc.exe" `
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-sm86-ninja --parallel
```

The verified prebuilt local artifact is [`artifacts/sm86/kudacuda.dll`](artifacts/sm86/kudacuda.dll). Use the `sm_89` artifact only after validating it on an actual L40S.

### Build on Linux

```bash
git clone https://github.com/lijeuki/KudaCuda.git
cd KudaCuda

cmake -S . -B build-sm86 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-sm86 --parallel
```

For an L40S build, replace `86` with `89`. The Python bridge looks for `libkudacuda.so` on Linux and `kudacuda.dll` on Windows.

### Run the contract and smoke tests

```powershell
$env:PYTHONPATH = (Resolve-Path ..).Path
python -m unittest discover -s tests -p 'test_*.py' -v

build-sm86/kudacuda_vector_add_smoke.exe
build-sm86/kudacuda_rmsnorm_smoke.exe
build-sm86/kudacuda_qwen2_gqa_smoke.exe
```

The Python suite runs without PyTorch or a GPU. CUDA smoke executables require the toolkit, compatible driver, and visible device.

### Python Transformers integration

```python
from pathlib import Path

import sys
import torch

sys.path.insert(0, "python")
from kudacuda_sidecar import KudaCudaBridge

root = Path(".")
bridge = KudaCudaBridge(
    manifest=root / "manifests" / "rtx3050-sm86.json",
    lib_path=root / "artifacts" / "sm86" / "kudacuda.dll",
    device_id=0,
)

try:
    # All tensors must already be contiguous CUDA tensors with the contract dtype.
    bridge.vector_add(
        a,
        b,
        output,
        stream_ptr=torch.cuda.current_stream().cuda_stream,
    )
finally:
    bridge.shutdown()
```

For model serving, use the Transformers adapter and enable only the exact tactic recorded by the manifest. Unsupported calls must remain framework fallbacks.

## Benchmark and validation

Run the long decode qualification rather than using a short generation whose prefill dominates the result:

```powershell
python tools/benchmark/benchmark_gemma.py `
  --model D:\AI\Models\HuggingFace\hub\models--Qwen--Qwen2.5-3B-Instruct\snapshots\<revision> `
  --manifest manifests/rtx3050-sm86.json `
  --library artifacts/sm86/kudacuda.dll `
  --load-in-4bit `
  --graph-decode `
  --graph-kv-bucket 512 `
  --max-new-tokens 128 `
  --output results/qwen25-3b-128tok.json
```

Use a paired eager control with the same model, prompt, token limit, and hardware. The benchmark reports TTFT, decode ITL, steady-state tok/s, peak VRAM, parity, dispatch counts, graph replays, runtime audit, and clock-state observations.

The current result is:

| Mode | Median ITL | Median steady-state | Peak VRAM | Parity |
|---|---:|---:|---:|---|
| Eager control | 66.269 ms | 15.090 tok/s | 2,113.4 MiB | Reference |
| KudaCuda static graph | 19.296 ms | 51.825 tok/s | 2,135.6 MiB | Exact, 7/7 |

Evidence links:

- [Tier-1 benchmark report](docs/benchmark-report-2026-08-25-tier1.md)
- [128-token aggregate](results/qwen25-3b-sm86-nf4-128tok-graph-aggregate.json)
- [Architecture and NVIDIA support map](docs/architecture-and-nvidia-support.md)
- [Linux and Windows guide](docs/getting-started-linux-windows.md)
- [Kernel lifecycle and promotion](docs/kernel-lifecycle.md)

### Experimental tactics

The following four-token graph experiments are retained as engineering evidence. They are not directly comparable to the 128-token qualification above and are not the default path:

| Tactic | Median throughput | Result |
|---|---:|---|
| Existing static graph | 21.209 tok/s | Current default in the historical four-token harness |
| NF4 fused QKV | 5.126 tok/s | Exact parity; rejected for performance |
| Tiled NF4 GEMV v2 | 19.382 tok/s | Exact parity; 8.61% below graph baseline |
| Multi-row NF4 GEMV v3 | 19.503 tok/s | Exact parity; 8.04% below graph baseline |
| NF4 fused SwiGLU | 13.335 tok/s | Exact parity; rejected for performance |
| BF16/NF4 fused SwiGLU | 12.305 tok/s | Exact parity; rejected for performance |
| Workspace GQA | 18.340 tok/s | Better than eager in its paired run, below static graph |

These results show why compiler-clean code, zero observed ptxas spills, and native correctness are necessary but not sufficient for promotion. The handwritten NF4 projection paths currently trail the framework's optimized quantized projection route.

## Current serving recommendation

Use the fixed-shape Qwen2 static CUDA Graph sidecar as the main local path when its complete contract matches. Keep the projection and SwiGLU kernels opt-in until they beat the active graph tactic under the long-run promotion gate.

The 300 tok/s goal remains a stretch target, not a current result. Achieving it on an RTX 3050 likely requires a different serving envelope—optimized tensor-core-compatible low-bit projections, larger-batch aggregate throughput, speculative or prompt-lookup decoding, and further graph/workspace integration. No 300 tok/s claim is made here.

## Documentation

- [Architecture and NVIDIA component support](docs/architecture-and-nvidia-support.md)
- [Architecture portability](docs/architecture-portability.md)
- [Linux and Windows getting started](docs/getting-started-linux-windows.md)
- [Kernel lifecycle and promotion](docs/kernel-lifecycle.md)
- [Gemma serving reconfiguration](docs/gemma-serving-reconfiguration.md)
- [Benchmark report: initial validation](docs/benchmark-report-2026-08-24.md)
- [Benchmark report: Tier-1 qualification](docs/benchmark-report-2026-08-25-tier1.md)
- [Transformers integration notes](integrations/transformers/README.md)

## Contributing

Contributions to the CUDA kernels, C ABI, runtime registry, manifests, Python adapters, benchmark harnesses, correctness tests, and documentation are welcome.

Every new tactic should include:

1. an explicit contract and guard predicate;
2. an independent numerical reference test;
3. an `sm_86`/`sm_89` build result where applicable;
4. ptxas register and spill evidence;
5. matched baseline and candidate measurements;
6. a documented promotion or rejection decision.

Keep unsupported contracts fail-closed and preserve the framework fallback. Read the existing architecture and lifecycle guides before changing a dispatch boundary.

## License

MIT © KudaCuda contributors.

KudaCuda is inspired by the contract-driven engineering patterns in [Netra Kernel](https://github.com/NetraRuntime/netra-kernel), adapted here for NVIDIA CUDA, PyTorch device pointers, CUDA streams, and CUDA Graph replay.
