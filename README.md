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

The first real kernel is `vector_add`. It is a complete smoke path for raw device pointers, stream forwarding, native error propagation, and architecture-specific compilation. Llama attention is not intercepted unless a real attention tactic is registered; unsupported attention always uses the original Transformers implementation.

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

The Python bridge loads `kudacuda.dll` on Windows and `libkudacuda.so` on Linux. The native launch API does not synchronize; the caller owns stream ordering and synchronization.

For the verified RTX 3050 build in this workspace, use [`artifacts/sm86/kudacuda.dll`](artifacts/sm86/kudacuda.dll). The companion smoke executables are in the same directory. The DLL loads, exports the complete public C ABI, and passes the vector-add graph and RMSNorm GPU smoke tests.

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

Select `manifests/l40s-sm89.json` and the `sm_89` build output on the L40S. The `gqa8_attn` tactic is intentionally absent from both manifests, so `patch_llama_attention` delegates to the original Transformers method until a separately validated attention kernel is added.

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
  --output results/rmsnorm-sm86.json
python tools/benchmark/benchmark_vector_add.py `
  --manifest manifests/rtx3050-sm86.json `
  --output results/vector-add-sm86.json
python tools/profiling/profile_command.py `
  --tool nsys --output results/nsys-sm86 -- <your-command>
```

The tools emit `skipped` evidence when PyTorch, a GPU, or Nsight is unavailable. Planned families remain framework fallbacks; see `results/kernel-status.json` before enabling a tactic.

## Shoutout
Thank you for netra runtime to make this project approachable https://github.com/NetraRuntime/netra-kernel/tree/main
