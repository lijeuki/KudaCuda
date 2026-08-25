# KudaCuda on Linux and Windows

This guide builds the native sidecar for an NVIDIA GPU and runs its correctness smoke tests. It supports the local RTX 3050 profile (`sm_86`) and the production L40S profile (`sm_89`).

## Requirements

- NVIDIA GPU and a display/compute driver compatible with the installed CUDA Toolkit;
- CUDA Toolkit with `nvcc`;
- CMake 3.24 or newer;
- Ninja or a supported native build generator;
- Python 3.10+ for the contract and benchmark tools;
- PyTorch with CUDA, Transformers, Accelerate, and bitsandbytes for model tests.

The native smoke tests require a visible GPU. The Python contract tests do not.

## Linux

From the KudaCuda repository root:

```bash
sudo apt-get install build-essential cmake ninja-build
export CUDACXX=/usr/local/cuda/bin/nvcc

# RTX 3050 / Ampere
cmake -S . -B build-sm86 -G Ninja \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_CUDA_COMPILER="$CUDACXX"
cmake --build build-sm86

./build-sm86/kudacuda_vector_add_smoke
./build-sm86/kudacuda_rmsnorm_smoke
```

For an L40S build, replace `86` with `89` and run on the L40S host. Linux links the CUDA Driver API through the driver-provided `libcuda.so`.

## Windows PowerShell

Open a Visual Studio Developer Command Prompt, or load the developer environment into the shell, then configure Ninja with the CUDA compiler and MSVC host compiler:

```powershell
$env:CUDACXX = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/nvcc.exe'

# If cl.exe is not already on PATH, start a developer shell and run the
# remaining commands in that new window:
cmd.exe /k call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64

cmake -S . -B build-sm86 -G Ninja `
  -DCMAKE_CUDA_ARCHITECTURES=86 `
  -DCMAKE_CUDA_COMPILER="$env:CUDACXX" `
  -DCMAKE_CUDA_HOST_COMPILER=cl.exe `
  -DCMAKE_CXX_COMPILER=cl.exe
cmake --build build-sm86

./build-sm86/kudacuda_vector_add_smoke.exe
./build-sm86/kudacuda_rmsnorm_smoke.exe
./build-sm86/kudacuda_qwen2_gqa_smoke.exe
```

For a normal CUDA 13.x Windows installation, the NVIDIA display driver provides `nvcuda.dll`. If the Toolkit does not ship `nvcuda.lib`, CMake generates the small import library from `runtime/nvcuda.def`; it does not copy or redistribute the driver DLL.

## Python validation

On Linux/macOS:

```bash
export PYTHONPATH="..${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest discover -s tests -p 'test_*.py' -v
python -m py_compile python/kudacuda_sidecar.py \
  tools/benchmark/benchmark_gemma.py \
  harness/reference.py harness/compare.py
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH = (Resolve-Path ..).Path
python -m unittest discover -s tests -p 'test_*.py' -v
python -m py_compile python/kudacuda_sidecar.py harness/reference.py harness/compare.py
```

Run a 4-bit Transformers benchmark with the native library:

```bash
python tools/benchmark/benchmark_gemma.py \
  --model google/gemma-2-2b-it \
  --load-in-4bit \
  --max-new-tokens 16 \
  --library artifacts/sm86/kudacuda.dll \
  --output results/gemma-sm86-sidecar-final.json
```

The script is model-generic despite its historical filename. Qwen2.5-3B uses the active `qwen2_gqa_decode_sm86_fp16` tactic when the exact decode contract matches; Gemma and unsupported profiles remain `fallback-only`. A DLL load without native launch counts is never a speedup claim.

For an opt-in fixed-shape Qwen2 decode graph, use a bucket large enough for the prompt plus generated tokens:

```powershell
python tools/benchmark/benchmark_gemma.py `
  --model D:\AI\Models\HuggingFace\hub\models--Qwen--Qwen2.5-3B-Instruct\snapshots\aa8e72537993ba99e69dfaafa59ed015b17504d1 `
  --manifest manifests\rtx3050-sm86.json `
  --library artifacts\sm86\kudacuda.dll `
  --load-in-4bit --device cuda --dtype float16 `
  --max-new-tokens 16 --graph-decode --graph-kv-bucket 512 `
  --output results\qwen25-3b-sm86-graph-model.json
```

Graph mode is experimental and requires exact token/text parity before it is considered an acceleration result. It uses PyTorch's graph allocator for the model capture; KudaCuda's native graph ABI remains available for CUDA-only callers and smoke tests.

The direct Qwen2 validation and microbenchmark are:

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

## Model sizing on a 4 GiB RTX 3050

Use weight quantization and short contexts first. Keep roughly 0.8–1.2 GiB available for activations, CUDA runtime state, and KV cache. The verified Qwen2.5-3B 4-bit run used about 2.11 GiB peak VRAM, while the 27B Nemotron GGUF could not allocate its 24.9 GiB host buffer on this laptop.

## L40S deployment

Build with `CMAKE_CUDA_ARCHITECTURES=89`, use the L40S manifest, and repeat correctness and performance evidence at the target batch/context sizes. The RTX 3050 artifact is not a substitute for an `sm_89` build. Planned L40S work includes FP8 E4M3, larger workspaces, paged KV cache, and continuous batching.
