# KudaCuda Benchmark Report — 2026-08-24

## Environment

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 4,095.5 MiB VRAM
- Driver: 610.88; CUDA UMD 13.3
- Native target: `sm_86`
- CUDA Toolkit: 13.3 / `nvcc 13.3.73`
- Platform: Windows, Python 3.13, PyTorch CUDA
- Generation: 16 new tokens, batch 1, fixed prompt
- Measurement: CUDA events for prefill/decode plus synchronized wall time

## Native smoke validation

Both published executables pass:

- `kudacuda_vector_add_smoke.exe`: raw device pointers, caller-owned non-default stream, and CUDA graph replay passed on the RTX 3050.
- `kudacuda_rmsnorm_smoke.exe`: output matched an independent FP64 reference within the declared tolerance.

## Transformers model results

The table uses the baseline and sidecar halves of the same DLL-enabled process, keeping model loading and prompt conditions paired:

| Model / precision | Mode | Peak VRAM | TTFT | ITL | Throughput | Native attention |
|---|---|---:|---:|---:|---:|---|
| Gemma 2 2B, 4-bit | Without sidecar | 2,263.8 MiB | 87.70 ms | 67.86 ms | 14.47 tok/s | Not dispatched |
| Gemma 2 2B, 4-bit | DLL-loaded sidecar | 2,263.8 MiB | 82.73 ms | 67.77 ms | 14.55 tok/s | `fallback-only` |
| Qwen2.5 3B, 4-bit | Without sidecar | 2,109.7 MiB | 252.19 ms | 223.97 ms | 4.43 tok/s | Not dispatched |
| Qwen2.5 3B, 4-bit | DLL-loaded sidecar | 2,109.7 MiB | 209.75 ms | 221.10 ms | 4.54 tok/s | `fallback-only` |

Qwen2.5-3B is larger than Gemma 2B and fits the 4 GiB card at this short benchmark context. The model snapshot was stored on `D:\AI\Models\HuggingFace` to keep the system drive free. Its exact baseline and sidecar evidence are:

- [Qwen2.5-3B paired run](../results/qwen25-3b-sm86-sidecar-final-rerun.json)
- [Qwen2.5-3B standalone baseline](../results/qwen25-3b-sm86-baseline-final.json)
- [Gemma 2B paired run](../results/gemma-sm86-sidecar-final-rerun.json)
- [Gemma 2B standalone baseline](../results/gemma-sm86-baseline-final.json)

Both sidecar runs produced exact token and text parity. They loaded the native handle successfully but reported `fallback-only`, `native_attention_dispatcher: false`, and `tactic_status: planned`. The observed timing differences are therefore framework/run variation, not KudaCuda attention speedups.

## Full verification matrix

- Python contract/tooling suite: 20/20 passed.
- Python compilation: passed for all tracked Python files.
- Native vector-add smoke: passed with a caller-owned non-default stream and CUDA Graph replay.
- Native RMSNorm smoke: passed against an independent FP64 reference.
- PyTorch RMSNorm harness: validated with maximum absolute error `5.004e-7`.
- PyTorch vector-add harness: exact output match; 0.0722 ms average GPU event time over 100 repetitions.
- Nsight Systems: validated; report generated at [nsys-sm86-vector-final.nsys-rep](../results/nsys-sm86-vector-final.nsys-rep).
- Nsight Compute: blocked by Windows GPU performance-counter permission `ERR_NVGPUCTRPERM`.
- `sm_89` compilation: blocked because `cl.exe` is not on `PATH` and the configured CMake launcher points to a missing WinGet package executable. See [sm89 build evidence](../results/sm89-build-final.json).

## Existing Ollama model

`nemotron-3.5-lightning:latest` was tested through Ollama. GPU startup failed because the runner could not allocate a 24.9 GiB CUDA-host model buffer. A CPU-only API run completed with 16 tokens at approximately 2.84 tok/s and 73.63 seconds total. See [Nemotron evidence](../results/nemotron-ollama-cpu.json).

This is not a KudaCuda result: Ollama's `llama-server` runner is separate from the Python/Transformers sidecar, and no Nemotron attention/DeltaNet dispatcher is implemented. A fresh 180-second CPU recheck timed out and was unloaded; see [recheck evidence](../results/nemotron-ollama-recheck.json).

## Interpretation and next work

The native ABI, stream forwarding, graph lifecycle, RMSNorm correctness, and GPU build are validated. The principal performance work remains model-critical:

1. Add a generic attention ABI with model-specific contracts.
2. Implement Gemma GQA4 (`Q=8`, `KV=4`, `head_dim=256`) and Qwen2 GQA (`Q=16`, `KV=2`, `head_dim=128`) separately.
3. Add RoPE, causal/sliding-window masks, and KV-cache updates.
4. Add quantized matmul and fused MLP paths before claiming end-to-end speedups.
5. Repeat the benchmark only after native dispatch, output parity, and no-fallback evidence pass.
