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

| Model / precision | Peak VRAM | TTFT | ITL | Throughput | Native attention |
|---|---:|---:|---:|---:|---|
| Gemma 2 2B, 4-bit | 2,263.8 MiB | 113.2 ms | 88.7 ms | 11.08 tok/s | Not dispatched |
| Qwen2.5 3B, 4-bit | 2,109.7 MiB | 260.0 ms | 200.6 ms | 4.90 tok/s | Not dispatched |

Qwen2.5-3B is larger than Gemma 2B and fits the 4 GiB card at this short benchmark context. The model snapshot was stored on `D:\AI\Models\HuggingFace` to keep the system drive free. Its exact baseline and sidecar evidence are:

- [Qwen2.5-3B baseline](../results/qwen25-3b-sm86-4bit.json)
- [Qwen2.5-3B DLL-loaded sidecar](../results/qwen25-3b-sm86-sidecar.json)
- [Gemma 2B baseline](../results/gemma-sm86-4bit.json)
- [Gemma 2B final sidecar](../results/gemma-sm86-sidecar-final.json)

The Qwen sidecar run produced exact token and text parity. It used the native handle successfully but reported `fallback-only` and `native_attention_dispatcher: false`. The observed timing difference is therefore framework/run variation, not a KudaCuda attention speedup.

## Existing Ollama model

`nemotron-3.5-lightning:latest` was tested through Ollama. GPU startup failed because the runner could not allocate a 24.9 GiB CUDA-host model buffer. A CPU-only API run completed with 16 tokens at approximately 2.84 tok/s and 73.63 seconds total. See [Nemotron evidence](../results/nemotron-ollama-cpu.json).

This is not a KudaCuda result: Ollama's `llama-server` runner is separate from the Python/Transformers sidecar, and no Nemotron attention/DeltaNet dispatcher is implemented.

## Interpretation and next work

The native ABI, stream forwarding, graph lifecycle, RMSNorm correctness, and GPU build are validated. The principal performance work remains model-critical:

1. Add a generic attention ABI with model-specific contracts.
2. Implement Gemma GQA4 (`Q=8`, `KV=4`, `head_dim=256`) and Qwen2 GQA (`Q=16`, `KV=2`, `head_dim=128`) separately.
3. Add RoPE, causal/sliding-window masks, and KV-cache updates.
4. Add quantized matmul and fused MLP paths before claiming end-to-end speedups.
5. Repeat the benchmark only after native dispatch, output parity, and no-fallback evidence pass.
