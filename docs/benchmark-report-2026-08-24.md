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

All three published smoke executables pass:

- `kudacuda_vector_add_smoke.exe`: raw device pointers, caller-owned non-default stream, and CUDA graph replay passed on the RTX 3050.
- `kudacuda_rmsnorm_smoke.exe`: output matched an independent FP64 reference within the declared tolerance.
- `kudacuda_qwen2_gqa_smoke.exe`: decode GQA output matched an independent FP32 score/softmax/value reference.
- `kudacuda_rmsnorm_qkv_smoke.exe`: fused FP16 RMSNorm-QKV output matched an independent FP32 reference.
- `kudacuda_cuda_graph_smoke.exe`: two native launches captured and replayed with an independent output match.

## Transformers model results

The table uses the baseline and sidecar halves of the same DLL-enabled process, keeping model loading and prompt conditions paired:

| Model / precision | Mode | Peak VRAM | TTFT | ITL | Throughput | Native attention |
|---|---|---:|---:|---:|---:|---|
| Gemma 2 2B, 4-bit | Without sidecar | 2,263.8 MiB | 87.70 ms | 67.86 ms | 14.47 tok/s | Not dispatched |
| Gemma 2 2B, 4-bit | DLL-loaded sidecar | 2,263.8 MiB | 82.73 ms | 67.77 ms | 14.55 tok/s | `fallback-only` |
| Qwen2.5 3B, 4-bit | Without sidecar | 2,109.7 MiB | 207.28 ms | 196.52 ms | 5.071 tok/s | Not dispatched |
| Qwen2.5 3B, 4-bit | Active native sidecar | 2,109.5 MiB | 217.61 ms | 159.66 ms | 6.125 tok/s | `completed`, 576 native launches |

Qwen2.5-3B is larger than Gemma 2B and fits the 4 GiB card at this short benchmark context. The model snapshot was stored on `D:\AI\Models\HuggingFace` to keep the system drive free. Its exact baseline and sidecar evidence are:

- [Qwen2.5-3B active native 4-bit paired run](../results/qwen25-3b-sm86-native-4bit-sidecar-final.json)
- [Qwen2.5-3B native attention correctness](../results/qwen2-gqa-sm86-native-optimized-final.json)
- [Qwen2.5-3B native attention microbenchmark](../results/qwen2-gqa-sm86-microbenchmark-optimized-final.json)
- [Qwen2.5-3B standalone baseline](../results/qwen25-3b-sm86-baseline-final.json)
- [Gemma 2B paired run](../results/gemma-sm86-sidecar-final-rerun.json)
- [Gemma 2B standalone baseline](../results/gemma-sm86-baseline-final.json)

The Qwen2 active sidecar run produced exact token and text parity, reported `native_attention_dispatcher: true`, `tactic_status: implemented`, and executed 576 native attention launches. Its decode ITL improved by approximately 18.7% and throughput by approximately 20.8% in this paired run; TTFT increased slightly because the first slice intentionally covers decode attention only and leaves prefill in Transformers. Gemma remains fallback-only because its GQA4 tactic is not implemented.

### Fresh paired retry — 2026-08-25

A new Qwen2.5-3B 4-bit run used the same model snapshot, prompt, token limit, and RTX 3050. The sidecar-enabled process measured its own baseline and active runs under identical conditions:

| Mode | TTFT | ITL | Throughput | Peak VRAM | Output parity |
|---|---:|---:|---:|---:|---|
| Without KudaCuda | 74.97 ms | 70.95 ms | 14.045 tok/s | 2,109.7 MiB | — |
| Active KudaCuda sidecar | 68.01 ms | 59.34 ms | 16.699 tok/s | 2,109.5 MiB | Exact; 576 native launches |

This retry improved throughput by 18.9%, reduced decode ITL by 16.4%, and reduced total generation time by 15.9%. The sidecar reported `native_attention_dispatcher: true`, `qwen2_gqa_decode_sm86_fp16`, and `last_qwen2_dispatch_reason: native`. Evidence: [fresh paired retry](../results/qwen25-3b-sm86-sidecar-retry.json). A separate baseline-only process measured 13.363 tok/s, confirming that the within-process baseline is the appropriate paired comparison.

### Opt-in static decode CUDA Graph — 2026-08-25

The graph path uses PyTorch `torch.cuda.CUDAGraph` and its graph memory pool around a complete fixed-shape Qwen2 decode forward. KudaCuda's graph-safe GQA tactic is called inside that capture, reads logical KV length from device memory, and keeps the cache allocation at a fixed 512-token bucket. Capture ran on a dedicated non-default stream; sampling stayed outside the graph.

| Mode | TTFT | ITL | Throughput | Peak VRAM | Graph evidence | Output parity |
|---|---:|---:|---:|---:|---|---|
| Without KudaCuda | 74.27 ms | 70.26 ms | 14.033 tok/s | 2,109.7 MiB | — | — |
| Static CUDA Graph sidecar | 73.04 ms | 38.81 ms | 21.110 tok/s | 2,135.6 MiB | `torch_cuda_graph`, 78.59 ms capture, 36 native dispatches, 3 replays | Exact |

Evidence: [Qwen2.5-3B graph run](../results/qwen25-3b-sm86-graph-model-final.json). This is a single short-decode validation, not a production promotion. The graph tactic remains experimental until repeated median measurements, longer contexts, graph-pool memory accounting, and an `sm_89` runtime run are available. The benchmark now marks native/graph output mismatch as `failed-parity` and exits non-zero.

### Attention optimization experiments

The existing per-query-head FP16 kernel remains the production default. Direct CUDA-event measurements at KV length 512 were:

| Tactic | Time | Status |
|---|---:|---|
| `qwen2_gqa_decode_sm86_fp16` | 0.0657 ms | Active default |
| `qwen2_gqa_decode_sm86_fp16_group_tiled_v2` | 0.1713 ms | Correct, experimental |
| `qwen2_gqa_decode_sm86_fp16_vectorized_v2` | 0.1140 ms | Correct, experimental |

Both variants passed KV lengths 1, 7, 128, and 512 with maximum absolute error `1.22e-4`. Group tiling improves reuse in principle, but the current two-KV-group grid does not provide enough parallel work on the RTX 3050 at long context. The vectorized variant retains the original query-head grid and is closer to the baseline, but is still slower. Neither tactic is promoted. Evidence: [group-tiled correctness](../results/qwen2-gqa-sm86-group-tiled-correctness.json), [group-tiled benchmark](../results/qwen2-gqa-sm86-group-tiled-microbenchmark.json), [vectorized correctness](../results/qwen2-gqa-sm86-vectorized-correctness.json), and [vectorized benchmark](../results/qwen2-gqa-sm86-vectorized-microbenchmark.json).

The fused RMSNorm-QKV smoke validates the FP16 dense-weight foundation. It is not yet connected to Qwen2's production NF4/int4 projection weights. The graph implementation now captures and replays the fixed-shape Qwen2 decode path with exact parity; Gemma, BF16, batching, paged KV, and quantized projection fusion remain unsupported.

## Full verification matrix

- Python contract/tooling suite: 30/30 passed.
- Python compilation: passed for all tracked Python files.
- Native vector-add smoke: passed with a caller-owned non-default stream and CUDA Graph replay.
- Native RMSNorm smoke: passed against an independent FP64 reference.
- Native Qwen2 GQA smoke: passed against an independent FP32 reference.
- Native fused RMSNorm-QKV smoke: passed against an independent FP32 reference.
- Native CUDA Graph multi-launch smoke: passed against an independent reference.
- Qwen2 static decode CUDA Graph: captured/replayed with exact token/text parity; 36 native attention dispatches were recorded during capture.
- Experimental Qwen2 GQA variants: correctness passed; group-tiled and vectorized tactics were not promoted after direct-event comparison.
- Qwen2 native correctness harness: passed for KV lengths 1, 7, 128, and 512; maximum absolute error `1.22e-4`.
- Qwen2 native microbenchmark: 6.25–10.08× GPU-event speedup versus PyTorch eager attention across the tested KV lengths.
- Nsight Systems: the `qwen2_gqa_decode_kernel` appears in the GPU kernel summary; see [native Qwen2 trace](../results/nsys-qwen2-sm86-native.nsys-rep).
- PyTorch RMSNorm harness: validated with maximum absolute error `5.004e-7`.
- PyTorch vector-add harness: exact output match; 0.0722 ms average GPU event time over 100 repetitions.
- Nsight Systems: validated; report generated at [nsys-sm86-vector-final.nsys-rep](../results/nsys-sm86-vector-final.nsys-rep).
- Nsight Compute: blocked by Windows GPU performance-counter permission `ERR_NVGPUCTRPERM`.
- `sm_89` compilation: completed with the installed MSVC Build Tools, CUDA 13.3, and Visual Studio CMake/Ninja; runtime validation remains pending on an L40S. See [sm89 build evidence](../results/sm89-build-final.json).

## Existing Ollama model

`nemotron-3.5-lightning:latest` was tested through Ollama. GPU startup failed because the runner could not allocate a 24.9 GiB CUDA-host model buffer. A CPU-only API run completed with 16 tokens at approximately 2.84 tok/s and 73.63 seconds total. See [Nemotron evidence](../results/nemotron-ollama-cpu.json).

This is not a KudaCuda result: Ollama's `llama-server` runner is separate from the Python/Transformers sidecar, and no Nemotron attention/DeltaNet dispatcher is implemented. A fresh 180-second CPU recheck timed out and was unloaded; see [recheck evidence](../results/nemotron-ollama-recheck.json).

## Interpretation and next work

The native ABI, stream forwarding, graph lifecycle, RMSNorm correctness, and GPU build are validated. The principal performance work remains model-critical:

1. Add a parallel tiled-workspace reduction variant so KV reuse does not collapse the grid to two blocks on small GPUs.
2. Implement Gemma GQA4 (`Q=8`, `KV=4`, `head_dim=256`) with its sliding-window contract.
3. Add prefill, BF16, paged KV cache, and batching profiles.
4. Connect the fused RMSNorm-QKV foundation to a Qwen2 NF4/int4 projection contract.
5. Add quantized matmul and fused MLP paths, which dominate total model time beyond attention.
6. Build and validate the same Qwen2 tactics on `sm_89`/L40S before production promotion.

## Phase 2/3 execution result — 2026-08-25

The approved NF4 fused RMSNorm-QKV and fixed-workspace GQA plan was implemented and tested against the real Qwen2.5-3B NF4 checkpoint stored on the D drive.

### Correctness and ABI evidence

- Python contract suite: `41/41` tests passed after updating the graph test double for the optional workspace pointer.
- Python compilation, manifest JSON parsing, and `git diff --check`: passed.
- RTX 3050 native smoke matrix: vector-add, FP32 RMSNorm, Qwen2 GQA, fused FP16 RMSNorm-QKV, NF4 fused RMSNorm-QKV, workspace GQA, and CUDA Graph replay all passed.
- Real projection parity against bitsandbytes dequantization plus FP32 matmul: Q max absolute error `0.0078125`, K `0.001953125`, V `0.00048828125`; the native ABI also includes the model's nonzero Q/K/V FP16 biases.
- Model graph runs preserved exact token IDs and text in all tested configurations.

### Model-level measurements

| Configuration | TTFT | ITL | Throughput | Peak VRAM | Native evidence |
|---|---:|---:|---:|---:|---|
| Latest paired baseline | 71.69 ms | 62.37 ms | 15.456 tok/s | 2,109.7 MiB | Framework only |
| Existing static graph | 67.23 ms | 35.21 ms | 23.139 tok/s | 2,135.6 MiB | 36 GQA dispatches, 3 replays |
| NF4 fused QKV graph | 67.42 ms | 236.30 ms | 5.153 tok/s | 2,135.8 MiB | 36 fused QKV + 36 GQA |
| Workspace GQA graph | 71.27 ms | 48.94 ms | 18.340 tok/s | 2,135.7 MiB | 36 workspace GQA dispatches |

Five paired NF4 graph runs confirm the regression: median throughput was `5.126 tok/s` with the fused path versus `15.358 tok/s` for its paired baseline, a `66.6%` decrease. All five runs had exact token/text parity and 36 native fused-QKV dispatches. The NF4 fused path is therefore functionally active but fails the no-regression promotion gate; its median is also `77.8%` below the latest static graph result. The workspace GQA path is correct and improves its paired eager baseline by about 18.0%, but is about 20.7% below the existing static graph. Both remain opt-in experimental; the existing static graph remains the local default.

The `sm_89` target compiled successfully with CUDA 13.3, MSVC, and Ninja. Runtime validation is explicitly pending on an L40S; the RTX 3050 cannot execute `sm_89` device code. The next performance step is a tiled NF4 GEMV with bias epilogue and a parallel workspace reduction, followed by repeated median runs before any promotion.
