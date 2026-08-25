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

The fused RMSNorm-QKV smoke validates the FP16 dense-weight foundation. It is not yet connected to Qwen2's production NF4/int4 projection weights. The graph implementation now captures and replays the fixed-shape Qwen2 decode path with exact parity; Gemma, BF16 native attention, batching, paged KV, and quantized projection fusion remain unsupported.

### Tiled NF4 GEMV v2 — 2026-08-25

The next NF4 experiment targets the real Qwen2.5-3B 4-bit decode path on the RTX 3050. It adds a versioned native ABI, a stable FP32 RMSNorm workspace, a 64-thread one-output-row GEMV, 128-bit `uint4` weight loads, a constant-memory NF4 codebook, FP32 reduction, blocksize-64 scales, and a fused bias epilogue. The sm86 `ptxas` audit reports 56 registers and zero spill stores/loads for the GEMV kernel, 16 bytes of shared memory, and 25 registers/zero spills for RMSNorm. The same source compiles for sm89; no L40S runtime was available.

Five paired four-token graph runs used the same D-drive checkpoint, prompt, 512-token graph bucket, and `--load-in-4bit` configuration:

| Mode | Median TTFT | Median ITL | Median throughput | Median peak VRAM | Dispatch/parity |
|---|---:|---:|---:|---:|---|
| Existing graph sidecar | 74.31 ms | 38.88 ms | 21.209 tok/s | 2,135.6 MiB | Exact |
| Tiled NF4 GEMV v2 | 70.85 ms | 43.83 ms | 19.382 tok/s | 2,136.1 MiB | 36 v2 dispatches; exact |

The v2 result is 8.61% below the existing graph sidecar, so it fails the 5% promotion gate and remains opt-in. It is substantially faster than the earlier v1 fused-QKV graph path (~5.1 tok/s), but the one-output-row NF4 GEMV still does not beat the framework's optimized quantized projection path. Evidence: [aggregate result](../results/qwen25-3b-sm86-nf4-tiled-gemv-final.json), [baseline repeat 1](../results/qwen25-3b-sm86-nf4-baseline-repeat-1.json), and [tiled repeat 1](../results/qwen25-3b-sm86-nf4-tiled-gemv-repeat-1.json).

### Multi-row activation-tiled NF4 GEMV v3 — 2026-08-25

The v3 follow-up maps two output rows to each 64-thread block and stages the normalized 2,048-element activation vector once in shared memory. It retains v2's constant-memory NF4 codebook, `uint4` packed-weight loads, blocksize-64 scales, FP32 accumulation, bias epilogue, stable workspace, and versioned fallback contract. The sm86 audit reports 92 registers, 8,224 bytes of shared memory, and zero spill stores/loads; sm89 compilation also completed, but no L40S runtime was available.

Five paired four-token graph runs used the same D-drive Qwen2.5-3B NF4 checkpoint and fixed 512-token graph bucket:

| Mode | Median throughput | Median peak VRAM | Dispatch/parity | Status |
|---|---:|---:|---|---|
| Existing graph sidecar | 21.209 tok/s | 2,135.6 MiB | Exact | Default |
| Tiled NF4 GEMV v2 | 19.382 tok/s | 2,136.1 MiB | 36 v2 dispatches; exact | Experimental |
| Multi-row NF4 GEMV v3 | 19.503 tok/s | 2,136.1 MiB | 36 v3 dispatches; exact | Experimental; not promoted |

V3 is 8.04% below the existing graph median, so it fails the `>=5%` promotion gate even though native correctness and zero-spill checks pass. Its paired vanilla comparison is not the promotion baseline because the graph tactic already removes the dominant host-launch overhead. A subsequent five-run graph-control batch was clock-state contaminated (median 8.577 tok/s and P8/210 MHz observed between processes); those files are retained as diagnostics and excluded from the aggregate. Evidence: [v3 aggregate](../results/qwen25-3b-sm86-nf4-gemv-v3-final.json), [v3 repeat 1](../results/qwen25-3b-sm86-nf4-gemv-v3-repeat-1.json), and [native smoke](../artifacts/sm86/kudacuda_nf4_tiled_gemv_smoke.exe).

The 300 tok/s goal is therefore still a stretch target, not an achieved result. The next measured phase must rank all decoder projections, MLP GEMVs, RMSNorm/residual boundaries, graph overhead, and clock/allocator variance before choosing the next kernel family. QKV-only optimization cannot provide the required whole-model multiplier.

### Fused NF4 SwiGLU experiment — 2026-08-25

The deeper operation profile ranked the MLP path as the next native-family candidate: post-attention RMSNorm, gate, up, and SiLU accounted for about 25.94 ms of the inclusive eager profile. The implementation therefore added an opt-in layer-boundary patch for Qwen2.5-3B that fuses post-attention RMSNorm, non-nested NF4 gate/up projections, SiLU, and the gate/up product, then leaves the down projection in the framework. The patch wraps the decoder layer rather than `mlp.forward`, because Hugging Face applies post-attention RMSNorm before entering the MLP; an MLP-only patch would normalize twice.

The independent sm86 reference smoke passed. Both `sm_86` and `sm_89` builds completed. The sm86 ptxas audit reported 56 registers, 8,264 bytes of shared memory, a 32-byte stack frame, and zero spill stores/loads. A four-output-row activation-tiled revision reused one normalized activation tile across rows, but it did not overcome the handwritten scalar NF4 projection cost.

Five four-token graph controls and five fused-SwiGLU graph runs on the RTX 3050 produced:

| Mode | Median throughput | Median peak VRAM | Parity | Promotion |
|---|---:|---:|---|---|
| Vanilla eager without sidecar | 15.091 tok/s | 2,109.7 MiB | — | Reference |
| Existing graph without fused SwiGLU | 22.333 tok/s | 2,135.6 MiB | Exact | Current default |
| Fused NF4 SwiGLU graph | 13.335 tok/s | 2,136.4 MiB | Exact; 36 native dispatches | Not promoted |

The fused path was 40.29% slower than its fresh graph control and 37.13% below the established 21.209 tok/s graph baseline. The kernel is functionally active and compiler-clean, but bitsandbytes' optimized quantized projection path is substantially faster than this scalar NF4 dot-product implementation. The tactic remains explicit opt-in; the static graph remains the main local path. Evidence: [SwiGLU aggregate](../results/qwen25-3b-sm86-nf4-swiglu-final.json), [v2 run](../results/qwen25-3b-sm86-nf4-swiglu-v2.json), [graph control](../results/qwen25-3b-sm86-nf4-swiglu-control-v2.json), and [native smoke](../artifacts/sm86/kudacuda_nf4_swiglu_smoke.exe).

### BF16/NF4 SwiGLU contract — 2026-08-25

The D-drive Qwen2.5-3B checkpoint was explicitly loaded with BF16 compute to freeze the real contract. The input and post-attention RMSNorm weight are contiguous CUDA BF16 `(1, 2048)` and `(2048,)`. Gate/up weights are non-nested NF4 with packed storage `(11272192, 1)`, logical shape `(11008, 2048)`, blocksize 64, float32 absmax, and no bias. This is a separate `qwen2_swiglu_sm86_bf16_nf4_v1` tactic; the existing FP16 tactic was not changed.

The BF16 bridge, C ABI, sm86/sm89 kernels, manifest entries, decoder-layer hook, and static workspace were implemented. The native smoke matched an independent FP32 reference. Both targets compiled; sm86 `ptxas` reported 48 registers, 8,264 bytes of shared memory, a 32-byte stack frame, and zero spill stores/loads. The graph runner now updates floating attention-mask future slots to dtype-appropriate negative infinity as logical KV length advances. This is required because BF16 attention currently uses framework/SDPA while the FP16 profile uses native GQA dispatch.

Five four-token runs on the RTX 3050 passed exact token/text parity and executed 36 BF16 SwiGLU dispatches per run:

| Mode | Median TTFT | Median ITL | Median throughput | Median peak VRAM | Dispatch/parity | Status |
|---|---:|---:|---:|---:|---|---|
| Static graph control without fused SwiGLU | 76.816 ms | 51.650 ms | 17.119 tok/s | 2,139.3 MiB | 0 BF16 dispatches | Fallback-only |
| Static graph + BF16 fused NF4 SwiGLU | 70.063 ms | 84.897 ms | 12.305 tok/s | 2,140.1 MiB | 36 BF16 dispatches; exact in 5/5 | Partial-native |

The BF16 tactic is therefore not promoted: throughput regressed 28.12% and decode ITL increased 64.37% versus the paired graph control, with a 0.756 MiB peak-VRAM increase. The result validates dtype alignment, pointer ABI, graph safety, numerical parity, and compiler cleanliness, but not performance. The scalar NF4 projection must be replaced by a tensor-core-compatible or framework-compatible packed path before this family can become a serving default. Evidence: [BF16 aggregate](../results/qwen25-3b-sm86-nf4-swiglu-bf16-final.json), [active repeat 1](../results/qwen25-3b-sm86-nf4-swiglu-bf16-active-1.json), [control repeat 1](../results/qwen25-3b-sm86-nf4-bf16-control-1.json), and [BF16 smoke](../artifacts/sm86/kudacuda_nf4_swiglu_bf16_smoke.exe).

## Full verification matrix

- Python contract/tooling suite: 49/49 passed with `pytest`.
- Python compilation: passed for all tracked Python files.
- Native vector-add smoke: passed with a caller-owned non-default stream and CUDA Graph replay.
- Native RMSNorm smoke: passed against an independent FP64 reference.
- Native Qwen2 GQA smoke: passed against an independent FP32 reference.
- Native fused RMSNorm-QKV smoke: passed against an independent FP32 reference.
- Native tiled NF4 GEMV v2/v3 smoke: passed against an independent FP32 reference; sm86 and sm89 targets compile.
- Native fused NF4 SwiGLU smoke: passed against an independent FP32 reference; sm86 and sm89 targets compile; five-run FP16 model parity exact.
- Native BF16 fused NF4 SwiGLU smoke: passed against an independent FP32 reference; sm86 and sm89 targets compile; five-run model parity exact; 48 registers and zero spill stores/loads on sm86.
- Native CUDA Graph multi-launch smoke: passed against an independent reference.
- Qwen2 static decode CUDA Graph: captured/replayed with exact token/text parity; FP16 native attention and BF16 framework-attention graph-mask replay are both covered.
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
3. Add BF16 native attention, prefill, paged KV cache, and batching profiles.
4. Connect the fused RMSNorm-QKV foundation to a Qwen2 NF4/int4 projection contract.
5. Add quantized matmul and fused MLP paths, which dominate total model time beyond attention.
6. Build and validate the same Qwen2 tactics on `sm_89`/L40S before production promotion.

## Phase 2/3 execution result — 2026-08-25

The approved NF4 fused RMSNorm-QKV, fixed-workspace GQA, and fused NF4 SwiGLU plans were implemented and tested against the real Qwen2.5-3B NF4 checkpoint stored on the D drive. The SwiGLU tactic is retained as experimental after the repeat benchmark failed the no-regression gate.

### Correctness and ABI evidence

- Python contract suite: `49/49` tests passed after adding the BF16 contract, graph-mask replay guard, and model test double.
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
