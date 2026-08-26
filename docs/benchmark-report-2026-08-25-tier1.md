# KudaCuda Tier-1 Decode Qualification — 2026-08-25

## Result

The truthful 128-token measurement clears the proposed Tier-1 throughput range without another CUDA kernel. On the RTX 3050, the static CUDA Graph sidecar measured a median 19.296 ms decode ITL, equivalent to **51.825 steady-state tok/s**. The paired eager control measured 66.269 ms ITL, or 15.090 steady-state tok/s.

All seven runs produced exact token-ID and decoded-text parity. The graph path executed 36 native GQA dispatches during capture and 127 graph replays per run. Peak VRAM was 2,135.6 MiB on a 4,095.5 MiB device.

| Mode | Median ITL | Median steady-state | p25/p75 gate | Peak VRAM | Parity |
|---|---:|---:|---:|---:|---|
| Eager control | 66.269 ms | 15.090 tok/s | control p75: 15.194 tok/s | 2,113.4 MiB | reference |
| Static Graph sidecar | 19.296 ms | 51.825 tok/s | candidate p25: 51.722 tok/s | 2,135.6 MiB | exact in 7/7 |

The promotion rule `candidate p25 > control p75` passes: `51.722 > 15.194`. This validates the graph sidecar against eager Transformers. It does not promote any experimental QKV, GEMV, or SwiGLU kernel; those remain excluded from the candidate.

## Hardware and runtime facts

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 4 GiB.
- Driver: 610.88; CUDA UMD: 13.3; PyTorch: 2.11.0+cu128.
- Observed state at benchmark setup: P0, 1500 MHz graphics/SM, 6001 MHz memory.
- Clock locking was not applied. The run captured read-only state and therefore should not be treated as a locked-clock publication benchmark.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` was set, but this Windows PyTorch build reported that expandable segments are unsupported. No allocator gain is credited.
- Transformers: 5.15.1; bitsandbytes: 0.50.1.

## bitsandbytes finding

The installation registers both `gemm_4bit` and `gemv_4bit`, but the live 0.50.1 `matmul_4bit` no-grad CUDA route selects `gemm_4bit_cuda_no_grad`. The operator name `gemv_4bit` therefore cannot be used as evidence that Qwen's projection path is using GEMV. The model contains 252 non-nested NF4 `Linear4bit` modules with blocksize 64.

This explains why a handwritten scalar NF4 GEMV can be mathematically correct and spill-free yet lose to the framework path: the comparison is against a different optimized bnb CUDA kernel family. A future projection optimization must first match the selected bnb `gemm_4bit` layout/tiling contract or use a tensor-core-compatible packed representation.

## Logical KV audit

The graph dispatcher receives both a fixed `kv_capacity` and a device `kv_len_device` pointer. In `kernels/common/qwen2_gqa_decode.cuh`, the graph kernel reads `*kv_len_device` when the graph template is enabled, while `kv_capacity` is passed as the cache stride. The attention loops use the logical scalar for score, softmax, and value traversal. The 512-token bucket is not the logical loop bound.

## Clock and sampling experiment

The laptop driver does not support persistence mode (`nvidia-smi -pm 1`), and clock locking (`nvidia-smi -lgc 1500,2100`) was rejected for lack of permission. A 10-second CUDA matmul warmup did raise the device from P8/210 MHz to P0/1927 MHz immediately after warmup, but the subsequent paired deferred-stop run returned to clock-contaminated behavior during its control leg: 3.911 control tok/s and 43.468 graph tok/s. Exact parity held, but this run is not a promotion result. The default seven-run P0 qualification remains the valid comparison.

The benchmark now exposes `--defer-stop-checks` to skip the per-token host `.item()` EOS check. It is disabled by default because it changes early-EOS length semantics. The experiment is retained for future serving integration; no speed claim is made from the clock-contaminated run.

## Evidence

- Aggregate: [qwen25-3b-sm86-nf4-128tok-graph-aggregate.json](../results/qwen25-3b-sm86-nf4-128tok-graph-aggregate.json)
- Repeat files: [repeat 1](../results/qwen25-3b-sm86-nf4-128tok-graph-repeat-1.json) through [repeat 7](../results/qwen25-3b-sm86-nf4-128tok-graph-repeat-7.json)
- Benchmark tooling: [benchmark_gemma.py](../tools/benchmark/benchmark_gemma.py)
- Runtime audit: [audit_runtime.py](../tools/benchmark/audit_runtime.py)
- Promotion gate: [analyze_promotion_gate.py](../tools/benchmark/analyze_promotion_gate.py)
- Deferred-stop experiment: [qwen25-3b-sm86-nf4-128tok-graph-deferred-warmup.json](../results/qwen25-3b-sm86-nf4-128tok-graph-deferred-warmup.json)

## Next measured work

1. Profile the host gap between CUDA-event decode time and wall time, with sampling `.item()` and per-token `torch.cat` explicitly measured.
2. Keep the graph GQA path as the local serving path; do not promote scalar NF4 projection experiments.
3. Evaluate prompt-lookup speculative decoding separately because it preserves greedy output parity when its verification contract is satisfied.
4. Treat lm_head NF4 as an accuracy-relaxed experiment only; Qwen's tied embeddings require a separate validation policy.
