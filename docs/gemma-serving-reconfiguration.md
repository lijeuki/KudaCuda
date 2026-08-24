# Gemma Serving Reconfiguration

## Current test result

The corrected benchmark is [`tools/benchmark/benchmark_gemma.py`](../tools/benchmark/benchmark_gemma.py). It records CUDA-event prefill/decode timing, synchronized wall time, peak allocation, generated tokens, and exact output comparison without synthetic fallback numbers.

The authenticated `google/gemma-2-2b-it` 4-bit baseline completed on the RTX 3050. The 16-token run measured approximately 113 ms TTFT, 89 ms ITL, 11 tokens/s, and 2,264 MiB peak allocated memory. The native `sm_86` DLL is available at [`artifacts/sm86/kudacuda.dll`](../artifacts/sm86/kudacuda.dll), and both native smoke tests now pass after the driver update. The final sidecar run is retained at [`results/gemma-sm86-sidecar-final.json`](../results/gemma-sm86-sidecar-final.json): it completed with exact token/text parity, approximately 74.6 ms TTFT, 66.3 ms ITL, and 15.0 tokens/s, but remains `fallback-only` because the Gemma attention tactic is still planned rather than natively dispatched. A public-model GPU harness run also completed, proving the measurement loop works independently of Gemma.

## Why the existing playbook overstates KudaCuda

The original playbook:

- calls `init_sidecar_backend` and `apply_monkey_patch`, which are not KudaCuda APIs;
- patches `LlamaAttention`, while Gemma 2 uses its Gemma attention module;
- does not pass a tokenizer streamer, so its TTFT callback never runs;
- times asynchronous CUDA work with `time.time()` without a reliable synchronization boundary;
- fabricates `1.82x` speed and `12%` memory savings when the sidecar fails;
- loads FP16 weights onto a 4 GiB card without a quantization or offload plan.

The corrected harness treats missing native dispatch as `skipped` or `fallback-only`. No speedup is valid until a native tactic actually executes and output comparison passes.

## Gemma 2 serving shape

Gemma 2 2B uses hidden size `2304`, intermediate size `9216`, 26 layers, 8 attention heads, 4 KV heads, head dimension `256`, and a sliding-window limit of `4096`. These values must be model-contract fields, not inferred from a generic Llama contract. See the [official Gemma 2 configuration](https://huggingface.co/docs/transformers/model_doc/gemma2).

Approximate FP16 KV-cache cost for a full 26-layer cache is:

```text
layers × kv_heads × head_dim × 2(K/V) × bytes_per_element × tokens
26 × 4 × 256 × 2 × 2 × tokens = 106,496 bytes/token
```

That is approximately 416 MiB at 4,096 tokens for batch 1, before allocator and metadata overhead. Weight-only INT4 is therefore required for a comfortable 4 GiB laptop configuration; longer context also needs a paged or quantized KV cache. Hugging Face documents dynamic, static, offloaded, and quantized cache trade-offs in its [KV-cache guide](https://huggingface.co/docs/transformers/kv_cache).

## Required KudaCuda reconfiguration

| Priority | Current state | Required change for larger models |
|---|---|---|
| P0 | Only vector-add and standalone FP32 RMSNorm are implemented | Add Gemma fused residual+RMSNorm for FP16/BF16 with Gemma epsilon `1e-6` |
| P0 | `gqa8_attn` is planned and has no dispatcher | Implement Gemma GQA4 prefill/decode attention, RoPE, sliding-window masking, and KV-cache update |
| P0 | No quantized matmul path | Add weight-only INT4/INT8 Tensor-Core kernels for RTX 3050; add FP8 E4M3 block-scaled kernels for L40S |
| P0 | No model-specific manifest | Add Gemma profiles containing dimensions, layer types, dtype, quantization, cache limits, and eligible shape contracts |
| P1 | No persistent workspace allocator | Pre-allocate activation, reduction, temporary, and KV-cache arenas before freeze; expose capacity and OOM diagnostics |
| P1 | One generic graph handle | Cache graphs by `(device, dtype, batch, prompt_length, decode_length, tactic set)` and invalidate only on shape/config changes |
| P1 | Registry exists but is not connected to dispatch | Bind descriptors to loaded `CUfunction` handles during initialization; prohibit module lookup and filesystem access after freeze |
| P1 | No serving scheduler | Add paged KV blocks, continuous batching, admission control, and a fallback budget so large requests do not fragment the laptop |
| P2 | No fused MLP | Fuse gate/up projection, gated SiLU, down projection, and quantize/dequantize boundaries; this is a larger win than RMSNorm alone |
| P2 | No observability of hot-path use | Record per-tactic dispatch count, fallback count/reason, shape, bytes moved, and kernel duration in evidence JSON |

## Target-specific policy

### RTX 3050 `sm_86`, 4 GiB

- INT4 weights with FP16 compute.
- Batch 1 first; cap context at 2K–4K until paged KV is validated.
- FP16/BF16 fused norms and attention; INT8 fallback when accuracy requires it.
- Keep the framework fallback for unsupported shapes instead of allocating a second large workspace.

### L40S `sm_89`, 48 GiB

- FP8 E4M3 or INT8/INT4 weight paths, with FP16/BF16 accumulation where required.
- Separate prefill and decode kernels; benchmark batch sizes 1, 2, 4, 8, 16 and long-context profiles.
- Use larger static graph pools, paged KV cache, and continuous batching.
- Add multi-request and concurrency evidence; single-request speed is not sufficient for server promotion.

## Promotion gates

An implementation can be marked `implemented` only when all of these are recorded for the exact model profile:

1. Logit comparison against framework output within declared dtype tolerance.
2. Greedy token and stop-token equality over fixed prompts and seeds.
3. Repeatability across at least 100 decode steps.
4. TTFT, median/p90 ITL, throughput, and peak VRAM from synchronized measurements.
5. Zero unexpected fallback on the advertised shape profile.
6. Graph replay and allocator stability under the advertised batch/context range.

The immediate bottleneck is not the Python wrapper. It is the absence of the model-critical attention, KV-cache, quantized matmul, and fused MLP tactics that dominate Gemma execution.
