# Qwen2.5-3B Decode GQA Attention — KudaCuda Design

Status: design approved for review

## Goal

Promote one real native attention tactic from `planned` to `active` for the
verified Qwen2.5-3B 4-bit serving path on the RTX 3050 `sm_86` target. The
tactic must execute attention score, softmax, and value aggregation on CUDA,
produce exact token/output parity with the Transformers eager reference, and
remain a safe framework fallback for every contract outside its profile.

This is the first vertical slice toward the larger Netra-style compiler and
engine workflow. It is deliberately narrow enough to validate the native
math and integration seam before adding QKV projection, quantized GEMM,
prefill, or multi-batch scheduling.

## Fixed serving contract

Model contract: `Qwen2ForCausalLM` from `Qwen/Qwen2.5-3B-Instruct`.

| Field | Contract |
|---|---|
| Target | NVIDIA RTX 3050, `sm_86` |
| Attention type | Qwen2 grouped-query attention |
| Query heads | 16 |
| KV heads | 2 |
| Head dimension | 128 |
| KV replication | 8 query heads per KV head |
| Batch | Exactly 1 |
| Query length | Exactly 1 token |
| KV length | 1 through 512 tokens in the first profile |
| Q/K/V dtype | `float16` |
| Accumulation | `float32` |
| Layout | Contiguous `[batch, heads, sequence, head_dim]` |
| Mask | Decode position is the final cached position; no arbitrary mask in this slice |
| Projection/RoPE/cache | Performed by the existing Transformers attention module |
| Output | Contiguous `[1, 16, 1, 128]` attention context, then existing `o_proj` |

The first profile is decode-only. It does not claim support for prefill,
batch sizes above one, BF16 tensors, arbitrary attention masks, sliding-window
profiles, paged KV caches, or quantized projection weights.

## Architecture and data flow

```text
Qwen2Attention.forward
  ├─ PyTorch q_proj / k_proj / v_proj
  ├─ Transformers rotary embedding
  ├─ Transformers Cache.update()
  ├─ KudaCuda guard: exact Qwen2 decode contract
  │    ├─ match: native GQA decode kernel through C ABI
  │    │          Q, cached K, cached V → context
  │    │          caller-owned CUDA stream
  │    └─ miss/error: original Transformers eager attention
  └─ existing o_proj
```

The Python adapter will patch the Qwen2 attention class at a model-specific
seam. It will not claim that a generic `LlamaAttention` patch covers Qwen2.
The native ABI will receive device pointers and explicit dimensions/strides;
it will never dereference host tensor data or allocate a hidden copy.

## Native kernel design

The first kernel uses one CUDA block per `(batch, query_head)` for the single
query token. Each block maps its query head to one of the two KV heads, computes
scaled dot products over the active KV length with FP32 accumulation, performs
a numerically stable max/sum-exp reduction, and accumulates the weighted V
rows into FP32 before converting the context to FP16.

The first implementation favors an independently auditable reference path:

- no fused QKV projection;
- no dynamic allocation inside the launch;
- no runtime graph capture requirement;
- no arbitrary mask branch;
- explicit error status for unsupported dimensions, dtype, layout, or stream;
- launch on the stream supplied by PyTorch.

The kernel may be slower than PyTorch SDPA initially. Promotion requires both
correctness and a measured improvement at the declared contract; correctness
alone can produce `verified` evidence but cannot produce `accepted` status.

## C ABI and tactic state

The runtime will add a dedicated Qwen2 decode entry point with:

- opaque KudaCuda handle;
- Q, K-cache, V-cache, and output device pointers;
- batch, query length, KV length, head counts, head dimension;
- explicit contiguous strides for the four tensors;
- scale and caller CUDA stream;
- integer status plus `kudacuda_last_error` diagnostics.

The manifest will describe the tactic as `qwen2_gqa_decode_sm86_fp16` with its
exact profile and evidence references. `native_attention_dispatcher` becomes
true only when all of these are true:

1. the DLL resolves the new symbol;
2. the manifest marks the tactic implemented;
3. the tensor metadata and dimensions match the profile;
4. the native launch returns success;
5. parity evidence for the same contract is present.

Any failure returns the original Transformers result and records a fallback
reason. A metadata-only switch is explicitly forbidden.

## Correctness and promotion gates

### RED tests first

Before production CUDA code, add tests that fail because the native tactic is
absent:

1. manifest recognizes the exact Qwen2 profile and rejects BF16, batch 2,
   query length 2, wrong head counts, and KV length above 512;
2. the adapter patches `Qwen2Attention`, not unrelated attention classes;
3. the adapter passes Q/K/V/output pointers, dimensions, and the active
   PyTorch stream to the native seam;
4. a fake native failure returns the original Transformers path;
5. an independent CPU reference validates grouped-head mapping and causal
   decode semantics.

### GPU correctness

For deterministic seeds and KV lengths `{1, 7, 32, 127, 512}`:

- compare native context against an independent FP64/FP32 CPU reference;
- compare native context against the unpatched Transformers eager path;
- require no NaN/Inf mismatches and declared absolute/relative tolerances;
- verify output-token and full-text parity for the Qwen2.5-3B harness;
- run under the caller-owned non-default stream;
- run repeated launches after warm-up and after handle freeze.

### Performance promotion

Measure paired baseline and native runs with identical model, prompt,
generation length, cache state, and warm-up. Record TTFT, decode ITL,
throughput, peak VRAM, launch count, and fallback count. The tactic remains
`verified` if it is correct but slower; it becomes `accepted` only if it
beats the matched eager baseline without changing outputs or increasing peak
VRAM beyond the declared budget.

## Error and fallback behavior

- Contract miss: no native call; return the original attention implementation.
- Missing symbol or unsupported DLL: mark native attention unavailable; do not
  crash model loading.
- Native nonzero status: capture the error text, discard the attempted native
  result, and execute the original attention path.
- Shape/layout/dtype mismatch: return fallback with a structured reason.
- CUDA launch failure: surface evidence and fallback for the current request;
  never silently report an accelerated result.

## Out of scope for this slice

The following remain separate delivery slices: QKV and output quantized GEMM,
prefill attention, arbitrary masks, BF16, paged KV cache, batch scheduling,
`sm_89` L40S compilation, deterministic engine packaging, static memory/layout
planning, and the general compiler CLI. The tactic metadata and evidence shape
will be designed so those capabilities can be added without changing the
Qwen2 runtime seam.

## Acceptance definition

This slice is complete only when the repository contains:

- a real `sm_86` Qwen2 GQA decode kernel;
- a production Qwen2 model adapter and fallback path;
- a manifest profile whose state is earned by symbol resolution and evidence;
- native-vs-reference and native-vs-Transformers correctness artifacts;
- paired baseline/native benchmark results;
- tests covering contract guards, pointer/stream forwarding, error fallback,
  and output parity;
- documentation that reports the result without claiming acceleration when
  the tactic is not accepted.
