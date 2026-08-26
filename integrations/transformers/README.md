# Transformers Integration

The integration remains opt-in. The adapters dispatch only when all of these conditions hold:

1. The tactic is listed under `implemented_tactics` for the selected target manifest.
2. The tensor is CUDA-resident, contiguous, and dtype-compatible.
3. The batch and sequence dimensions satisfy the exact contract.
4. The bridge exports the matching native symbol and has recorded a successful native dispatch.

`patch_qwen2_attention` supports the validated Qwen2.5-3B decode profile on `sm_86` (`qwen2_gqa_decode_sm86_fp16`) and `sm_89` (`qwen2_gqa_decode_sm89_fp16`). The direct profile accepts logical KV lengths 1..2,048; the graph and experimental profiles retain their own fixed 512-token contracts. It runs after QKV projection, RoPE, and `Cache.update`, then sends the contiguous FP16 Q/K/V pointers and the caller-owned CUDA stream to the native kernel. Prefill, non-zero masks, unsupported dimensions, and Gemma/Llama attention remain framework fallback paths.

`patch_llama_attention` still leaves `gqa8_attn` planned. No placeholder output is returned for planned or unknown tactics.

Promotion of an attention tactic requires independent output comparison, deterministic repeated runs, matched framework timing, and end-to-end serving evidence. The patches are reversible through the callable they return.
