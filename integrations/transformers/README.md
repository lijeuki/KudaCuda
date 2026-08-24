# Transformers Integration

The integration remains opt-in. `patch_llama_attention` only dispatches when all of these conditions hold:

1. The tactic is listed under `implemented_tactics` for the selected target manifest.
2. The tensor is CUDA-resident, contiguous, and dtype-compatible.
3. The batch and sequence dimensions satisfy the exact contract.
4. The bridge exposes a real `dispatch_attention` implementation.

`gqa8_attn` is currently listed under `planned_tactics`, so it always uses the original Hugging Face forward path. No placeholder output is returned for planned or unknown tactics.

Promotion of an attention tactic requires independent output comparison, deterministic repeated runs, CUDA graph capture/replay, matched framework timing, and end-to-end serving evidence. The patch is reversible through the callable returned by `patch_llama_attention`.
