# KudaCuda Kernel Lifecycle

Every kernel family follows the same promotion gates:

1. **Idea:** define the input/output contract, target profile, expected precision, and independent reference.
2. **Correctness:** compare real outputs against the reference over representative shapes and edge cases.
3. **Determinism:** repeat identical inputs and verify the declared tolerance or exact-match rule.
4. **Graph:** capture and replay the operation on the caller stream when the integration uses CUDA graphs.
5. **Matched benchmark:** compare the same shape, dtype, and workload against the framework baseline using CUDA events and host launch timing.
6. **Serving gate:** compare end-to-end model outputs, token/logit behavior, and fallback behavior.
7. **Promotion:** mark the tactic implemented in the target manifest only after all required evidence exists.

Rejected or slower variants remain reproducible under `results/` with their command, target, and measurements. A passing isolated kernel benchmark is not sufficient evidence for serving promotion.
