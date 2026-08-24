# KudaCuda Results

Each result file is JSON and records the target architecture, device, input contract, reference implementation, measurement method, error metrics, and pass/fail status.

Required correctness fields:

- `target_arch`
- `shape`
- `dtype`
- `reference`
- `metrics.passed`
- `metrics.max_abs_error`
- `metrics.max_rel_error`
- `metrics.rms_error`
- `metrics.nan_actual`
- `metrics.inf_mismatch`

Performance evidence must identify whether a number came from CUDA events, host launch timing, Nsight Systems, or Nsight Compute. Profiler counter output is not a substitute for wall-time evidence.
