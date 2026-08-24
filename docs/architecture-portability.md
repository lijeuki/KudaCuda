# KudaCuda AMD-to-NVIDIA Architecture Portability

KudaCuda borrows Netra’s runtime and validation boundaries while implementing NVIDIA-native device code. Netra’s `gfx1151` and `gfx950` artifacts are AMD HIP/AMDGPU targets; they are architectural references, not binaries or source code that can be relinked for CUDA.

## Target mapping

| Netra concept | KudaCuda target | Porting rule |
|---|---|---|
| `gfx1151` RDNA wave32 | RTX 3050 `sm_86` warp32 | Lane-width patterns may map directly when they do not depend on AMD ISA behavior |
| `gfx950` CDNA4 wave64 | L40S `sm_89` warp32 | Split or retile wave-wide operations across two warps |
| AMD workgroup | CUDA thread block | Preserve block-level ownership and barriers, then retune block size |
| AMD LDS | CUDA shared memory | Preserve scratchpad intent; retune bank layout and capacity |
| AMD MFMA | NVIDIA Tensor Core MMA | Rebuild tile shape, accumulator ownership, and precision conversion |
| HIP stream | CUDA stream | Pass the caller stream unchanged |
| HIP module/function | `CUmodule`/`CUfunction` | Load and resolve once before freeze |
| HIP graph | CUDA graph | Capture, instantiate, replay, and destroy with the same lifecycle |
| HSACO | CUBIN/PTX/fatbin | Build separately for `sm_86` and `sm_89` |
| `rocprofv3` | Nsight Systems/Compute | Record tool, command, target, and measurement type in evidence |

## Hardware facts that affect tuning

- NVIDIA compute capabilities `8.6` and `8.9` use warp32. NVIDIA’s compute-capability tables report up to 100 KiB of shared memory per SM for both profiles, with 32 shared-memory banks.
- AMD documents Radeon wavefronts as wave32 and Instinct wavefronts as wave64. AMD CDNA4 LDS uses a different bank organization and serves matrix instructions through MFMA.
- NVIDIA Ampere/Ada Tensor Cores and AMD CDNA Matrix Cores support overlapping AI precisions but not identical instruction shapes, accumulator layouts, or sparsity semantics.
- NVIDIA L40S has a different memory system and workload profile from the RTX 3050. A kernel that is correct on both targets still requires target-specific launch and tiling measurements.

## Compiler targets

```text
AMD reference:   hipcc --offload-arch=gfx1151 or --offload-arch=gfx950
NVIDIA local:    nvcc -arch=sm_86
NVIDIA server:   nvcc -arch=sm_89
```

The host-side C ABI, Python contract model, manifest schema, reference harness, and evidence format are shared. Kernel sources remain under common code plus target-specific wrappers whenever `sm_86` and `sm_89` need different tuning.

## Correctness rules

The harness compares each output against an independent reference and records absolute error, relative error, RMS error, NaN/Inf counts, and deterministic repeatability. FP32 elementwise operations should be exact where the operation order is identical. Fused reductions and lower-precision paths use declared tolerances rather than an unqualified bitwise claim.

## Primary references

- [Netra Kernel targets and design](https://github.com/NetraRuntime/netra-kernel#supported-targets)
- [NVIDIA compute capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
- [NVIDIA CUDA GPU list](https://developer.nvidia.com/cuda/gpus)
- [CUDA Driver modules and launches](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/driver-api.html)
- [HIP programming model](https://rocm.docs.amd.com/projects/HIP/en/docs-6.1.0/understand/programming_model_reference.html)
- [AMD device hardware glossary](https://rocm.docs.amd.com/en/latest/reference/glossary/device-hardware.html)
- [AMD CDNA4 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)
