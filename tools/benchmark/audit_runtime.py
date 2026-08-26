"""Collect runtime facts that distinguish registered and selected 4-bit paths."""

from __future__ import annotations

import importlib.metadata
import inspect
import subprocess
import time
from typing import Any


_GPU_STATE_FIELDS = (
    "name",
    "pstate",
    "graphics_clock",
    "sm_clock",
    "memory_clock",
    "temperature",
    "power",
)


def capture_gpu_state() -> dict[str, Any]:
    """Capture read-only NVIDIA state without making clock changes."""

    command = [
        "nvidia-smi",
        "--query-gpu=name,pstate,clocks.current.graphics,clocks.current.sm,"
        "clocks.current.memory,temperature.gpu,power.draw",
        "--format=csv,noheader",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "unavailable", "reason": f"{type(error).__name__}: {error}"}
    raw = completed.stdout.strip().splitlines()[0].strip() if completed.stdout.strip() else ""
    if completed.returncode != 0 or not raw:
        return {
            "status": "unavailable",
            "reason": completed.stderr.strip() or f"nvidia-smi exit code {completed.returncode}",
        }
    values = [value.strip() for value in raw.split(",")]
    return {
        "status": "completed",
        "raw": raw,
        "fields": dict(zip(_GPU_STATE_FIELDS, values)),
    }


def warm_up_cuda(torch: Any, device: Any, seconds: float) -> dict[str, Any]:
    """Keep a CUDA device busy for a requested interval before timing."""

    if getattr(device, "type", str(device)) != "cuda":
        return {"status": "skipped", "reason": "non_cuda_device"}
    seconds = float(seconds)
    if seconds <= 0.0:
        return {"status": "skipped", "reason": "disabled"}

    matrix = torch.randn((4096, 4096), dtype=torch.float16, device=device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    deadline = started + seconds
    iterations = 0
    while time.perf_counter() < deadline:
        matrix @ matrix
        torch.cuda.synchronize(device)
        iterations += 1
    elapsed = time.perf_counter() - started
    del matrix
    empty_cache = getattr(torch.cuda, "empty_cache", None)
    if callable(empty_cache):
        empty_cache()
    return {
        "status": "completed",
        "requested_seconds": seconds,
        "actual_seconds": elapsed,
        "iterations": iterations,
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _registered_bitsandbytes_ops(torch: Any) -> list[str]:
    namespace = getattr(getattr(torch, "ops", None), "bitsandbytes", None)
    if namespace is None:
        return []
    return sorted(
        name
        for name in ("gemm_4bit", "gemv_4bit")
        if getattr(namespace, name, None) is not None
    )


def _quantized_modules(model: Any) -> list[dict[str, Any]]:
    modules = getattr(model, "modules", None)
    if not callable(modules):
        return []
    result: list[dict[str, Any]] = []
    for module in modules():
        weight = getattr(module, "weight", None)
        state = getattr(weight, "quant_state", None)
        if state is None:
            continue
        result.append(
            {
                "module_type": type(module).__name__,
                "quant_type": str(getattr(state, "quant_type", "unknown")),
                "blocksize": getattr(state, "blocksize", None),
                "nested": bool(getattr(state, "nested", False)),
            }
        )
    return result


def _cuda_matmul_route() -> str:
    """Describe the installed bnb CUDA no-grad route from its live source."""

    try:
        from bitsandbytes.autograd import _functions

        source = inspect.getsource(_functions.matmul_4bit)
    except (ImportError, OSError, TypeError):
        return "unknown"
    if "torch.ops.bitsandbytes.gemm_4bit.default" in source:
        return "gemm_4bit_cuda_no_grad"
    if "torch.ops.bitsandbytes.gemv_4bit" in source:
        return "gemv_4bit"
    return "dequantize_matmul_or_unknown"


def audit_runtime(torch: Any, model: Any, args: Any) -> dict[str, Any]:
    """Return JSON-safe version, quantization, and operator-dispatch evidence."""

    quantized = _quantized_modules(model)
    return {
        "torch_version": str(getattr(torch, "__version__", "unknown")),
        "cuda_version": str(getattr(getattr(torch, "version", None), "cuda", "unknown")),
        "bitsandbytes_version": _package_version("bitsandbytes"),
        "transformers_version": _package_version("transformers"),
        "requested_compute_dtype": str(getattr(args, "dtype", "unknown")),
        "quantization_requested": bool(getattr(args, "load_in_4bit", False)),
        "quantized_module_count": len(quantized),
        "quantized_modules": quantized,
        "registered_bitsandbytes_ops": _registered_bitsandbytes_ops(torch),
        "cuda_matmul_4bit_route": _cuda_matmul_route(),
        "gpu_state": capture_gpu_state(),
    }
