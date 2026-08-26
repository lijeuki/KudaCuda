"""Benchmark the native Qwen2 GQA decode kernel against PyTorch eager attention."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from KudaCuda.harness.compare import compare_values, write_json_evidence
from KudaCuda.python.kudacuda_sidecar import KudaCudaBridge


def _event_time(torch: Any, callback: Callable[[], Any], repetitions: int) -> tuple[float, float]:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    host_durations: list[float] = []
    start.record()
    for _ in range(repetitions):
        host_start = time.perf_counter()
        callback()
        host_durations.append((time.perf_counter() - host_start) * 1000.0)
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / repetitions, statistics.median(host_durations)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument(
        "--kv-lens",
        type=int,
        nargs="+",
        default=[1, 7, 128, 512, 585, 1024, 2048],
    )
    parser.add_argument(
        "--tactic",
        default=None,
        help="Exact Qwen tactic name; defaults to the architecture baseline tactic.",
    )
    args = parser.parse_args(argv)

    try:
        import torch
    except ImportError as error:
        payload = {"status": "skipped", "reason": f"PyTorch unavailable: {error}"}
        write_json_evidence(args.output, payload)
        print(json.dumps(payload, sort_keys=True))
        return 2
    if not torch.cuda.is_available():
        payload = {"status": "skipped", "reason": "CUDA device unavailable to PyTorch"}
        write_json_evidence(args.output, payload)
        print(json.dumps(payload, sort_keys=True))
        return 2

    device = torch.device("cuda")
    torch.manual_seed(20260824)
    bridge = KudaCudaBridge(manifest=args.manifest, lib_path=args.library)
    target_arch = str(bridge.manifest["target_arch"]).replace("sm_", "sm")
    tactic_name = args.tactic or f"qwen2_gqa_decode_{target_arch}_fp16"
    contract = bridge.qwen_contracts.get(tactic_name)
    max_kv_len = int(getattr(contract, "kv_len_max", 512))
    stream_ptr = int(torch.cuda.current_stream(device).cuda_stream)
    cases: list[dict[str, Any]] = []
    try:
        for kv_len in args.kv_lens:
            if kv_len < 1 or kv_len > max_kv_len:
                raise ValueError(f"kv_len must be in 1..{max_kv_len}, got {kv_len}")
            query = torch.randn((1, 16, 1, 128), device=device, dtype=torch.float16).contiguous()
            key_cache = torch.randn((1, 2, kv_len, 128), device=device, dtype=torch.float16).contiguous()
            value_cache = torch.randn((1, 2, kv_len, 128), device=device, dtype=torch.float16).contiguous()
            expanded_key = key_cache.repeat_interleave(8, dim=1).contiguous()
            expanded_value = value_cache.repeat_interleave(8, dim=1).contiguous()
            native_output = torch.empty_like(query)

            def native() -> None:
                bridge.dispatch_qwen2_gqa_decode(
                    query,
                    key_cache,
                    value_cache,
                    native_output,
                    kv_len=kv_len,
                    stream_ptr=stream_ptr,
                    tactic_name=tactic_name,
                )

            def eager() -> None:
                scores = torch.matmul(query, expanded_key.transpose(-1, -2)) * (128.0 ** -0.5)
                probabilities = torch.softmax(scores, dim=-1)
                torch.matmul(probabilities, expanded_value, out=native_output)

            for _ in range(args.warmup):
                native()
            torch.cuda.synchronize(device)
            native_gpu_ms, native_host_ms = _event_time(torch, native, args.repetitions)

            for _ in range(args.warmup):
                eager()
            torch.cuda.synchronize(device)
            eager_gpu_ms, eager_host_ms = _event_time(torch, eager, args.repetitions)

            reference_scores = torch.matmul(query.float(), expanded_key.float().transpose(-1, -2)) * (128.0 ** -0.5)
            reference = torch.matmul(torch.softmax(reference_scores, dim=-1), expanded_value.float()).to(torch.float16)
            native()
            torch.cuda.synchronize(device)
            metrics = compare_values(
                native_output.detach().cpu().tolist(),
                reference.detach().cpu().tolist(),
                atol=3.5e-3,
                rtol=3.5e-3,
            )
            cases.append(
                {
                    "kv_len": kv_len,
                    "native_gpu_ms": native_gpu_ms,
                    "native_host_ms_p50": native_host_ms,
                    "pytorch_eager_gpu_ms": eager_gpu_ms,
                    "pytorch_eager_host_ms_p50": eager_host_ms,
                    "gpu_speedup_vs_pytorch": eager_gpu_ms / native_gpu_ms if native_gpu_ms else None,
                    "metrics": metrics,
                }
            )
    except Exception as error:
        payload = {"status": "failed", "reason": f"{type(error).__name__}: {error}", "cases": cases}
        write_json_evidence(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    finally:
        bridge.shutdown()

    passed = all(case["metrics"]["passed"] for case in cases)
    payload = {
        "status": "validated" if passed else "failed",
        "target_arch": bridge.manifest["target_arch"],
        "gpu": torch.cuda.get_device_name(device),
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "tactic_name": tactic_name,
        "cases": cases,
        "reference": "PyTorch FP32 score/softmax/value correctness; PyTorch eager FP16 timing comparison",
    }
    write_json_evidence(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
