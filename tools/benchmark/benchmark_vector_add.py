"""Measure vector_add with CUDA events and host launch timing."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

# Keep both `python -m ...` and direct file execution usable from the project
# checkout, which is convenient for CI and local CUDA bring-up.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from KudaCuda.harness.compare import compare_values, write_json_evidence
from KudaCuda.python.kudacuda_sidecar import KudaCudaBridge


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=False)
    parser.add_argument("--count", type=int, default=1 << 20)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
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

    a = torch.linspace(0.0, 1.0, args.count, device="cuda", dtype=torch.float32)
    b = torch.linspace(1.0, 0.0, args.count, device="cuda", dtype=torch.float32)
    output = torch.empty_like(a)
    bridge = KudaCudaBridge(manifest=args.manifest, lib_path=args.library)
    stream_ptr = torch.cuda.current_stream().cuda_stream
    try:
        for _ in range(args.warmup):
            bridge.vector_add(a, b, output, stream_ptr=stream_ptr)
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        stop_event = torch.cuda.Event(enable_timing=True)
        host_durations = []
        start_event.record()
        for _ in range(args.repetitions):
            host_start = time.perf_counter()
            bridge.vector_add(a, b, output, stream_ptr=stream_ptr)
            host_durations.append((time.perf_counter() - host_start) * 1000.0)
        stop_event.record()
        stop_event.synchronize()

        expected = (a + b).cpu().tolist()
        metrics = compare_values(output.cpu().tolist(), expected, exact=True)
        payload = {
            "status": "validated" if metrics["passed"] else "failed",
            "target_arch": json.loads(args.manifest.read_text(encoding="utf-8"))["target_arch"],
            "count": args.count,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "gpu_ms_average": start_event.elapsed_time(stop_event) / args.repetitions,
            "host_ms_p50": statistics.median(host_durations),
            "reference": "torch.float32 addition",
            "metrics": metrics,
        }
        write_json_evidence(args.output, payload)
        print(json.dumps(payload, sort_keys=True))
        return 0 if metrics["passed"] else 1
    finally:
        bridge.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
