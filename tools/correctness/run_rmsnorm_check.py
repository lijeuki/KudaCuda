"""Run the native RMSNorm tactic against an independent CPU reference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Keep both `python -m ...` and direct file execution usable from the project
# checkout, which is convenient for CI and local CUDA bring-up.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from KudaCuda.harness.compare import compare_values, write_json_evidence
from KudaCuda.harness.reference import rmsnorm_reference
from KudaCuda.python.kudacuda_sidecar import KudaCudaBridge


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=False)
    parser.add_argument("--rows", type=int, default=7)
    parser.add_argument("--hidden", type=int, default=513)
    parser.add_argument("--epsilon", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=17)
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

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    input_tensor = torch.randn((args.rows, args.hidden), device=device, dtype=torch.float32)
    weight = torch.randn((args.hidden,), device=device, dtype=torch.float32)
    output = torch.empty_like(input_tensor)
    bridge = None
    try:
        bridge = KudaCudaBridge(manifest=args.manifest, lib_path=args.library)
        bridge.rmsnorm(
            input_tensor,
            weight,
            output,
            epsilon=args.epsilon,
            stream_ptr=torch.cuda.current_stream().cuda_stream,
        )
        torch.cuda.synchronize()
        expected = rmsnorm_reference(input_tensor.cpu().tolist(), weight.cpu().tolist(), args.epsilon)
        metrics = compare_values(output.cpu().tolist(), expected, atol=5.0e-5, rtol=5.0e-5)
        payload: dict[str, Any] = {
            "status": "validated" if metrics["passed"] else "failed",
            "target_arch": json.loads(args.manifest.read_text(encoding="utf-8"))["target_arch"],
            "shape": [args.rows, args.hidden],
            "dtype": "float32",
            "seed": args.seed,
            "reference": "KudaCuda.harness.reference.rmsnorm_reference",
            "metrics": metrics,
        }
        write_json_evidence(args.output, payload)
        print(json.dumps(payload, sort_keys=True))
        return 0 if metrics["passed"] else 1
    except Exception as error:
        payload = {"status": "error", "reason": str(error)}
        write_json_evidence(args.output, payload)
        print(json.dumps(payload, sort_keys=True))
        return 1
    finally:
        if bridge is not None:
            bridge.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
