"""Validate the native Qwen2 GQA decode tactic against a PyTorch reference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from KudaCuda.harness.compare import compare_values, write_json_evidence
from KudaCuda.python.kudacuda_sidecar import KudaCudaBridge


def reference_qwen2_gqa(torch: Any, query: Any, key_cache: Any, value_cache: Any) -> Any:
    """Compute the independent FP32-accumulation reference for one decode step."""

    scale = 1.0 / (int(query.shape[-1]) ** 0.5)
    group_size = int(query.shape[1]) // int(key_cache.shape[1])
    expanded_key = key_cache.repeat_interleave(group_size, dim=1)
    expanded_value = value_cache.repeat_interleave(group_size, dim=1)
    scores = torch.matmul(query.float(), expanded_key.float().transpose(-1, -2)) * scale
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, expanded_value.float()).to(dtype=query.dtype)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tactic",
        default=None,
        help="Exact Qwen tactic name; defaults to the architecture baseline tactic.",
    )
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--kv-lens", type=int, nargs="+", default=[1, 7, 128, 512])
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
    bridge = None
    tactic_name = args.tactic
    cases: list[dict[str, Any]] = []
    try:
        bridge = KudaCudaBridge(manifest=args.manifest, lib_path=args.library)
        stream_ptr = int(torch.cuda.current_stream(device).cuda_stream)
        tactic_name = args.tactic or (
            f"qwen2_gqa_decode_{str(bridge.manifest['target_arch']).replace('sm_', 'sm')}_fp16"
        )
        for kv_len in args.kv_lens:
            if kv_len < 1 or kv_len > 512:
                raise ValueError(f"kv_len must be in 1..512, got {kv_len}")
            query = torch.randn((1, 16, 1, 128), device=device, dtype=torch.float16).contiguous()
            key_cache = torch.randn((1, 2, kv_len, 128), device=device, dtype=torch.float16).contiguous()
            value_cache = torch.randn((1, 2, kv_len, 128), device=device, dtype=torch.float16).contiguous()
            output = torch.empty_like(query)
            expected = reference_qwen2_gqa(torch, query, key_cache, value_cache)
            bridge.dispatch_qwen2_gqa_decode(
                query,
                key_cache,
                value_cache,
                output,
                kv_len=kv_len,
                stream_ptr=stream_ptr,
                tactic_name=tactic_name,
            )
            torch.cuda.synchronize(device)
            metrics = compare_values(
                output.detach().cpu().tolist(),
                expected.detach().cpu().tolist(),
                atol=3.5e-3,
                rtol=3.5e-3,
            )
            cases.append({"kv_len": kv_len, "metrics": metrics})
    except Exception as error:
        payload = {
            "status": "failed",
            "reason": f"{type(error).__name__}: {error}",
            "cases": cases,
        }
        write_json_evidence(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    finally:
        if bridge is not None:
            bridge.shutdown()

    passed = all(case["metrics"]["passed"] for case in cases)
    payload = {
        "status": "validated" if passed else "failed",
        "target_arch": json.loads(args.manifest.read_text(encoding="utf-8"))["target_arch"],
        "gpu": torch.cuda.get_device_name(device),
        "seed": args.seed,
        "cases": cases,
        "native_attention_dispatches": getattr(bridge, "native_attention_dispatches", None),
        "tactic_name": tactic_name,
        "reference": "torch FP32 score/softmax/value accumulation, FP16 output",
    }
    write_json_evidence(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
