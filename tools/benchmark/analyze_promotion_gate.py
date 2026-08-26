"""Evaluate the KudaCuda percentile promotion gate from measured samples."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable


def _percentile(values: Iterable[float], fraction: float) -> float | None:
    samples = sorted(float(value) for value in values)
    if not samples:
        return None
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0 and 1")
    index = (len(samples) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return samples[lower]
    weight = index - lower
    return samples[lower] + (samples[upper] - samples[lower]) * weight


def analyze_promotion_gate(
    *,
    candidate_values: Iterable[float],
    control_values: Iterable[float],
) -> dict[str, float | int | bool | str | None]:
    """Require candidate p25 throughput to exceed control p75 throughput."""

    candidate = [float(value) for value in candidate_values]
    control = [float(value) for value in control_values]
    if any(not math.isfinite(value) for value in (*candidate, *control)):
        raise ValueError("promotion samples must be finite numbers")

    candidate_p25 = _percentile(candidate, 0.25)
    control_p75 = _percentile(control, 0.75)
    if candidate_p25 is None or control_p75 is None:
        return {
            "candidate_p25": candidate_p25,
            "control_p75": control_p75,
            "candidate_samples": len(candidate),
            "control_samples": len(control),
            "promoted": False,
            "reason": "missing_samples",
        }

    promoted = candidate_p25 > control_p75
    return {
        "candidate_p25": candidate_p25,
        "control_p75": control_p75,
        "candidate_samples": len(candidate),
        "control_samples": len(control),
        "promoted": promoted,
        "reason": "promoted" if promoted else "candidate_p25_not_above_control_p75",
    }


def read_samples(path: Path, *, section: str | None = None) -> list[float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [float(value) for value in payload]
    if isinstance(payload, dict):
        if section is not None:
            payload = payload.get(section)
            if not isinstance(payload, dict):
                raise ValueError(f"{path} has no object section named {section!r}")
        for key in ("steady_state_tokens_per_second", "tokens_per_second", "itl_ms"):
            values = payload.get(key)
            if isinstance(values, list):
                if key == "itl_ms":
                    return [1000.0 / float(value) for value in values]
                return [float(value) for value in values]
    raise ValueError(f"{path} must contain a JSON list or a named sample list")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate-section")
    parser.add_argument("--control-section")
    args = parser.parse_args()
    result = analyze_promotion_gate(
        candidate_values=read_samples(args.candidate, section=args.candidate_section),
        control_values=read_samples(args.control, section=args.control_section),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["promoted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
