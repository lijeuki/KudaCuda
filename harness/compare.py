"""Independent output comparison and evidence serialization helpers."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _flatten(values: Any):
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        for value in values:
            yield from _flatten(value)
        return
    yield values


def compare_values(
    actual: Any,
    expected: Any,
    *,
    atol: float = 1.0e-5,
    rtol: float = 1.0e-5,
    exact: bool = False,
) -> dict[str, Any]:
    """Compare scalar or nested sequence values and return evidence metrics."""

    actual_values = list(_flatten(actual))
    expected_values = list(_flatten(expected))
    shape_match = len(actual_values) == len(expected_values)
    count = min(len(actual_values), len(expected_values))
    max_abs_error = 0.0
    max_rel_error = 0.0
    squared_error = 0.0
    mismatches = 0
    nan_actual = 0
    nan_expected = 0
    inf_mismatch = 0

    for actual_value, expected_value in zip(actual_values, expected_values):
        actual_float = float(actual_value)
        expected_float = float(expected_value)
        actual_nan = math.isnan(actual_float)
        expected_nan = math.isnan(expected_float)
        if actual_nan:
            nan_actual += 1
        if expected_nan:
            nan_expected += 1
        if actual_nan or expected_nan:
            mismatches += int(not (actual_nan and expected_nan))
            continue

        actual_inf = math.isinf(actual_float)
        expected_inf = math.isinf(expected_float)
        if actual_inf or expected_inf:
            if not (actual_inf and expected_inf and actual_float == expected_float):
                inf_mismatch += 1
                mismatches += 1
            continue

        if exact:
            matches = actual_value == expected_value
            absolute_error = 0.0 if matches else abs(actual_float - expected_float)
        else:
            absolute_error = abs(actual_float - expected_float)
            matches = absolute_error <= atol + rtol * abs(expected_float)
        if not matches:
            mismatches += 1
        relative_error = absolute_error / max(abs(expected_float), 1.0e-12)
        max_abs_error = max(max_abs_error, absolute_error)
        max_rel_error = max(max_rel_error, relative_error)
        squared_error += absolute_error * absolute_error

    if not shape_match:
        mismatches += abs(len(actual_values) - len(expected_values))
    denominator = max(count, 1)
    return {
        "passed": shape_match and mismatches == 0 and nan_actual == 0 and nan_expected == 0 and inf_mismatch == 0,
        "exact": exact,
        "atol": atol,
        "rtol": rtol,
        "count": count,
        "actual_count": len(actual_values),
        "expected_count": len(expected_values),
        "shape_match": shape_match,
        "mismatches": mismatches,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "rms_error": math.sqrt(squared_error / denominator),
        "nan_actual": nan_actual,
        "nan_expected": nan_expected,
        "inf_mismatch": inf_mismatch,
    }


def write_json_evidence(path: str | Path, payload: dict[str, Any]) -> None:
    """Write deterministic, human-readable JSON evidence."""

    evidence_path = Path(path)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
