"""Small independent CPU references used by correctness gates."""

from __future__ import annotations

import math
from typing import Sequence


def rmsnorm_reference(
    rows: Sequence[Sequence[float]],
    weight: Sequence[float],
    epsilon: float = 1.0e-5,
) -> list[list[float]]:
    """Compute row-wise RMSNorm using host double-precision accumulation."""

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    hidden = len(weight)
    if hidden == 0:
        raise ValueError("weight must not be empty")
    result: list[list[float]] = []
    for row in rows:
        if len(row) != hidden:
            raise ValueError("each input row must match the weight length")
        mean_square = sum(float(value) * float(value) for value in row) / hidden
        inverse_rms = 1.0 / math.sqrt(mean_square + epsilon)
        result.append([
            float(value) * inverse_rms * float(scale)
            for value, scale in zip(row, weight)
        ])
    return result
