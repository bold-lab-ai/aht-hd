"""Numerical helpers extracted from the paper experiment implementation."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


def pava_nonincreasing(values: Sequence[float]) -> np.ndarray:
    """Least-squares non-increasing fit using pooled adjacent violators."""

    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("PAVA requires a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(y)):
        raise ValueError("PAVA values must be finite")

    # Apply the conventional non-decreasing PAVA to -y.
    blocks: list[list[float]] = []
    for value in -y:
        blocks.append([float(value), 1.0])
        while len(blocks) >= 2:
            left_mean = blocks[-2][0] / blocks[-2][1]
            right_mean = blocks[-1][0] / blocks[-1][1]
            if left_mean <= right_mean:
                break
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([left[0] + right[0], left[1] + right[1]])

    fitted: list[float] = []
    for total, weight in blocks:
        fitted.extend([-total / weight] * int(weight))
    result = np.asarray(fitted, dtype=np.float64)
    if np.any(np.diff(result) > 1e-12):
        raise AssertionError("Internal PAVA error: fit is not non-increasing")
    return result


def monotone_piecewise_fit(points: np.ndarray) -> dict[str, list[float]]:
    """Fit decreasing target KL as a piecewise-linear function of quality.

    Input columns are ``[optimal_target_kl, reference_quality]``. Duplicate
    quality values are averaged before PAVA.
    """

    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or len(array) < 2:
        raise ValueError("Calibration points must have shape [N, 2] with N >= 2")
    if not np.all(np.isfinite(array)):
        raise ValueError("Calibration points must be finite")
    if np.any(array[:, 0] < 0):
        raise ValueError("KL targets must be non-negative")

    order = np.argsort(array[:, 1], kind="stable")
    target = array[order, 0]
    quality = array[order, 1]

    unique_quality: list[float] = []
    mean_target: list[float] = []
    for x in np.unique(quality):
        unique_quality.append(float(x))
        mean_target.append(float(np.mean(target[quality == x])))
    fitted = pava_nonincreasing(mean_target)
    return {
        "quality": unique_quality,
        "target_kl": [float(value) for value in fitted],
    }


def piecewise_predict(
    quality: float, fit: Mapping[str, Sequence[float]], clamp: bool = True
) -> float:
    x = np.asarray(fit["quality"], dtype=np.float64)
    y = np.asarray(fit["target_kl"], dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError("Invalid piecewise fit")
    if np.any(np.diff(x) <= 0) or np.any(np.diff(y) > 1e-12):
        raise ValueError("Fit knots must have increasing x and non-increasing y")
    if not math.isfinite(quality):
        raise ValueError("Prediction quality must be finite")
    if clamp:
        return float(np.interp(quality, x, y))
    if quality < x[0]:
        lo, hi = 0, 1
    elif quality > x[-1]:
        lo, hi = len(x) - 2, len(x) - 1
    else:
        return float(np.interp(quality, x, y))
    slope = (y[hi] - y[lo]) / (x[hi] - x[lo])
    return float(y[lo] + slope * (quality - x[lo]))
