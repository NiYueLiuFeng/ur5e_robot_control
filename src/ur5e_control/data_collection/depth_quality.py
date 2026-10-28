"""Raw depth-frame quality checks used before recording an episode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class DepthQualityResult:
    """Invalid-pixel statistics for one depth stream."""

    key: str
    sample_count: int
    minimum_invalid_ratio: float
    median_invalid_ratio: float
    maximum_invalid_ratio: float
    limit: float

    @property
    def ok(self) -> bool:
        return self.median_invalid_ratio <= self.limit


def depth_invalid_ratio(depth: Any) -> float:
    """Return the fraction of invalid pixels in a raw RealSense Z16 frame."""

    array = np.asarray(depth)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"depth frame must have shape HxW or HxWx1, got {array.shape}")
    invalid = ~np.isfinite(array) | (array <= 0)
    if np.issubdtype(array.dtype, np.integer):
        invalid |= array == np.iinfo(array.dtype).max
    return float(np.mean(invalid))


def evaluate_depth_quality(
    observations: Iterable[Mapping[str, Any]],
    max_invalid_ratio: Mapping[str, float],
) -> list[DepthQualityResult]:
    """Evaluate median invalid ratios over several raw observations."""

    samples = list(observations)
    if not samples:
        raise ValueError("at least one observation is required")
    results = []
    for key, limit in max_invalid_ratio.items():
        if not 0.0 <= float(limit) <= 1.0:
            raise ValueError(f"invalid ratio limit for {key!r} must be in [0, 1]")
        ratios = []
        for sample in samples:
            if key not in sample:
                raise ValueError(f"depth observation {key!r} is missing")
            ratios.append(depth_invalid_ratio(sample[key]))
        results.append(
            DepthQualityResult(
                key=key,
                sample_count=len(ratios),
                minimum_invalid_ratio=min(ratios),
                median_invalid_ratio=float(np.median(ratios)),
                maximum_invalid_ratio=max(ratios),
                limit=float(limit),
            )
        )
    return results
