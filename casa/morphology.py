"""Morphology analysis — Phase 5, not implemented.

Head shape from the detection box and head/neck geometry: length, width,
ellipticity, and the head-neck insertion angle. Ranking combines morphology
with motility to shortlist candidate cells.
"""

from __future__ import annotations

from dataclasses import dataclass

from detection.detector import Detection


@dataclass(frozen=True)
class MorphologyMetrics:
    """Per-sperm head geometry in micrometres."""

    track_id: int
    head_length: float
    head_width: float
    ellipticity: float      # length / width
    neck_angle: float       # degrees between head-neck axis and head major axis
    is_normal: bool


def analyze(detection: Detection, microns_per_pixel: float) -> MorphologyMetrics:
    """Measure one sperm head."""
    raise NotImplementedError("Phase 5")


def rank(metrics: list[MorphologyMetrics]) -> list[MorphologyMetrics]:
    """Order cells best-first for selection."""
    raise NotImplementedError("Phase 5")
