"""Motility classification.

WHO 5th/6th edition grades: progressive, non-progressive, immotile. Thresholds
are laboratory-tunable, so they live in a config object rather than as
constants baked into the classifier.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum

from casa.metrics import KinematicMetrics


class MotilityGrade(str, Enum):
    PROGRESSIVE = "progressive"
    NON_PROGRESSIVE = "non_progressive"
    IMMOTILE = "immotile"
    UNRELIABLE = "unreliable"  # tracking artifact, not a real motility class


@dataclass
class MotilityThresholds:
    """WHO-style cut-offs; adjust per laboratory."""

    immotile_vcl: float = 10.0     # um/s below which a cell counts as immotile
    progressive_vsl: float = 25.0  # um/s
    progressive_str: float = 0.8   # VSL/VAP ratio


@dataclass(frozen=True)
class MotilityReport:
    """Sample-level summary."""

    counts: dict[MotilityGrade, int]
    percentages: dict[MotilityGrade, float]
    total: int


def classify(metrics: KinematicMetrics, thresholds: MotilityThresholds) -> MotilityGrade:
    """Grade a single sperm.

    Unreliable is checked first: a track flagged implausible has an ID switch
    in it, so its VSL/STR would otherwise sail past the progressive
    thresholds and report a tracking artifact as the healthiest cell in the
    sample.

    Immotile is checked next on VCL alone — a cell with almost no raw path
    length is immotile regardless of what LIN/STR say, since those ratios are
    undefined (0/0-ish) at near-zero speed and computing them first can
    misclassify jitter as any of the other two grades.
    """
    if not metrics.plausible:
        return MotilityGrade.UNRELIABLE
    if metrics.vcl < thresholds.immotile_vcl:
        return MotilityGrade.IMMOTILE
    if metrics.vsl >= thresholds.progressive_vsl and metrics.str_ >= thresholds.progressive_str:
        return MotilityGrade.PROGRESSIVE
    return MotilityGrade.NON_PROGRESSIVE


def summarize(metrics: list[KinematicMetrics], thresholds: MotilityThresholds) -> MotilityReport:
    """Aggregate grades across the sample."""
    grades = [classify(m, thresholds) for m in metrics]
    counts = Counter(grades)
    total = len(metrics)
    return MotilityReport(
        counts={g: counts.get(g, 0) for g in MotilityGrade},
        percentages={g: (100.0 * counts.get(g, 0) / total if total else 0.0) for g in MotilityGrade},
        total=total,
    )


if __name__ == "__main__":
    # ponytail: one self-check instead of a suite — one cell per grade,
    # exercised through classify() and summarize() together.
    fast = KinematicMetrics(track_id=1, vcl=120.0, vsl=100.0, vap=110.0,
                            lin=0.83, str_=0.91, wob=0.92, alh=2.0, bcf=8.0)
    slow = KinematicMetrics(track_id=2, vcl=40.0, vsl=5.0, vap=15.0,
                            lin=0.13, str_=0.33, wob=0.38, alh=3.0, bcf=5.0)
    still = KinematicMetrics(track_id=3, vcl=2.0, vsl=0.5, vap=1.0,
                             lin=0.25, str_=0.5, wob=0.5, alh=0.1, bcf=0.0)

    # An ID switch produces high VSL and high STR — exactly the progressive
    # signature — so it must be caught by the plausible flag, not the
    # thresholds, or it gets reported as the best cell in the sample.
    artifact = KinematicMetrics(track_id=4, vcl=4363.0, vsl=838.0, vap=1381.0,
                                lin=0.19, str_=0.61, wob=0.32, alh=4.3, bcf=26.0,
                                plausible=False)

    thresholds = MotilityThresholds()
    assert classify(fast, thresholds) == MotilityGrade.PROGRESSIVE
    assert classify(slow, thresholds) == MotilityGrade.NON_PROGRESSIVE
    assert classify(still, thresholds) == MotilityGrade.IMMOTILE
    assert classify(artifact, thresholds) == MotilityGrade.UNRELIABLE

    report = summarize([fast, slow, still, artifact], thresholds)
    assert report.total == 4
    assert report.counts[MotilityGrade.PROGRESSIVE] == 1
    assert report.counts[MotilityGrade.UNRELIABLE] == 1
    assert abs(report.percentages[MotilityGrade.IMMOTILE] - 25.0) < 0.01

    print("motility.py self-check passed")
