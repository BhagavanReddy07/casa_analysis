"""Trajectory accumulation.

Holds the per-track path that CASA metrics are computed from. Both the head
path (drives VCL/VSL/VAP) and the neck point (drives orientation and, later,
head rotation) are kept.

Everything here stays in pixels. Conversion to micrometres happens once, in
the metrics stage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from tracking.tracker import Track

logger = logging.getLogger(__name__)


@dataclass
class Trajectory:
    """One sperm's path across frames, in pixel coordinates."""

    track_id: int
    frames: list[int] = field(default_factory=list)
    head_points: list[tuple[float, float]] = field(default_factory=list)
    neck_points: list[tuple[float, float]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def head_array(self) -> np.ndarray:
        """Head path as an (N, 2) array."""
        return np.array(self.head_points, dtype=np.float64).reshape(-1, 2)

    @property
    def gaps(self) -> int:
        """Frames the track was lost and later recovered."""
        return 0 if len(self) < 2 else (self.frames[-1] - self.frames[0] + 1) - len(self)


class TrajectoryBuilder:
    """Collects tracks into per-ID trajectories."""

    def __init__(self) -> None:
        self._trajectories: dict[int, Trajectory] = {}

    def add(self, tracks: list[Track]) -> None:
        """Append one frame's tracks."""
        for track in tracks:
            traj = self._trajectories.setdefault(track.track_id, Trajectory(track.track_id))
            traj.frames.append(track.frame_index)
            traj.head_points.append(track.detection.head)
            traj.neck_points.append(track.detection.neck)

    @property
    def trajectories(self) -> dict[int, Trajectory]:
        """Live view, including tracks too short to keep. Used for drawing."""
        return self._trajectories

    def finalize(self, min_length: int = 10) -> dict[int, Trajectory]:
        """Return trajectories long enough to measure."""
        kept = {k: v for k, v in self._trajectories.items() if len(v) >= min_length}
        logger.info("kept %d/%d trajectories at min_length=%d",
                    len(kept), len(self._trajectories), min_length)
        return kept

    def summary(
        self, min_length: int = 10,
        frame_size: tuple[int, int] | None = None, edge_margin: int = 30,
    ) -> dict[str, float]:
        """Track-quality figures — the numbers that say whether tracking held.

        ``frame_size`` (width, height), if given, adds ``edge_birth_fraction``:
        the share of tracks whose first point sits within ``edge_margin`` px
        of the frame border. On a full-length clip most new IDs are cells
        swimming into or out of frame — normal turnover, not a tracking
        failure — and this is what tells the two apart. A track that starts
        near-edge is presumed to be a real entry; a track starting mid-frame
        has no such excuse and is the actual fragmentation signal.
        """
        lengths = np.array([len(t) for t in self._trajectories.values()])
        if lengths.size == 0:
            return {"tracks": 0, "usable": 0, "mean_length": 0.0,
                    "median_length": 0.0, "max_length": 0.0, "mean_gaps": 0.0,
                    "edge_birth_fraction": 1.0}

        stats = {
            "tracks": int(lengths.size),
            "usable": int((lengths >= min_length).sum()),
            "mean_length": float(lengths.mean()),
            "median_length": float(np.median(lengths)),
            "max_length": float(lengths.max()),
            "mean_gaps": float(np.mean([t.gaps for t in self._trajectories.values()])),
        }

        if frame_size is not None:
            width, height = frame_size
            edge_births = sum(
                1 for t in self._trajectories.values() if t.head_points and (
                    t.head_points[0][0] < edge_margin or t.head_points[0][0] > width - edge_margin
                    or t.head_points[0][1] < edge_margin or t.head_points[0][1] > height - edge_margin
                )
            )
            stats["edge_birth_fraction"] = edge_births / len(self._trajectories)

        return stats
