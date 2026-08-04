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
    # False where the point was interpolated across a detection gap rather
    # than measured. Left empty by callers that never fill gaps, and then read
    # as "everything here was observed".
    observed: list[bool] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def head_array(self) -> np.ndarray:
        """Head path as an (N, 2) array, filled points included."""
        return np.array(self.head_points, dtype=np.float64).reshape(-1, 2)

    @property
    def observed_mask(self) -> np.ndarray:
        """Which points were actually detected."""
        if not self.observed:
            return np.ones(len(self.frames), dtype=bool)
        return np.array(self.observed, dtype=bool)

    @property
    def filled(self) -> int:
        """Points interpolated across a gap."""
        return int((~self.observed_mask).sum())

    @property
    def gaps(self) -> int:
        """Frames the track was lost and later recovered.

        Counts real absences, so filling a gap does not hide it.
        """
        if len(self) < 2:
            return 0
        seen = int(self.observed_mask.sum())
        return (self.frames[-1] - self.frames[0] + 1) - seen

    def fill_gaps(self, max_gap: int = 30) -> int:
        """Interpolate the frames this cell was not detected in.

        ByteTrack holds a lost identity for ``track_buffer * fps/30`` frames —
        49 at our rate — so a cell can vanish and come back under the *same*
        number. Measured across the four VISEM clips that happens 249 times for
        1,311 missing frames, against only 6 cases where the identity was
        actually lost and restarted. So the common defect is a hole in an
        otherwise intact trajectory, not a renamed track.

        The filled points are marked, and :func:`casa.metrics.compute_metrics`
        ignores them. They are a straight line, and ALH and BCF measure exactly
        the wobble a straight line does not have, so letting them into the
        kinematics would quietly flatten both.

        Gaps longer than ``max_gap`` are left alone: over a long absence a
        straight line stops being a fair guess at where the cell went.
        """
        if len(self) < 2:
            return 0

        frames, heads, necks, observed = [], [], [], []
        for index in range(len(self.frames) - 1):
            frames.append(self.frames[index])
            heads.append(self.head_points[index])
            necks.append(self.neck_points[index])
            observed.append(bool(self.observed_mask[index]))

            span = self.frames[index + 1] - self.frames[index]
            if not 1 < span <= max_gap:
                continue
            for step in range(1, span):
                weight = step / span
                frames.append(self.frames[index] + step)
                heads.append(_lerp(self.head_points[index], self.head_points[index + 1], weight))
                necks.append(_lerp(self.neck_points[index], self.neck_points[index + 1], weight))
                observed.append(False)

        frames.append(self.frames[-1])
        heads.append(self.head_points[-1])
        necks.append(self.neck_points[-1])
        observed.append(bool(self.observed_mask[-1]))

        added = len(frames) - len(self.frames)
        self.frames, self.head_points, self.neck_points, self.observed = (
            frames, heads, necks, observed)
        return added


def _lerp(a: tuple[float, float], b: tuple[float, float], weight: float) -> tuple[float, float]:
    return (a[0] + (b[0] - a[0]) * weight, a[1] + (b[1] - a[1]) * weight)


def _heading(trajectory: Trajectory, lag: int, window: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Last clean position, velocity and frame of a trajectory.

    The final frames before a track dies are the ones where the detector was
    already blending this cell with whatever it collided into, so they are
    dropped before the velocity is fitted. This is the same correction that
    decided the 4/32 crossing in 22.mp4: fitting through the contaminated tail
    put both tracks on the wrong cell.
    """
    frames = trajectory.frames
    points = trajectory.head_array
    cutoff = frames[-1] - lag
    usable = [i for i, f in enumerate(frames) if f <= cutoff][-window:]
    if len(usable) < 2:
        usable = list(range(len(frames)))[-2:]
    if len(usable) < 2 or frames[usable[-1]] == frames[usable[0]]:
        return points[-1], np.zeros(2), frames[-1]
    first, last = usable[0], usable[-1]
    velocity = (points[last] - points[first]) / (frames[last] - frames[first])
    return points[last], velocity, frames[last]


def repair_fragments(
    trajectories: dict[int, Trajectory],
    max_gap: int = 20,
    max_distance: int = 20,
    lag: int = 8,
    window: int = 10,
) -> dict[int, int]:
    """Join a track that is plainly the continuation of a dead one.

    ByteTrack already re-acquires a lost identity for ``track_buffer * fps/30``
    frames — 49 at our rate — so this only ever sees the residue. Measured on
    the four VISEM clips: 249 same-identity gaps that never needed renaming,
    against a handful of births with a plausible predecessor. Do not expect it
    to fire often; on those clips it fires **once** in 5,880 frames.

    Two tracks alive in the same frame are two cells, so any overlap in time
    disqualifies the pair outright — that is the guard that stops this
    inventing an identity switch while claiming to repair one.

    ``max_distance`` is the whole safety margin, and it was set by measurement
    rather than taste. Scored on repaired identities against the VISEM key:

    ======  ========  ========  ==========================================
    limit   switches  mean IDF1  note
    ======  ========  ========  ==========================================
    off           29    0.9030  baseline
    30 px         26    0.8934  6 joins, but clip 60 IDF1 0.794 -> 0.753
    **20 px**     28    0.9036  1 join, no clip worse
    15 px         28    0.9036  same single join
    10 px         29    0.9030  never fires
    ======  ========  ========  ==========================================

    At 30 px it buys three switches and pays for them by welding two different
    cells together on clip 60, which the switch count barely notices and IDF1
    punishes. 20 px is the widest gate where no clip regresses.

    Returns the joins made, as ``{old_id: canonical_id}``. This is not just a
    count: the video overlay is drawn from live per-frame tracker output, in
    a separate pass, before this ever runs — a stitch here does nothing to
    what has already been rendered unless the renderer is told which old IDs
    now mean which canonical one. That is what this mapping is for.
    """
    renamed: dict[int, int] = {}
    # Oldest first, so A->B->C collapses in one pass: B is merged into A before
    # C is considered, and C then matches the extended A.
    for track_id in sorted(trajectories, key=lambda t: trajectories[t].frames[0]):
        candidate = trajectories.get(track_id)
        if candidate is None or len(candidate) < 2:
            continue
        start = candidate.frames[0]

        best, best_gap = None, float("inf")
        for other_id, other in trajectories.items():
            if other_id == track_id or len(other) < 2:
                continue
            if not 0 < start - other.frames[-1] <= max_gap:
                continue
            if set(other.frames) & set(candidate.frames):     # co-existed: different cells
                continue
            last, velocity, last_frame = _heading(other, lag, window)
            predicted = last + velocity * (start - last_frame)
            distance = float(np.linalg.norm(predicted - candidate.head_array[0]))
            if distance <= max_distance and distance < best_gap:
                best, best_gap = other_id, distance

        if best is None:
            continue
        target = trajectories[best]
        gap = start - target.frames[-1]
        target.frames += candidate.frames
        target.head_points += candidate.head_points
        target.neck_points += candidate.neck_points
        target.observed = list(target.observed_mask) + list(candidate.observed_mask)
        del trajectories[track_id]
        renamed[track_id] = best
        logger.info("track %d continues track %d (%.1f px from prediction, %d frame gap)",
                    track_id, best, best_gap, gap)
    return renamed


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
            traj.observed.append(True)

    def finalize(
        self, min_length: int = 10, repair: bool = True,
    ) -> tuple[dict[int, Trajectory], dict[int, int]]:
        """Return trajectories long enough to measure, gaps repaired.

        Order matters. Fragments are joined *before* the length filter, so a
        cell broken into two short pieces is measured as one long track instead
        of being discarded twice; and gaps are filled last, so a join's own gap
        is filled too.

        Also returns the fragment-repair rename map, ``{old_id: canonical_id}``
        — empty when ``repair`` is False. The video overlay is drawn from live
        per-frame tracker output *before* this ever runs (see
        ``detection/inference.py``), so without this map a stitch made here
        would exist in the metrics and never appear on screen.
        """
        kept = dict(self._trajectories)
        renamed: dict[int, int] = {}
        if repair:
            renamed = repair_fragments(kept)
            filled = sum(t.fill_gaps() for t in kept.values())
            logger.info("repaired %d fragment(s), filled %d frame(s) across %d gap(s)",
                        len(renamed), filled, sum(1 for t in kept.values() if t.filled))

        kept = {k: v for k, v in kept.items() if len(v) >= min_length}
        logger.info("kept %d/%d trajectories at min_length=%d",
                    len(kept), len(self._trajectories), min_length)
        return kept, renamed

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


if __name__ == "__main__":
    # ponytail: one self-check instead of a suite. The two things that must
    # hold are that filling a gap does not move any kinematic number, and that
    # the stitcher refuses two cells that were alive at the same time — the
    # failure mode where "repairing fragmentation" quietly invents an identity
    # switch instead.
    from casa.metrics import compute_metrics

    straight = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]
    gapped = Trajectory(
        track_id=1, frames=list(straight),
        head_points=[(float(i), 0.0) for i in straight],
        neck_points=[(float(i) - 2, 0.0) for i in straight],
        observed=[True] * len(straight))

    before = compute_metrics(gapped, 49.0, 0.495)
    added = gapped.fill_gaps()
    after = compute_metrics(gapped, 49.0, 0.495)

    assert added == 5, f"expected 5 filled points, got {added}"
    assert gapped.frames == list(range(15)), f"path not continuous: {gapped.frames}"
    assert gapped.filled == 5, f"filled points not marked: {gapped.filled}"
    assert gapped.gaps == 5, f"gaps must still report the real absence, got {gapped.gaps}"
    assert abs(before.vcl - after.vcl) < 1e-9, "interpolated points changed VCL"
    assert abs(before.alh - after.alh) < 1e-9, "interpolated points changed ALH"

    def _track(track_id: int, frames: list[int]) -> Trajectory:
        return Trajectory(track_id=track_id, frames=list(frames),
                          head_points=[(float(i), 0.0) for i in frames],
                          neck_points=[(float(i) - 2, 0.0) for i in frames],
                          observed=[True] * len(frames))

    broken = {2: _track(2, [0, 1, 2, 3, 4]), 3: _track(3, [7, 8, 9, 10])}
    remap = repair_fragments(broken)
    assert remap == {3: 2}, f"a clean continuation was not joined correctly: {remap}"
    assert list(broken) == [2], f"join kept the wrong id: {list(broken)}"
    assert broken[2].frames == [0, 1, 2, 3, 4, 7, 8, 9, 10]

    overlapping = {4: _track(4, [0, 1, 2, 3, 4]), 5: _track(5, [3, 4, 5, 6])}
    assert repair_fragments(overlapping) == {}, "joined two cells that co-existed"
    assert len(overlapping) == 2

    # A chain: 7 continues 6, then 8 continues the now-extended 6. The map
    # must point both 7 and 8 straight at 6, not 8 at 7 — a video overlay
    # relabels every frame in one dict lookup, so a two-hop chain would leave
    # 8's frames unrelabelled.
    chained = {6: _track(6, [0, 1, 2, 3, 4]), 7: _track(7, [7, 8, 9, 10]),
              8: _track(8, [14, 15, 16, 17])}
    chain_remap = repair_fragments(chained)
    assert chain_remap == {7: 6, 8: 6}, f"chained merge did not collapse to one id: {chain_remap}"
    assert list(chained) == [6]

    # A gap longer than the limit stays open: a straight line across a long
    # absence is a guess, not a measurement.
    far = _track(6, [0, 1, 2])
    far.frames += [90, 91]
    far.head_points += [(90.0, 0.0), (91.0, 0.0)]
    far.neck_points += [(88.0, 0.0), (89.0, 0.0)]
    far.observed += [True, True]
    assert far.fill_gaps(max_gap=30) == 0, "filled a gap beyond max_gap"

    print("trajectory.py self-check passed")
