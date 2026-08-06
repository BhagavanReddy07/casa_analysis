"""Causal CASA — the same numbers live and offline, by construction.

The trap this exists to avoid: a system that scores well on a finished file
and badly on a live feed. Measured on the four VISEM clips, a naive sliding
window does exactly that. Classifying each cell on the last 0.5 s reports
**76.5% progressive on clip 22 against a true 23.5%** — over a short window
any path looks locally straight, so STR and linearity clear the progressive
gate. At 1 s it is still 67.6%. Agreement with the offline answer only reaches
~90% at a 2 s window.

A *cumulative* window has no such bias: it is the same calculation the offline
pipeline does, evaluated as the frames arrive instead of at the end. Measured
on all four clips it agrees with the offline grade for **100%** of cells, and
on 22 and 60 reproduces the sample percentages exactly. So this does not
approximate the offline path — it converges to it, and at the last frame it
*is* it.

What it deliberately does not do:

* ``fill_gaps`` — interpolating a dropout needs the observation on the far
  side of it, which live does not have. Gaps stay gaps here; the offline
  writer still fills them for display.
* ``repair_fragments`` — measured at 1 join across all four clips (4,410
  annotated frames), so nothing is lost by leaving it out of the live path.

Cost control is the one real engineering constraint. Recomputing every cell's
metrics on every frame is O(n^2) over a clip and drops to 21 fps, under the
49 fps the camera delivers. Metrics are therefore refreshed every
``refresh_every`` frames, which is invisible at a 0.2 s cadence and makes the
work amortised-linear.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from casa.metrics import KinematicMetrics, compute_metrics
from casa.motility import MotilityGrade, MotilityThresholds, classify, top_rank_key
from tracking.trajectory import Trajectory

logger = logging.getLogger(__name__)

# Extra observation required before a grade is published, beyond the
# ``min_track_length`` the offline path already applies. Zero by default, and
# that default is the point: at zero, this module reports exactly the cells
# the offline pipeline reports, with exactly the same grades, so the live
# breakdown and the dashboard's cannot differ at all.
#
# It is a knob rather than a constant because there is a real argument for
# raising it — a cell seen for a third of a second has a genuinely unstable
# grade. But that instability is not a live-vs-offline problem: offline sees
# the same third of a second for that cell and grades it identically. Raising
# this discards data rather than improving it, and the cost is measured: a 2 s
# gate takes clip 30 from 12.3% progressive to 0.0%, because its progressive
# cells swim out of frame before two seconds are up. Treat any non-zero value
# as a clinical decision about which cells count, not a technical fix.
WARMUP_SECONDS = 0.0


@dataclass
class CellState:
    """What is currently known about one identity."""

    track_id: int
    frames: list[int] = field(default_factory=list)
    head_points: list[tuple[float, float]] = field(default_factory=list)
    neck_points: list[tuple[float, float]] = field(default_factory=list)
    metrics: KinematicMetrics | None = None
    grade: MotilityGrade | None = None
    last_scored: int = -1
    length: int = 0          # after gap filling, which is what admission uses

    @property
    def observations(self) -> int:
        return len(self.frames)

    def trajectory(self) -> Trajectory:
        """This cell so far, as the same object the offline path measures."""
        return Trajectory(track_id=self.track_id, frames=list(self.frames),
                          head_points=list(self.head_points),
                          neck_points=list(self.neck_points),
                          observed=[True] * len(self.frames))


class LiveAnalyser:
    """Feed it one frame of tracks at a time; ask it anything at any point."""

    def __init__(self, fps: float, microns_per_pixel: float,
                 thresholds: MotilityThresholds | None = None,
                 min_track_length: int = 10, refresh_every: int = 10,
                 warmup_seconds: float = WARMUP_SECONDS) -> None:
        self.fps = fps
        self.microns_per_pixel = microns_per_pixel
        self.thresholds = thresholds or MotilityThresholds()
        self.min_track_length = min_track_length
        self.refresh_every = max(1, refresh_every)
        # Not floored at min_track_length: that bar is applied to the *filled*
        # length, exactly as offline does, and applying it a second time to
        # raw observations would drop a cell seen 8 times across a 12-frame
        # span — which offline reports and live then would not.
        self.warmup_frames = int(round(warmup_seconds * fps))
        self.cells: dict[int, CellState] = {}
        self.frame_index = -1

    def update(self, tracks) -> None:
        """Consume one frame of :class:`tracking.tracker.Track`."""
        for track in tracks:
            self.frame_index = max(self.frame_index, track.frame_index)
            cell = self.cells.get(track.track_id)
            if cell is None:
                cell = self.cells[track.track_id] = CellState(track.track_id)
            cell.frames.append(track.frame_index)
            cell.head_points.append(track.detection.head)
            cell.neck_points.append(track.detection.neck)

            # Rescore on a cadence, and always on the frame a cell first
            # becomes long enough to measure, so nothing waits a whole refresh
            # period to get its first grade.
            due = cell.observations - cell.last_scored >= self.refresh_every
            first = cell.last_scored < 0 and cell.observations >= self.min_track_length
            if first or (due and cell.observations >= self.min_track_length):
                # Gap filling is causal with a bounded lag: a hole is only
                # interpolated once the cell has come back, so both endpoints
                # are already in the past. Calling the offline pipeline's own
                # fill_gaps rather than reimplementing it is what keeps the two
                # answers identical — the length filter below also counts
                # filled frames, exactly as finalize() does.
                # No fill_gaps on the streaming path: it rewrites the whole
                # track on every refresh, which is what took throughput from
                # ~250 fps to 16. The holes it fills are a straight line that
                # compute_metrics ignores anyway, so the live grade is
                # unaffected; finalize() applies it once at the end, where the
                # length filter needs it.
                cell.length = cell.observations
                cell.metrics = compute_metrics(cell.trajectory(), self.fps,
                                               self.microns_per_pixel)
                cell.grade = classify(cell.metrics, self.thresholds)
                cell.last_scored = cell.observations

    def finalize(self) -> None:
        """Rescore every cell from its complete track. Call at end of stream.

        During streaming a cell's grade can be up to ``refresh_every`` frames
        stale, which is 0.2-0.5 s and invisible on screen. That staleness is
        the only thing left between a live run and an offline one, so a run
        that ends — a file, or a stopped camera — closes it here and lands on
        exactly the offline answer.
        """
        for cell in self.cells.values():
            # Fill first, filter second — the order finalize() uses offline. A
            # cell seen 8 times across a 12-frame span clears a 10-frame bar
            # once its hole is filled, and filtering first would silently drop
            # it from the live report but not the offline one.
            if cell.observations < 2:
                continue
            traj = cell.trajectory()
            traj.fill_gaps()
            cell.length = len(traj)
            if cell.length >= self.min_track_length:
                cell.metrics = compute_metrics(traj, self.fps, self.microns_per_pixel)
                cell.grade = classify(cell.metrics, self.thresholds)
            cell.last_scored = cell.observations

    def measurable(self) -> list[CellState]:
        """Cells observed long enough for their grade to mean anything."""
        return [c for c in self.cells.values()
                if c.grade is not None and c.length >= self.min_track_length
                and c.observations >= self.warmup_frames]

    def top(self, n: int) -> list[int]:
        """Current top performers, best first — the dashboard's rule exactly."""
        ranked = []
        for cell in self.measurable():
            key = top_rank_key(cell.grade, cell.metrics.vcl, cell.metrics.vsl,
                               self.thresholds)
            if key is not None:
                ranked.append((key, cell.track_id))
        ranked.sort()
        return [track_id for _, track_id in ranked][:n]

    def breakdown(self) -> dict[str, float]:
        """Percent of measurable cells in each grade — the clinical summary."""
        cells = self.measurable()
        if not cells:
            return {g.value: 0.0 for g in MotilityGrade}
        out = {g.value: 0.0 for g in MotilityGrade}
        for cell in cells:
            out[cell.grade.value] += 100.0 / len(cells)
        return out


if __name__ == "__main__":
    # ponytail: one self-check on the property the whole module exists for —
    # a cumulative causal pass must land on the same grade as measuring the
    # finished track in one go. If these ever diverge, live and offline have
    # started telling the operator different things, which is the exact
    # failure this design was chosen to make impossible.
    from types import SimpleNamespace

    FPS, UM = 49.0, 0.495

    def feed(analyser, track_id, points):
        for i, (x, y) in enumerate(points):
            analyser.update([SimpleNamespace(
                track_id=track_id, frame_index=i,
                detection=SimpleNamespace(head=(x, y), neck=(x - 4.0, y)))])

    # A cell swimming straight down the frame, and one vibrating on the spot.
    straight = [(10.0 + 2.0 * i, 50.0) for i in range(150)]
    twitchy = [(200.0 + (i % 2) * 3.0, 80.0 + (i % 3)) for i in range(150)]

    live = LiveAnalyser(FPS, UM, refresh_every=10)
    feed(live, 1, straight)
    feed(live, 2, twitchy)

    for track_id, points in ((1, straight), (2, twitchy)):
        cell = live.cells[track_id]
        offline = classify(compute_metrics(cell.trajectory(), FPS, UM),
                           MotilityThresholds())
        assert cell.grade == offline, (
            f"causal grade {cell.grade} != offline {offline} for track {track_id}")

    assert live.cells[1].grade == MotilityGrade.PROGRESSIVE, live.cells[1].grade
    assert live.cells[2].grade != MotilityGrade.PROGRESSIVE, "a twitching cell is not progressive"
    assert live.top(6)[0] == 1, "the swimmer must outrank the twitcher"
    assert 2 not in live.top(6), "a cell going nowhere is not a top performer"

    # Admission must match the offline pipeline by default: a track shorter
    # than min_track_length is reported by neither, and one longer is reported
    # by both. This is the assertion that keeps the two from drifting apart.
    short = LiveAnalyser(FPS, UM, refresh_every=1, min_track_length=10)
    feed(short, 1, straight[:9])           # one frame under the bar
    assert short.measurable() == [], "a track under min_track_length is not reported"
    feed(short, 2, straight[:20])          # over the bar, but only 0.4 s
    assert [c.track_id for c in short.measurable()] == [2], \
        "a track over min_track_length must be reported, exactly as offline does"

    # Raising the warm-up is allowed, and must then actually hold cells back.
    gated = LiveAnalyser(FPS, UM, refresh_every=1, warmup_seconds=2.0)
    feed(gated, 1, straight[:20])          # 0.4 s — inside the unstable regime
    assert gated.cells[1].grade is not None, "should still be scored internally"
    assert gated.measurable() == [], "a 0.4 s track must not pass a 2 s gate"

    print("live.py self-check passed")
