"""Multi-object tracking for sperm.

Wraps the ByteTrack implementation that ships with ultralytics rather than
reimplementing it. Two project-specific pieces sit around it:

* near-duplicate head suppression, because the detector occasionally fires
  twice on one agglutinated cell (~7% of detections on 38.mp4) and each copy
  would otherwise claim its own identity;
* a detector-agnostic interface, so the tracker consumes :class:`Detection`
  objects instead of ultralytics result tensors.

ByteTrack associates on bounding-box IoU. That is valid here: at 0.495 um/px
and 49 fps a progressive sperm moves ~2 px per frame against a ~16 px box, so
consecutive-frame boxes overlap heavily.

IoU alone is *not* enough when two cells cross: both detections then overlap
both predicted boxes by a similar amount, the cost matrix is near-degenerate
and the assignment is free to swap the two identities. So we add a
Kalman-prediction distance term to the cost (see :class:`_MotionBYTETracker`),
which stays discriminative while the boxes overlap.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker

from detection.detector import Detection

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Track:
    """A detection bound to a persistent identity."""

    track_id: int
    frame_index: int
    detection: Detection


@dataclass
class TrackerConfig:
    """ByteTrack parameters plus our duplicate-suppression radius.

    ``track_buffer`` is scaled internally by ``frame_rate / 30``, so at 49 fps
    a buffer of 30 keeps a lost sperm alive for about one second before its
    identity is retired.

    ``match_thresh`` is a *distance* threshold (1 - IoU), not a required
    overlap â€” despite the name, higher is more lenient. The ultralytics
    default of 0.8 is tuned for boxes tens of pixels wide; our sperm boxes are
    only ~15 px, so a fast cell's frame-to-frame displacement (measured ~11 px
    on 38.mp4) can be a large fraction of the box width and drop IoU below
    what 0.8 accepts. That killed and respawned a continuously-detected,
    never-lost cell under three different IDs in 30 frames (25 -> 54 -> 60).
    Sweeping 0.8-1.0 on all four clips: 0.95-0.99 plateau at the same, much
    lower track count with low mean_gaps (real fix); 1.0 removes the
    threshold's filtering entirely and mean_gaps triples, i.e. it starts
    accepting bad matches and can splice two different cells into one
    trajectory. 0.95 is the highest value that still rejects those.
    """

    track_high_thresh: float = 0.25   # first association
    track_low_thresh: float = 0.10    # second association (the "BYTE" pass)
    new_track_thresh: float = 0.25    # confidence needed to start an identity
    track_buffer: int = 30
    match_thresh: float = 0.95
    fuse_score: bool = True

    dedupe_distance: float = 10.0     # px between head keypoints
    min_track_length: int = 10        # frames, used when reporting

    # Crossing cells: how much of the association cost comes from the distance
    # to the Kalman-predicted position instead of from box overlap, and the
    # displacement that scores a full unit of it. The two are *blended*, not
    # summed â€” match_thresh already sits near its ceiling at 0.95, so adding a
    # term on top pushes legitimate re-acquisitions after an occlusion over the
    # threshold and spawns a fresh ID, which is the failure it was meant to
    # fix. Calibration knob: raise motion_weight if IDs still swap at
    # crossings, lower it if fast cells fragment.
    #
    # Replaying 600 cached frames of all four clips through the tracker,
    # against the baseline of no motion term and no crossing exemption below
    # (112 tracks / 89 usable / 18.0 mean gaps / 29 heading reversals / 884
    # detections suppressed as duplicates): 118 / 91 / 17.5 / 23 / 627. A
    # heading reversal mid-track is a cell handing its ID to another one, so
    # that count is the swap proxy. Weight is flat between 0.25 and 0.5; the
    # gate matters more â€” 20 px gives 25 reversals, 25 px gives 24 plus a
    # position jump, so a gate near one box width wins.
    motion_weight: float = 0.35
    motion_gate: float = 15.0         # px, ~one box width

    # A detection this close to a live track's last head continues that
    # identity. Measured frame-to-frame head displacement is ~11 px on 38.mp4.
    claim_distance: float = 12.0


# Keys BYTETracker reads off its args namespace; the rest are ours.
_BYTETRACK_KEYS = (
    "track_high_thresh", "track_low_thresh", "new_track_thresh",
    "track_buffer", "match_thresh", "fuse_score",
    "motion_weight", "motion_gate",
)


class _DetectionBoxes:
    """Duck-types the ultralytics ``Boxes`` fields that BYTETracker reads.

    Must not expose ``xywhr`` â€” BYTETracker treats its presence as the oriented
    bounding box case.
    """

    def __init__(self, detections: list[Detection]) -> None:
        boxes = np.array([d.bbox for d in detections], dtype=np.float32).reshape(-1, 4)
        x1, y1, x2, y2 = boxes.T
        self.xywh = np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)
        self.conf = np.array([d.confidence for d in detections], dtype=np.float32)
        self.cls = np.zeros(len(detections), dtype=np.float32)


class _MotionBYTETracker(BYTETracker):
    """ByteTrack with the Kalman-predicted position folded into the cost.

    Overlap-only costs tie when two cells cross; the distance between a
    detection and each track's *predicted* centre does not, because the two
    tracks predict forward along their own velocities. Measured against the
    ground truth in ``evaluation/``: dropping this term takes 22.mp4 from 2
    identity switches to 5.

    Three richer ideas were tried here and removed, each with numbers, so they
    are not retried by accident:

    * **a stationary rule** ("a dead cell cannot suddenly move") — no effect on
      any metric at any setting;
    * **a heading rule** ("a swimming cell cannot reverse") — direction has no
      signal at 49 fps, where the median step is 0.4 px and 43% of measured
      turn angles exceed 90 degrees. Forcing it cost 79 position jumps;
    * **orientation as an identity fingerprint** — a cell's head-to-neck axis
      moves a median 0.9 degrees per frame while two neighbouring cells differ
      by a median 83, which makes it the most discriminative signal available.
      It still changed nothing, because at the frames that fail the association
      is already correct and the errors lie elsewhere. Worth revisiting if
      identities are ever matched across a gap rather than frame to frame.
    """

    def get_dists(self, tracks, detections):
        dists = super().get_dists(tracks, detections)
        if not len(tracks) or not len(detections):
            return dists
        pred = np.array([t.xywh[:2] for t in tracks], dtype=np.float32)
        obs = np.array([d.xywh[:2] for d in detections], dtype=np.float32)
        gap = np.linalg.norm(pred[:, None, :] - obs[None, :, :], axis=2)
        motion = np.minimum(gap / self.args.motion_gate, 1.0)

        # Geometry is a blend, so the total stays in [0, 1] and match_thresh
        # keeps meaning what it meant. The stationary rule is added on top
        # instead: blending it in would shrink the geometry weight, and a
        # track with no history yet (rule cost 0) would then have a maximum
        # possible cost below match_thresh â€” the assignment would accept
        # literally any pairing. Measured: 200 position jumps against 0. As a
        # penalty it is zero for a plausible match and only ever pushes an
        # implausible one out of reach.
        w_m = self.args.motion_weight
        return (1.0 - w_m) * dists + w_m * motion


class SpermTracker:
    """Assigns stable IDs to per-frame detections."""

    def __init__(self, config: TrackerConfig | None = None, frame_rate: float = 30.0) -> None:
        self.config = config or TrackerConfig()
        self.frame_rate = frame_rate
        self._tracker = self._new_tracker()
        self.duplicates_removed = 0
        self._last_heads: dict[int, tuple[int, tuple[float, float]]] = {}

    def _new_tracker(self) -> BYTETracker:
        args = SimpleNamespace(**{k: v for k, v in asdict(self.config).items()
                                  if k in _BYTETRACK_KEYS})
        return _MotionBYTETracker(args, frame_rate=int(round(self.frame_rate)))

    def reset(self) -> None:
        """Drop all state â€” call between videos."""
        self._tracker = self._new_tracker()
        self.duplicates_removed = 0
        self._last_heads = {}

    def _claims(self, detections: list[Detection]) -> list[int | None]:
        """Which live identity each detection continues, at most one each.

        Greedy nearest-first so two converging cells claim their own two IDs
        instead of both claiming the nearer one.
        """
        pairs = sorted(
            (np.hypot(det.head[0] - head[0], det.head[1] - head[1]), i, tid)
            for i, det in enumerate(detections)
            for tid, (_, head) in self._last_heads.items()
        )
        claims: list[int | None] = [None] * len(detections)
        taken: set[int] = set()
        for dist, i, tid in pairs:
            if dist < self.config.claim_distance and claims[i] is None and tid not in taken:
                claims[i] = tid
                taken.add(tid)
        return claims

    def _dedupe(self, detections: list[Detection]) -> list[Detection]:
        """Drop the weaker of any two detections whose heads nearly coincide.

        Two detections that each continue a *different* live identity are two
        real cells passing close, not one cell detected twice â€” suppressing
        either one kills its track and respawns it under a new ID on the way
        out of the crossing, so both are kept.

        ponytail: O(n^2) over ~15 cells per frame. Switch to a KD-tree only if
        cell counts reach the hundreds.
        """
        if len(detections) < 2:
            return detections

        claims = self._claims(detections)
        order = sorted(range(len(detections)),
                       key=lambda i: detections[i].confidence, reverse=True)
        kept: list[tuple[Detection, int | None]] = []
        for i in order:
            det = detections[i]
            if all(np.hypot(det.head[0] - k.head[0], det.head[1] - k.head[1])
                   >= self.config.dedupe_distance or (claims[i] is not None and kc is not None)
                   for k, kc in kept):
                kept.append((det, claims[i]))
        self.duplicates_removed += len(detections) - len(kept)
        return [det for det, _ in kept]

    def update(self, detections: list[Detection], frame_index: int) -> list[Track]:
        """Associate this frame's detections with existing tracks."""
        detections = self._dedupe(detections)

        # Column layout of BYTETracker.update: x1,y1,x2,y2,track_id,score,cls,idx
        rows = self._tracker.update(_DetectionBoxes(detections))

        tracks = [
            Track(track_id=int(row[4]), frame_index=frame_index,
                  detection=detections[int(row[7])])
            for row in rows
        ]

        # Heads of live identities, so _dedupe can tell a crossing from a
        # double-fire. Retire them on the same buffer the tracker uses.
        self._last_heads.update({t.track_id: (frame_index, t.detection.head) for t in tracks})
        buffer = self.config.track_buffer * self.frame_rate / 30.0
        self._last_heads = {tid: v for tid, v in self._last_heads.items()
                            if frame_index - v[0] <= buffer}
        return tracks


if __name__ == "__main__":
    # ponytail: one self-check instead of a suite â€” a cell drifting 2 px/frame
    # (our measured speed) must keep a single ID, survive a two-frame dropout,
    # and duplicate heads must collapse to one identity.
    cfg = TrackerConfig()
    tracker = SpermTracker(cfg, frame_rate=49.0)

    def cell(x: float, conf: float = 0.9, y: float = 100.0) -> Detection:
        return Detection(bbox=(x - 8, y - 8, x + 8, y + 8), head=(x, y),
                         neck=(x - 10, y), confidence=conf)

    ids = []
    for i in range(40):
        dets = [] if i in (20, 21) else [cell(50.0 + 2.0 * i)]   # dropout at 20-21
        tracks = tracker.update(dets, i)
        ids += [t.track_id for t in tracks]

    assert ids, "tracker produced no tracks at all"
    assert len(set(ids)) == 1, f"identity was not stable across the dropout: {sorted(set(ids))}"

    tracker.reset()
    assert tracker.duplicates_removed == 0, "reset did not clear counters"
    tracker.update([cell(100.0, 0.9), cell(104.0, 0.6)], 0)   # 4 px apart -> duplicate
    assert tracker.duplicates_removed == 1, "duplicate head was not suppressed"

    # Two cells crossing head-on, 4 px apart at closest approach: both must
    # survive the crossing and come out the other side with the ID they went in
    # with (the failure this guards is a swap, or a respawn under a new ID).
    tracker.reset()
    seen: dict[int, list[float]] = {}
    for i in range(40):
        left, right = 40.0 + 3.0 * i, 160.0 - 3.0 * i
        for t in tracker.update([cell(left, y=98.0), cell(right, y=102.0)], i):
            seen.setdefault(t.track_id, []).append(t.detection.head[0])

    long_lived = {tid: xs for tid, xs in seen.items() if len(xs) >= 30}
    assert len(long_lived) == 2, f"crossing did not yield two stable IDs: {
        {tid: len(xs) for tid, xs in seen.items()}}"
    for tid, xs in long_lived.items():
        steps = np.diff(xs)
        assert (steps > 0).all() or (steps < 0).all(), \
            f"ID {tid} reversed direction â€” it was swapped onto the other cell"

    # A motile cell swims straight over a dead one. The dead cell has not moved
    # for 30 frames, so it must not inherit the swimmer's motion: both IDs stay
    # where they belong, and the stationary one still sits at x=100 at the end.
    tracker.reset()
    parked = 100.0
    swimmer_id = parked_id = None
    for i in range(60):
        x = 40.0 + 2.0 * i                       # crosses x=100 around frame 30
        dets = [cell(parked, y=100.0), cell(x, y=100.5)] if abs(x - parked) > 3 else [cell(x, y=100.5)]
        for t in tracker.update(dets, i):
            if abs(t.detection.head[0] - parked) < 3 and i > 40:
                parked_id = t.track_id
            elif t.detection.head[0] > 130:
                swimmer_id = t.track_id

    assert parked_id is not None, "the immotile cell lost its identity to the swimmer"
    assert swimmer_id != parked_id, "the swimmer and the dead cell ended up as one identity"

    print("tracker.py self-check passed")
