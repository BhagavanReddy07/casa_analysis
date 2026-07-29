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
    overlap — despite the name, higher is more lenient. The ultralytics
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


# Keys BYTETracker reads off its args namespace; the rest are ours.
_BYTETRACK_KEYS = (
    "track_high_thresh", "track_low_thresh", "new_track_thresh",
    "track_buffer", "match_thresh", "fuse_score",
)


class _DetectionBoxes:
    """Duck-types the ultralytics ``Boxes`` fields that BYTETracker reads.

    Must not expose ``xywhr`` — BYTETracker treats its presence as the oriented
    bounding box case.
    """

    def __init__(self, detections: list[Detection]) -> None:
        boxes = np.array([d.bbox for d in detections], dtype=np.float32).reshape(-1, 4)
        x1, y1, x2, y2 = boxes.T
        self.xywh = np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)
        self.conf = np.array([d.confidence for d in detections], dtype=np.float32)
        self.cls = np.zeros(len(detections), dtype=np.float32)


class SpermTracker:
    """Assigns stable IDs to per-frame detections."""

    def __init__(self, config: TrackerConfig | None = None, frame_rate: float = 30.0) -> None:
        self.config = config or TrackerConfig()
        self.frame_rate = frame_rate
        self._tracker = self._new_tracker()
        self.duplicates_removed = 0

    def _new_tracker(self) -> BYTETracker:
        args = SimpleNamespace(**{k: v for k, v in asdict(self.config).items()
                                  if k in _BYTETRACK_KEYS})
        return BYTETracker(args, frame_rate=int(round(self.frame_rate)))

    def reset(self) -> None:
        """Drop all state — call between videos."""
        self._tracker = self._new_tracker()
        self.duplicates_removed = 0

    def _dedupe(self, detections: list[Detection]) -> list[Detection]:
        """Drop the weaker of any two detections whose heads nearly coincide.

        ponytail: O(n^2) over ~15 cells per frame. Switch to a KD-tree only if
        cell counts reach the hundreds.
        """
        if len(detections) < 2:
            return detections

        ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept: list[Detection] = []
        for det in ordered:
            if all(np.hypot(det.head[0] - k.head[0], det.head[1] - k.head[1])
                   >= self.config.dedupe_distance for k in kept):
                kept.append(det)
        self.duplicates_removed += len(detections) - len(kept)
        return kept

    def update(self, detections: list[Detection], frame_index: int) -> list[Track]:
        """Associate this frame's detections with existing tracks."""
        detections = self._dedupe(detections)

        # Column layout of BYTETracker.update: x1,y1,x2,y2,track_id,score,cls,idx
        rows = self._tracker.update(_DetectionBoxes(detections))
        return [
            Track(track_id=int(row[4]), frame_index=frame_index,
                  detection=detections[int(row[7])])
            for row in rows
        ]


if __name__ == "__main__":
    # ponytail: one self-check instead of a suite — a cell drifting 2 px/frame
    # (our measured speed) must keep a single ID, survive a two-frame dropout,
    # and duplicate heads must collapse to one identity.
    cfg = TrackerConfig()
    tracker = SpermTracker(cfg, frame_rate=49.0)

    def cell(x: float, conf: float = 0.9) -> Detection:
        return Detection(bbox=(x - 8, 92.0, x + 8, 108.0), head=(x, 100.0),
                         neck=(x - 10, 100.0), confidence=conf)

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

    print("tracker.py self-check passed")
