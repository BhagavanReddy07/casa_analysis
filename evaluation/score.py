"""Score the tracker against the key: identity switches, IDF1, MOTA.

    python -m evaluation.score                 # current settings
    python -m evaluation.score --sweep         # every knob, one line each

Matching is head-to-head within ``MATCH_GATE`` pixels. Not box overlap: the
annotations are whole-cell boxes that overlap their neighbours, so an
overlap-based match flips between two cells at exactly the crossings we are
measuring — it reported 13 switches where head matching reports 2, and it hid
which of them were real.

The detector output is cached (``data/raw/detections.pkl``) because tracking
settings change far more often than the model does; a sweep of twenty settings
is seconds after the first run.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from dataclasses import replace
from pathlib import Path

import motmetrics as mm
import numpy as np

from detection.detector import SpermDetector
from evaluation import key as key_module
from evaluation.track_eval import DETECTIONS_PATH, FRAMES_DIR, load_boxes
from tracking.tracker import SpermTracker, TrackerConfig
from utils.config import Config

logger = logging.getLogger(__name__)

MATCH_GATE = 15.0     # px between a reported head and the true head
FRAME_RATE = 49.0
KEY_CACHE = Path("data/raw/key.pkl")


def load_key(frames: list[Path]) -> key_module.Key:
    if KEY_CACHE.exists():
        return pickle.loads(KEY_CACHE.read_bytes())
    built = key_module.build(frames, load_boxes(640, 480))
    KEY_CACHE.write_bytes(pickle.dumps(built))
    return built


def load_detections(frames: list[Path]) -> list[list]:
    if DETECTIONS_PATH.exists():
        return pickle.loads(DETECTIONS_PATH.read_bytes())
    import cv2
    detector = SpermDetector(Config(conf=TrackerConfig().track_low_thresh))
    per_frame = [detector.detect(cv2.imread(str(path))) for path in frames]
    DETECTIONS_PATH.write_bytes(pickle.dumps(per_frame))
    return per_frame


def evaluate(config: TrackerConfig, key: key_module.Key, detections: list[list]):
    """Run the tracker over cached detections and accumulate MOT metrics."""
    tracker = SpermTracker(config, frame_rate=FRAME_RATE)
    accumulator = mm.MOTAccumulator(auto_id=True)

    for index, frame_detections in enumerate(detections):
        tracks = tracker.update(list(frame_detections), index)
        true_ids, true_heads = key.ids.get(index, []), key.heads.get(index, [])

        distances = np.full((len(true_heads), len(tracks)), np.nan)
        for row, point in enumerate(true_heads):
            for column, track in enumerate(tracks):
                gap = np.hypot(track.detection.head[0] - point[0],
                               track.detection.head[1] - point[1])
                if gap <= MATCH_GATE:
                    distances[row, column] = gap / MATCH_GATE
        accumulator.update(true_ids, [t.track_id for t in tracks], distances)

    summary = mm.metrics.create().compute(
        accumulator, name="tracker",
        metrics=["idf1", "num_switches", "mota", "num_fragmentations",
                 "num_misses", "num_false_positives"])
    return summary.iloc[0], accumulator


def report(label: str, row, accumulator=None, show_switches: bool = False) -> None:
    print(f"{label:32s} switches={int(row.num_switches):3d}  IDF1={row.idf1:.4f}  "
          f"MOTA={row.mota:.4f}  frag={int(row.num_fragmentations):3d}  "
          f"missed={int(row.num_misses):4d}  false={int(row.num_false_positives):4d}",
          flush=True)
    if show_switches and accumulator is not None:
        events = accumulator.mot_events
        for (frame, _), event in events[events.Type == "SWITCH"].iterrows():
            print(f"      frame {frame:3d}: cell {int(event.OId)} reported as {int(event.HId)}")


# Everything worth turning, and what it is worth turning to. Kept here rather
# than in a shell script so a regression shows up as a changed line of output.
SWEEP = {
    "match_thresh": (0.80, 0.90, 0.95, 0.99),
    "track_buffer": (10, 30, 60, 120),
    "motion_weight": (0.0, 0.2, 0.35, 0.5),
    "motion_gate": (10.0, 15.0, 25.0),
    "claim_distance": (0.0, 8.0, 12.0, 20.0),
    "stationary_penalty": (0.0, 0.5, 1.0),
    "orientation_penalty": (0.0, 0.3, 0.6, 1.0),
    "orientation_gate": (20.0, 45.0, 90.0),
    "dedupe_distance": (0.0, 5.0, 10.0, 15.0),
    "new_track_thresh": (0.15, 0.25, 0.4),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep", action="store_true", help="try every knob one at a time")
    parser.add_argument("--switches", action="store_true", help="list where identity moved")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    frames = sorted(FRAMES_DIR.glob("*.png"))
    key = load_key(frames)
    detections = load_detections(frames)
    logger.info("key: %d identities, %d teleports | %d frames",
                key.identities, key.teleports(), len(frames))

    base = TrackerConfig()
    row, accumulator = evaluate(base, key, detections)
    report("current settings", row, accumulator, show_switches=True)

    if not args.sweep:
        return

    print()
    for field, values in SWEEP.items():
        for value in values:
            candidate = replace(base, **{field: value})
            row, _ = evaluate(candidate, key, detections)
            marker = " <- current" if getattr(base, field) == value else ""
            report(f"{field}={value}{marker}", row)
        print()


if __name__ == "__main__":
    main()
