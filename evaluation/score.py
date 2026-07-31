"""Score the tracker against the key: identity switches, IDF1, MOTA.

    python -m evaluation.score                 # current settings
    python -m evaluation.score --sweep         # every knob, one line each

Matching is head-to-head within ``MATCH_GATE`` pixels. Not box overlap: the
annotations are whole-cell boxes that overlap their neighbours, so an
overlap-based match flips between two cells at exactly the crossings we are
measuring — it reported 13 switches where head matching reports 2, and it hid
which of them were real.

    python -m evaluation.score --dataset sperm2 # 30.mp4 instead of 22.mp4

The detector output is cached per clip because tracking settings change far
more often than the model does; a sweep of twenty settings is seconds after the
first run.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import motmetrics as mm
import numpy as np

from detection.detector import SpermDetector
from evaluation import key as key_module
from tracking.tracker import SpermTracker, TrackerConfig
from utils.config import Config

logger = logging.getLogger(__name__)

MATCH_GATE = 15.0     # px between a reported head and the true head
FRAME_RATE = 49.0
CACHE_DIR = Path("data/raw")


@dataclass(frozen=True)
class Dataset:
    """One annotated clip: where its frames, labels and caches live.

    The CVAT exports land as ``<name>/<name>_frames`` and
    ``<name>/<name>/obj_train_data``, so the layout is derived rather than
    configured — a new clip only needs its folder dropped in.
    """

    name: str

    @property
    def frames_dir(self) -> Path:
        return Path(self.name) / f"{self.name}_frames"

    @property
    def labels_dir(self) -> Path:
        return Path(self.name) / self.name / "obj_train_data"

    @property
    def frames(self) -> list[Path]:
        return sorted(self.frames_dir.glob("*.png"))

    def cache(self, kind: str) -> Path:
        return CACHE_DIR / f"{self.name}_{kind}.pkl"


def load_key(dataset: Dataset) -> key_module.Key:
    cached = dataset.cache("key")
    if cached.exists():
        return pickle.loads(cached.read_bytes())
    frames = dataset.frames
    height, width = cv2.imread(str(frames[0])).shape[:2]
    built = key_module.build(frames, key_module.load_boxes(dataset.labels_dir, width, height))
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(pickle.dumps(built))
    return built


def load_detections(dataset: Dataset) -> list[list]:
    """Detector output per frame, cached — it dwarfs everything else here."""
    cached = dataset.cache("detections")
    if cached.exists():
        return pickle.loads(cached.read_bytes())
    detector = SpermDetector(Config(conf=TrackerConfig().track_low_thresh))
    per_frame = [detector.detect(cv2.imread(str(path))) for path in dataset.frames]
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(pickle.dumps(per_frame))
    logger.info("cached detections for %d frames -> %s", len(per_frame), cached)
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
    "dedupe_distance": (0.0, 5.0, 10.0, 15.0),
    "new_track_thresh": (0.15, 0.25, 0.4),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="sperm1",
                        help="annotated clip folder: sperm1 is 22.mp4, sperm2 is 30.mp4")
    parser.add_argument("--sweep", action="store_true", help="try every knob one at a time")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    dataset = Dataset(args.dataset)
    key = load_key(dataset)
    detections = load_detections(dataset)
    logger.info("%s: %d identities, %d teleports | %d frames",
                dataset.name, key.identities, key.teleports(), len(dataset.frames))

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
