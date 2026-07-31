"""Find the frames where the detector merges two cells, ready to be annotated.

Every identity error left on 22.mp4 happens at a frame where the model emits
one detection and the truth is two cells. That is a detection problem, not a
tracking one — no assignment can give one box two identities — so the way
forward is to teach the model to split them, and the way to do that is to
retrain on exactly these frames rather than on random ones.

This writes an annotation package: the offending frames as images, the model's
current reading as editable YOLO-pose labels, and a manifest saying why each
frame was picked. Correcting ~500 of these is worth more than labelling ten
thousand frames of cells swimming alone, which the model already handles at
99.4% recall.

    python -m evaluation.mine_crossings --videos videos/input

Output under ``data/raw/crossings/``::

    images/38_000123.png    the frame
    labels/38_000123.txt    class cx cy w h  headx heady 2  neckx necky 2
    manifest.csv            frame, reason, how many cells are involved

The label format is what ultralytics expects for pose training (kpt_shape
[2, 3]), so the corrected folder can be pointed at directly by a training run.
Keypoint visibility is written as 2 (labelled and visible) for what the model
found; a cell you add by hand needs the same three numbers per keypoint.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from detection.detector import SpermDetector
from tracking.tracker import SpermTracker, TrackerConfig
from utils.config import Config

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/raw/crossings")


def _write_labels(path: Path, detections, width: int, height: int) -> None:
    """One line per cell: box then both keypoints, all normalised 0-1."""
    lines = []
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        w, h = (x2 - x1) / width, (y2 - y1) / height
        hx, hy = det.head[0] / width, det.head[1] / height
        nx, ny = det.neck[0] / width, det.neck[1] / height
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} "
                     f"{hx:.6f} {hy:.6f} 2 {nx:.6f} {ny:.6f} 2")
    path.write_text("\n".join(lines) + "\n")


def mine(video: Path, config: Config, tracker_config: TrackerConfig,
         merge_radius: float, context: int) -> list[tuple[int, str, int]]:
    """Frames worth annotating, with the reason each was chosen.

    Three symptoms, all of which mean "two cells, one detection" or its
    aftermath:

    * an identity vanishes while another identity's detection sits within a
      cell's width of where it was — it was absorbed, not lost;
    * two identities are within ``merge_radius`` of each other, which is the
      frame before or after an absorption;
    * the duplicate filter fired, meaning two detections landed on one cell —
      the same confusion in the other direction.
    """
    detector = SpermDetector(config)
    tracker = SpermTracker(tracker_config, frame_rate=49.0)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video}")

    picked: dict[int, tuple[str, int]] = {}
    previous: dict[int, tuple[float, float]] = {}
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            duplicates_before = tracker.duplicates_removed
            tracks = tracker.update(detector.detect(frame), index)
            heads = {t.track_id: t.detection.head for t in tracks}

            for tid, head in previous.items():
                if tid in heads:
                    continue
                stolen = [other for other, position in heads.items()
                          if np.hypot(position[0] - head[0], position[1] - head[1]) <= merge_radius]
                if stolen:
                    picked[index] = (f"identity {tid} absorbed by {stolen[0]}", len(stolen) + 1)

            ids = sorted(heads)
            for n, first in enumerate(ids):
                for second in ids[n + 1:]:
                    gap = np.hypot(heads[first][0] - heads[second][0],
                                   heads[first][1] - heads[second][1])
                    if gap <= merge_radius:
                        picked.setdefault(index, (f"identities {first} and {second} "
                                                  f"{gap:.0f} px apart", 2))

            if tracker.duplicates_removed > duplicates_before:
                picked.setdefault(index, ("two detections on one cell", 2))

            previous = heads
            index += 1
    finally:
        capture.release()

    # Neighbouring frames of a merge are just as valuable — the model has to
    # learn the approach and the separation, not only the worst moment.
    with_context = dict(picked)
    for frame, (reason, count) in picked.items():
        for offset in range(-context, context + 1):
            with_context.setdefault(frame + offset, (f"context for frame {frame}", count))
    return sorted((f, r, c) for f, (r, c) in with_context.items() if 0 <= f < index)


def export(video: Path, events: list[tuple[int, str, int]], config: Config) -> int:
    """Save the picked frames and the model's current reading of them."""
    images, labels = OUTPUT_DIR / "images", OUTPUT_DIR / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    detector = SpermDetector(config)
    capture = cv2.VideoCapture(str(video))
    wanted = {frame for frame, _, _ in events}
    manifest = []
    index = written = 0
    try:
        while wanted:
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted:
                stem = f"{video.stem}_{index:06d}"
                cv2.imwrite(str(images / f"{stem}.png"), frame)
                _write_labels(labels / f"{stem}.txt", detector.detect(frame),
                              frame.shape[1], frame.shape[0])
                reason = next(r for f, r, _ in events if f == index)
                manifest.append(f"{stem},{video.name},{index},{reason}")
                wanted.discard(index)
                written += 1
            index += 1
    finally:
        capture.release()

    path = OUTPUT_DIR / "manifest.csv"
    header = "" if path.exists() else "name,video,frame,reason\n"
    with path.open("a") as handle:
        handle.write(header + "\n".join(manifest) + "\n")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--videos", type=Path, default=Path("videos/input"))
    parser.add_argument("--merge-radius", type=float, default=25.0,
                        help="px between heads that counts as a crossing")
    parser.add_argument("--context", type=int, default=2,
                        help="frames either side of each event to include")
    parser.add_argument("--limit", type=int, default=150,
                        help="most frames to export per video, evenly spread")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tracker_config = TrackerConfig()
    config = Config(conf=tracker_config.track_low_thresh)

    total = 0
    for video in sorted(args.videos.glob("*.mp4")):
        events = mine(video, config, tracker_config, args.merge_radius, args.context)
        if len(events) > args.limit:
            # Evenly spread rather than the first N, so one long tangle in the
            # opening seconds cannot swallow the whole budget.
            step = len(events) / args.limit
            events = [events[int(n * step)] for n in range(args.limit)]
        written = export(video, events, config)
        total += written
        logger.info("%s: %d frames worth annotating", video.name, written)

    logger.info("%d frames -> %s (correct the labels, then retrain)", total, OUTPUT_DIR)


if __name__ == "__main__":
    main()
