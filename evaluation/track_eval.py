"""Ground truth for identity: build it, then score the tracker against it.

The 501 hand-annotated frames in ``sperm1/`` are frames 0-500 of 22.mp4. They
carry boxes but no identities, and the boxes are whole-cell (median 58x53 px)
while the model predicts head-only boxes (~16 px), so nothing here matches on
IoU — a head box inside a cell box scores about 0.08. Everything below matches
on **is the predicted head inside the annotated box**, which is the right
question for this data and needs no re-annotation.

Two modes::

    python -m evaluation.track_eval prefill    # writes a reviewable gt.txt
    python -m evaluation.track_eval score      # IDF1 / ID switches / MOTA

``prefill`` runs the tracker and copies its identity onto each annotated box,
so the review job is correcting numbers rather than inventing them. It flags
every box it was unsure about; those flagged frames are the ones worth a human
eye, and they are exactly the crossings we are trying to fix.

Format is MOT 1.1 (``frame,id,x,y,w,h,conf,cls,vis``) because CVAT imports it
for review and motmetrics reads it for scoring — one file, both jobs.
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

FRAMES_DIR = Path("sperm1/sperm1_frames")
LABELS_DIR = Path("sperm1/sperm1/obj_train_data")
GT_PATH = Path("data/raw/gt.txt")
FLAGS_PATH = Path("data/raw/gt_flags.csv")
CROPS_PATH = Path("data/raw/unlabelled_crops.png")


def load_boxes(width: int, height: int) -> dict[int, list[tuple[float, float, float, float]]]:
    """YOLO-normalised label files -> pixel x1,y1,x2,y2 per frame index."""
    boxes: dict[int, list[tuple[float, float, float, float]]] = {}
    for path in sorted(LABELS_DIR.glob("*.txt")):
        index = int(path.stem.split("_")[-1])
        rows = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            _, cx, cy, w, h = (float(v) for v in line.split())
            cx, cy, w, h = cx * width, cy * height, w * width, h * height
            rows.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        boxes[index] = rows
    return boxes


def _inside(box, point) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def _centre(box) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def prefill(source: Path, config: Config, tracker_config: TrackerConfig) -> None:
    """Copy tracker identities onto the annotated boxes and flag the doubtful."""
    frames = sorted(FRAMES_DIR.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"no annotated frames under {FRAMES_DIR}")
    height, width = cv2.imread(str(frames[0])).shape[:2]
    boxes = load_boxes(width, height)

    detector = SpermDetector(config)
    tracker = SpermTracker(tracker_config, frame_rate=49.0)

    lines: list[str] = []
    flags: list[str] = ["frame,box_index,reason"]
    previous: dict[int, tuple[float, float, float, float]] = {}   # id -> box
    unmatched_heads: list[tuple[int, tuple[float, float]]] = []

    for index, path in enumerate(frames):
        frame = cv2.imread(str(path))
        tracks = tracker.update(detector.detect(frame), index)
        claimed: set[int] = set()   # track ids used this frame

        for box_index, box in enumerate(boxes.get(index, [])):
            candidates = [t for t in tracks if _inside(box, t.detection.head)]
            if not candidates:
                flags.append(f"{index},{box_index},no detection inside the box")
                continue
            if len(candidates) > 1:
                flags.append(f"{index},{box_index},{len(candidates)} detections inside the box")
            cx, cy = _centre(box)
            candidates.sort(key=lambda t: np.hypot(t.detection.head[0] - cx,
                                                   t.detection.head[1] - cy))
            # One identity cannot be two cells in the same frame. If the nearest
            # is already spoken for, the annotator saw two cells where the model
            # saw one — leave this box unnumbered for review rather than
            # duplicating an identity, which would make the score meaningless.
            track = next((t for t in candidates if t.track_id not in claimed), None)
            if track is None:
                flags.append(f"{index},{box_index},its detection is already claimed "
                             f"by another box (two cells, one detection)")
                continue
            claimed.add(track.track_id)
            # An identity that lands on a box far from where the same identity
            # sat last frame is the swap we are hunting — worth a human eye.
            was = previous.get(track.track_id)
            if was is not None:
                moved = np.hypot(*(np.array(_centre(box)) - np.array(_centre(was))))
                if moved > 0.5 * (box[2] - box[0]):
                    flags.append(f"{index},{box_index},identity {track.track_id} jumped {moved:.0f} px")
            previous[track.track_id] = box

            x, y = box[0], box[1]
            lines.append(f"{index + 1},{track.track_id},{x:.1f},{y:.1f},"
                         f"{box[2] - x:.1f},{box[3] - y:.1f},1,1,1")

        for track in tracks:
            if track.track_id not in claimed:
                unmatched_heads.append((index, track.detection.head))

    GT_PATH.parent.mkdir(parents=True, exist_ok=True)
    GT_PATH.write_text("\n".join(lines) + "\n")
    FLAGS_PATH.write_text("\n".join(flags) + "\n")
    _crop_sheet(frames, unmatched_heads)

    logger.info("%d boxes given an identity -> %s", len(lines), GT_PATH)
    logger.info("%d flagged for review -> %s", len(flags) - 1, FLAGS_PATH)
    logger.info("%d detections outside every annotated box -> %s (are these cells?)",
                len(unmatched_heads), CROPS_PATH)


def _crop_sheet(frames: list[Path], heads: list[tuple[int, tuple[float, float]]],
                size: int = 64, columns: int = 20, limit: int = 200) -> None:
    """Contact sheet of what the model found but the annotator did not label."""
    if not heads:
        return
    step = max(1, len(heads) // limit)
    sample = heads[::step][:limit]
    rows = (len(sample) + columns - 1) // columns
    sheet = np.zeros((rows * size, columns * size, 3), dtype=np.uint8)
    cache: dict[int, np.ndarray] = {}
    for n, (index, (hx, hy)) in enumerate(sample):
        image = cache.setdefault(index, cv2.imread(str(frames[index])))
        x, y = int(hx) - size // 2, int(hy) - size // 2
        x, y = max(0, min(x, image.shape[1] - size)), max(0, min(y, image.shape[0] - size))
        r, c = divmod(n, columns)
        sheet[r * size:(r + 1) * size, c * size:(c + 1) * size] = image[y:y + size, x:x + size]
    cv2.imwrite(str(CROPS_PATH), sheet)


def score(config: Config, tracker_config: TrackerConfig) -> None:
    """Compare the tracker against the reviewed gt.txt: IDF1, ID switches, MOTA."""
    import motmetrics as mm

    if not GT_PATH.exists():
        raise FileNotFoundError(f"{GT_PATH} not found — run prefill and review it first")

    truth: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = {}
    for line in GT_PATH.read_text().splitlines():
        frame, tid, x, y, w, h, *_ = line.split(",")
        truth.setdefault(int(frame) - 1, []).append(
            (int(tid), (float(x), float(y), float(x) + float(w), float(y) + float(h))))

    frames = sorted(FRAMES_DIR.glob("*.png"))
    detector = SpermDetector(config)
    tracker = SpermTracker(tracker_config, frame_rate=49.0)
    accumulator = mm.MOTAccumulator(auto_id=True)

    for index, path in enumerate(frames):
        tracks = tracker.update(detector.detect(cv2.imread(str(path))), index)
        gt_ids = [tid for tid, _ in truth.get(index, [])]
        gt_boxes = [box for _, box in truth.get(index, [])]
        hypothesis_ids = [t.track_id for t in tracks]

        # Distance in [0, 1] when the head sits inside the box, else no match.
        distances = np.full((len(gt_boxes), len(tracks)), np.nan)
        for i, box in enumerate(gt_boxes):
            cx, cy = _centre(box)
            radius = max(box[2] - box[0], box[3] - box[1]) / 2
            for j, track in enumerate(tracks):
                head = track.detection.head
                if _inside(box, head):
                    distances[i, j] = min(1.0, np.hypot(head[0] - cx, head[1] - cy) / radius)
        accumulator.update(gt_ids, hypothesis_ids, distances)

    metrics = mm.metrics.create()
    summary = metrics.compute(accumulator, metrics=[
        "idf1", "idp", "idr", "num_switches", "mota", "motp",
        "num_false_positives", "num_misses", "num_fragmentations",
    ], name="tracker")
    print(mm.io.render_summary(summary, namemap=mm.io.motchallenge_metric_names))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["prefill", "score"])
    parser.add_argument("--source", type=Path, default=Path("videos/input/22.mp4"))
    parser.add_argument("--weights", type=Path, default=Path("models/best.pt"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tracker_config = TrackerConfig()
    config = Config(weights=args.weights, conf=tracker_config.track_low_thresh)

    if args.mode == "prefill":
        prefill(args.source, config, tracker_config)
    else:
        score(config, tracker_config)


if __name__ == "__main__":
    main()
