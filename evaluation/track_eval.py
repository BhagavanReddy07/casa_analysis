"""Ground truth for identity: build it, then score the tracker against it.

The 501 hand-annotated frames in ``sperm1/`` are frames 0-500 of 22.mp4. They
carry boxes but no identities, and the boxes are whole-cell (median 58x53 px)
while the model predicts head-only boxes (~16 px), so nothing here matches on
IoU — a head box inside a cell box scores about 0.08. Everything below matches
on **is the predicted head inside the annotated box**, which is the right
question for this data and needs no re-annotation.

Two modes::

    python -m evaluation.track_eval prefill    # writes a reviewable gt.txt
    python -m evaluation.track_eval review     # video of just the doubtful moments
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
REVIEW_PATH = Path("data/raw/review.mp4")
ADDED_BOX = (58.0, 53.0)          # median hand-drawn box, for machine-added cells
REVIEW_CONTEXT = 5                # frames either side of a flagged moment


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

        # Cells the model found but the annotator never boxed. Inspection of
        # data/raw/unlabelled_crops.png says these are real sperm, not debris,
        # and leaving them out would mark every tracker down for finding real
        # cells. They go in with conf 0.5 so a reviewer can tell them from the
        # hand-drawn ones, sized like the median human box so the head-inside
        # test behaves identically.
        for track in tracks:
            if track.track_id in claimed:
                continue
            hx, hy = track.detection.head
            unmatched_heads.append((index, (hx, hy)))
            lines.append(f"{index + 1},{track.track_id},{hx - ADDED_BOX[0] / 2:.1f},"
                         f"{hy - ADDED_BOX[1] / 2:.1f},{ADDED_BOX[0]:.1f},{ADDED_BOX[1]:.1f},"
                         f"0.5,1,1")

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


def review() -> None:
    """Render only the doubtful moments, so the 207 fixes can be eyeballed first.

    Green box with a number = an identity the machine is confident about. Red =
    the box it was unsure of, with the reason printed. A few frames either side
    are included so the crossing can be watched, not just glimpsed.
    """
    frames = sorted(FRAMES_DIR.glob("*.png"))
    height, width = cv2.imread(str(frames[0])).shape[:2]
    boxes = load_boxes(width, height)

    identities: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = {}
    for line in GT_PATH.read_text().splitlines():
        frame, tid, x, y, w, h, conf, *_ = line.split(",")
        identities.setdefault(int(frame) - 1, []).append(
            (int(tid), (float(x), float(y), float(x) + float(w), float(y) + float(h))))

    trouble: dict[int, list[str]] = {}
    for line in FLAGS_PATH.read_text().splitlines()[1:]:
        frame, box_index, reason = line.split(",", 2)
        trouble.setdefault(int(frame), []).append(f"box {box_index}: {reason}")

    wanted = sorted({f for frame in trouble
                     for f in range(max(0, frame - REVIEW_CONTEXT),
                                    min(len(frames), frame + REVIEW_CONTEXT + 1))})
    REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(REVIEW_PATH), cv2.VideoWriter_fourcc(*"mp4v"),
                             5.0, (width, height))
    try:
        for index in wanted:
            image = cv2.imread(str(frames[index]))
            for tid, box in identities.get(index, []):
                p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                cv2.rectangle(image, p1, p2, (0, 220, 0), 1)
                cv2.putText(image, str(tid), (p1[0], p1[1] - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 0), 1)
            for note in trouble.get(index, []):
                box_index = int(note.split()[1].rstrip(":"))
                if box_index < len(boxes.get(index, [])):
                    box = boxes[index][box_index]
                    cv2.rectangle(image, (int(box[0]), int(box[1])),
                                  (int(box[2]), int(box[3])), (0, 0, 255), 2)
            cv2.putText(image, f"frame {index}", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            for row, note in enumerate(trouble.get(index, [])[:3]):
                cv2.putText(image, note[:70], (5, height - 8 - 14 * row),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            writer.write(image)
    finally:
        writer.release()

    logger.info("%d frames covering %d doubtful moments -> %s",
                len(wanted), sum(len(v) for v in trouble.values()), REVIEW_PATH)


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
    parser.add_argument("mode", choices=["prefill", "review", "score"])
    parser.add_argument("--source", type=Path, default=Path("videos/input/22.mp4"))
    parser.add_argument("--weights", type=Path, default=Path("models/best.pt"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tracker_config = TrackerConfig()
    config = Config(weights=args.weights, conf=tracker_config.track_low_thresh)

    if args.mode == "prefill":
        prefill(args.source, config, tracker_config)
    elif args.mode == "review":
        review()
    else:
        score(config, tracker_config)


if __name__ == "__main__":
    main()
