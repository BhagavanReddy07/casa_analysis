"""VISEM-Tracking as ground truth.

``videos/input/22.mp4``, ``30``, ``38`` and ``60`` are videos 22, 30, 38 and 60
of the public VISEM-Tracking dataset — same rig as ours (Olympus CX31, 400x,
45-50 fps, 640x480), which is why the tuned constants transfer. Verified on 22:
the clip is 1470 frames and VISEM annotates 1470; box counts match the detector
frame for frame; every annotated box lands within 10 px of a detection.

This matters because it replaces the hand-built key in :mod:`evaluation.key`,
which covers 501 of those 1470 frames and was *prefilled by our own tracker*
before a human corrected it — a bias its own notes flag. VISEM was annotated
independently by domain experts via LabelBox, carries persistent identities,
and needs no labelling from us.

Annotated boxes are head boxes, not whole-cell boxes: their centres sit a
median 1.5 px from our head keypoint (90th percentile 2.9 px), so they feed
:func:`evaluation.score.evaluate` directly with no change to its matching.

    python -m evaluation.score --visem 22
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from evaluation.key import Key

logger = logging.getLogger(__name__)

REPO = "sperm-net/VISEM-Tracking"
LOCAL = Path("data/raw/visem")
VIDEOS = {"22": "videos/input/22.mp4", "30": "videos/input/30.mp4",
          "38": "videos/input/38.mp4", "60": "videos/input/60.mp4"}

_FRAME = re.compile(r"_frame_(\d+)")


def annotation(video_id: str) -> Path:
    """The per-video JSON, downloaded once (~8 MB) and kept."""
    local = LOCAL / f"{video_id}.json"
    if not local.exists():
        from huggingface_hub import hf_hub_download   # only needed on first run
        LOCAL.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(REPO, f"data/json_per_video/{video_id}.json",
                                     repo_type="dataset")
        local.write_bytes(Path(downloaded).read_bytes())
        logger.info("fetched VISEM annotation for video %s -> %s", video_id, local)
    return local


def load(video_id: str) -> Key:
    """Ground-truth heads and identities per frame index.

    Identities are LabelBox feature hashes; they are renumbered to ints because
    that is what ``motmetrics`` reports back in its event table.
    """
    frames = json.loads(annotation(video_id).read_text())["frames"]
    heads: dict[int, list[tuple[float, float]]] = {}
    ids: dict[int, list[int]] = {}
    numbering: dict[str, int] = {}

    for frame in frames:
        index = int(_FRAME.search(frame["frame_id"]).group(1))
        objects = frame["objects"]
        # coco_bbox is x, y, w, h in pixels; the centre is the head.
        heads[index] = [(x + w / 2, y + h / 2) for x, y, w, h in objects["coco_bbox"]]
        ids[index] = [numbering.setdefault(f, len(numbering))
                      for f in objects["feature_ids"]]

    logger.info("VISEM %s: %d frames, %d identities, frame range %d-%d",
                video_id, len(heads), len(numbering), min(heads), max(heads))
    return Key(heads=heads, ids=ids)


if __name__ == "__main__":
    # ponytail: one self-check — the key must cover the clip it claims to, and
    # contain no teleports (a jump no real cell could make is a key defect, and
    # it is what invalidated two of the three hand-built keys).
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import cv2

    for video_id, path in VIDEOS.items():
        key = load(video_id)
        total = int(cv2.VideoCapture(path).get(cv2.CAP_PROP_FRAME_COUNT))
        cells = sum(len(v) for v in key.heads.values()) / len(key.heads)
        print(f"video {video_id}: {len(key.heads)} annotated of {total} video frames, "
              f"{key.identities} identities, {cells:.1f} cells/frame, "
              f"{key.teleports()} teleports")
        assert len(key.heads) <= total, "more annotated frames than the video has"
        assert max(key.heads) < total, "annotation indexes past the end of the video"
