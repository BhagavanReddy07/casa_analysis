"""Ground truth for identity, built from hand-drawn boxes without a tracker.

The annotations in ``sperm1/`` are one whole-cell box per sperm per frame, with
no identities and no keypoints. Two problems follow, and this module solves
both without asking for a single new annotation:

**Identities.** An earlier version copied them from our own tracker, which
makes the key agree with whatever the tracker did — and worse, the copy was
made by asking which head sits inside each box. Boxes here are 58x53 px and
overlap their neighbours, so that mapping flipped between two overlapping boxes
while the tracker itself was provably stable, inventing swaps in the key. Now
identities come from linking the annotations to each other over time, with no
tracker involved.

**Precision.** A whole-cell box cannot say which cell a detection belongs to
when two boxes overlap, which is exactly the situation every identity error
occurs in. But the head is the brightest point of a sperm under phase contrast,
so the box only has to *locate* a head that the pixels then pin down precisely.
Linking those points instead of the boxes gives a key with zero teleports on
22.mp4, against 2 when linking boxes.

The result is a key that scores identity to the pixel, from box annotations
that never mentioned heads.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

# A head cannot move further than this between frames (measured maximum on
# these clips is 11 px), and an identity survives this many frames of the
# annotator drawing nothing — they skip cells whose head is not visible.
LINK_GATE = 14.0
LINK_MEMORY = 10


@dataclass
class Key:
    """Per frame: the head point of every annotated cell, and its identity."""

    heads: dict[int, list[tuple[float, float]]]
    ids: dict[int, list[int]]

    @property
    def identities(self) -> int:
        return len({i for row in self.ids.values() for i in row})

    def teleports(self, limit: float = 15.0) -> int:
        """Consecutive-frame jumps that no real cell could make — key defects."""
        paths: dict[int, dict[int, tuple[float, float]]] = {}
        for frame in self.heads:
            for identity, point in zip(self.ids[frame], self.heads[frame]):
                paths.setdefault(identity, {})[frame] = point
        return sum(
            1 for path in paths.values()
            for a, b in zip(sorted(path), sorted(path)[1:])
            if b - a == 1 and np.hypot(path[b][0] - path[a][0], path[b][1] - path[a][1]) > limit
        )


def load_shapes(root: Path, width: int, height: int) -> dict[int, list[np.ndarray]]:
    """Annotated cells per frame as 4-corner polygons, from whichever export exists.

    CVAT's YOLO 1.1 export **silently drops rotated boxes** — there is no angle
    field in the format. On 38.mp4 that lost 1,204 of 3,618 shapes, a third of
    the annotation, and made the clip look under-labelled by 5 cells a frame.
    So ``annotations.xml`` (CVAT for images 1.1) is preferred wherever it
    exists, and the YOLO text files are the fallback for the sets exported
    before this was noticed.

    Polygons rather than boxes because a rotated box around a diagonal sperm is
    half empty space once axis-aligned, and the head is found by brightness —
    the tighter the shape, the less chance of grabbing a neighbour.
    """
    xml = next(iter(root.rglob("annotations.xml")), None)
    if xml is not None:
        return _shapes_from_cvat(xml)
    labels = max({p.parent for p in root.rglob("*.txt")},
                 key=lambda p: len(list(p.glob("*.txt"))))
    return {frame: [_corners(*box) for box in boxes]
            for frame, boxes in load_boxes(labels, width, height).items()}


def _corners(x1: float, y1: float, x2: float, y2: float, rotation: float = 0.0) -> np.ndarray:
    """The four corners of a box, rotated clockwise about its centre."""
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    if not rotation:
        return points
    angle = np.radians(rotation)
    rotate = np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]], dtype=np.float32)
    return (points - (cx, cy)) @ rotate.T + (cx, cy)


def _shapes_from_cvat(path: Path) -> dict[int, list[np.ndarray]]:
    import xml.etree.ElementTree as ElementTree

    shapes: dict[int, list[np.ndarray]] = {}
    for image in ElementTree.parse(path).getroot().findall("image"):
        frame = int(image.get("id"))
        shapes[frame] = [
            _corners(float(box.get("xtl")), float(box.get("ytl")),
                     float(box.get("xbr")), float(box.get("ybr")),
                     float(box.get("rotation", 0.0)))
            for box in image.findall("box")
        ]
    return shapes


def load_boxes(labels_dir: Path, width: int, height: int
               ) -> dict[int, list[tuple[float, float, float, float]]]:
    """YOLO-normalised label files -> pixel x1,y1,x2,y2 per frame index.

    Frame index comes from the trailing number in the filename, so both
    ``frame_000012.txt`` and ``s2_frame_000012.txt`` land on frame 12.
    """
    boxes: dict[int, list[tuple[float, float, float, float]]] = {}
    for path in sorted(labels_dir.glob("*.txt")):
        index = int(path.stem.split("_")[-1])
        rows = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            _, cx, cy, w, h = (float(value) for value in line.split()[:5])
            cx, cy, w, h = cx * width, cy * height, w * width, h * height
            rows.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        boxes[index] = rows
    return boxes


def head_in_shape(gray: np.ndarray, polygon: np.ndarray) -> tuple[float, float] | None:
    """Brightest point inside an annotated cell — its head, under phase contrast.

    Masked to the polygon, so a rotated box around a diagonal sperm cannot pick
    up a neighbour sitting in the corner of its axis-aligned extent.
    """
    x0, y0 = int(max(0, polygon[:, 0].min())), int(max(0, polygon[:, 1].min()))
    x1 = int(min(gray.shape[1], np.ceil(polygon[:, 0].max())))
    y1 = int(min(gray.shape[0], np.ceil(polygon[:, 1].max())))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None

    patch = cv2.GaussianBlur(gray[y0:y1, x0:x1].astype(np.float32), (3, 3), 0)
    mask = np.zeros(patch.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(polygon - (x0, y0)).astype(np.int32)], 1)
    if not mask.any():
        return None
    patch = np.where(mask.astype(bool), patch, -np.inf)
    y, x = np.unravel_index(int(np.argmax(patch)), patch.shape)
    return (x0 + float(x), y0 + float(y))


def build(frames: list[Path], shapes: dict[int, list[np.ndarray]]) -> Key:
    """Extract a head per annotated cell, then link the heads into identities."""
    heads: dict[int, list[tuple[float, float]]] = {}
    for index in sorted(shapes):
        if index >= len(frames):
            continue
        gray = cv2.cvtColor(cv2.imread(str(frames[index])), cv2.COLOR_BGR2GRAY)
        heads[index] = [point for point in
                        (head_in_shape(gray, shape) for shape in shapes[index])
                        if point is not None]

    ids: dict[int, list[int]] = {}
    live: dict[int, tuple[int, tuple[float, float]]] = {}
    next_id = 1
    for frame in sorted(heads):
        points = heads[frame]
        alive = [(identity, point) for identity, (seen, point) in live.items()
                 if frame - seen <= LINK_MEMORY]
        row = [0] * len(points)

        if alive and points:
            cost = np.array([[np.hypot(a[0] - b[0], a[1] - b[1]) for b in points]
                             for _, a in alive])
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                identity = alive[r][0]
                # A cell unseen for k frames may legitimately have moved k times
                # as far, so the gate opens with the gap it has to bridge.
                elapsed = max(1, frame - live[identity][0])
                if cost[r, c] <= LINK_GATE * elapsed:
                    row[c] = identity
                    live[identity] = (frame, points[c])

        for index, identity in enumerate(row):
            if not identity:
                row[index] = next_id
                live[next_id] = (frame, points[index])
                next_id += 1
        ids[frame] = row

    key = _drop_unstable(Key(heads=heads, ids=ids))
    logger.info("key: %d identities from %d annotated cells, %d teleports",
                key.identities, sum(len(v) for v in key.heads.values()), key.teleports())
    return key


def _drop_unstable(key: Key, jump: float = 8.0, repeats: int = 3,
                   travel: float = 25.0) -> Key:
    """Remove identities whose head hops about while the cell stays put.

    Agglutinated cells are boxed as one sperm — correctly — but hold two or
    three heads, and "the brightest point inside the box" then alternates
    between them. That reads as an identity switch which never happened: one
    such clump on 38.mp4 produced 7 of 10 reported switches.

    The signature is specific: several large jumps, yet no net travel. A real
    fast cell jumps once or twice and ends up somewhere else; a clump jumps
    repeatedly and ends where it started. Nothing is guessed at — these cells
    are still tracked, they are just not used as evidence.
    """
    paths: dict[int, dict[int, tuple[float, float]]] = {}
    for frame in key.heads:
        for identity, point in zip(key.ids[frame], key.heads[frame]):
            paths.setdefault(identity, {})[frame] = point

    unstable = set()
    for identity, path in paths.items():
        frames = sorted(path)
        hops = sum(1 for a, b in zip(frames, frames[1:])
                   if b - a == 1 and np.hypot(path[b][0] - path[a][0],
                                              path[b][1] - path[a][1]) > jump)
        net = np.hypot(path[frames[-1]][0] - path[frames[0]][0],
                       path[frames[-1]][1] - path[frames[0]][1])
        if hops >= repeats and net < travel:
            unstable.add(identity)

    if not unstable:
        return key
    logger.info("excluded %d identity(ies) whose head oscillates in place "
                "(agglutinated cells boxed as one)", len(unstable))
    heads, ids = {}, {}
    for frame in key.heads:
        keep = [n for n, identity in enumerate(key.ids[frame]) if identity not in unstable]
        heads[frame] = [key.heads[frame][n] for n in keep]
        ids[frame] = [key.ids[frame][n] for n in keep]
    return Key(heads=heads, ids=ids)
