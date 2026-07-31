"""Recover two cells from one detection, using the pixels the model discarded.

Every identity error left on 22.mp4 happens where the model emits one detection
and the truth is two cells touching. The tracker cannot fix that — one box
cannot carry two identities — but the image still contains both cells: under
phase contrast a sperm head is a bright blob, and two touching heads are two
bright blobs inside one box.

So when the tracker finds two identities competing for a single detection, this
looks at the frame and asks whether there are really two heads there. If it
finds two peaks far enough apart, both identities get their own measurement and
neither has to be guessed.

Deliberately narrow: it runs only where identities are contested, never on
every detection, because splitting a genuinely single cell would invent a
sperm. The tracker's contention is the evidence that a split is plausible; the
pixels are the evidence that it is real.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from detection.detector import Detection

logger = logging.getLogger(__name__)

WINDOW = 34          # px around the head to search, ~2 head widths
BLUR = 3             # smoothing before peak finding, kills sensor speckle
MIN_SEPARATION = 7.0  # px between two heads to call them separate cells
MIN_PROMINENCE = 0.12  # peak must stand this far above the window's range


def _peaks(patch: np.ndarray) -> list[tuple[float, float, float]]:
    """Local maxima as (x, y, brightness), brightest first.

    A 3x3 dilation marks every pixel that is the maximum of its neighbourhood;
    comparing against it is the standard cheap peak finder and needs no scipy.
    """
    smooth = cv2.GaussianBlur(patch.astype(np.float32), (BLUR, BLUR), 0)
    local_max = cv2.dilate(smooth, np.ones((5, 5), np.uint8))
    lo, hi = float(smooth.min()), float(smooth.max())
    if hi - lo < 1e-6:
        return []
    floor = lo + MIN_PROMINENCE * (hi - lo)

    ys, xs = np.where((smooth >= local_max - 1e-6) & (smooth > floor))
    found = sorted(((float(x), float(y), float(smooth[y, x])) for x, y in zip(xs, ys)),
                   key=lambda p: -p[2])

    kept: list[tuple[float, float, float]] = []
    for x, y, value in found:
        if all(np.hypot(x - kx, y - ky) >= MIN_SEPARATION for kx, ky, _ in kept):
            kept.append((x, y, value))
    return kept


def split(frame: np.ndarray, detection: Detection,
          wanted: int = 2) -> list[Detection] | None:
    """Two detections if the pixels show two heads, otherwise None.

    The replacements keep the original box and confidence — only the head moves
    to its own peak. The neck is carried over as an offset from the head, which
    holds because the pair are near-parallel at the moment they touch; the
    tracker's orientation term then has something to work with even though the
    axis is approximate.
    """
    height, width = frame.shape[:2]
    hx, hy = detection.head
    half = WINDOW // 2
    x0, y0 = int(max(0, hx - half)), int(max(0, hy - half))
    x1, y1 = int(min(width, hx + half)), int(min(height, hy + half))
    if x1 - x0 < MIN_SEPARATION * 2 or y1 - y0 < MIN_SEPARATION * 2:
        return None

    patch = frame[y0:y1, x0:x1]
    if patch.ndim == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    peaks = _peaks(patch)[:wanted]
    if len(peaks) < wanted:
        return None

    offset = (detection.neck[0] - detection.head[0], detection.neck[1] - detection.head[1])
    x_min, y_min, x_max, y_max = detection.bbox
    box_half = ((x_max - x_min) / 2, (y_max - y_min) / 2)

    replacements = []
    for x, y, _ in peaks:
        head = (x0 + x, y0 + y)
        replacements.append(Detection(
            bbox=(head[0] - box_half[0], head[1] - box_half[1],
                  head[0] + box_half[0], head[1] + box_half[1]),
            head=head,
            neck=(head[0] + offset[0], head[1] + offset[1]),
            confidence=detection.confidence,
        ))
    return replacements
