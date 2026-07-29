"""CASA-style overlay rendering.

Draws head point, neck point, the head-neck segment and a small confidence
figure. Bounding boxes and class labels are intentionally never drawn — the
overlay should stay readable at 40x magnification with 50+ cells in frame.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import cv2
import numpy as np

from utils.config import DrawConfig

if TYPE_CHECKING:  # avoids importing detection/tracking just for types
    from detection.detector import Detection
    from tracking.tracker import Track
    from tracking.trajectory import Trajectory

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_detection(
    frame: np.ndarray,
    det: "Detection",
    cfg: DrawConfig | None = None,
) -> np.ndarray:
    """Draw a single sperm overlay in place and return the frame."""
    cfg = cfg or DrawConfig()
    head = tuple(int(round(v)) for v in det.head)
    neck = tuple(int(round(v)) for v in det.neck)

    cv2.line(frame, head, neck, cfg.line_color, cfg.line_thickness, cv2.LINE_AA)
    cv2.circle(frame, neck, cfg.neck_radius, cfg.neck_color, -1, cv2.LINE_AA)
    cv2.circle(frame, head, cfg.head_radius, cfg.head_color, -1, cv2.LINE_AA)

    if cfg.show_conf:
        origin = (head[0] + cfg.conf_offset[0], head[1] + cfg.conf_offset[1])
        cv2.putText(
            frame,
            f"{det.confidence:.2f}",
            origin,
            FONT,
            cfg.font_scale,
            cfg.text_color,
            1,
            cv2.LINE_AA,
        )
    return frame


def draw_detections(
    frame: np.ndarray,
    detections: Iterable["Detection"],
    cfg: DrawConfig | None = None,
) -> np.ndarray:
    """Draw every detection on a copy of the frame."""
    cfg = cfg or DrawConfig()
    canvas = frame.copy()
    for det in detections:
        draw_detection(canvas, det, cfg)
    return canvas


def draw_count(frame: np.ndarray, count: int, cfg: DrawConfig | None = None) -> np.ndarray:
    """Stamp the per-frame detection count in the top-left corner."""
    cfg = cfg or DrawConfig()
    cv2.putText(frame, f"n = {count}", (10, 22), FONT, 0.5, cfg.text_color, 1, cv2.LINE_AA)
    return frame


def track_color(track_id: int) -> tuple[int, int, int]:
    """Stable, well-separated colour per track ID.

    The stride of 47 is coprime with 180, so neighbouring IDs land far apart on
    the hue circle instead of shading into each other.
    """
    hue = (int(track_id) * 47) % 180
    bgr = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_tracks(
    frame: np.ndarray,
    tracks: Iterable["Track"],
    trajectories: dict[int, "Trajectory"],
    cfg: DrawConfig | None = None,
) -> np.ndarray:
    """Draw trails, head/neck marks and track IDs on a copy of the frame.

    Trails are drawn first so the keypoint marks stay legible on top of them.
    """
    cfg = cfg or DrawConfig()
    canvas = frame.copy()
    tracks = list(tracks)

    if cfg.trail_length > 0:
        for track in tracks:
            traj = trajectories.get(track.track_id)
            if traj is None or len(traj) < 2:
                continue
            tail = np.array(traj.head_points[-cfg.trail_length:], dtype=np.int32)
            cv2.polylines(canvas, [tail], False, track_color(track.track_id),
                          cfg.trail_thickness, cv2.LINE_AA)

    for track in tracks:
        draw_detection(canvas, track.detection, cfg)
        if cfg.show_id:
            head = tuple(int(round(v)) for v in track.detection.head)
            cv2.putText(canvas, str(track.track_id),
                        (head[0] + cfg.conf_offset[0], head[1] - cfg.conf_offset[1] + 8),
                        FONT, cfg.font_scale, track_color(track.track_id), 1, cv2.LINE_AA)
    return canvas


if __name__ == "__main__":
    # ponytail: one self-check instead of a test suite — proves the overlay
    # marks pixels, leaves the source frame untouched, and never draws a box.
    from detection.detector import Detection

    blank = np.zeros((80, 80, 3), np.uint8)
    det = Detection(bbox=(20.0, 20.0, 60.0, 60.0), head=(30.0, 40.0),
                    neck=(50.0, 40.0), confidence=0.87)

    out = draw_detections(blank, [det])
    assert blank.sum() == 0, "source frame must not be mutated"
    assert out.sum() > 0, "overlay drew nothing"
    assert out[40, 30].tolist() != [0, 0, 0], "head point missing"
    assert out[40, 50].tolist() != [0, 0, 0], "neck point missing"
    assert out[40, 40].tolist() != [0, 0, 0], "skeleton line missing"
    assert out[20, 20].tolist() == [0, 0, 0], "bbox corner must stay empty"
    print("draw.py self-check passed")
