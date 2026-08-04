"""Configuration objects for the CASA pipeline.

Plain dataclasses with defaults. No file loading — the CLI in ``main.py``
overrides what it needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Keypoint indices as trained in Phase 1 (kpt_shape = [2, 3]).
HEAD = 0
NECK = 1

BGR = tuple[int, int, int]

# --- Spatial calibration ----------------------------------------------------
# Rig: Olympus CX31, 40x objective (the quoted "400x" is 40x objective x 10x
# eyepiece — eyepiece magnification does not reach the camera).
# Camera: IDS uEye UI-2210C, 1/2" CCD, 640x480, 9.9 um pixel pitch.
#   640 px * 9.9 um = 6.34 mm = the 1/2" sensor width, so 640x480 is native.
#   The videos were not downscaled and the calibration applies as-is.
#
#   um/px = pixel_pitch / (objective * c-mount adapter)
#     0.5x adapter -> 9.9 / 20 = 0.495   <-- assumed
#     1.0x adapter -> 9.9 / 40 = 0.2475
#
# The adapter is not documented, so it is inferred from cell size: the mean
# head-neck vector is 10.3 px, which is 5.1 um at 0.495 (consistent with a
# 4-5 um sperm head) but only 2.6 um at 0.2475 — smaller than any real head.
#
# ponytail: one global constant. Replace with a stage-micrometer measurement
# before any clinical or published use, and make it per-video if clips ever
# come from different rigs.
MICRONS_PER_PIXEL = 0.495


@dataclass
class DrawConfig:
    """Overlay style. Deliberately minimal — no boxes, no class labels."""

    head_color: BGR = (0, 240, 255)      # amber
    neck_color: BGR = (255, 200, 0)      # cyan
    line_color: BGR = (0, 255, 120)      # green
    text_color: BGR = (235, 235, 235)

    head_radius: int = 3
    neck_radius: int = 2
    line_thickness: int = 1

    # Off by default: a clinical view shows geometry, not numbers. Turn on
    # with --show-conf for debugging.
    show_conf: bool = False
    font_scale: float = 0.3
    conf_offset: tuple[int, int] = (5, -5)

    # Tracking overlay
    show_id: bool = True
    # 0 = no trail. Dense fields (10+ cells) get unreadable with trails on;
    # opt in with --trail N when you want to inspect one cell's path.
    trail_length: int = 0
    trail_thickness: int = 1


@dataclass
class Config:
    """Everything the detection stage needs."""

    # The deployed Streamlit app now uses the retrained v2 checkpoint by
    # default for new uploads. The legacy baseline remains available as
    # models/best.pt for reference and comparison.
    weights: Path = Path("models/best_v2.pt")
    conf: float = 0.25
    iou: float = 0.5
    # YOLO-pose emits (0, 0) for a keypoint it cannot localize. Those land in
    # the trajectory as a jump to the frame corner and fabricate enormous
    # distances. The split is clean: on 30.mp4 head-keypoint confidence has
    # mean 0.976 and 5th centile 0.992, while the bad ones sit near 0.03 —
    # 2.4% of detections, and exactly those are the (0, 0) ones. 0.5 cuts
    # through the empty middle of that gap.
    min_keypoint_conf: float = 0.5
    imgsz: int = 640
    device: str | None = None            # None -> ultralytics picks CUDA if present
    output_dir: Path = Path("videos/output")
    draw: DrawConfig = field(default_factory=DrawConfig)
