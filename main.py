"""Sperm CASA — detection, tracking and CASA kinematics.

    python main.py --source videos/input/22.mp4                    # detection only
    python main.py --source videos/input/22.mp4 --track            # + persistent IDs
    python main.py --source videos/input/22.mp4 --track --metrics  # + VCL/VSL/... CSV
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from casa.motility import MotilityThresholds
from detection.detector import SpermDetector
from detection.inference import run
from tracking.tracker import TrackerConfig
from utils.config import MICRONS_PER_PIXEL, Config
from utils.helpers import setup_logging

logger = logging.getLogger("casa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sperm detection and CASA-style overlay")
    parser.add_argument("--source", default="videos/input/22.mp4",
                        help="video path, image path, or camera index")
    parser.add_argument("--weights", type=Path, default=Path("models/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("videos/output"))
    parser.add_argument("--conf", type=float, default=None,
                        help="confidence threshold (default 0.25, or 0.10 with --track)")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="cuda, cuda:0, cpu (default: auto)")
    parser.add_argument("--max-frames", type=int, default=None, help="stop early, for quick checks")
    parser.add_argument("--show", action="store_true", help="preview window; q to quit")
    parser.add_argument("--show-conf", action="store_true",
                        help="overlay per-cell confidence (off by default — clutters a clinical view)")
    parser.add_argument("--verbose", action="store_true")

    group = parser.add_argument_group("tracking")
    group.add_argument("--track", action="store_true", help="assign persistent IDs")
    group.add_argument("--track-buffer", type=int, default=30,
                       help="frames a lost cell keeps its identity, scaled by fps/30")
    group.add_argument("--match-thresh", type=float, default=0.95,
                       help="association leniency, 1-IoU distance (higher = more lenient; "
                            "0.8 fragments IDs on our small sperm boxes, see tracker.py)")
    group.add_argument("--trail", type=int, default=0,
                       help="frames of path drawn behind each cell (0 = off, dense fields get unreadable with it on)")
    group.add_argument("--min-track-len", type=int, default=10,
                       help="frames a track needs to count as usable")

    metrics_group = parser.add_argument_group("CASA metrics")
    metrics_group.add_argument("--metrics", action="store_true",
                               help="compute VCL/VSL/VAP/LIN/STR/WOB/ALH/BCF and write a CSV (implies --track)")
    metrics_group.add_argument("--um-per-px", type=float, default=MICRONS_PER_PIXEL,
                               help="calibration; see utils/config.py for how the default was derived")
    metrics_group.add_argument("--immotile-vcl", type=float, default=10.0, help="um/s")
    metrics_group.add_argument("--progressive-vsl", type=float, default=25.0, help="um/s")
    metrics_group.add_argument("--progressive-str", type=float, default=0.8, help="VSL/VAP ratio")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    tracker_config = TrackerConfig(
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        min_track_length=args.min_track_len,
    )

    # ByteTrack's second association pass exists to recover low-confidence
    # detections, so tracking feeds the detector's marginal output through and
    # lets the tracker thresholds decide. Detection-only mode keeps 0.25.
    conf = args.conf if args.conf is not None else (
        tracker_config.track_low_thresh if args.track else 0.25
    )

    config = Config(
        weights=args.weights,
        conf=conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        output_dir=args.output,
    )
    config.draw.show_conf = args.show_conf
    config.draw.trail_length = args.trail

    thresholds = MotilityThresholds(
        immotile_vcl=args.immotile_vcl,
        progressive_vsl=args.progressive_vsl,
        progressive_str=args.progressive_str,
    )

    detector = SpermDetector(config)
    destination = run(detector, args.source, config, show=args.show,
                      max_frames=args.max_frames, track=args.track,
                      tracker_config=tracker_config, metrics=args.metrics,
                      microns_per_pixel=args.um_per_px, motility_thresholds=thresholds)
    logger.info("output written to %s", destination)


if __name__ == "__main__":
    main()
