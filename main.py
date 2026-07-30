"""Sperm CASA — detection, tracking and CASA kinematics.

    python main.py --source videos/input/22.mp4                    # detection only
    python main.py --source videos/input/22.mp4 --track            # + persistent IDs
    python main.py --source videos/input/22.mp4 --track --metrics  # + VCL/VSL/... CSV
    python main.py --rebuild                                       # redo whatever the code outdated
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

    parser.add_argument("--rebuild", action="store_true",
                        help="re-analyse every video in videos/input whose results are older than "
                             "the detection/tracking/CASA code, and ignore --source")
    parser.add_argument("--input-dir", type=Path, default=Path("videos/input"),
                        help="where --rebuild looks for source videos")
    return parser.parse_args()


# Directories whose .py files change what a result looks like. The dashboard
# only reads finished files, so nothing else notices that a fix has landed and
# every clip in videos/output stays as it was until this is run.
PIPELINE_DIRS = ("detection", "tracking", "casa", "utils")


def stale_videos(input_dir: Path, output_dir: Path) -> list[Path]:
    """Inputs with no metrics CSV, or one written before the last code change."""
    code_mtime = max(p.stat().st_mtime
                     for d in PIPELINE_DIRS for p in Path(d).glob("*.py"))
    stale = []
    for source in sorted(input_dir.glob("*.mp4")):
        csv = output_dir / f"{source.stem}_metrics.csv"
        if not csv.exists() or csv.stat().st_mtime < code_mtime:
            stale.append(source)
    return stale


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
        tracker_config.track_low_thresh if args.track or args.rebuild else 0.25
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

    if args.rebuild:
        sources = stale_videos(args.input_dir, args.output)
        if not sources:
            logger.info("every video in %s is already up to date", args.input_dir)
            return
        logger.info("re-analysing %d video(s): %s",
                    len(sources), ", ".join(s.stem for s in sources))
    else:
        sources = [args.source]

    for source in sources:
        destination = run(detector, str(source), config, show=args.show,
                          max_frames=args.max_frames, track=args.track or args.rebuild,
                          tracker_config=tracker_config, metrics=args.metrics or args.rebuild,
                          microns_per_pixel=args.um_per_px, motility_thresholds=thresholds)
        logger.info("output written to %s", destination)


if __name__ == "__main__":
    main()
