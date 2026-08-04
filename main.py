"""Sperm CASA — detection, tracking and CASA kinematics.

    python main.py --source videos/input/22.mp4                    # detection only
    python main.py --source videos/input/22.mp4 --track            # + persistent IDs
    python main.py --source videos/input/22.mp4 --track --metrics  # + VCL/VSL/... CSV
    python main.py --rebuild                                       # redo whatever the code outdated
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

from casa.motility import MotilityThresholds
from detection.detector import SpermDetector
from detection.inference import run
from tracking.tracker import TrackerConfig
from utils import remote_analysis
from utils.config import MICRONS_PER_PIXEL, Config
from utils.helpers import setup_logging

logger = logging.getLogger("casa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sperm detection and CASA-style overlay")
    parser.add_argument("--source", default="videos/input/22.mp4",
                        help="video path, image path, or camera index")
    parser.add_argument("--weights", type=Path, default=Path("models/best_v2.pt"))
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

    # Defaults come from TrackerConfig itself rather than being retyped here —
    # a second hardcoded copy is exactly how this drifted before: match_thresh
    # was tuned 0.95 -> 0.99 in TrackerConfig, this parser kept defaulting to
    # 0.95, and because argparse always supplies *some* value, every run of
    # `main.py --track` without an explicit --match-thresh silently rebuilt
    # every video on the stale setting — including, at the time this was
    # caught, every clip in videos/output.
    _defaults = TrackerConfig()
    group = parser.add_argument_group("tracking")
    group.add_argument("--track", action="store_true", help="assign persistent IDs")
    group.add_argument("--track-buffer", type=int, default=_defaults.track_buffer,
                       help="frames a lost cell keeps its identity, scaled by fps/30")
    group.add_argument("--match-thresh", type=float, default=_defaults.match_thresh,
                       help="association leniency, 1-IoU distance (higher = more lenient; "
                            "see tracker.py for how this was tuned)")
    group.add_argument("--trail", type=int, default=0,
                       help="frames of path drawn behind each cell (0 = off, dense fields get unreadable with it on)")
    group.add_argument("--min-track-len", type=int, default=_defaults.min_track_length,
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


# Files whose changes actually affect the detection/tracking/CASA outputs.
# UI/frontend changes, docs, or other viewer-only code should not invalidate
# the rebuild stamp and force every clip to be reprocessed.
REBUILD_FILES = (
    Path("detection/detector.py"),
    Path("detection/inference.py"),
    Path("tracking/tracker.py"),
    Path("tracking/trajectory.py"),
    Path("casa/metrics.py"),
    Path("casa/motility.py"),
    Path("utils/config.py"),
)

# What counts as a clip anywhere in the pipeline. WMV and MKV are here because
# converting footage before upload is what broke a real recording: the desktop
# converter crushed its contrast to 10 grey levels and the detector found
# nothing. Reading the original is always safer than reading a re-encode.
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".wmv", ".mkv"}


def pipeline_fingerprint() -> str:
    """Hash of the code and model weights that decide what a result looks like.

    Content, not mtime: a deploy checks the repo out fresh, so every file's
    mtime is "now" on the server and a timestamp comparison would re-analyse
    every clip on every push, including README-only ones. Model changes must
    also invalidate the rebuild stamp so old videos are reprocessed with the
    new checkpoint.
    """
    digest = hashlib.sha256()
    for path in REBUILD_FILES:
        if path.exists():
            digest.update(path.read_bytes())
    for path in sorted(Path("models").glob("*.pt")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def stale_videos(input_dir: Path, output_dir: Path) -> list[Path]:
    """Inputs never analysed, or analysed by a different version of the code.

    Covers the preloaded samples and dashboard uploads alike — they are the
    same files to this, and the suffixes are the ones the uploader accepts.
    """
    current = pipeline_fingerprint()
    stale = []
    sources = (p for p in sorted(input_dir.iterdir())
               if p.suffix.lower() in VIDEO_SUFFIXES)
    for source in sources:
        stamp = output_dir / f"{source.stem}.build"
        csv = output_dir / f"{source.stem}_metrics.csv"
        if not csv.exists() or not stamp.exists() or stamp.read_text().strip() != current:
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

    # --rebuild runs on the always-on frontend by design — it is dispatched
    # from the deploy pipeline, which has no GPU. Checked once, not per clip:
    # the GPU box does not come and go within the few minutes a batch takes,
    # and one probe instead of N is one fewer way to be unlucky on a flaky
    # connection.
    use_remote = args.rebuild and remote_analysis.available()
    if args.rebuild:
        logger.info("GPU box %s for this batch", "available — dispatching there" if use_remote
                    else "not reachable — running on this machine")

    for source in sources:
        remote_ok = False
        if use_remote:
            remote_ok = remote_analysis.run_remote(
                Path(source), tracker_config.min_track_length, output_dir=args.output)
            if not remote_ok:
                logger.warning("%s: GPU run failed, falling back to local", Path(source).stem)

        if remote_ok:
            destination = args.output / f"{Path(source).stem}_tracked.mp4"
        else:
            destination = run(detector, str(source), config, show=args.show,
                              max_frames=args.max_frames, track=args.track or args.rebuild,
                              tracker_config=tracker_config, metrics=args.metrics or args.rebuild,
                              microns_per_pixel=args.um_per_px, motility_thresholds=thresholds)
        if args.rebuild:
            # Written last, so an interrupted rebuild leaves the clip stale and
            # the next deploy picks it up again. The fingerprint describes
            # *this* machine's code — correct either way, since a remote run
            # only happens once the GPU box has been confirmed to match (see
            # the deploy pipeline's "Sync code to the GPU box" step).
            (args.output / f"{Path(source).stem}.build").write_text(pipeline_fingerprint())
        logger.info("output written to %s", destination)


if __name__ == "__main__":
    main()
