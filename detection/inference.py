"""Inference loops for images, video files and live cameras.

Detection + drawing + writing. Nothing is accumulated across frames — that is
the tracking stage's job (Phase 3).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
from pathlib import Path

import cv2
import pandas as pd

from casa.metrics import MAX_PLAUSIBLE_VCL, compute_batch
from casa.motility import MotilityGrade, MotilityThresholds, classify, summarize
from detection.detector import SpermDetector
from tracking.tracker import SpermTracker, TrackerConfig
from tracking.trajectory import Trajectory, TrajectoryBuilder
from utils.config import Config, MICRONS_PER_PIXEL
from utils.draw import draw_count, draw_detections, draw_tracks
from utils.helpers import is_image, output_path, resolve_source

logger = logging.getLogger(__name__)

FALLBACK_FPS = 30.0
LOG_EVERY = 100


def run_image(detector: SpermDetector, source: str, config: Config) -> Path:
    """Annotate a single image and save it to the output directory."""
    frame = cv2.imread(source)
    if frame is None:
        raise FileNotFoundError(f"cannot read image: {source}")

    detections = detector.detect(frame)
    annotated = draw_count(draw_detections(frame, detections, config.draw), len(detections), config.draw)

    dst = output_path(source, config.output_dir, Path(source).suffix)
    cv2.imwrite(str(dst), annotated)
    logger.info("%s -> %s (%d detections)", source, dst, len(detections))
    return dst


def run_video(
    detector: SpermDetector,
    source: str | int,
    config: Config,
    show: bool = False,
    max_frames: int | None = None,
    track: bool = False,
    tracker_config: TrackerConfig | None = None,
    metrics: bool = False,
    microns_per_pixel: float = MICRONS_PER_PIXEL,
    motility_thresholds: MotilityThresholds | None = None,
) -> Path:
    """Annotate a video file or camera stream frame by frame.

    With ``track`` enabled each cell keeps an identity and its recent path is
    drawn behind it. With ``metrics`` also enabled, CASA kinematics are
    computed once tracking finishes and written to
    ``<output>/<stem>_metrics.csv``. Press ``q`` to stop early when ``show``
    is enabled.

    Tracking runs in two passes, not one. A cell the detector loses briefly
    can come back from ``tracking.trajectory.repair_fragments`` under its
    *old* identity rather than the new one ByteTrack gave it — but only after
    the whole clip has been seen once, since the fix depends on where the cell
    goes next, not just where it has been. Drawing frame-by-frame during
    tracking, as a single pass necessarily does, bakes in whichever id ByteTrack
    had assigned *at that moment* — the video would keep showing the swap
    ``repair_fragments`` was built to undo. So detection and tracking finish
    completely first; the corrected identities come out of that; only then is
    anything drawn, from those corrected identities, in a second read of the
    source. Re-decoding the file is far cheaper than the detector pass it
    follows, and cheaper than holding every raw frame in memory to avoid it.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video source: {source}")

    # Every velocity in the CASA report is frames-per-second times pixels, so a
    # wrong frame rate scales every result with it. Cameras and converters do
    # write nonsense here — an uploaded clip declared 1000 fps, which would
    # have reported VCL twenty times too high had any cell been detected.
    # Anything outside what a microscope camera plausibly produces is refused
    # in favour of the known rig rate.
    fps = cap.get(cv2.CAP_PROP_FPS) or FALLBACK_FPS
    if not 5.0 <= fps <= 240.0:
        logger.warning("video claims %.1f fps, which is not a plausible capture rate — "
                       "using %.1f instead; velocities would otherwise be wrong by %.0fx",
                       fps, FALLBACK_FPS, fps / FALLBACK_FPS)
        fps = FALLBACK_FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    dst = output_path(source, config.output_dir, ".mp4", "tracked" if track else "annotated")
    tracker_config = tracker_config or TrackerConfig()
    tracker = SpermTracker(tracker_config, frame_rate=fps) if track else None
    builder = TrajectoryBuilder() if track else None

    logger.info("%s | %dx%d @ %.1f fps | %s frames | %s", source, width, height, fps,
                total if total > 0 else "unknown", "tracking" if track else "detection only")

    frames = 0
    detected = 0
    started = time.perf_counter()

    if tracker is not None and builder is not None:
        # Pass 1 — detect and track only. No drawing, no writer: whatever gets
        # marked on screen has to reflect the *repaired* identities, which
        # do not exist until this pass is over.
        per_frame_tracks: list[list] = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                tracks = tracker.update(detector.detect(frame), frames)
                builder.add(tracks)
                per_frame_tracks.append(tracks)

                frames += 1
                detected += len(tracks)
                if frames % LOG_EVERY == 0:
                    logger.info("frame %d/%s  mean %.1f cells/frame",
                                frames, total if total > 0 else "?", detected / frames)
                if max_frames is not None and frames >= max_frames:
                    break
        finally:
            cap.release()

        trajectories, id_map = builder.finalize(tracker_config.min_track_length)

        # Pass 2 — draw from the repaired identities. id_map.get(old, old)
        # relabels a track exactly where repair_fragments joined it to an
        # earlier one; every other id passes through unchanged.
        writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"cannot open video writer: {dst}")
        replay = cv2.VideoCapture(source)
        try:
            for frame_index, raw_tracks in enumerate(per_frame_tracks):
                ok, frame = replay.read()
                if not ok:
                    break
                corrected = [replace(t, track_id=id_map.get(t.track_id, t.track_id))
                            for t in raw_tracks]
                annotated = draw_count(
                    draw_tracks(frame, corrected, trajectories, config.draw),
                    len(corrected), config.draw,
                )
                writer.write(annotated)
                if show:
                    cv2.imshow("Sperm CASA - detection", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("stopped by user at frame %d", frame_index)
                        break
        finally:
            replay.release()
            writer.release()
            if show:
                cv2.destroyAllWindows()
    else:
        # Detection-only: nothing to repair, so a single pass is exactly
        # right — this is the path a plain --source run without --track uses.
        writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"cannot open video writer: {dst}")
        trajectories, id_map = {}, {}
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                detections = detector.detect(frame)
                annotated = draw_count(
                    draw_detections(frame, detections, config.draw), len(detections), config.draw
                )
                writer.write(annotated)

                frames += 1
                detected += len(detections)
                if show:
                    cv2.imshow("Sperm CASA - detection", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("stopped by user at frame %d", frames)
                        break
                if frames % LOG_EVERY == 0:
                    logger.info("frame %d/%s  mean %.1f cells/frame",
                                frames, total if total > 0 else "?", detected / frames)
                if max_frames is not None and frames >= max_frames:
                    break
        finally:
            cap.release()
            writer.release()
            if show:
                cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    logger.info(
        "done: %d frames in %.1fs (%.1f fps), mean %.1f cells/frame -> %s",
        frames, elapsed, frames / elapsed if elapsed else 0.0,
        detected / frames if frames else 0.0, dst,
    )
    if detector.dropped_keypoints:
        logger.info("dropped %d detection(s) with unlocalized keypoints (model emits (0,0) "
                    "when it cannot place one)", detector.dropped_keypoints)

    if builder is not None and tracker is not None:
        stats = builder.summary(tracker_config.min_track_length, frame_size=(width, height))
        logger.info(
            "tracks: %d total, %d lasting >=%d frames | length mean %.0f median %.0f max %.0f "
            "| mean gaps %.1f | duplicate heads removed %d | %.0f%% of new IDs born at the frame edge",
            stats["tracks"], stats["usable"], tracker_config.min_track_length,
            stats["mean_length"], stats["median_length"], stats["max_length"],
            stats["mean_gaps"], tracker.duplicates_removed, 100 * stats["edge_birth_fraction"],
        )
        # A full-length clip naturally accumulates new IDs as cells swim into
        # or out of frame — that alone isn't fragmentation. Only warn when
        # most new tracks start away from the edge, i.e. an existing,
        # still-in-frame cell was dropped and re-born under a new ID. This
        # counts raw ByteTrack output, before repair_fragments — it is meant
        # to describe how much the tracker itself fragmented, independent of
        # how much of that the repair pass then papered over.
        ideal = detected / frames if frames else 0.0
        if ideal and stats["tracks"] > 3 * ideal and stats["edge_birth_fraction"] < 0.5:
            logger.warning(
                "track count is %.1fx the mean cell count and most new IDs start mid-frame "
                "(not at the edge) — identities are fragmenting; raise track_buffer or match_thresh",
                stats["tracks"] / ideal,
            )

        if metrics:
            _write_metrics(trajectories, fps, microns_per_pixel,
                           motility_thresholds or MotilityThresholds(), dst)

    return dst


def _write_metrics(
    trajectories: dict[int, Trajectory],
    fps: float,
    microns_per_pixel: float,
    thresholds: MotilityThresholds,
    video_dst: Path,
) -> Path | None:
    """Compute CASA kinematics for every usable trajectory and save a CSV.

    Takes the already-repaired, already-length-filtered trajectories —
    ``run_video`` produces these once, from ``TrajectoryBuilder.finalize``,
    and both the video overlay and this CSV are drawn from that same result
    rather than each calling ``finalize`` (and re-running the repair) again.

    Returns None (and logs why) if no trajectory was long enough to measure —
    a bad calibration or overly strict min_track_length shouldn't crash the
    run, since the video output above is still valid.
    """
    if not trajectories:
        logger.warning("no trajectory reached the length required to measure — "
                       "skipping CASA metrics")
        return None

    kinematics = compute_batch(trajectories, fps, microns_per_pixel)
    report = summarize(kinematics, thresholds)

    rows = []
    for k, (track_id, traj) in zip(kinematics, sorted(trajectories.items())):
        row = {"track_id": track_id, "frames": len(traj),
              "motility": classify(k, thresholds).value,
              "plausible": k.plausible,
              "vcl_um_s": k.vcl, "vsl_um_s": k.vsl, "vap_um_s": k.vap,
              "lin": k.lin, "str": k.str_, "wob": k.wob,
              "alh_um": k.alh, "bcf_hz": k.bcf}
        rows.append(row)

    csv_path = video_dst.with_name(video_dst.stem.replace("_tracked", "") + "_metrics.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # Persist the raw per-frame points so the dashboard can replay or plot a
    # single cell without re-running detection and tracking.
    trajectory_path = video_dst.with_name(video_dst.stem.replace("_tracked", "") + "_trajectories.json")
    trajectory_path.write_text(json.dumps({
        str(track_id): {
            "frames": traj.frames,
            "head": [list(p) for p in traj.head_points],
            "neck": [list(p) for p in traj.neck_points],
        }
        for track_id, traj in sorted(trajectories.items())
    }))

    flagged = [k.track_id for k in kinematics if not k.plausible]
    if flagged:
        logger.warning(
            "%d track(s) exceed %.0f um/s and are graded unreliable (ID switch at a crossing, "
            "not a fast cell): %s", len(flagged), MAX_PLAUSIBLE_VCL, flagged,
        )

    pct = report.percentages
    logger.info(
        "motility (n=%d, %.4f um/px): progressive %.0f%%  non-progressive %.0f%%  immotile %.0f%%  "
        "unreliable %.0f%% -> %s",
        report.total, microns_per_pixel,
        pct[MotilityGrade.PROGRESSIVE], pct[MotilityGrade.NON_PROGRESSIVE],
        pct[MotilityGrade.IMMOTILE], pct[MotilityGrade.UNRELIABLE],
        csv_path,
    )
    return csv_path


def run(
    detector: SpermDetector,
    source: str,
    config: Config,
    show: bool = False,
    max_frames: int | None = None,
    track: bool = False,
    tracker_config: TrackerConfig | None = None,
    metrics: bool = False,
    microns_per_pixel: float = MICRONS_PER_PIXEL,
    motility_thresholds: MotilityThresholds | None = None,
) -> Path:
    """Dispatch to the image or video loop based on the source string."""
    resolved = resolve_source(source)
    if is_image(resolved):
        if track or metrics:
            logger.warning("--track/--metrics ignored: a single image has no temporal dimension")
        return run_image(detector, str(resolved), config)

    if metrics and not track:
        logger.warning("--metrics requires --track (kinematics need identities) — enabling tracking")
        track = True

    return run_video(detector, resolved, config, show=show, max_frames=max_frames,
                     track=track, tracker_config=tracker_config, metrics=metrics,
                     microns_per_pixel=microns_per_pixel, motility_thresholds=motility_thresholds)
