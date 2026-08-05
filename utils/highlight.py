"""Single-cell highlight clips.

Given a persisted trajectory, re-render just that one cell from the source
video: seek to the frames it appears in, draw only its head/neck/skeleton
plus its path so far, and write a short clip.

This is a seek-and-redraw over stored points — detection and tracking are
not re-run, which is the whole reason ``_write_metrics`` persists
``*_trajectories.json``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd

from casa.motility import MotilityThresholds
from utils.config import DrawConfig
from utils.draw import FONT, draw_count, track_color
from utils.video import (MAX_PLAUSIBLE_FPS, MIN_PLAUSIBLE_FPS, ensure_browser_playable,
                         plausible_fps)

logger = logging.getLogger(__name__)

# Keep highlight outputs close to the source video length unless a caller
# explicitly requests a shorter clip.

# Never mark more than this at once. The point of the view is to pick out the
# few cells worth watching; past half a dozen marks the frame reads as noise
# again and there is no advantage over the full overlay.
MAX_TOP_MARKS = 6

# A non-progressive cell only qualifies if it is genuinely swimming, not just
# clearing the immotile floor. 2.5x that floor (25 um/s VCL against the 10 um/s
# cut-off in casa/motility.py) drops the bottom ~30% of non-progressive cells
# in the reference samples — the ones twitching in place.
DECENT_VCL_MULTIPLE = 2.5


def load_trajectories(path: Path) -> dict[int, dict]:
    """Load a ``*_trajectories.json`` file, keyed by int track_id."""
    raw = json.loads(Path(path).read_text())
    return {int(k): v for k, v in raw.items()}


def top_performer_ranking(df: pd.DataFrame) -> list[int]:
    """Cells worth marking as top performers, best first.

    Eligibility, not just ordering: immotile cells and tracks with rejected
    measurements are excluded outright, and a non-progressive cell is only
    admitted if its VCL clears ``DECENT_VCL_MULTIPLE`` times the immotile
    floor. An empty list is a real answer — it means nothing in the sample is
    swimming well enough to call a top performer, and nothing gets marked.

    Progressive cells always outrank non-progressive ones. Within progressive
    the tiebreak is VSL, which measures net forward progress; within
    non-progressive it is VCL, since by definition they are not progressing
    and what distinguishes them is raw vigour.
    """
    if "motility" not in df or df.empty:
        return []

    ok = df[df["plausible"].astype(bool)] if "plausible" in df else df
    decent_vcl = MotilityThresholds().immotile_vcl * DECENT_VCL_MULTIPLE
    is_progressive = ok["motility"] == "progressive"
    is_decent_non_prog = (ok["motility"] == "non_progressive") & (ok["vcl_um_s"] >= decent_vcl)
    ok = ok[is_progressive | is_decent_non_prog]
    if ok.empty:
        return []

    ok = ok.assign(
        _grade=(ok["motility"] != "progressive").astype(int),
        _score=ok["vsl_um_s"].where(ok["motility"] == "progressive", ok["vcl_um_s"]),
    ).sort_values(["_grade", "_score"], ascending=[True, False])
    return [int(t) for t in ok["track_id"]]


# Bumped whenever what gets *drawn* on the top-N clip changes. The filename
# is otherwise keyed only on the ranking, so a drawing change would leave
# every cached clip looking valid (it is newer than the trajectories) and the
# old overlay would stay on screen forever. v2 dropped the "#rank" prefix.
TOP_LABEL_VERSION = 2


def top_clip_path(stem: str, highlight_dir: Path, ranking: Sequence[int], top_n: int) -> Path:
    """Where the top-N clip for this exact ranking lives.

    Keyed by the whole ranking, not just N — the per-frame selection can pull
    in any cell, so every ID in the order affects what ends up on screen.
    hashlib, not hash(): the builtin is seeded per process, so the filename
    would change on every restart and the cached clip would never be reused.
    Shared with the dashboard so both agree on the name; they used to compute
    it separately, which meant any change here silently orphaned the cache.
    """
    key = f"v{TOP_LABEL_VERSION}-" + "-".join(str(t) for t in ranking)
    digest = hashlib.sha1(key.encode()).hexdigest()[:8]
    return highlight_dir / f"{stem}_top{top_n}_{digest}.mp4"


def _stale(clip: Path, trajectories_path: Path) -> bool:
    """Whether a cached highlight clip predates the data it should reflect.

    A clip's filename is keyed on track_id (and, for the top-N clip, the
    ranking) but not on *when* it was rendered, so re-running tracking — a
    tuning change, a fragment-repair fix, anything — can leave an
    already-cached clip on disk that still shows the old, uncorrected
    identities while the video and metrics next to it have moved on. Same
    fix as ``load_metrics``: compare against the source's mtime instead of
    trusting "a file with this name exists".
    """
    return not clip.exists() or clip.stat().st_mtime < trajectories_path.stat().st_mtime


def _wrong_fps(clip: Path) -> bool:
    """Whether an existing clip was written with an unusable frame rate.

    Clips rendered before the rate was sanitised are *newer* than the
    trajectory file, so the mtime check above calls them fresh and they stay
    broken — a few hundred frames written at the source's claimed 1000 fps,
    which a browser reports as 0:00 and plays as one flashing frame. Checked
    only when (re)building the cache, never on the per-click path, so viewing
    stays a plain stat.
    """
    cap = cv2.VideoCapture(str(clip))
    try:
        if not cap.isOpened():
            return True
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    return not MIN_PLAUSIBLE_FPS <= fps <= MAX_PLAUSIBLE_FPS


# Where the dashboard's player fetches clips from. Streamlit serves this
# folder at /app/static/, which is the only way to hand the browser a URL it
# can swap between without a rerun.
STATIC_DIR = Path("static")


def prerender(stem: str, source: Path, output_dir: Path, top_n: int = MAX_TOP_MARKS) -> None:
    """Render every per-cell clip, the top-N clip, and the browser-playable
    transcodes right after analysis finishes.

    Without this, the dashboard renders each clip lazily the first time a
    viewer requests it — a completed analysis then looks "instant" only for
    the Overview tab; clicking a new sperm ID or toggling "Top sperms" still
    pays OpenCV seek+draw+encode cost on that click. Paying it once here,
    right after the already-slow analysis step, makes every later view a
    plain file read.
    """
    trajectories_path = output_dir / f"{stem}_trajectories.json"
    csv_path = output_dir / f"{stem}_metrics.csv"
    tracked_path = output_dir / f"{stem}_tracked.mp4"
    if not (trajectories_path.exists() and csv_path.exists() and tracked_path.exists()):
        return

    # Everything the player can show needs a URL, so the source is transcoded
    # too. It goes to STATIC_DIR, never beside the source: written into the
    # input directory it would read back as a separate un-analysed video.
    ensure_browser_playable(tracked_path, out_dir=STATIC_DIR)
    ensure_browser_playable(source, out_dir=STATIC_DIR)

    trajectories = load_trajectories(trajectories_path)
    highlight_dir = output_dir / "highlights"
    highlight_dir.mkdir(parents=True, exist_ok=True)

    # The top-N clip first, and only the ranked cells after it. Rendering a
    # clip per trajectory is fine at 40 cells and pathological when tracking
    # fragments: one deployed sample produced 1663 tracks, which at ~14s each
    # is over six hours -- and because the top-N clip used to be rendered
    # last, the "Top sperms" switch stayed dead for that whole time. The
    # unranked cells still render on first click, which is what the lazy path
    # in the dashboard has always done.
    ranking = top_performer_ranking(pd.read_csv(csv_path))
    if ranking:
        clip = top_clip_path(stem, highlight_dir, ranking, top_n)
        if _stale(clip, trajectories_path) or _wrong_fps(clip):
            render_top_n(source, trajectories, ranking, top_n, clip,
                        DrawConfig(trail_length=30, font_scale=0.4))
        if clip.exists():
            ensure_browser_playable(clip, out_dir=STATIC_DIR)
            # The name carries the ranking and the label version, so a
            # re-analysis or a drawing change mints a new one and orphans the
            # last. Without this they accumulate one dead pair per change.
            keep = clip.stem
            for old in highlight_dir.glob(f"{stem}_top{top_n}_*.mp4"):
                if old.stem != keep:
                    old.unlink(missing_ok=True)
            for old in STATIC_DIR.glob(f"{stem}_top{top_n}_*.h264.mp4"):
                if not old.name.startswith(keep + "."):
                    old.unlink(missing_ok=True)

    for track_id in ranking[:top_n]:
        trajectory = trajectories.get(track_id)
        if trajectory is None:
            continue
        clip = highlight_dir / f"{stem}_id{track_id}.mp4"
        if _stale(clip, trajectories_path) or _wrong_fps(clip):
            render_highlight(source, trajectory, track_id, clip)
        if clip.exists():
            ensure_browser_playable(clip, out_dir=STATIC_DIR)


def render_highlight(
    source_video: Path,
    trajectory: dict,
    track_id: int,
    destination: Path,
    cfg: DrawConfig | None = None,
    max_frames: int | None = None,
) -> Path | None:
    """Write a clip showing only ``track_id``, with its path drawn behind it.

    Returns None if the source video can't be opened or the trajectory is
    empty, so the caller can fall back to the full tracked video.
    """
    cfg = cfg or DrawConfig()
    frames = trajectory.get("frames") or []
    heads = trajectory.get("head") or []
    necks = trajectory.get("neck") or []
    if not frames:
        return None

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        logger.error("cannot open %s for highlight rendering", source_video)
        return None

    fps = plausible_fps(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Trim from the middle outward only when a caller explicitly requests a
    # shorter clip; otherwise preserve the trajectory length.
    if max_frames is not None and len(frames) > max_frames:
        start = (len(frames) - max_frames) // 2
        frames = frames[start:start + max_frames]
        heads = heads[start:start + max_frames]
        necks = necks[start:start + max_frames]

    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        cap.release()
        logger.error("cannot open writer for %s", destination)
        return None

    colour = track_color(track_id)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
        expected = frames[0]
        for i, (frame_index, head, neck) in enumerate(zip(frames, heads, necks)):
            # Trajectories can have gaps (a briefly lost cell). Only seek
            # when the next wanted frame isn't the one we'd read anyway —
            # sequential reads are far cheaper than repeated seeks.
            if frame_index != expected:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            expected = frame_index + 1
            if not ok:
                break

            trail = np.array(heads[:i + 1], dtype=np.int32)
            if len(trail) > 1:
                cv2.polylines(frame, [trail], False, colour, 1, cv2.LINE_AA)

            head_pt = (int(round(head[0])), int(round(head[1])))
            neck_pt = (int(round(neck[0])), int(round(neck[1])))
            cv2.line(frame, head_pt, neck_pt, cfg.line_color, cfg.line_thickness, cv2.LINE_AA)
            cv2.circle(frame, neck_pt, cfg.neck_radius, cfg.neck_color, -1, cv2.LINE_AA)
            cv2.circle(frame, head_pt, cfg.head_radius, cfg.head_color, -1, cv2.LINE_AA)
            cv2.circle(frame, head_pt, cfg.head_radius + 6, colour, 1, cv2.LINE_AA)
            cv2.putText(frame, f"ID {track_id}", (head_pt[0] + 10, head_pt[1] - 8),
                        FONT, 0.4, colour, 1, cv2.LINE_AA)
            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    return destination


def render_top_n(
    source_video: Path,
    trajectories: dict[int, dict],
    ranking: Sequence[int],
    top_n: int,
    destination: Path,
    cfg: DrawConfig | None = None,
    max_frames: int | None = None,
) -> Path | None:
    """Write a clip marking the best ``top_n`` cells *present in each frame*.

    ``ranking`` is every track ordered best-to-worst for the whole video; the
    selection is then made per frame, not once up front. So a cell that swims
    out of view frees its slot for the next-best cell still on screen, and a
    strong cell entering later displaces a weaker one — the marks re-rank
    continuously instead of being fixed to one global top-N set.

    ``render_highlight`` follows one cell and seeks to just the frames it
    appears in; this keeps the original timeline so the clip stays in step
    with the full tracked video.

    Returns None if the source can't be opened or none of the ranked IDs have
    a stored trajectory, so the caller can fall back.
    """
    cfg = cfg or DrawConfig()
    known = [int(t) for t in ranking if int(t) in trajectories]
    if not known or top_n < 1:
        return None
    rank_of = {track_id: position for position, track_id in enumerate(known)}

    # frame index -> every cell visible on it. Building this once up front
    # turns rendering into a single sequential read with no seeking, and lets
    # each frame pick its own top N out of whoever is actually there.
    heads_by_id = {t: (trajectories[t].get("head") or []) for t in known}
    per_frame: dict[int, list[tuple[int, list, list, int]]] = {}
    for track_id in known:
        traj = trajectories[track_id]
        for i, (frame_index, head, neck) in enumerate(zip(traj.get("frames") or [],
                                                          heads_by_id[track_id],
                                                          traj.get("neck") or [])):
            per_frame.setdefault(int(frame_index), []).append((track_id, head, neck, i))
    if not per_frame:
        return None

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        logger.error("cannot open %s for top-N rendering", source_video)
        return None

    fps = plausible_fps(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if max_frames is None:
        last_frame = total_frames - 1 if total_frames > 0 else max(per_frame)
    else:
        last_frame = min(max(per_frame), max_frames - 1)

    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        cap.release()
        logger.error("cannot open writer for %s", destination)
        return None

    try:
        for frame_index in range(last_frame + 1):
            ok, frame = cap.read()
            if not ok:
                break

            # The re-ranking itself: whoever is on screen right now, best
            # first, capped at N. Recomputed every frame, so the marks follow
            # cells in and out of view.
            visible = per_frame.get(frame_index, [])
            marks = sorted(visible, key=lambda m: rank_of[m[0]])[:top_n]

            for track_id, head, neck, i in marks:
                colour = track_color(track_id)
                if cfg.trail_length > 0 and i > 0:
                    start = max(0, i - cfg.trail_length)
                    trail = np.array(heads_by_id[track_id][start:i + 1], dtype=np.int32)
                    if len(trail) > 1:
                        cv2.polylines(frame, [trail], False, colour,
                                      cfg.trail_thickness, cv2.LINE_AA)

                head_pt = (int(round(head[0])), int(round(head[1])))
                neck_pt = (int(round(neck[0])), int(round(neck[1])))
                cv2.line(frame, head_pt, neck_pt, cfg.line_color,
                         cfg.line_thickness, cv2.LINE_AA)
                cv2.circle(frame, neck_pt, cfg.neck_radius, cfg.neck_color, -1, cv2.LINE_AA)
                cv2.circle(frame, head_pt, cfg.head_radius, cfg.head_color, -1, cv2.LINE_AA)
                cv2.circle(frame, head_pt, cfg.head_radius + 6, colour, 1, cv2.LINE_AA)
                # ID only. The rank is recomputed every frame, so a prefix
                # flickers between numbers as cells enter and leave — it reads
                # as the cell's identity changing when only the ordering did.
                cv2.putText(frame, f"ID {track_id}",
                            (head_pt[0] + 10, head_pt[1] - 8),
                            FONT, cfg.font_scale, colour, 1, cv2.LINE_AA)

            draw_count(frame, len(marks), cfg)
            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    return destination
