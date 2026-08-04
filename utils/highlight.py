"""Single-cell highlight clips.

Given a persisted trajectory, re-render just that one cell from the source
video: seek to the frames it appears in, draw only its head/neck/skeleton
plus its path so far, and write a short clip.

This is a seek-and-redraw over stored points — detection and tracking are
not re-run, which is the whole reason ``_write_metrics`` persists
``*_trajectories.json``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from utils.config import DrawConfig
from utils.draw import FONT, draw_count, track_color

logger = logging.getLogger(__name__)

MAX_CLIP_FRAMES = 400  # ~8 s at 49 fps; keeps render time and file size sane
MAX_SUBSET_FRAMES = 900  # ~18 s; a top-N pass keeps the whole timeline


def load_trajectories(path: Path) -> dict[int, dict]:
    """Load a ``*_trajectories.json`` file, keyed by int track_id."""
    raw = json.loads(Path(path).read_text())
    return {int(k): v for k, v in raw.items()}


def render_highlight(
    source_video: Path,
    trajectory: dict,
    track_id: int,
    destination: Path,
    cfg: DrawConfig | None = None,
    max_frames: int = MAX_CLIP_FRAMES,
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

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Trim from the middle outward if the track is long, so the clip shows
    # the cell mid-swim rather than only its first moments.
    if len(frames) > max_frames:
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
    max_frames: int = MAX_SUBSET_FRAMES,
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

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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

            for position, (track_id, head, neck, i) in enumerate(marks, start=1):
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
                # Rank prefix as well as the ID: with the set changing frame to
                # frame, "who is leading right now" is the thing being watched,
                # and the ID alone doesn't show it.
                cv2.putText(frame, f"#{position} ID {track_id}",
                            (head_pt[0] + 10, head_pt[1] - 8),
                            FONT, cfg.font_scale, colour, 1, cv2.LINE_AA)

            draw_count(frame, len(marks), cfg)
            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    return destination
