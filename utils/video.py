"""Browser-playable video conversion.

OpenCV writes ``mp4v``/``FMP4``-encoded MP4s, which browsers' native
``<video>`` element does not decode — Streamlit would show an empty player.
This re-encodes to H.264 (``libx264`` + ``yuv420p``), which every browser
supports.

Kept out of the inference pipeline on purpose: ``main.py`` output format
stays as-is for anything consuming it non-interactively, and the dashboard
pays the transcode cost lazily, once per video.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

SUFFIX = ".h264.mp4"

# The reference rig's rate, used whenever a file's own claim is not believable.
FALLBACK_FPS = 30.0

# What a microscope camera can plausibly produce. Converters really do write
# nonsense outside this — an uploaded clip declared 1000 fps.
MIN_PLAUSIBLE_FPS, MAX_PLAUSIBLE_FPS = 5.0, 240.0


def plausible_fps(raw: float, fallback: float = FALLBACK_FPS) -> float:
    """A believable frame rate for ``raw``, or ``fallback`` if it isn't one.

    Shared by every writer in the pipeline on purpose. The tracked overlay
    sanitised its rate while the highlight clips used the raw value, so a
    clip declaring 1000 fps produced a full-length tracked video next to
    per-cell clips written at 1000 fps — a few hundred frames lasting a
    fraction of a second, which plays as a single flashing frame and reports
    a 0:00 duration. Velocities scale with this too, so the same number has
    to reach ``compute_batch``.
    """
    fps = raw or fallback
    if not MIN_PLAUSIBLE_FPS <= fps <= MAX_PLAUSIBLE_FPS:
        logger.warning("video claims %.1f fps, which is not a plausible capture rate — "
                       "using %.1f instead; timings would otherwise be wrong by %.0fx",
                       fps, fallback, fps / fallback if fallback else 0)
        return fallback
    return fps


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def ensure_browser_playable(source: Path, timeout: int = 600,
                            out_dir: Path | None = None) -> Path:
    """Return a browser-playable H.264 copy of ``source``, creating it if needed.

    ``out_dir`` places the copy somewhere other than beside the source — the
    dashboard points it at the statically-served folder so the player can
    fetch clips by URL. Defaults to alongside the source, which is what the
    CLI wants. Nothing is duplicated either way: this is the only place the
    H.264 copy is written.

    Falls back to returning ``source`` unchanged when ffmpeg is missing —
    a missing preview is better than a crashed dashboard, and the caller
    can still offer the file for download.
    """
    source = Path(source)
    stem = source.with_suffix("").name
    destination = (Path(out_dir) if out_dir is not None else source.parent) / (stem + SUFFIX)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_mtime >= source.stat().st_mtime:
        return destination

    if not ffmpeg_available():
        logger.warning("ffmpeg not found — serving %s as-is; it may not play in a browser", source)
        return source

    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
         "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(destination)],
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        logger.error("ffmpeg failed on %s: %s", source, result.stderr.strip()[:400])
        return source

    logger.info("transcoded %s -> %s", source.name, destination.name)
    return destination
