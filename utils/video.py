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


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def ensure_browser_playable(source: Path, timeout: int = 600) -> Path:
    """Return a browser-playable H.264 copy of ``source``, creating it if needed.

    Falls back to returning ``source`` unchanged when ffmpeg is missing —
    a missing preview is better than a crashed dashboard, and the caller
    can still offer the file for download.
    """
    source = Path(source)
    destination = source.with_suffix("")
    destination = destination.with_name(destination.stem + SUFFIX)

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
