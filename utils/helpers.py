"""Small shared utilities."""

from __future__ import annotations

import logging
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once, with a compact format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def resolve_source(source: str) -> str | int:
    """Return a camera index for digit-only input, otherwise the path unchanged."""
    return int(source) if source.isdigit() else source


def is_image(source: str | int) -> bool:
    return isinstance(source, str) and Path(source).suffix.lower() in IMAGE_SUFFIXES


def output_path(source: str | int, out_dir: Path, suffix: str, tag: str = "annotated") -> Path:
    """Build ``<out_dir>/<stem>_<tag><suffix>``, creating the directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"camera{source}" if isinstance(source, int) else Path(source).stem
    return out_dir / f"{stem}_{tag}{suffix}"
