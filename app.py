"""Sperm CASA dashboard.

    streamlit run app.py

Browses videos that have already been processed, lets a viewer inspect any
individual sperm, and accepts new uploads which run through the same
pipeline and then behave identically to the preloaded ones.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from casa.metrics import MAX_PLAUSIBLE_VCL
from main import VIDEO_SUFFIXES
from detection.detector import SpermDetector
from detection.inference import run
from tracking.tracker import TrackerConfig
from utils import charts
from utils.config import MICRONS_PER_PIXEL, Config
from utils.explain import (GRADE_HELP, GRADE_LABEL, METRIC_HELP, WHO_DISCLAIMER,
                           WHO_REFERENCE)
from utils.helpers import setup_logging
from utils.highlight import load_trajectories, render_highlight
from utils.video import ensure_browser_playable

INPUT_DIR = Path("videos/input")
OUTPUT_DIR = Path("videos/output")
HIGHLIGHT_DIR = OUTPUT_DIR / "highlights"
MAX_UPLOAD_MB = 50

setup_logging(logging.INFO)
st.set_page_config(page_title="Sperm CASA", page_icon="🔬", layout="wide")

STYLE = """
<style>
  .block-container { padding-top: 2.5rem; max-width: 1400px; }
  h1, h2, h3 { letter-spacing: -0.01em; }

  /* Metric cards: give them a surface so they read as tiles, not floating text */
  div[data-testid="stMetric"] {
    background: #1e1e1c;
    border: 1px solid #2e2e2c;
    border-radius: 10px;
    padding: 14px 16px;
  }
  div[data-testid="stMetricLabel"] p {
    font-size: 0.78rem !important;
    color: #9a9a94 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  div[data-testid="stMetricValue"] { font-size: 1.65rem; }

  /* Colour bar down the left of each grade tile — identity is still carried
     by the text label; this only reinforces it. */
  .grade-progressive     div[data-testid="stMetric"] { border-left: 3px solid #199e70; }
  .grade-non_progressive div[data-testid="stMetric"] { border-left: 3px solid #3987e5; }
  .grade-immotile        div[data-testid="stMetric"] { border-left: 3px solid #d95926; }
  .grade-unreliable      div[data-testid="stMetric"] { border-left: 3px solid #6b6b68; }

  .stTabs [data-baseweb="tab-list"] { gap: 4px; }
  .stTabs [data-baseweb="tab"] { padding: 8px 18px; }

  /* The source footage is 640x480. Left alone it stretches to ~1240px — a
     2x upscale that looks soft and is tall enough to push the title and tabs
     off screen at 100% zoom. Cap it at a mild upscale and bound it by
     viewport height so the page fits on a 1080p laptop.
     The <video> element itself carries data-testid="stVideo" (it is not a
     child of it) and an inline width:100%, so the selector must target the
     element directly and !important is required to beat the inline style. */
  video[data-testid="stVideo"], video.stVideo {
    /* Height drives the size and width follows the 4:3 ratio, so the player
       always clears the fold instead of overflowing it. 58vh leaves room for
       the title, tabs and controls above; the 620px cap stops it ballooning
       on tall monitors. Setting width instead (and letting height follow)
       overflowed a 1080p screen by ~90px, and width:auto alone collapses to
       the native 640px, which is too small. */
    height: min(58vh, 620px) !important;
    width: auto !important;
    max-width: 100% !important;
    display: block;
    border-radius: 8px;
  }

  .casa-caption { color: #9a9a94; font-size: 0.85rem; line-height: 1.5; }
  .casa-ref {
    background: #1e1e1c; border: 1px solid #2e2e2c; border-left: 3px solid #6b6b68;
    border-radius: 8px; padding: 12px 16px; margin-top: 8px;
  }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Access gate
# --------------------------------------------------------------------------

def check_password() -> bool:
    """Single shared password, read from CASA_DASHBOARD_PASSWORD.

    Skipped entirely when the variable is unset, so local development is
    unaffected; set it on the deployed instance. ponytail: one shared secret,
    not an auth framework — swap for real auth only if this stops being a
    demo.
    """
    expected = os.environ.get("CASA_DASHBOARD_PASSWORD")
    if not expected:
        return True
    if st.session_state.get("authenticated"):
        return True

    st.title("Sperm CASA")
    st.caption("Computer-assisted sperm analysis")
    entered = st.text_input("Password", type="password")
    if entered:
        if entered == expected:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


# --------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------

def processed_videos() -> dict[str, dict[str, Path]]:
    """Every video with a finished metrics CSV, preloaded or uploaded alike."""
    found: dict[str, dict[str, Path]] = {}
    for csv_path in sorted(OUTPUT_DIR.glob("*_metrics.csv")):
        stem = csv_path.stem.replace("_metrics", "")
        tracked = OUTPUT_DIR / f"{stem}_tracked.mp4"
        source = next((p for p in INPUT_DIR.glob(f"{stem}.*")), None)
        if tracked.exists() and source is not None:
            found[stem] = {
                "csv": csv_path,
                "tracked": tracked,
                "source": source,
                "trajectories": OUTPUT_DIR / f"{stem}_trajectories.json",
            }
    return found


@st.cache_data(show_spinner=False)
def load_metrics(csv_path: str, mtime: float) -> pd.DataFrame:
    """Read a metrics CSV. ``mtime`` busts the cache when the file changes."""
    return pd.read_csv(csv_path)


def unprocessed_videos() -> tuple[list[Path], list[Path]]:
    """Clips with no results, split into never-run and ran-but-found-nothing.

    A clip the detector found no sperm in produces a tracked video but no
    metrics CSV, so it can never join the sample list. Calling that "not yet
    analysed" invites the viewer to upload it again and get the same silence,
    which is why the two cases are told apart here.
    """
    done = set(processed_videos())
    waiting, empty = [], []
    for path in sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES):
        if path.stem in done:
            continue
        (empty if (OUTPUT_DIR / f"{path.stem}_tracked.mp4").exists() else waiting).append(path)
    return waiting, empty


def grade_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = df["motility"].value_counts().to_dict() if "motility" in df else {}
    return {g: int(counts.get(g, 0)) for g in GRADE_LABEL}


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------

def render_dashboard(df: pd.DataFrame, paths: dict[str, Path], stem: str) -> None:
    counts = grade_counts(df)
    graded_total = sum(v for k, v in counts.items() if k != "unreliable")

    if graded_total == 0:
        st.warning("No reliably measured cells in this video.")
        return

    progressive_pct = 100.0 * counts["progressive"] / graded_total
    motile_pct = 100.0 * (counts["progressive"] + counts["non_progressive"]) / graded_total

    st.markdown("#### Motility")
    cols = st.columns(4)
    for col, grade in zip(cols, ["progressive", "non_progressive", "immotile", "unreliable"]):
        n = counts[grade]
        # Unreliable cells are excluded from the denominator: they are
        # measurement failures, not a motility category, so folding them in
        # would deflate every real percentage.
        with col:
            st.markdown(f'<div class="grade-{grade}">', unsafe_allow_html=True)
            if grade == "unreliable":
                st.metric(GRADE_LABEL[grade], f"{n}", help=GRADE_HELP[grade])
            else:
                st.metric(GRADE_LABEL[grade], f"{100.0 * n / graded_total:.0f}%",
                          delta=f"{n} cells", delta_color="off", help=GRADE_HELP[grade])
            st.markdown("</div>", unsafe_allow_html=True)

    st.altair_chart(charts.motility_bar(df), use_container_width=True)

    # WHO orientation
    prog_limit, prog_note = WHO_REFERENCE["progressive_pct"]
    total_limit, total_note = WHO_REFERENCE["total_motile_pct"]
    prog_mark = "at or above" if progressive_pct >= prog_limit else "below"
    total_mark = "at or above" if motile_pct >= total_limit else "below"
    st.markdown(
        f'<div class="casa-ref"><b>Against WHO 6th edition reference limits</b><br>'
        f'<span class="casa-caption">'
        f'Progressive {progressive_pct:.0f}% — <b>{prog_mark}</b> the {prog_limit:.0f}% limit.<br>'
        f'Total motile {motile_pct:.0f}% — <b>{total_mark}</b> the {total_limit:.0f}% limit.<br><br>'
        f'{WHO_DISCLAIMER}</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### Average kinematics")
    st.caption(f"Across the {graded_total} reliably measured cells.")
    reliable = df[df["plausible"]] if "plausible" in df else df
    cols = st.columns(4)
    for col, column in zip(cols, ["vcl_um_s", "vsl_um_s", "vap_um_s", "lin"]):
        if column not in reliable:
            continue
        label, explanation = METRIC_HELP[column]
        # LIN is a 0-1 ratio: one decimal throws away most of its resolution.
        value = (f"{reliable[column].mean():.2f}" if column == "lin"
                 else f"{reliable[column].mean():.1f} µm/s")
        col.metric(label.split(" (")[0], value, help=explanation)

    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Speed vs. straightness")
        st.caption("Each dot is one cell. Fast and straight sits top-right; "
                   "fast but circling sits bottom-right. Hover for details.")
        st.altair_chart(charts.velocity_scatter(df), use_container_width=True)
    with right:
        st.markdown("#### Speed distribution")
        st.caption("How the sample spreads across velocities.")
        st.altair_chart(charts.velocity_histogram(df), use_container_width=True)

    st.markdown("#### Downloads")
    cols = st.columns(3)
    cols[0].download_button("Metrics CSV", paths["csv"].read_bytes(),
                            file_name=paths["csv"].name, mime="text/csv",
                            use_container_width=True)
    cols[1].download_button("Tracked video", paths["tracked"].read_bytes(),
                            file_name=paths["tracked"].name, mime="video/mp4",
                            use_container_width=True)
    if paths["trajectories"].exists():
        cols[2].download_button("Trajectories JSON", paths["trajectories"].read_bytes(),
                                file_name=paths["trajectories"].name,
                                mime="application/json", use_container_width=True)


ALL_CELLS = "All cells"


def render_video_viewer(df: pd.DataFrame, paths: dict[str, Path], stem: str) -> None:
    """One main video that changes with the selection.

    Picking a sperm swaps the main player to a clip of *only* that cell —
    there is no second video anywhere. The layout starts full width and
    narrows to make room for the analysis panel only once a cell is chosen.
    """
    def label_for(value) -> str:
        if value == ALL_CELLS:
            return "All cells"
        row = df[df["track_id"] == value].iloc[0]
        return f"ID {value} — {GRADE_LABEL.get(row.get('motility', ''), '?')}"

    header_left, header_right = st.columns([3, 2], gap="large")
    with header_right:
        options = [ALL_CELLS] + (df["track_id"].tolist() if not df.empty else [])
        selection = st.selectbox("Inspect a single sperm", options, format_func=label_for)

    inspecting = selection != ALL_CELLS

    with header_left:
        views = (["This cell", "All cells", "Original"] if inspecting
                 else ["All cells", "Original"])
        default = "This cell" if inspecting else "All cells"
        # Key varies with the selection so the control resets to "This cell"
        # when a new sperm is picked, instead of holding a stale view.
        view = st.segmented_control(
            "View", views, default=default, label_visibility="collapsed",
            key=f"view-{selection}",
        ) or default

    video_col, detail_col = (st.columns([3, 2], gap="large") if inspecting
                             else (st.container(), None))

    with video_col:
        _render_main_video(view, selection, paths, stem, inspecting)

    if inspecting and detail_col is not None:
        with detail_col:
            _render_cell_detail(df, paths, selection)


def _render_main_video(view: str, selection, paths: dict[str, Path], stem: str,
                       inspecting: bool) -> None:
    """Render whichever video the current view calls for, autoplaying."""
    if view == "Original":
        caption, source = "Raw microscope footage, no overlay.", paths["source"]
    elif view == "This cell" and inspecting:
        caption = (f"Only sperm {selection} is marked — every other cell is left "
                   "unlabelled. Its path builds up behind it.")
        source = _highlight_clip(paths, stem, int(selection))
        if source is None:
            st.info("Could not render a clip for this cell; showing all cells instead.")
            source = ensure_browser_playable(paths["tracked"])
            caption = "Amber = head, cyan = neck, with each cell's tracking ID."
    else:
        caption = "Amber = head, cyan = neck, with each cell's tracking ID."
        source = ensure_browser_playable(paths["tracked"])

    st.caption(caption)
    # muted is required — browsers block autoplay on videos with audio.
    st.video(str(source), autoplay=True, muted=True, loop=True)


def _highlight_clip(paths: dict[str, Path], stem: str, track_id: int) -> Path | None:
    """Path to this cell's clip, rendering and transcoding it on first request."""
    if not paths["trajectories"].exists():
        return None
    trajectory = load_trajectories(paths["trajectories"]).get(track_id)
    if trajectory is None:
        return None

    clip = HIGHLIGHT_DIR / f"{stem}_id{track_id}.mp4"
    if not clip.exists():
        with st.spinner(f"Preparing the clip for sperm {track_id}…"):
            if render_highlight(paths["source"], trajectory, track_id, clip) is None:
                return None
    return ensure_browser_playable(clip)


def _render_cell_detail(df: pd.DataFrame, paths: dict[str, Path], track_id) -> None:
    """The selected cell's grade, kinematics and path."""
    row = df[df["track_id"] == track_id].iloc[0]
    grade = row.get("motility", "")

    st.markdown(f"##### Sperm {track_id} — {GRADE_LABEL.get(grade, grade)}")
    st.markdown(f'<span class="casa-caption">{GRADE_HELP.get(grade, "")}</span>',
                unsafe_allow_html=True)

    if not row.get("plausible", True):
        st.warning(
            f"**Measurement rejected** — {row['vcl_um_s']:.0f} µm/s exceeds the "
            f"{MAX_PLAUSIBLE_VCL:.0f} µm/s physical limit for human sperm. The tracker most "
            "likely swapped this cell with another crossing its path, so these numbers are not "
            "trustworthy. Excluded from the sample percentages."
        )

    numeric = ["vcl_um_s", "vsl_um_s", "vap_um_s", "lin", "str", "wob", "alh_um", "bcf_hz"]
    present = [c for c in numeric if c in row.index]
    # Two per row: this column is narrow, four would wrap the labels.
    for start in range(0, len(present), 2):
        cols = st.columns(2)
        for col, column in zip(cols, present[start:start + 2]):
            label, explanation = METRIC_HELP[column]
            unit = " µm/s" if column.endswith("_um_s") else (
                " µm" if column.endswith("_um") else (" Hz" if column.endswith("_hz") else ""))
            col.metric(label.split(" (")[0], f"{row[column]:.2f}{unit}", help=explanation)

    st.caption(f"Tracked across {int(row['frames'])} frames "
               f"({int(row['frames']) / 49.0:.1f} seconds).")

    if not paths["trajectories"].exists():
        return
    trajectory = load_trajectories(paths["trajectories"]).get(int(track_id))
    if trajectory is not None:
        with st.expander("Path travelled", expanded=True):
            st.altair_chart(charts.trajectory_chart(trajectory["head"]),
                            use_container_width=True)


def render_table(df: pd.DataFrame) -> None:
    st.markdown("#### All measured cells")
    st.caption("Click a column header to sort. Every figure is explained in the "
               "Overview tab's tooltips.")

    grades = ["All"] + [GRADE_LABEL[g] for g in GRADE_LABEL if (df["motility"] == g).any()]
    chosen = st.radio("Filter", grades, horizontal=True, label_visibility="collapsed")
    view = df if chosen == "All" else df[
        df["motility"] == next(k for k, v in GRADE_LABEL.items() if v == chosen)]

    display = view.rename(columns={
        "track_id": "ID", "frames": "Frames", "motility": "Grade", "plausible": "Reliable",
        "vcl_um_s": "VCL µm/s", "vsl_um_s": "VSL µm/s", "vap_um_s": "VAP µm/s",
        "lin": "LIN", "str": "STR", "wob": "WOB", "alh_um": "ALH µm", "bcf_hz": "BCF Hz",
    })
    if "Grade" in display:
        display["Grade"] = display["Grade"].map(GRADE_LABEL).fillna(display["Grade"])

    st.dataframe(
        display, use_container_width=True, hide_index=True,
        column_config={c: st.column_config.NumberColumn(format="%.2f")
                       for c in display.columns
                       if display[c].dtype.kind == "f"},
    )
    st.caption(f"{len(view)} of {len(df)} cells shown. "
               "`Reliable = false` marks tracks rejected as measurement artifacts.")


def render_upload(tracker_config: TrackerConfig) -> None:
    """Sidebar upload. Runs the same pipeline the samples went through."""
    # WMV is accepted so footage never has to be converted before upload.
    # A clip converted by a desktop tool arrived with its contrast crushed into
    # 10 grey levels (against 35-63 in the reference clips), its frame rate
    # written as 1000 fps and its colour range dropped — the detector found
    # nothing in it. Decoding the original here avoids that whole class of
    # damage; ffmpeg is already installed for the browser transcode.
    uploaded = st.file_uploader(f"Video file (max {MAX_UPLOAD_MB} MB)",
                                type=["mp4", "avi", "mov", "wmv", "mkv"])
    if uploaded is None:
        return

    size_mb = uploaded.size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        st.error(f"{size_mb:.0f} MB — over the {MAX_UPLOAD_MB} MB limit.")
        return

    st.caption(f"**{uploaded.name}** · {size_mb:.1f} MB")
    if not st.button("Run analysis", type="primary", use_container_width=True):
        st.caption("The page is frozen while this runs and the tab must stay "
                   "open. On the GPU a 30-second clip takes about half a "
                   "minute; long recordings take proportionally longer.")
        return

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = INPUT_DIR / uploaded.name
    destination.write_bytes(uploaded.getbuffer())

    config = Config(conf=tracker_config.track_low_thresh, output_dir=OUTPUT_DIR)
    config.draw.show_conf = False
    config.draw.trail_length = 0

    with st.spinner("Analysing… the page stays frozen until this finishes."):
        try:
            detector = SpermDetector(config)
            run(detector, str(destination), config, track=True, metrics=True,
                tracker_config=tracker_config, microns_per_pixel=MICRONS_PER_PIXEL)
        except Exception as exc:  # surface the failure instead of a blank page
            st.error(f"Analysis failed: {exc}")
            return

    # A finished run is not the same as a usable one. When the detector finds
    # nothing — footage at a different magnification, say — there are no
    # trajectories, no CSV is written, and the clip cannot appear in the sample
    # list. Saying "Done" and leaving the list unchanged reads as a broken
    # upload, so the reason is stated instead.
    if not (OUTPUT_DIR / f"{destination.stem}_metrics.csv").exists():
        st.error(
            f"**{destination.stem}** was analysed but no sperm were found, so "
            "there is nothing to report.\n\n"
            "The model was trained on 400x phase-contrast footage where a head "
            "is a bright spot roughly 16 px across with a visible tail. Video "
            "at a lower magnification — cells as small dark dots — looks "
            "nothing like that to it and will come back empty. The annotated "
            "video is still saved if you want to look at it."
        )
        return

    load_metrics.clear()
    st.success(f"Done — **{destination.stem}** is now in the sample list.")
    st.rerun()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    if not check_password():
        return

    videos = processed_videos()
    pending, found_nothing = unprocessed_videos()

    with st.sidebar:
        st.markdown("### 🔬 Sperm CASA")
        st.markdown('<span class="casa-caption">Computer-assisted sperm analysis</span>',
                    unsafe_allow_html=True)
        st.divider()

        if videos:
            stem = st.selectbox("Sample", sorted(videos))
        else:
            stem = None
            st.info("No analysed videos yet — upload one below.")

        if pending:
            st.caption("Not yet analysed: " + ", ".join(p.stem for p in pending))
        if found_nothing:
            st.caption("Analysed, no sperm found: " + ", ".join(p.stem for p in found_nothing))
            st.caption("Re-uploading will not help — the footage is at a "
                       "different magnification from what the model was "
                       "trained on. See the upload panel for details.")

        st.divider()
        st.markdown("**Analyse a new video**")
        min_track_len = st.session_state.get("min_track_len", 10)
        render_upload(TrackerConfig(min_track_length=int(min_track_len)))

        st.divider()
        with st.expander("Analysis settings"):
            st.caption("Applies to new uploads only.")
            st.number_input("µm per pixel", value=float(MICRONS_PER_PIXEL), format="%.4f",
                            step=0.005, disabled=True,
                            help="Calibrated for the reference rig: Olympus CX31 at 400x with "
                                 "an IDS UI-2210C camera. Change it in utils/config.py if your "
                                 "setup differs — every velocity scales with it.")
            st.number_input(
                "Minimum track length (frames)", value=10, min_value=2, key="min_track_len",
                help="Cells tracked for fewer frames than this are ignored — too few points to "
                     "measure a velocity from.")

    if stem is None:
        st.title("Sperm CASA")
        st.caption("Upload a video in the sidebar to get started.")
        return

    paths = videos[stem]
    df = load_metrics(str(paths["csv"]), paths["csv"].stat().st_mtime)

    st.title(f"Sample {stem}")
    st.markdown(
        f'<span class="casa-caption">{len(df)} cells tracked · '
        f'{MICRONS_PER_PIXEL} µm/pixel · 49 fps</span>',
        unsafe_allow_html=True,
    )
    st.write("")

    tabs = st.tabs(["Overview", "Video & cells", "All cells"])
    with tabs[0]:
        render_dashboard(df, paths, stem)
    with tabs[1]:
        render_video_viewer(df, paths, stem)
    with tabs[2]:
        render_table(df)


if __name__ == "__main__":
    main()
