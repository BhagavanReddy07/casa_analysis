"""Altair charts for the dashboard.

Altair ships with Streamlit, renders SVG, and gives tooltips for free — so
every chart here is hoverable without extra work.

Palette note: the three motility colours are the first three slots of a
validated categorical palette (aqua / blue / orange). They were checked with
an all-pairs colour-vision-deficiency validator in both light and dark mode
because the scatter puts all three on screen at once:

    dark   worst CVD dE 9.4, worst normal-vision dE 20.9  — pass
    light  worst CVD dE 9.2, worst normal-vision dE 24.0  — pass

The obvious green/amber/red choice was tried first and *failed* — amber and
red sit at normal-vision dE 13.0, under the 15 floor, so a full-colour viewer
struggles to tell them apart. Every chart also carries text labels or a
legend, so grade is never conveyed by colour alone.

Charts encode colour on a pretty-printed ``Grade`` column rather than the raw
``motility`` value, so every legend reads the same and no chart leaks
``non_progressive`` at a viewer.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

# Clinical order, worst-to-best reading left to right in a stack.
GRADE_ORDER_RAW = ["progressive", "non_progressive", "immotile", "unreliable"]
GRADE_LABELS = {
    "progressive": "Progressive",
    "non_progressive": "Non-progressive",
    "immotile": "Immotile",
    "unreliable": "Unreliable",
}
GRADE_ORDER = [GRADE_LABELS[g] for g in GRADE_ORDER_RAW]

# Dark-surface steps; the dashboard renders on a dark surface.
GRADE_COLORS: dict[str, str] = {
    "Progressive": "#199e70",      # aqua
    "Non-progressive": "#3987e5",  # blue
    "Immotile": "#d95926",         # orange
    "Unreliable": "#6b6b68",       # muted — excluded data, deliberately recessive
}

SURFACE = "#14140f"
TEXT_PRIMARY = "#e8e8e4"
TEXT_MUTED = "#9a9a94"
GRID = "#2e2e2c"


def _theme() -> dict:
    """Recessive axes and grid, so the data carries the ink."""
    return {
        "config": {
            "background": "transparent",
            "view": {"stroke": "transparent"},
            "axis": {
                "labelColor": TEXT_MUTED, "titleColor": TEXT_MUTED,
                "gridColor": GRID, "domainColor": GRID, "tickColor": GRID,
                "labelFontSize": 11, "titleFontSize": 11, "titleFontWeight": "normal",
            },
            "legend": {
                "labelColor": TEXT_PRIMARY, "titleColor": TEXT_MUTED,
                "labelFontSize": 11, "titleFontSize": 11, "titleFontWeight": "normal",
                "symbolType": "circle", "symbolSize": 90,
            },
            "text": {"color": TEXT_PRIMARY},
        }
    }


alt.theme.register("casa", enable=True)(_theme)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the pretty label and a numeric sort key for stack ordering."""
    out = df.copy()
    out["Grade"] = out["motility"].map(GRADE_LABELS).fillna(out["motility"])
    out["_order"] = out["motility"].map({g: i for i, g in enumerate(GRADE_ORDER_RAW)})
    return out


def _color(present: list[str], legend: alt.Legend | None) -> alt.Color:
    """Colour scale limited to the grades actually in this chart.

    Without the restriction the legend advertises grades the chart filtered
    out — e.g. the scatter drops unreliable tracks but would still show an
    'Unreliable' swatch.
    """
    domain = [g for g in GRADE_ORDER if g in present]
    return alt.Color(
        "Grade:N",
        scale=alt.Scale(domain=domain, range=[GRADE_COLORS[g] for g in domain]),
        sort=domain,
        legend=legend,
    )


def _empty(mark: str = "bar") -> alt.Chart:
    return getattr(alt.Chart(pd.DataFrame({"x": []})), f"mark_{mark}")()


def motility_bar(df: pd.DataFrame) -> alt.Chart:
    """Single stacked bar: what fraction of measured cells fall in each grade.

    Unreliable tracks are excluded — they are measurement failures, not a
    motility category, and including them would deflate every real share.
    """
    graded = _prepare(df[df["motility"] != "unreliable"])
    if graded.empty:
        return _empty("bar")

    counts = (graded.groupby(["Grade", "_order"], as_index=False)
              .size().rename(columns={"size": "count"})
              .sort_values("_order"))
    total = counts["count"].sum()
    counts["share"] = counts["count"] / total
    counts["pct"] = (100 * counts["share"]).round(0).astype(int).astype(str) + "%"

    # Explicit midpoints. Altair positions stacked text at the segment's
    # upper edge, which clips the label; computing the centre keeps each
    # percentage inside its own band.
    counts["end"] = counts["share"].cumsum()
    counts["mid"] = counts["end"] - counts["share"] / 2

    present = counts["Grade"].tolist()
    # No axis: the segments are directly labelled and the tiles above already
    # give the counts, so a 0-100% scale would be redundant ink.
    x_scale = alt.Scale(domain=[0, 1])

    bar = (
        alt.Chart(counts)
        .mark_bar(height=46, stroke=SURFACE, strokeWidth=2, cornerRadius=3)
        .encode(
            x=alt.X("share:Q", stack="normalize", axis=None, scale=x_scale),
            color=_color(present, None),
            order=alt.Order("_order:Q"),
            tooltip=[alt.Tooltip("Grade:N"), alt.Tooltip("count:Q", title="Cells"),
                     alt.Tooltip("pct:N", title="Share")],
        )
    )
    labels = (
        alt.Chart(counts)
        .mark_text(color="#ffffff", fontWeight="bold", fontSize=13, baseline="middle")
        .encode(
            x=alt.X("mid:Q", scale=x_scale, axis=None),
            # Hide the label on slivers too narrow to hold it.
            text=alt.condition(alt.datum.share > 0.07, alt.Text("pct:N"), alt.value("")),
        )
    )
    return (bar + labels).properties(height=50)


def velocity_scatter(df: pd.DataFrame) -> alt.Chart:
    """VCL against LIN — the standard CASA subpopulation view.

    Fast-and-straight cells sit top-right, fast-but-circling bottom-right,
    immotile bottom-left. Reveals structure a bar chart of averages hides.
    """
    plot = _prepare(df[df["plausible"]] if "plausible" in df else df)
    if plot.empty:
        return _empty("point")

    return (
        alt.Chart(plot)
        .mark_circle(size=120, opacity=0.85, stroke=SURFACE, strokeWidth=1.5)
        .encode(
            x=alt.X("vcl_um_s:Q", title="Curvilinear velocity VCL (µm/s)",
                    scale=alt.Scale(nice=True, zero=True)),
            y=alt.Y("lin:Q", title="Linearity LIN  (0 = thrashing, 1 = straight)",
                    scale=alt.Scale(domain=[0, 1])),
            color=_color(plot["Grade"].unique().tolist(),
                         alt.Legend(title=None, orient="top", direction="horizontal")),
            tooltip=[alt.Tooltip("track_id:Q", title="Sperm ID"),
                     alt.Tooltip("Grade:N"),
                     alt.Tooltip("vcl_um_s:Q", title="VCL µm/s", format=".1f"),
                     alt.Tooltip("vsl_um_s:Q", title="VSL µm/s", format=".1f"),
                     alt.Tooltip("lin:Q", title="LIN", format=".2f"),
                     alt.Tooltip("frames:Q", title="Frames tracked")],
        )
        .properties(height=330)
        .interactive()
    )


def velocity_histogram(df: pd.DataFrame, column: str = "vcl_um_s") -> alt.Chart:
    """Distribution of a velocity measure across reliably measured cells."""
    plot = _prepare(df[df["plausible"]] if "plausible" in df else df)
    if plot.empty:
        return _empty("bar")

    return (
        alt.Chart(plot)
        .mark_bar(stroke=SURFACE, strokeWidth=1.5, cornerRadiusTopLeft=3,
                  cornerRadiusTopRight=3)
        .encode(
            x=alt.X(f"{column}:Q", bin=alt.Bin(maxbins=20),
                    title="Curvilinear velocity VCL (µm/s)"),
            y=alt.Y("count():Q", title="Cells"),
            color=_color(plot["Grade"].unique().tolist(),
                         alt.Legend(title=None, orient="top", direction="horizontal")),
            order=alt.Order("_order:Q"),
            tooltip=[alt.Tooltip("Grade:N"), alt.Tooltip("count():Q", title="Cells")],
        )
        .properties(height=330)
    )


def trajectory_chart(head_points: list[list[float]], width: int = 640,
                     height: int = 480) -> alt.Chart:
    """One cell's swum path, in image coordinates.

    y is inverted because the image origin is top-left; the domain is held to
    the full frame so short paths are not blown up to look like long ones.
    """
    path = pd.DataFrame(head_points, columns=["x", "y"])
    path["step"] = range(len(path))

    x_enc = alt.X("x:Q", scale=alt.Scale(domain=[0, width]), title="x (pixels)")
    y_enc = alt.Y("y:Q", scale=alt.Scale(domain=[height, 0]), title="y (pixels)")

    line = (
        alt.Chart(path)
        .mark_line(color="#3987e5", strokeWidth=2, opacity=0.9)
        .encode(x=x_enc, y=y_enc, order="step:Q",
                tooltip=[alt.Tooltip("step:Q", title="Frame"),
                         alt.Tooltip("x:Q", format=".0f"),
                         alt.Tooltip("y:Q", format=".0f")])
    )
    ends = pd.DataFrame([
        {"x": path.iloc[0]["x"], "y": path.iloc[0]["y"], "Point": "Start"},
        {"x": path.iloc[-1]["x"], "y": path.iloc[-1]["y"], "Point": "End"},
    ])
    markers = (
        alt.Chart(ends)
        .mark_point(size=150, filled=True, stroke=SURFACE, strokeWidth=2)
        .encode(
            x=x_enc, y=y_enc,
            shape=alt.Shape("Point:N",
                            scale=alt.Scale(domain=["Start", "End"],
                                            range=["circle", "square"]),
                            legend=alt.Legend(title=None, orient="top")),
            color=alt.Color("Point:N",
                            scale=alt.Scale(domain=["Start", "End"],
                                            range=["#199e70", "#d95926"]),
                            legend=None),
            tooltip=[alt.Tooltip("Point:N"), alt.Tooltip("x:Q", format=".0f"),
                     alt.Tooltip("y:Q", format=".0f")],
        )
    )
    return (line + markers).properties(height=330)
