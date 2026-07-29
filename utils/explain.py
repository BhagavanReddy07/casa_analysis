"""Plain-language descriptions of CASA metrics.

The dashboard is for people who do not read VCL/STR/WOB as words. Every
number shown to a viewer gets one of these next to it.
"""

from __future__ import annotations

METRIC_HELP: dict[str, tuple[str, str]] = {
    # column name -> (display label, plain-language explanation)
    "vcl_um_s": ("Curvilinear velocity (VCL)",
                 "Speed along the actual wiggly path the cell swam. High means energetic."),
    "vsl_um_s": ("Straight-line velocity (VSL)",
                 "Speed measured start-point to end-point, ignoring the wiggle. "
                 "High means it actually got somewhere."),
    "vap_um_s": ("Average path velocity (VAP)",
                 "Speed along a smoothed version of the path, with the frame-to-frame "
                 "jitter removed."),
    "lin": ("Linearity (LIN)",
            "VSL divided by VCL. Near 1 = swims almost straight. Near 0 = thrashes in place."),
    "str": ("Straightness (STR)",
            "VSL divided by VAP. How directly the cell followed its own average course."),
    "wob": ("Wobble (WOB)",
            "VAP divided by VCL. How much the head oscillates side to side while moving."),
    "alh": ("Head displacement (ALH)",
            "How far the head swings sideways off its average path, in micrometres."),
    "alh_um": ("Head displacement (ALH)",
               "How far the head swings sideways off its average path, in micrometres."),
    "bcf_hz": ("Beat frequency (BCF)",
               "How many times per second the head crosses its own average path. "
               "Undersampled at 49 fps — treat as indicative only."),
    "frames": ("Frames tracked",
               "How many video frames this cell was followed for. More frames = more reliable."),
}

GRADE_HELP: dict[str, str] = {
    "progressive": "Swimming actively and making real forward progress. The cells that matter most for fertility.",
    "non_progressive": "Moving, but not getting anywhere — swimming in tight circles or thrashing in place.",
    "immotile": "Not moving.",
    "unreliable": ("Measurement rejected: this cell's recorded speed is physically impossible, "
                   "usually because the tracker briefly confused it with another cell crossing "
                   "its path. Excluded from the grades above."),
}

GRADE_LABEL: dict[str, str] = {
    "progressive": "Progressive",
    "non_progressive": "Non-progressive",
    "immotile": "Immotile",
    "unreliable": "Unreliable",
}

# WHO 6th edition (2021) lower reference limits — the 5th centile of a
# fertile reference population, NOT a pass/fail line for an individual. A
# sample below them is not "abnormal"; it is below the reference range and
# needs clinical interpretation. Shown for orientation only.
#
# Also: these limits are defined for a full manual/CASA assay of a whole
# ejaculate. This tool measures the cells visible in one ~30 s field, which
# is a much smaller and non-random sample, so the comparison is indicative
# rather than diagnostic.
WHO_REFERENCE = {
    "progressive_pct": (30.0, "WHO 6th ed. lower reference limit for progressive motility"),
    "total_motile_pct": (42.0, "WHO 6th ed. lower reference limit for total motility "
                               "(progressive + non-progressive)"),
}

WHO_DISCLAIMER = (
    "WHO reference limits describe the 5th centile of a fertile population and apply to a "
    "full ejaculate assay. This measures one microscope field, so treat the comparison as "
    "orientation, not diagnosis."
)
