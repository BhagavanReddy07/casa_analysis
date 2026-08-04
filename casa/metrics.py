"""CASA kinematic metrics.

Standard WHO set: VCL, VSL, VAP, LIN, STR, WOB, ALH, BCF. All of them need a
micrometres-per-pixel scale and the capture frame rate, so those are inputs
here rather than assumptions — see MICRONS_PER_PIXEL in utils/config.py for
where the scale comes from and how uncertain it still is.

Definitions (WHO 6th edition / standard CASA practice):
  VCL  curvilinear velocity  - speed along the actual, frame-by-frame path.
  VSL  straight-line velocity - speed of net displacement, start to end.
  VAP  average path velocity - speed along a smoothed version of the path,
       which removes pixel-level detection jitter that VCL would otherwise
       count as extra distance travelled.
  LIN  VSL / VCL  - how straight the raw path is.
  STR  VSL / VAP  - how straight the smoothed path is (STR >= LIN).
  WOB  VAP / VCL  - how much the raw path wobbles around the smoothed one.
  ALH  amplitude of lateral head displacement - how far the head swings
       side to side around the smoothed path, in micrometres.
  BCF  beat-cross frequency - how often per second the raw path crosses the
       smoothed path, in Hz.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tracking.trajectory import Trajectory

# 5-frame running average for the smoothed (VAP) path. This is the window
# size common commercial CASA systems use; it is not derived from this rig.
# ponytail: fixed constant, expose as a parameter if a lab needs to tune it
# against a reference instrument.
SMOOTHING_WINDOW = 5

# Even hyperactivated human sperm top out around 250-300 um/s in the
# literature. A track reporting more than this has not found a fast cell —
# something teleported its head between frames.
#
# The dominant cause turned out to be unlocalized keypoints: YOLO-pose emits
# (0, 0) when it cannot place one, which dragged trajectories to the frame
# corner. That is now fixed at source (Config.min_keypoint_conf), so this
# threshold is no longer the primary defence.
#
# It stays as a backstop for the genuinely hard case this project cannot
# fully solve: two cells crossing paths and swapping identities. Such tracks
# are flagged rather than dropped, so the count stays visible instead of the
# sample silently shrinking.
MAX_PLAUSIBLE_VCL = 300.0


@dataclass(frozen=True)
class KinematicMetrics:
    """Per-sperm kinematics in micrometres and micrometres/second."""

    track_id: int
    vcl: float   # curvilinear velocity
    vsl: float   # straight-line velocity
    vap: float   # average path velocity
    lin: float   # VSL / VCL
    str_: float  # VSL / VAP
    wob: float   # VAP / VCL
    alh: float   # amplitude of lateral head displacement
    bcf: float   # beat-cross frequency
    plausible: bool = True  # False when VCL exceeds MAX_PLAUSIBLE_VCL


def _smooth_path(points: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average, same length as the input.

    Edge-pads by repeating the boundary point before convolving, then trims
    back to the original length. Zero-padding (``np.convolve(..., "same")``
    on its own) would pull the smoothed path toward the origin at the
    endpoints — for a track running through (60, 30) that drags the last
    smoothed point toward 0 instead of 30, so a perfectly straight path would
    come out with STR != 1.
    """
    window = min(window, len(points))
    if window < 2:
        return points.copy()
    pad = window // 2
    kernel = np.ones(window) / window
    padded = np.pad(points, ((pad, pad), (0, 0)), mode="edge")
    smoothed = np.stack([np.convolve(padded[:, d], kernel, mode="valid") for d in range(2)], axis=1)
    smoothed = smoothed[:len(points)]

    # Anchor the endpoints to the true detected positions. Edge-replicate
    # padding still blends real neighbouring points into the first/last
    # outputs, which can pull them off the true start/end — enough, on short
    # tracks, to shrink the smoothed path below the raw straight-line
    # distance and produce STR = VSL/VAP > 1, which is impossible (VAP can
    # never be less than the straight-line distance it's supposed to
    # approximate). Observed on 30.mp4: STR = 1.21 on a 17-frame track.
    smoothed[0] = points[0]
    smoothed[-1] = points[-1]
    return smoothed


def _path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _lateral_deviation(raw: np.ndarray, smoothed: np.ndarray) -> np.ndarray:
    """Signed distance of each raw point from the smoothed path, perpendicular
    to the smoothed path's local direction of travel.

    The sign is what makes a zero-crossing count meaningful for BCF — without
    it, "deviation" would just be a distance and could never cross zero.
    """
    n = len(smoothed)
    tangents = np.zeros_like(smoothed)
    tangents[1:-1] = smoothed[2:] - smoothed[:-2]
    tangents[0] = smoothed[1] - smoothed[0] if n > 1 else [1.0, 0.0]
    tangents[-1] = smoothed[-1] - smoothed[-2] if n > 1 else [1.0, 0.0]

    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tangents /= norms
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)  # rotate 90 degrees

    return np.sum((raw - smoothed) * normals, axis=1)


def compute_metrics(trajectory: Trajectory, fps: float, microns_per_pixel: float) -> KinematicMetrics:
    """Compute the full kinematic set for one trajectory.

    Requires at least 2 points; metrics that need a notion of "wobble" (ALH,
    BCF) degrade gracefully to 0 on very short tracks rather than raising.
    """
    if len(trajectory) < 2:
        raise ValueError(f"track {trajectory.track_id}: need >= 2 points, got {len(trajectory)}")

    # Measured points only. Gaps are interpolated for display and for a
    # continuous stored path (see Trajectory.fill_gaps), but a filled point sits
    # exactly on the straight line between its neighbours, and ALH and BCF are
    # defined by the wobble around that line — letting them in would report
    # less lateral movement the more frames the detector missed. VCL is
    # unaffected either way, since a straight interpolation has the same length
    # as the chord it replaces.
    mask = trajectory.observed_mask
    if int(mask.sum()) < 2:
        raise ValueError(f"track {trajectory.track_id}: fewer than 2 observed points")
    frames = np.array(trajectory.frames, dtype=np.float64)[mask]
    raw = trajectory.head_array[mask] * microns_per_pixel
    smoothed = _smooth_path(raw, SMOOTHING_WINDOW)

    # Use the real elapsed time between the first and last sample, not the
    # sample count. ByteTrack's second association pass exists to recover a
    # cell after a few missed frames, so a trajectory can hold points with
    # real gaps between them (see Trajectory.gaps) — treating those as
    # consecutive frames divides by too little time and inflates every
    # velocity by however many frames were actually missed. This is what
    # produced a reported 4363 um/s on 30.mp4 (15x the fastest speed any
    # human sperm has been recorded at) on a 14-point track that actually
    # spanned far more real frames than that.
    duration_s = (frames[-1] - frames[0]) / fps

    vcl = _path_length(raw) / duration_s
    vsl = float(np.linalg.norm(raw[-1] - raw[0])) / duration_s
    vap = _path_length(smoothed) / duration_s

    lin = vsl / vcl if vcl else 0.0
    str_ = vsl / vap if vap else 0.0
    wob = vap / vcl if vcl else 0.0

    deviation = _lateral_deviation(raw, smoothed)
    alh = 2.0 * float(np.mean(np.abs(deviation)))  # peak-to-peak approximation
    crossings = int(np.sum(np.diff(np.sign(deviation)) != 0))
    bcf = crossings / duration_s if duration_s else 0.0

    return KinematicMetrics(
        track_id=trajectory.track_id, vcl=vcl, vsl=vsl, vap=vap,
        lin=lin, str_=str_, wob=wob, alh=alh, bcf=bcf,
        plausible=vcl <= MAX_PLAUSIBLE_VCL,
    )


def compute_batch(
    trajectories: dict[int, Trajectory], fps: float, microns_per_pixel: float,
) -> list[KinematicMetrics]:
    """Compute metrics for every trajectory, in track_id order."""
    return [compute_metrics(t, fps, microns_per_pixel) for _, t in sorted(trajectories.items())]


if __name__ == "__main__":
    # ponytail: one self-check instead of a suite — a straight-line track
    # must read as maximally linear, and a zigzag around that same line must
    # show reduced LIN/STR and a nonzero ALH/BCF, which is the whole point of
    # separating VCL from VAP.
    FPS = 49.0
    MPP = 0.495
    N = 60

    straight = Trajectory(track_id=1, frames=list(range(N)),
                          head_points=[(float(i), 0.0) for i in range(N)])
    # 5% tolerance, not exact equality: any moving-average filter distorts a
    # straight line slightly at its boundary points (edge-replicate padding
    # is not the same as continuing the line), so STR/LIN land close to but
    # not exactly 1.0 even for perfectly straight motion.
    m = compute_metrics(straight, FPS, MPP)
    assert abs(m.lin - 1.0) < 0.05, f"straight-line LIN should be ~1.0, got {m.lin}"
    assert abs(m.str_ - 1.0) < 0.05, f"straight-line STR should be ~1.0, got {m.str_}"
    assert m.alh < 0.5, f"straight-line ALH should be near 0, got {m.alh}"
    assert m.vcl > 0, "straight-line VCL should be positive"

    zigzag = Trajectory(track_id=2, frames=list(range(N)),
                        head_points=[(float(i), 3.0 if i % 2 == 0 else -3.0) for i in range(N)])
    z = compute_metrics(zigzag, FPS, MPP)
    assert z.lin < m.lin, "zigzag should be less linear than a straight path"
    assert z.alh > 0, "zigzag should show nonzero lateral amplitude"
    assert z.bcf > 0, "zigzag should show nonzero beat-cross frequency"
    assert z.vcl > z.vap, "raw zigzag path must be longer than its smoothed version"
    assert m.str_ <= 1.0 + 1e-9 and z.str_ <= 1.0 + 1e-9, \
        "STR = VSL/VAP can never exceed 1.0 (VAP is the smoothed path, always >= straight-line distance)"

    # A track that goes 0->10 at t=0, is lost, then reappears at t=0 + 10
    # frames must read the same VCL/VSL/VAP as one with no gap at all — the
    # duration is the true elapsed time between samples, not the number of
    # samples. Getting this wrong (using len(trajectory)-1 instead of
    # frames[-1]-frames[0]) inflated a real 14-point track to a reported
    # 4363 um/s, ~15x the fastest speed ever recorded for human sperm.
    no_gap = Trajectory(track_id=3, frames=[0, 10],
                        head_points=[(0.0, 0.0), (10.0, 0.0)])
    with_gap = Trajectory(track_id=4, frames=[0, 20],
                          head_points=[(0.0, 0.0), (10.0, 0.0)])
    g_no_gap = compute_metrics(no_gap, FPS, MPP)
    g_with_gap = compute_metrics(with_gap, FPS, MPP)
    assert g_with_gap.vcl == g_no_gap.vcl / 2, \
        f"a track spanning 2x the frames for the same displacement should read half the speed: " \
        f"{g_with_gap.vcl} vs {g_no_gap.vcl}"

    # An ID switch teleports the head across the frame in one step. The
    # resulting VCL is physically impossible and must be flagged, not
    # reported as an extremely fast cell.
    assert m.plausible, "a normal straight-line track should be plausible"
    # Short track, like the real 30.mp4 failure: one jump dominates the
    # average instead of being diluted across many frames.
    teleport = Trajectory(track_id=5, frames=list(range(10)),
                          head_points=[(float(i), 0.0) for i in range(5)]
                          + [(float(i) + 600.0, 0.0) for i in range(5)])
    t_metrics = compute_metrics(teleport, FPS, MPP)
    assert not t_metrics.plausible, \
        f"a track with a 600px single-frame jump should be flagged, got vcl={t_metrics.vcl}"

    print("metrics.py self-check passed")
