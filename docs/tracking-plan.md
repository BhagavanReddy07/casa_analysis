# Plan: stop the IDs from swapping

The problem, in one line: when two sperm meet, the tracker sometimes hands cell
A's number to cell B, and when a swimming cell passes a dead one it sometimes
steals the dead one's number.

This plan fixes that in six steps. Each step says **what you do by hand** and
**what I build**. The steps are in dependency order — step 1 makes step 4
possible, and nothing can be judged at all until step 0 exists.

Effort is my estimate of calendar time, assuming you do your part when it comes
up.

---

## Where we are today

Already done (commits `1d3fb82` … `ef95cb3`):

* the association cost now includes how far a detection sits from where the
  track was predicted to be, not just box overlap;
* two cells passing close are no longer collapsed into one detection, so an
  identity stops dying at every crossing;
* a track that has not moved refuses to be dragged along by a passer-by.

Measured on 600 frames of each clip: on `38.mp4` (the crowded one) identity
reversals went **8 → 2** and 220 fewer real cells were deleted as duplicates.
On the other three clips it was roughly neutral.

What is still broken, and why the current approach cannot fix it: **when two
cells overlap, the detector emits one box, not two.** At that moment the
tracker is not choosing the wrong cell out of two candidates — there is only
one candidate. No amount of cost tuning fixes that. Steps 1 and 4 do.

---

## Where this actually landed (2026-07-31)

**The tracker makes 2-3 identity errors in 501 frames of 22.mp4**, against a
ground truth that owes nothing to the tracker. That is the headline, and it is
much better than it looked all of the previous day, because the measurement was
wrong before it was right.

Run it with::

    python -m evaluation.score            # switches, IDF1, MOTA, and where they are
    python -m evaluation.score --sweep    # every setting, one line each

### The measurement was the hard part

Three keys were built before one was trustworthy:

| key | how identities were assigned | switches reported | verdict |
|---|---|---|---|
| v1 | copied from our tracker onto each box | 6 | circular, and unstable where boxes overlap |
| v2 | boxes linked to each other over time | 13 | tracker-independent, but overlap still flips the match |
| v3 | head point per box from the pixels, then linked | **2-3** | zero teleports; trustworthy |

The trap in v1 and v2 is the same: the annotations are whole-cell boxes about
58 px across that overlap their neighbours, so "which cell does this detection
belong to" has no stable answer from boxes alone — and the ambiguity is worst
at crossings, which is precisely what is being measured. v3 fixes it without
new annotation: a sperm head is the brightest point under phase contrast, so
each box only has to *locate* a head the pixels then pin down.

A day of tuning was done against v1 before this was noticed. Every conclusion
from it was re-derived against v3, and most did not survive.

### What survived, with numbers

| change | effect on 22.mp4 |
|---|---|
| Kalman-prediction distance in the association cost | switches 5 -> 3 |
| keeping both cells when two identities pass close | 9 fewer missed cell-frames, 3 fewer fragmentations, +1 switch |

### What was tried and removed, with numbers

Kept here so none of it is attempted again by accident.

* **A stationary rule** — "a cell that has not moved cannot suddenly move".
  Correct in principle, and 44% of near-stationary tracks did contain an
  impossible jump. No effect on any metric at any setting.
* **A heading rule** — "a swimming cell cannot reverse". Direction has no
  signal at 49 fps: the median step is 0.4 px and 43% of measured turn angles
  exceed 90 degrees, i.e. jitter. Forcing it cost 79 position jumps and half
  again as many identities.
* **Orientation as a fingerprint** — the strongest signal found all night: a
  cell's head-to-neck axis moves a median **0.9 degrees** per frame while two
  neighbouring cells differ by a median **83 degrees**, so 87% of neighbouring
  pairs are further apart than one cell ever drifts. It changed nothing,
  because at the frames that fail the per-frame association is already correct.
  It is the obvious tool for matching identities *across a gap*, which is where
  it should be tried next.
* **Withholding contested detections** during a merge — no switches gained, 7
  more missed cell-frames.
* **Offline swap repair** at crossings — every setting that fired made things
  worse (6 switches became 8-10); the only safe threshold was one that never
  fired.
* **Detector settings** — NMS from 0.5 to 0.9 cut merges 45 to 36 but added
  45% more detections; imgsz 1280 cut merges to 29 while quadrupling false
  positives. Neither is a win.

### The three remaining errors

* **frame 36** — two cells 34 px apart during a long crossing.
* **frame 129** — two cells 23 px apart; caused by keeping a real cell that
  then takes a fresh identity. The alternative is deleting it, which trades one
  switch for nine missed cell-frames.
* **frame 164** — a cell at the top edge of the frame (y = 3), half out of
  view. Arguably not an error at all: a cell leaving the frame has ended.

None of these is a cost-function problem. Two are genuine ambiguity at a
crossing, one is an edge artefact.

### Validated on a second clip

A 301-frame annotation of 30.mp4 arrived after the above and was scored without
retuning anything. The key built clean again (32 identities, zero teleports),
and **every setting in the sweep was flat at 4 switches** — no alternative beat
the defaults, and the defaults were best-or-equal on fragmentation and IDF1. So
they are not overfitted to 22.mp4.

| clip | frames | identity switches | what they are |
|---|---|---|---|
| 22.mp4 | 501 | 3 | two crossings 23-34 px apart, one cell at the frame edge |
| 30.mp4 | 301 | 4 | one close pair (cells 6 and 9, 11-19 px apart), one last-frame artefact |

Roughly one error per 150-200 frames, every one of them at genuine near-contact
between two cells. There is no setting left to turn: the sweep is flat on both
clips.

### Honest limits

* Two clips, both of the calmer ones. 38.mp4 has the most crossings and no
  ground truth at all. Numbers here do not transfer to it.
* The "false positives" are real sperm the annotator never boxed — 520 on
  22.mp4, and 1,385 on 30.mp4 where only about 60% of cells were labelled.
  They are constant across settings so comparisons hold, but absolute MOTA is
  meaningless and should not be quoted.
* The tail of the key — 26 identities from 4,814 boxes — is only as good as
  the head extraction, which assumes the brightest pixel in a box is that
  cell's head. It has zero teleports, which is the best available check.

---

## Step 0 — Ground truth — **done for 22.mp4**

The 501 hand-annotated frames in `sperm1/` turned out to be frames 0-500 of
22.mp4. They had boxes but no identities, so `evaluation/track_eval.py`
prefilled identities from the tracker, flagged the 207 doubtful moments,
rendered them as a review video, and a human corrected five of them
(`data/raw/corrections.csv`). 489 cells the annotator never boxed were added
from the model's own detections after checking the crops were real sperm.

First real measurement, on the corrected key:

| | ID switches | IDF1 | MOTA | fragmentations |
|---|---|---|---|---|
| tracker before 2026-07-30 | 10 | 0.982 | 0.978 | 11 |
| tracker after | **6** | 0.975 | 0.979 | 9 |

**−40% identity switches**, which is the number the whole plan is about. IDF1
is marginally worse because the new version reports a few more false positives
(58 against 52); it also misses fewer real cells (44 against 54).

One caveat to remember when reading these: the key was prefilled *by* the
current tracker and then corrected, which biases in its favour. The bias is
mild — the older tracker still scores better on IDF1 — but a second video
labelled from scratch would settle it.

The rest of this section is what it took, and what a second video would need.

### The original plan

**Why.** Right now neither of us can prove a change helped. Everything I
measured today used stand-ins: "a track reversed direction" or "a dead cell
jumped". The real measure is **ID switches** — how many times a number moved to
the wrong cell — and it needs videos where the correct answer is known.

**You do.** Two things:

1. Download the free **VISEM-Tracking** dataset (20 clips, 29,196 frames, every
   sperm already numbered by hand). No labelling needed — it exists so people
   can compare trackers.
2. Label ~300 frames of *your own* clips: draw a box on each sperm and give it
   a number that stays the same across frames. Use CVAT (free, browser-based) —
   it interpolates between frames, so 300 frames is a few hours, not days.
   Your rig and optics differ from the public dataset, which is why this matters.

**I do.** Wire in **TrackEval** (the standard scoring library) and add one
command that prints HOTA, IDF1 and ID-switch counts for any tracker setting.
After that, every change is a number instead of an opinion.

**Done when.** `python -m eval.track` prints a table like
`IDF1 78.2  HOTA 71.4  IDSW 46` and we can re-run it in one minute.

**Effort.** Me: 1 day. You: half a day of labelling.

---

## Step 1 — Teach the detector to see two cells, not one blob

**Why.** This is the ceiling on everything else. When cells overlap the model
outputs a single detection, so one identity is guaranteed to be lost.

**You do.** Correct labels on the frames I give you — I'll pre-fill the boxes,
so mostly you're splitting one box into two and fixing the head points. Budget
~500 frames, and they will be the *hard* ones only (crossings, clumps, cells
touching), because those are the frames that break tracking. Random frames
teach the model nothing new.

**I do.** Three things:

1. A **crossing miner**: a script that replays your existing videos and dumps
   every frame where two identities came within a few pixels, a track died
   mid-frame, or the duplicate filter deleted something — as ready-to-edit
   label files. This is what turns "annotate the hard cases" into a concrete
   folder of ~500 frames instead of a vague idea.
2. Change the detector's overlap filter (NMS) to work on **head keypoints**
   instead of boxes. Two crossing sperm have overlapping boxes but clearly
   separate heads, and right now the box rule deletes one of them before
   tracking ever sees it.
3. Fine-tune the model on your corrected frames and compare against step 0's
   numbers.

**Done when.** On the mined crossing frames, the detector emits two boxes where
it used to emit one, and IDSW from step 0 drops.

**Effort.** Me: 2 days plus training time. You: a day of correcting labels.

---

## Step 2 — A motion model that knows how each cell behaves (IMM)

**Why.** This is your idea, done properly. Today every track uses one motion
model: "it keeps going the way it was going". A dead cell and a fast swimmer
get the same treatment. An **Interacting Multiple Model** filter runs three
models per cell at once — *not moving*, *swimming straight*, *turning* — and
keeps a running probability of which one is true. A cell that has been dead for
200 frames carries a strong "not moving" belief, and one frame of a swimmer
passing over it cannot overturn that.

That accumulated belief is exactly what my quick version lacked: I used a hard
threshold over 10 frames, and at 49 fps that window is mostly camera noise.

**You do.** Nothing. This is internal.

**I do.** Replace the single Kalman filter with the three-model bank, and
report per-cell model probabilities so the dashboard can show *why* a cell was
graded immotile.

**Done when.** IDSW drops again on step 0's benchmark, and immotile cells stop
acquiring velocity when traffic passes them.

**Effort.** Me: 2-3 days.

---

## Step 3 — Gates that size themselves

**Why.** I currently hard-code "15 pixels" as the distance beyond which a match
looks wrong. That is too tight for a fast cell and far too loose for a dead
one. With step 2 in place, each track knows its own uncertainty, so the gate
can be derived from it (Mahalanobis distance) instead of guessed.

**You do.** Nothing.

**I do.** Swap the fixed threshold for the statistical one, and delete the
hand-tuned constants that replaces.

**Effort.** Me: half a day (only after step 2).

---

## Step 4 — Say "occluded" instead of guessing

**Why.** This is your second idea, and the highest-value change that does not
need retraining. When two cells merge into one blob, the honest answer is *"I
cannot see them separately right now"* — not "give the blob to whichever track
is nearest". Today somebody always gets that blob, and that is the swap.

**You do.** Nothing, except decide one policy question: during an occlusion,
should the CASA metrics (VCL etc.) treat the gap as interpolated motion, or
exclude those frames? My recommendation is exclude — invented positions
inflate velocity.

**I do.** Detect the occlusion (two tracks' gates overlap, one detection
available), assign the detection to **neither**, coast both cells on their
predicted motion, and mark those frames as occluded. When the cells separate,
each re-attaches to the one that matches its own motion. Occluded frames get
flagged in the trajectory file so metrics can skip them.

**Done when.** In a crossing, both numbers survive and stay on their original
cells, with the overlap frames marked rather than faked.

**Effort.** Me: 1 day.

---

## Step 5 — Decide identities with the whole video in hand

**Why.** The tracker currently decides frame by frame, in order, and can never
revise. But you are analysing files, not a live camera. Offline, the standard
approach is to build short reliable fragments first and then stitch them
together using the whole clip — a global optimisation (min-cost flow) rather
than a greedy choice. Direction also becomes usable again here, because it is
measured over 10-30 frames instead of one, which is where my per-frame heading
rule failed.

**You do.** Nothing.

**I do.** Add an offline pass: build tracklets, score every possible
join by motion continuity, solve the assignment globally, and fill the
occlusion gaps by interpolation.

**Effort.** Me: 2-3 days.

---

## Step 6 — Audit and repair (do this early, it pays immediately)

**Why.** Published work gets **−25% ID switches** from a verification layer
alone, without touching the tracker. The idea: after tracking finishes, look
for things that cannot physically happen — a cell that teleported, a dead cell
that suddenly moved, two cells that exchanged velocities at the moment they
touched — and re-solve just those windows.

**You do.** Review what the audit flags. It should be a handful of cells per
clip, not hundreds. This is the last mile to "no wrong IDs at all" — a machine
cannot certify that, a person can.

**I do.** The audit rules, the local re-solve, and a review screen in the
dashboard: it shows the flagged crossing, the two candidate readings, and you
click the correct one. Corrections are saved alongside the trajectories.

**Effort.** Me: 1-2 days (audit) plus 1 day (review screen).

---

## What we are deliberately not doing

* **Appearance / re-identification networks (DeepSORT-style).** Sperm heads are
  interchangeable grey blobs. The published attempt with a ResNet50 ReID model
  went from 4 ID switches to 5 — its real gains came from the detector. Skip.
* **Transformer trackers (MOTR and friends).** They need far more labelled
  video than you will have.
* **Per-frame heading rules.** Tried and removed today, with numbers: at 49 fps
  the median step is 0.4 px, 43% of measured turn angles exceed 90 degrees, and
  the neck-to-head axis sits a median 46-63 degrees off the true travel
  direction. Direction only works over longer baselines — hence step 5.

---

## Suggested order

If you want results soonest rather than the full rebuild:

**0 → 4 → 6 → 1 → 2 → 3 → 5**

Step 4 and step 6 are a day each and need no retraining. Step 1 needs your
labelling day but lifts the ceiling everything else runs into. Steps 2, 3 and 5
are the deeper work, worth doing once the measurements from step 0 show what is
left.

---

## References

* [EKF identity reassignment on BoT-SORT (Sensors, 2025)](https://doi.org/10.3390/s25247539) — the verification layer in step 6; −25% ID switches, IDF1 80.3 → 84.8.
* [Enhanced YOLOv4 + improved DeepSORT (Sci Rep, 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12660916/) — evidence that appearance features do not help identity; detector work does.
* [VISEM-Tracking dataset](https://arxiv.org/abs/2212.02842) — the labelled video for step 0.
* [Multi-sperm tracking with an interactive motion model (2025)](https://www.sciencedirect.com/science/article/abs/pii/S1746809425006834) — the IMM approach in step 2.
* [Bull sperm tracking / Tracking-Grid](https://www.researchgate.net/publication/351026283_Bull_Sperm_Tracking_and_Machine_Learning-Based_Motility_Classification) — coasting through occlusion on motion angle, step 4.
* [Global data association via network flows (Zhang et al.)](http://vision.cse.psu.edu/courses/Tracking/vlpr12/lzhang_cvpr08global.pdf) — the offline stitching in step 5.
