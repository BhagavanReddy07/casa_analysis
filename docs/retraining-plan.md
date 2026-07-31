# Plan: make the model work on any microscope, not just VISEM

The model today reads one imaging setup — 400x phase contrast, bright heads,
strong contrast — because that is all it has ever seen. On footage from a
different rig it finds nothing at all, which is what happened with the first
clip from the new microscope: 18,525 frames, zero detections.

This is the plan to fix that. It is written so the parts you do by hand and the
parts I build are never tangled.

---

## The rule that matters most

**One annotation convention, used everywhere, forever.**

Your three existing sets already disagree with each other:

| set | clip | boxes |
|---|---|---|
| sperm1 | 22.mp4 | head-sized, 7-20 px |
| sperm2 | 30.mp4 | 58 x 31 px |
| sperm3 | 38.mp4 | whole-cell, rotated, ~58 px |

Training on a mixture teaches the model two contradictory ideas of what a box
means. It also breaks tracking silently: `dedupe_distance`, `motion_gate` and
`claim_distance` are all measured in pixels against **head-sized** boxes, so a
model that starts predicting whole-cell boxes invalidates every one of them.

**The convention, from now on:**

* a **tight box around the head only** — the bright (or dark) oval, not the tail;
* **two keypoints**: point 1 on the head centre, point 2 on the neck where the
  head meets the tail;
* an **agglutinated clump is one cell**, one box, one head point — the head you
  judge to be most in focus;
* a cell whose head is **not fully visible** (cut by the frame edge, or hidden
  under another cell) is **not annotated**;
* nothing else in the frame is annotated — no debris, no epithelial cells.

The keypoints are not optional. Every CASA figure — VCL, VSL, VAP, ALH, BCF —
is computed from the head point's path. A box-only dataset would produce a
detector that cannot drive this pipeline.

---

## Step 1 — Collect footage (you)

For each **imaging type** — meaning a distinct combination of microscope,
objective, camera and illumination — gather **5 to 9 videos**.

Deliberately vary what you can: focus, illumination, cell density, amount of
debris, and samples from different patients or animals. A model fails on the
conditions it never saw, not on the ones it saw fewer times.

Two acquisition checks before you record a lot of footage. Both were real
problems on the first clip from the new rig:

1. **Contrast.** Standard deviation of pixel values should be **8 or higher**
   across the field. The reference clips measure 8.2-10.4; the new rig
   measured 3.2, which is why nothing was visible. Low contrast usually means
   a phase ring that does not match the objective, or under-illumination with
   camera gain compensating. Fix it at the microscope — no software recovers
   detail that was never recorded.
2. **Frame rate.** Confirm what the camera *actually* records. The new clip
   claims 1000 fps in its metadata, which is certainly wrong. Every velocity
   scales directly with this number, so a clip labelled 1000 that really ran
   at 30 would report speeds 30x too high.

I will check both for you on any clip — one command, one minute.

---

## Step 2 — Annotate (you)

**Per video: 25-50 frames, spread across the clip, never consecutive.**

In CVAT, when creating the task, set **Frame step = 10** (or 20 for long
clips). CVAT then shows only every 10th frame; track mode still interpolates,
so the work is identical to annotating 50 consecutive frames but covers ten
times as much of the recording. Fifty frames in a row is one second of footage
where nothing changes — perhaps ten genuinely different examples for fifty
frames of clicking.

**Per imaging type: 200-400 frames total**, across 4-8 videos.

**Hold one whole video per type completely out.** Do not annotate it, do not
train on it. It is the only honest test of whether the model works on footage
it has never seen.

Label setup in CVAT: use a **skeleton** label with two nodes named `head` and
`neck` (your sperm3 project already has this defined), plus the box. Export as
**`CVAT for images 1.1`** — *never* YOLO 1.1, which silently drops rotated
shapes and lost a third of your sperm3 annotation.

Effort: roughly an evening per imaging type.

---

## Step 3 — Build the dataset (me)

A script that takes the CVAT exports and produces a training set:

* reads `annotations.xml` — boxes, rotated boxes and skeletons alike;
* converts to YOLO-pose format (`class cx cy w h  headx heady 2  neckx necky 2`);
* **splits by video, never by frame** — frames from one clip are near
  duplicates, so a frame-level split lets the validation set memorise the
  training set and report a score that means nothing;
* **keeps the existing VISEM data in the mix.** Fine-tuning only on new footage
  makes the model forget the old rig, quickly and completely. Training on both
  keeps it working everywhere;
* writes `data.yaml` and reports the composition so imbalances are visible
  before training rather than after.

---

## Step 4 — Fine-tune (me)

Start from `models/best.pt` rather than from scratch — the model already knows
what a sperm is; this is adaptation, not new learning. Lower learning rate,
early stopping on the validation set, and the result is written to a new file
so the current model is never overwritten until it is beaten.

---

## Step 5 — Judge it honestly (me)

Three tests, and the model must pass all three:

1. **Detection on the held-out video.** Box mAP50 and pose mAP50 on footage
   never trained on.
2. **No regression on the old rig.** The current model gets 99.4% recall on
   22.mp4. If a retrained model does worse there, it has forgotten more than it
   learned.
3. **Tracking, end to end.** `python -m evaluation.score --dataset spermN`
   reports identity switches against the ground truth you already annotated.
   Detection changes move these numbers; a model with better mAP that produces
   more identity switches is not an improvement.

If the box convention changes despite the rule above, the tracking thresholds
must be re-swept (`--sweep`) before these numbers mean anything.

---

## Step 6 — Ship it (me, automatic)

Replace `models/best.pt`, push. The deploy syncs the code, the pipeline
fingerprint changes, and every clip on the server re-analyses itself on the
GPU — about 100 seconds for four clips. The dashboard updates as each one
finishes.

---

## What can still go wrong

* **Preprocessing parity.** If contrast enhancement is used to make annotation
  bearable, the identical enhancement must run at inference. Train on enhanced
  frames, feed raw ones in production, and the model fails exactly as it does
  today.
* **The clump decision.** Labelling an agglutinated cell as one sperm teaches
  the model to stop splitting it into two or three detections — which is what
  currently inflates counts on 38.mp4 (904 duplicate suppressions against 78 on
  22.mp4). Worth doing, but only if it is done consistently.
* **Too little diversity per type.** 200 frames from four near-identical videos
  is roughly four examples. Vary the conditions, not the frame count.
* **Contrast below the floor.** If a rig cannot produce cells distinguishable
  from noise, annotation cannot fix it. Check the contrast figure before
  spending an evening labelling.

---

## Order of work

| # | who | what | effort |
|---|---|---|---|
| 1 | you | collect 5-9 videos per imaging type | — |
| 1b | me | check contrast and frame rate on each | minutes |
| 2 | you | annotate 200-400 frames per type, frame step 10 | ~1 evening per type |
| 3 | me | dataset conversion, split, merge with VISEM | 1 day |
| 4 | me | fine-tune | hours on the T4 |
| 5 | me | evaluate: held-out, old rig, tracking | 1 day |
| 6 | me | deploy | automatic |

Start with **one** imaging type end to end before annotating the rest. If
something in this plan is wrong, it is much cheaper to find out after one
evening of labelling than after four.
