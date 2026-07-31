# Sperm CASA

Computer Assisted Sperm Analysis. YOLO11-pose detects each sperm and its head/neck
keypoints; later phases add tracking, trajectories and CASA motility metrics.

## Status

| Phase | Scope | State |
|-------|-------|-------|
| 1 | Dataset + YOLO11-pose training | done — `models/best.pt`, box mAP50 0.994 / pose mAP50 0.989 |
| 2 | Inference + custom CASA-style visualization | done |
| 3 | ByteTrack multi-object tracking | done |
| 4 | Trajectories + CASA metrics + motility classes | done |
| 5 | Streamlit dashboard | done |
| 6 | Morphology, PDF reports | not started |

## Installation

Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

GPU is optional. `pip install -r requirements.txt` pulls CPU torch; for CUDA install
torch first from https://pytorch.org, then the requirements.

## Folder structure

```
Sperm_CASA/
├── models/best.pt          trained YOLO11-pose weights (2 keypoints: head, neck)
├── videos/
│   ├── input/              source clips
│   └── output/             annotated results
├── data/
│   ├── raw/                original frames
│   └── yolo_pose/          training dataset (lives on Colab, not synced here)
├── detection/              detector wrapper + inference loop
├── tracking/               ByteTrack wrapper + trajectory store (Phase 3)
├── casa/                   metrics, motility, morphology (Phase 4-5)
├── utils/                  drawing, config, helpers
├── api/                    FastAPI backend (Phase 5)
├── frontend/               React app (Phase 5)
├── notebooks/              experiments
└── main.py                 entry point
```

`sperm_pose_baseline/` holds the raw Phase 1 training run (curves, confusion matrix,
`weights/best.pt` and `last.pt`). Kept for reference; nothing in the pipeline reads it.

## Running inference

### Dashboard (recommended)

```bash
streamlit run app.py
```

Opens at http://localhost:8501. Pick a video in the sidebar, then:

- **Overview** — motility breakdown and average kinematics, with a plain-language
  tooltip on every figure. CSV and video downloads.
- **Videos** — original and tracked footage side by side.
- **Sperm explorer** — pick any cell to see its metrics, its path plot, and an
  auto-generated clip showing only that cell.
- **Data table** — the full per-cell CSV, sortable.
- **Upload** — analyse a new video (50 MB cap); it then behaves like the
  preloaded ones.

Set `CASA_DASHBOARD_PASSWORD` to require a password; unset means open access.

### Command line

```bash
python main.py --source videos/input/22.mp4                    # detection only
python main.py --source videos/input/22.mp4 --track            # + persistent IDs
python main.py --source videos/input/22.mp4 --track --metrics  # + CASA CSV
python main.py --source videos/input/22.mp4 --show             # preview window, q to quit
python main.py --source 0 --track                              # camera index 0
python main.py --source videos/input/22.mp4 --max-frames 60    # quick check
```

Writes `videos/output/<name>_{annotated,tracked}.mp4`, and with `--metrics`
also `<name>_metrics.csv` and `<name>_trajectories.json`.

Useful flags: `--conf` (0.25, or 0.10 with `--track`), `--iou` (0.5),
`--trail N` (path behind each cell, 0 = off), `--show-conf`, `--device cuda`,
`--verbose`.

The overlay draws an amber head point, a cyan neck point, the head-neck
segment and (when tracking) a small ID. No bounding boxes, no class labels.

On CPU expect 3-6 fps, so a full 1470-frame clip takes 4-7 minutes.

Self-checks:

```bash
python -m utils.draw
python -m tracking.tracker
python -m casa.metrics
python -m casa.motility
```

## Calibration

Source footage is the VISEM dataset: Olympus CX31 at 400x visual (40x objective),
IDS uEye UI-2210C camera, 1/2" CCD, 640x480 native, 9.9 um pixels, ~49 fps.

`MICRONS_PER_PIXEL = 0.495` in [utils/config.py](utils/config.py) — derived as
9.9 / (40 x 0.5). The 0.5x C-mount adapter is inferred from cell size, not
documented; see the comment there. Field of view is ~317 x 238 um.

Confirm with a stage micrometer before publishing any measurement.

## Reading the metrics

Per cell, in `<name>_metrics.csv`:

| Column | Meaning |
|--------|---------|
| `motility` | progressive / non_progressive / immotile / unreliable |
| `plausible` | `False` when the measurement was rejected — see below |
| `vcl_um_s` | speed along the actual wiggly path |
| `vsl_um_s` | speed start-point to end-point, ignoring wiggle |
| `vap_um_s` | speed along the smoothed path |
| `lin`, `str`, `wob` | VSL/VCL, VSL/VAP, VAP/VCL — how straight the swim was |
| `alh_um`, `bcf_hz` | side-to-side head swing and beat frequency |

### Two known limits

**Rejected tracks.** Any track exceeding `MAX_PLAUSIBLE_VCL` (300 um/s, above
the literature maximum for human sperm) is graded `unreliable` and excluded
from the motility percentages, but kept in the CSV so nothing is hidden
silently.

The original cause of these was a detector bug, now fixed: YOLO-pose is
trained with unannotated keypoints encoded as `(0, 0, visible=0)`, so it
predicts the frame origin whenever it cannot localize one, and ultralytics
returns that unmasked in `.xy`. Reading those without checking `.conf` drew
straight lines from the corner of the frame into 2.4% of detections and
inflated their velocities tenfold. `Config.min_keypoint_conf` now discards
them at source.

What remains is the genuinely hard case: two cells crossing and swapping
identities. The threshold stays as a backstop for those.

**ALH and BCF are undersampled.** The flagellar beat is roughly 2 frames per
cycle at 49 fps — at or below the Nyquist limit (WHO guidance wants >=60 fps
for this). VCL/VSL/VAP/LIN/STR/WOB are unaffected, but ALH is likely
underestimated and BCF should be treated as indicative only. This is a
camera frame-rate limit, not something fixable in software.

## Continuous deployment

Every push to `main` redeploys automatically via
[.github/workflows/deploy.yml](.github/workflows/deploy.yml): rsync the code,
reinstall requirements, restart the service, then poll `/_stcore/health` for
up to two minutes. If the app does not come back the run fails red and prints
the last 40 lines of `journalctl` — a green tick on a dead service would be
worse than no CI at all.

Repository secrets required: `EC2_HOST`, `EC2_USER`, `EC2_SSH_KEY`.

The instance uses an Elastic IP (`13.204.5.64`), so the address survives a
stop/start. If it is ever re-allocated, `EC2_HOST` must be updated to match
or every deploy will fail trying to reach the old address.

`videos/` is excluded from the sync, and rsync protects excluded paths from
`--delete`, so a deploy can never remove analysed results from the server —
and no local clip is ever uploaded by a deploy. Only what already sits in
`videos/input` on the instance, i.e. what was uploaded through the dashboard,
is analysed there.

Once the app is healthy the deploy dispatches `main.py --rebuild` on the
instance and returns without waiting. That re-analyses every clip whose
results came from different pipeline code — otherwise a tracking fix would
land while the dashboard kept serving the results it invalidated. Staleness
is decided by a hash of `detection/ tracking/ casa/ utils/*.py` stored in
`videos/output/<name>.build`, so a docs-only push rebuilds nothing. Each clip
takes 15-20 minutes on two vCPUs and appears in the dashboard as it finishes;
watch `~/sperm_CASA/videos/rebuild.log`. A deploy landing mid-rebuild skips
(`flock -n`) rather than starting a second one.

To redeploy the current `main` without a commit, use "Run workflow" on the
Actions tab.

## First-time AWS setup

For a shared demo link. CPU instance is fine — inference is single-threaded,
so a bigger box changes little; the point is a stable URL, not speed.

1. Launch EC2: **c6i.large** or **c7i.large**, Ubuntu 22.04 LTS, 30 GB gp3.
   Security group: inbound TCP 8501. Attach an Elastic IP so the link
   survives stop/start.
2. Copy the repo, skipping training artifacts:
   ```bash
   rsync -av --exclude sperm_pose_baseline --exclude .venv \
         ./ ubuntu@<ip>:~/sperm_CASA/
   ```
3. On the instance:
   ```bash
   sudo apt update && sudo apt install -y python3-pip ffmpeg
   cd ~/sperm_CASA
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```
   The CPU wheel index matters — the default pulls ~2 GB of CUDA that does
   nothing on a CPU box.

### Switching between CPU and GPU instances

Stop the instance, change its type, start it. Nothing else — the pieces that
matter persist on the root volume and the deploy repairs the rest.

The first move to a GPU type needs the driver installed once:

```bash
sudo apt-get install -y linux-headers-$(uname -r) nvidia-driver-575-server
sudo reboot
nvidia-smi          # must list the GPU
```

That driver stays on the EBS volume, so later switches in either direction need
nothing. The CUDA torch build is correct on both kinds of instance — on a CPU
box it simply reports no GPU — and if a deploy ever leaves a CPU-only wheel on
a GPU box, the workflow detects the mismatch and reinstalls CUDA torch by
itself.

Nothing in the code changes either: `device` is left unset in
`utils/config.py`, so ultralytics picks CUDA when it is genuinely available.
Every run names the device it actually used, which is the quickest way to tell
a working GPU from a silent fallback:

```
classes={0: 'sperm'} device=cuda (Tesla T4)
classes={0: 'sperm'} device=cpu (torch 2.7.0+cpu is the CPU-only build)
```

The Elastic IP survives the stop/start, so `EC2_HOST` stays valid and the next
push deploys as usual. Deploys do not disturb the CUDA build: `torch` is not
pinned in `requirements.txt`, and pip leaves a satisfying version alone.
4. Run as a service so it survives reboots:
   ```ini
   # /etc/systemd/system/casa.service
   [Unit]
   Description=Sperm CASA dashboard
   After=network.target

   [Service]
   User=ubuntu
   WorkingDirectory=/home/ubuntu/sperm_CASA
   Environment=CASA_DASHBOARD_PASSWORD=<choose-one>
   ExecStart=/usr/local/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```
   ```bash
   sudo systemctl enable --now casa
   ```
5. Open `http://<elastic-ip>:8501`.

**Cost:** stop the instance between demos. Compute billing stops when
stopped; only EBS storage continues, at cents per month for this size.

## Roadmap

```
YOLO detection → ByteTrack → trajectories → CASA metrics → motility class → morphology
```

Each stage is a separate package so it can be tested and swapped on its own.

Identity stability at crossings is the open problem — see
[docs/tracking-plan.md](docs/tracking-plan.md) for the six steps that address
it, what each one needs by hand, and the measurements behind the ordering.
