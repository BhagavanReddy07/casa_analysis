"""YOLO11-pose sperm detector.

Thin wrapper around Ultralytics. Returns geometry only — bounding box, head
keypoint, neck keypoint, confidence. No velocities, no angles, no counting;
those belong to the tracking and CASA stages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from utils.config import HEAD, NECK, Config

logger = logging.getLogger(__name__)

Point = tuple[float, float]


@dataclass(frozen=True)
class Detection:
    """One detected sperm in one frame, in pixel coordinates."""

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    head: Point
    neck: Point
    confidence: float

    @property
    def center(self) -> Point:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class SpermDetector:
    """Loads ``best.pt`` and turns frames into :class:`Detection` lists."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        weights = Path(self.config.weights)
        if not weights.exists():
            raise FileNotFoundError(f"weights not found: {weights.resolve()}")

        logger.info("loading %s", weights)
        self.model = YOLO(str(weights))
        self.dropped_keypoints = 0
        # Report the device actually in use, not what was asked for. A CPU-only
        # torch wheel on a GPU instance runs silently on the CPU at a twentieth
        # of the speed, and nothing else in the logs would say so.
        logger.info("classes=%s device=%s", self.model.names, self._device())

    def _device(self) -> str:
        import torch

        if self.config.device:
            return str(self.config.device)
        if torch.cuda.is_available():
            return f"cuda ({torch.cuda.get_device_name(0)})"
        return f"cpu — torch {torch.__version__} has no CUDA support" \
            if "+cpu" in torch.__version__ else "cpu (no GPU visible)"

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run the model on a single BGR frame."""
        result = self.model.predict(
            frame,
            conf=self.config.conf,
            iou=self.config.iou,
            imgsz=self.config.imgsz,
            device=self.config.device,
            verbose=False,
        )[0]

        if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        keypoints = result.keypoints.xy.cpu().numpy()

        # Keypoint visibility must be checked, not assumed. YOLO-pose was
        # trained with unannotated keypoints encoded as (0, 0, visible=0), so
        # it predicts the frame origin whenever it cannot localize one, and
        # ultralytics passes that straight through in .xy. Used unchecked,
        # those points teleport a trajectory to the corner and fabricate huge
        # velocities. If .conf is absent (a model exported without the
        # visibility dimension) treat every keypoint as valid rather than
        # silently discarding the whole frame.
        if result.keypoints.conf is not None:
            kpt_conf = result.keypoints.conf.cpu().numpy()
        else:
            kpt_conf = np.ones((len(boxes), 2), dtype=np.float32)

        detections = []
        for box, score, kpts, kconf in zip(boxes, scores, keypoints, kpt_conf):
            if float(kconf.min()) < self.config.min_keypoint_conf:
                self.dropped_keypoints += 1
                continue
            detections.append(Detection(
                bbox=tuple(box.tolist()),
                head=tuple(kpts[HEAD].tolist()),
                neck=tuple(kpts[NECK].tolist()),
                confidence=float(score),
            ))
        return detections


if __name__ == "__main__":
    # ponytail: one self-check instead of a suite — a detection whose keypoint
    # the model could not localize must be dropped, not passed through as a
    # position at the frame origin. That bug drew straight lines from the
    # corner of the video into every affected trajectory and inflated VCL by
    # an order of magnitude.
    from types import SimpleNamespace

    class _FakeTensor:
        def __init__(self, array): self._a = array
        def cpu(self): return self
        def numpy(self): return self._a
        def __len__(self): return len(self._a)

    class _FakeBoxes(SimpleNamespace):
        def __len__(self): return len(self.xyxy)

    class _StubDetector(SpermDetector):
        def __init__(self, result):  # no weights, no model load
            self.config = Config()
            self.dropped_keypoints = 0
            self._result = result
        def detect(self, frame):
            return SpermDetector.detect(self, frame)

    good_xy = [[100.0, 100.0], [110.0, 100.0]]
    lost_xy = [[0.0, 0.0], [0.0, 0.0]]          # what the model emits when unsure
    result = SimpleNamespace(
        boxes=_FakeBoxes(xyxy=_FakeTensor(np.array([[90.0, 90.0, 120.0, 115.0],
                                                    [10.0, 10.0, 40.0, 35.0]])),
                         conf=_FakeTensor(np.array([0.9, 0.8]))),
        keypoints=SimpleNamespace(xy=_FakeTensor(np.array([good_xy, lost_xy])),
                                  conf=_FakeTensor(np.array([[0.99, 0.98], [0.03, 0.02]]))),
    )
    detector = _StubDetector(result)
    detector.model = SimpleNamespace(predict=lambda *a, **k: [result])

    found = detector.detect(None)
    assert len(found) == 1, f"the unlocalized detection should be dropped, got {len(found)}"
    assert found[0].head == (100.0, 100.0), f"wrong detection kept: {found[0].head}"
    assert detector.dropped_keypoints == 1, "drop was not counted"

    # Without a visibility channel every keypoint is trusted, rather than the
    # whole frame being silently discarded.
    result.keypoints.conf = None
    detector.dropped_keypoints = 0
    assert len(detector.detect(None)) == 2, "missing .conf should not drop detections"

    print("detector.py self-check passed")
