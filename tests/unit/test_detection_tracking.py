"""Sprint 4 — EMA smoothing + loss-hysteresis wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from irisflow.core.types import BoundingBox, FaceDetection, Frame
from irisflow.detection.tracking import TrackedFaceDetector


def _make_frame(frame_id: int = 0) -> Frame:
    return Frame(
        data=np.zeros((100, 100, 3), dtype=np.uint8),
        frame_id=frame_id,
        timestamp=float(frame_id),
    )


_DEFAULT_FACE = BoundingBox(x=10, y=10, w=50, h=50)
_DEFAULT_LEFT = BoundingBox(x=15, y=25, w=15, h=10)
_DEFAULT_RIGHT = BoundingBox(x=40, y=25, w=15, h=10)


def _make_detection(
    face: BoundingBox = _DEFAULT_FACE,
    left: BoundingBox = _DEFAULT_LEFT,
    right: BoundingBox = _DEFAULT_RIGHT,
) -> FaceDetection:
    return FaceDetection(
        face_bbox=face,
        left_eye_bbox=left,
        right_eye_bbox=right,
        landmarks=np.zeros((478, 3), dtype=np.float32),
        confidence=1.0,
    )


@dataclass
class ScriptedDetector:
    """Returns a scripted sequence of ``FaceDetection | None`` values."""

    results: list[FaceDetection | None]
    calls: list[int] = field(default_factory=list)

    def detect(self, frame: Frame) -> FaceDetection | None:
        self.calls.append(frame.frame_id)
        if not self.results:
            return None
        return self.results.pop(0)


def test_first_detection_is_returned_unchanged() -> None:
    det = _make_detection()
    tracker = TrackedFaceDetector(ScriptedDetector([det]), smoothing_alpha=0.4)
    result = tracker.detect(_make_frame(0))
    assert result is not None
    assert result.face_bbox == det.face_bbox


def test_ema_smoothing_pulls_new_box_toward_previous() -> None:
    first = _make_detection(face=BoundingBox(x=10, y=10, w=50, h=50))
    second = _make_detection(face=BoundingBox(x=30, y=30, w=50, h=50))
    tracker = TrackedFaceDetector(ScriptedDetector([first, second]), smoothing_alpha=0.5)
    r1 = tracker.detect(_make_frame(0))
    r2 = tracker.detect(_make_frame(1))
    assert r1 is not None
    assert r2 is not None
    # With alpha=0.5, the smoothed x should be halfway between 10 and 30.
    assert r2.face_bbox.x == 20
    assert r2.face_bbox.y == 20


def test_alpha_one_disables_smoothing() -> None:
    d1 = _make_detection(face=BoundingBox(x=0, y=0, w=10, h=10))
    d2 = _make_detection(face=BoundingBox(x=100, y=100, w=10, h=10))
    tracker = TrackedFaceDetector(ScriptedDetector([d1, d2]), smoothing_alpha=1.0)
    tracker.detect(_make_frame(0))
    r2 = tracker.detect(_make_frame(1))
    assert r2 is not None
    assert r2.face_bbox.x == 100
    assert r2.face_bbox.y == 100


def test_still_face_produces_stable_smoothed_boxes() -> None:
    """Anti-jitter: identical inputs → identical outputs after the first."""
    det = _make_detection()
    inner = ScriptedDetector([det for _ in range(10)])
    tracker = TrackedFaceDetector(inner, smoothing_alpha=0.4)
    outs = [tracker.detect(_make_frame(i)) for i in range(10)]
    assert all(o is not None for o in outs)
    xs = [o.face_bbox.x for o in outs if o is not None]
    # After the first frame the EMA converges immediately since input is constant.
    assert max(xs[1:]) - min(xs[1:]) < 3


def test_hysteresis_returns_last_detection_for_configured_frames() -> None:
    good = _make_detection()
    tracker = TrackedFaceDetector(
        ScriptedDetector([good, None, None, None]),
        smoothing_alpha=1.0,
        lost_hysteresis=2,
    )
    r0 = tracker.detect(_make_frame(0))  # detected
    r1 = tracker.detect(_make_frame(1))  # miss 1 → held
    r2 = tracker.detect(_make_frame(2))  # miss 2 → held
    r3 = tracker.detect(_make_frame(3))  # miss 3 → truly LOST
    assert r0 is not None
    assert r1 is not None
    assert r1.face_bbox == r0.face_bbox
    assert r2 is not None
    assert r3 is None


def test_reacquisition_starts_fresh_after_true_loss() -> None:
    a = _make_detection(face=BoundingBox(x=10, y=10, w=20, h=20))
    b = _make_detection(face=BoundingBox(x=80, y=80, w=20, h=20))
    tracker = TrackedFaceDetector(
        ScriptedDetector([a, None, None, None, b]),
        smoothing_alpha=0.5,
        lost_hysteresis=1,
    )
    tracker.detect(_make_frame(0))
    tracker.detect(_make_frame(1))  # held
    tracker.detect(_make_frame(2))  # LOST
    tracker.detect(_make_frame(3))  # still LOST
    r = tracker.detect(_make_frame(4))
    # After LOST reset, the EMA state is gone — new detection returns unchanged.
    assert r is not None
    assert r.face_bbox == b.face_bbox


def test_zero_hysteresis_reports_loss_immediately() -> None:
    good = _make_detection()
    tracker = TrackedFaceDetector(
        ScriptedDetector([good, None]),
        smoothing_alpha=1.0,
        lost_hysteresis=0,
    )
    assert tracker.detect(_make_frame(0)) is not None
    assert tracker.detect(_make_frame(1)) is None


def test_no_detection_at_all_returns_none() -> None:
    tracker = TrackedFaceDetector(ScriptedDetector([None]), smoothing_alpha=1.0)
    assert tracker.detect(_make_frame(0)) is None


def test_reset_clears_smoothing_and_hysteresis_state() -> None:
    det = _make_detection()
    inner = ScriptedDetector([det, None, det])
    tracker = TrackedFaceDetector(inner, smoothing_alpha=0.5, lost_hysteresis=5)
    tracker.detect(_make_frame(0))
    tracker.detect(_make_frame(1))  # miss but held
    assert tracker.missed_frames == 1
    tracker.reset()
    assert tracker.missed_frames == 0
    assert not tracker.is_in_hysteresis
    r = tracker.detect(_make_frame(2))
    assert r is not None


def test_invalid_smoothing_alpha_raises() -> None:
    with pytest.raises(ValueError, match="smoothing_alpha"):
        TrackedFaceDetector(ScriptedDetector([]), smoothing_alpha=0.0)
    with pytest.raises(ValueError, match="smoothing_alpha"):
        TrackedFaceDetector(ScriptedDetector([]), smoothing_alpha=1.5)


def test_invalid_hysteresis_raises() -> None:
    with pytest.raises(ValueError, match="lost_hysteresis"):
        TrackedFaceDetector(ScriptedDetector([]), lost_hysteresis=-1)
