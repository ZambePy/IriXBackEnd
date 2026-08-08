"""Temporal smoothing and loss-hysteresis on top of any :class:`FaceDetector`.

The stage exists for two reasons:

* **Anti-jitter.** MediaPipe's per-frame landmarks wobble by 1-3 px even
  when the head is completely still. An EMA on the four bbox corners
  reduces that to sub-pixel, which is what the CNN needs to see the
  *same face* across frames rather than a shivering approximation.
* **Anti-blink loss.** A single-frame detection miss (blink, brief
  occlusion, hand across the mouth) shouldn't yank the pipeline into
  ``LOST``. We hold onto the last smoothed detection for ``lost_hysteresis``
  frames before finally reporting absence.

The wrapper is a plain class implementing the same
:class:`~irisflow.core.interfaces.FaceDetector` Protocol as the raw
adapter, so callers cannot tell the two apart — that is the point.
"""

from __future__ import annotations

from irisflow.core.interfaces import FaceDetector
from irisflow.core.types import BoundingBox, FaceDetection, Frame

__all__ = ["TrackedFaceDetector"]


class TrackedFaceDetector:
    """Wrap a :class:`FaceDetector` with EMA smoothing and loss hysteresis.

    Args:
        inner: The concrete detector to consult each frame.
        smoothing_alpha: Weight of the new observation in the EMA. ``1.0``
            means "no smoothing"; ``0.0`` means "freeze at first detection".
            Sensible range: ``0.2`` to ``0.6``.
        lost_hysteresis: Frames the tracker keeps returning the last
            detection after a miss before finally returning ``None``. Set
            to ``0`` for no hysteresis (report loss immediately).
    """

    __slots__ = (
        "_inner",
        "_last_detection",
        "_lost_hysteresis",
        "_missed_frames",
        "_smoothed_boxes",
        "_smoothing_alpha",
    )

    def __init__(
        self,
        inner: FaceDetector,
        *,
        smoothing_alpha: float = 0.4,
        lost_hysteresis: int = 5,
    ) -> None:
        if not (0.0 < smoothing_alpha <= 1.0):
            raise ValueError(f"smoothing_alpha must be in (0, 1], got {smoothing_alpha}")
        if lost_hysteresis < 0:
            raise ValueError(f"lost_hysteresis must be >= 0, got {lost_hysteresis}")
        self._inner = inner
        self._smoothing_alpha = float(smoothing_alpha)
        self._lost_hysteresis = int(lost_hysteresis)
        self._smoothed_boxes: dict[str, BoundingBox] = {}
        self._missed_frames = 0
        self._last_detection: FaceDetection | None = None

    # ------------------------------------------------------------------ API
    def detect(self, frame: Frame) -> FaceDetection | None:
        raw = self._inner.detect(frame)
        if raw is None:
            return self._handle_miss()
        return self._handle_hit(raw)

    def reset(self) -> None:
        """Discard smoothing state and miss counter — call on state change."""
        self._smoothed_boxes.clear()
        self._missed_frames = 0
        self._last_detection = None

    # ------------------------------------------------------------------ stats
    @property
    def missed_frames(self) -> int:
        """How many consecutive frames the inner detector has returned ``None``."""
        return self._missed_frames

    @property
    def is_in_hysteresis(self) -> bool:
        """True while we are returning stale detections after a miss."""
        return 0 < self._missed_frames <= self._lost_hysteresis

    # ------------------------------------------------------------------ core
    def _handle_hit(self, det: FaceDetection) -> FaceDetection:
        self._missed_frames = 0
        smoothed = FaceDetection(
            face_bbox=self._smooth("face", det.face_bbox),
            left_eye_bbox=self._smooth("left_eye", det.left_eye_bbox),
            right_eye_bbox=self._smooth("right_eye", det.right_eye_bbox),
            landmarks=det.landmarks,
            confidence=det.confidence,
        )
        self._last_detection = smoothed
        return smoothed

    def _handle_miss(self) -> FaceDetection | None:
        self._missed_frames += 1
        if self._missed_frames <= self._lost_hysteresis and self._last_detection is not None:
            # Hold onto the last known good detection. Do NOT touch
            # smoothing state so that when tracking recovers on the next
            # frame the EMA continues without a jump.
            return self._last_detection
        # Truly lost — clear everything so recovery starts from a clean slate.
        self.reset()
        return None

    def _smooth(self, key: str, box: BoundingBox) -> BoundingBox:
        prev = self._smoothed_boxes.get(key)
        if prev is None:
            self._smoothed_boxes[key] = box
            return box
        a = self._smoothing_alpha
        smoothed = BoundingBox(
            x=round(a * box.x + (1.0 - a) * prev.x),
            y=round(a * box.y + (1.0 - a) * prev.y),
            w=round(a * box.w + (1.0 - a) * prev.w),
            h=round(a * box.h + (1.0 - a) * prev.h),
        )
        self._smoothed_boxes[key] = smoothed
        return smoothed
