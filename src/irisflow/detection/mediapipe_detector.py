"""MediaPipe FaceLandmarker adapter — the concrete side of the seam.

MediaPipe 1.0 dropped the legacy ``mp.solutions.face_mesh`` API; we use
the new :mod:`mediapipe.tasks.python.vision` API, which requires a
``.task`` model asset on disk. Download once:

    https://storage.googleapis.com/mediapipe-models/face_landmarker/
        face_landmarker/float16/1/face_landmarker.task

into the path pointed to by ``detection.face_model_path``
(``models/face_landmarker.task`` by default).

The class conforms structurally to
:class:`~irisflow.core.interfaces.FaceDetector` and never raises inside
:meth:`detect` — a failure to detect a face is an expected, per-frame
condition and returns ``None``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import cv2
import numpy as np
from numpy.typing import NDArray

from irisflow.core.exceptions import DetectionError
from irisflow.core.types import FaceDetection, Frame
from irisflow.detection.landmarks import (
    FACE_LANDMARK_COUNT_LEGACY,
    FACE_LANDMARK_COUNT_WITH_IRIS,
    FACE_OVAL_INDICES,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    landmarks_to_pixel_array,
)
from irisflow.detection.roi import bbox_from_indices
from irisflow.logging import get_logger

__all__ = [
    "FaceMeshFactory",
    "FaceMeshLike",
    "MediaPipeFaceDetector",
]


@runtime_checkable
class FaceMeshLike(Protocol):
    """Structural interface every FaceMesh backend implements.

    ``detect`` receives an ``(H, W, 3)`` uint8 RGB array and returns
    ``(N, 3)`` normalized landmarks (``x``/``y`` in ``[0, 1]``, ``z`` in
    world units), or ``None`` when nothing was found. This is the exact
    boundary tests substitute across.
    """

    def detect(self, rgb_image: NDArray[np.uint8]) -> NDArray[np.float32] | None: ...

    def close(self) -> None: ...


FaceMeshFactory = Callable[..., FaceMeshLike]
"""Callable that produces a :class:`FaceMeshLike`. Injected by the CLI /
pipeline; tests use a fake to avoid loading the real model."""


class MediaPipeFaceDetector:
    """Face + eye ROI detector backed by a :class:`FaceMeshLike`.

    Args:
        face_mesh: An already-open FaceMesh backend. Ownership transfers
            to this detector: :meth:`close` will close it.
        roi_margin: Fractional expansion applied to each ROI. Matches
            :attr:`~irisflow.config.schema.DetectionConfig.roi_margin`.
        square_rois: When true, expand each ROI to a square before
            clipping — the CNN was trained on square crops.
        min_confidence: Face detections with reported confidence below
            this are treated as absence. FaceMesh itself doesn't give a
            per-detection confidence; this parameter is retained for
            future backends that do.
    """

    __slots__ = ("_face_mesh", "_log", "_min_confidence", "_roi_margin", "_square_rois")

    def __init__(
        self,
        face_mesh: FaceMeshLike,
        *,
        roi_margin: float = 0.20,
        square_rois: bool = True,
        min_confidence: float = 0.0,
    ) -> None:
        if not (0.0 <= roi_margin <= 1.0):
            raise ValueError(f"roi_margin must be in [0, 1], got {roi_margin}")
        if not (0.0 <= min_confidence <= 1.0):
            raise ValueError(f"min_confidence must be in [0, 1], got {min_confidence}")
        self._face_mesh = face_mesh
        self._roi_margin = float(roi_margin)
        self._square_rois = bool(square_rois)
        self._min_confidence = float(min_confidence)
        self._log = get_logger("irisflow.detection.mediapipe")

    # ------------------------------------------------------------------ API
    def detect(self, frame: Frame) -> FaceDetection | None:
        try:
            rgb = np.ascontiguousarray(
                cv2.cvtColor(frame.data, cv2.COLOR_BGR2RGB), dtype=np.uint8
            )
            landmarks_norm = self._face_mesh.detect(rgb)
        except Exception as exc:
            self._log.warning(
                "detection.face_mesh_raised",
                frame_id=frame.frame_id,
                error=str(exc),
            )
            return None
        if landmarks_norm is None or landmarks_norm.size == 0:
            return None
        if landmarks_norm.shape[0] not in (
            FACE_LANDMARK_COUNT_LEGACY,
            FACE_LANDMARK_COUNT_WITH_IRIS,
        ):
            self._log.warning(
                "detection.unexpected_landmark_count",
                count=landmarks_norm.shape[0],
                frame_id=frame.frame_id,
            )
            return None
        try:
            landmarks_px = landmarks_to_pixel_array(
                landmarks_norm, frame.width, frame.height
            )
            face_bbox = bbox_from_indices(
                landmarks_px,
                FACE_OVAL_INDICES,
                margin=self._roi_margin,
                frame_width=frame.width,
                frame_height=frame.height,
                square=self._square_rois,
            )
            left_eye_bbox = bbox_from_indices(
                landmarks_px,
                LEFT_EYE_INDICES,
                margin=self._roi_margin,
                frame_width=frame.width,
                frame_height=frame.height,
                square=self._square_rois,
            )
            right_eye_bbox = bbox_from_indices(
                landmarks_px,
                RIGHT_EYE_INDICES,
                margin=self._roi_margin,
                frame_width=frame.width,
                frame_height=frame.height,
                square=self._square_rois,
            )
        except ValueError as exc:
            self._log.warning(
                "detection.bbox_failed",
                frame_id=frame.frame_id,
                error=str(exc),
            )
            return None
        return FaceDetection(
            face_bbox=face_bbox,
            left_eye_bbox=left_eye_bbox,
            right_eye_bbox=right_eye_bbox,
            landmarks=landmarks_px,
            confidence=1.0,
        )

    def close(self) -> None:
        self._face_mesh.close()


# ---------------------------------------------------------------------------
# Real MediaPipe backend — lazy import so tests using a fake factory never
# pay the ~500ms MediaPipe module import cost.
# ---------------------------------------------------------------------------
def build_default_face_mesh(
    *,
    model_path: Path,
    refine_landmarks: bool = True,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> FaceMeshLike:
    """Construct the production FaceMesh backend from a ``.task`` file.

    Raises:
        DetectionError: The model file is missing. The message includes
            the download URL so the user can fix it without leaving the
            terminal.
    """
    if not model_path.exists():
        raise DetectionError(
            f"MediaPipe face landmarker model not found at {model_path!r}. "
            "Download it from "
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task "
            "and place it at that path (or override "
            "detection.face_model_path in your config)."
        )
    # Deferred import: keeps the module cheap to import in test environments
    # where the mediapipe C-extension is heavy and not needed.
    from mediapipe import Image, ImageFormat
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceLandmarker,
        FaceLandmarkerOptions,
    )

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        num_faces=1,
        min_face_detection_confidence=min_detection_confidence,
        min_face_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    landmarker = FaceLandmarker.create_from_options(options)
    return _MediaPipeFaceMesh(landmarker, Image, ImageFormat, refine_landmarks)


class _MediaPipeFaceMesh:
    """Thin adapter: MediaPipe's ``FaceLandmarker`` → :class:`FaceMeshLike`.

    Kept internal because callers only ever see the :class:`FaceMeshLike`
    Protocol — the concrete class is a MediaPipe implementation detail.
    """

    __slots__ = ("_image_cls", "_image_format", "_landmarker", "_refine_landmarks")

    def __init__(
        self,
        landmarker: Any,
        image_cls: Any,
        image_format: Any,
        refine_landmarks: bool,
    ) -> None:
        self._landmarker = landmarker
        self._image_cls = image_cls
        self._image_format = image_format
        self._refine_landmarks = refine_landmarks

    def detect(self, rgb_image: NDArray[np.uint8]) -> NDArray[np.float32] | None:
        mp_image = self._image_cls(
            image_format=self._image_format.SRGB,
            data=np.ascontiguousarray(rgb_image, dtype=np.uint8),
        )
        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        landmarks = result.face_landmarks[0]
        arr = np.array(
            [(lm.x, lm.y, lm.z) for lm in landmarks],
            dtype=np.float32,
        )
        return arr

    def close(self) -> None:
        self._landmarker.close()
