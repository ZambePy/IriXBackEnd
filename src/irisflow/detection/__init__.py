"""Detection layer — MediaPipe adapter + pure ROI/tracking helpers.

Public surface:

* :class:`MediaPipeFaceDetector` — production face + eye ROI detector.
* :class:`TrackedFaceDetector` — smoothing + loss-hysteresis wrapper.
* :func:`build_default_face_mesh` — factory for the production MediaPipe
  backend (requires the ``.task`` model file on disk).
* :mod:`roi`, :mod:`landmarks` — pure helpers usable without MediaPipe.
"""

from irisflow.detection.landmarks import (
    FACE_LANDMARK_COUNT_LEGACY,
    FACE_LANDMARK_COUNT_WITH_IRIS,
    FACE_OVAL_INDICES,
    LEFT_EYE_INDICES,
    LEFT_IRIS_INDICES,
    RIGHT_EYE_INDICES,
    RIGHT_IRIS_INDICES,
    landmarks_to_pixel_array,
)
from irisflow.detection.mediapipe_detector import (
    FaceMeshFactory,
    FaceMeshLike,
    MediaPipeFaceDetector,
    build_default_face_mesh,
)
from irisflow.detection.roi import bbox_from_indices
from irisflow.detection.tracking import TrackedFaceDetector

__all__ = [
    "FACE_LANDMARK_COUNT_LEGACY",
    "FACE_LANDMARK_COUNT_WITH_IRIS",
    "FACE_OVAL_INDICES",
    "LEFT_EYE_INDICES",
    "LEFT_IRIS_INDICES",
    "RIGHT_EYE_INDICES",
    "RIGHT_IRIS_INDICES",
    "FaceMeshFactory",
    "FaceMeshLike",
    "MediaPipeFaceDetector",
    "TrackedFaceDetector",
    "bbox_from_indices",
    "build_default_face_mesh",
    "landmarks_to_pixel_array",
]
