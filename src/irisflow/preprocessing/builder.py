"""Compose crops + resize + normalize + rect vector into one
:class:`ModelInput`.

**Zero-allocation contract.** The builder holds the four output buffers
(``face``, ``left_eye``, ``right_eye``, ``rect``) for its lifetime and
overwrites them on every :meth:`build` call. The returned
:class:`ModelInput` **references** those buffers — it does not copy them.

**Consequence for callers.** A :class:`ModelInput` returned by
:meth:`build` is only valid until the next :meth:`build` call. This is
fine for the pipeline (which passes it straight to
:class:`~irisflow.core.interfaces.GazeEstimator` and drops the reference)
but would surprise a caller who stashed the object for later use. If you
need to keep one around, snapshot with :func:`numpy.copy`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from irisflow.core.types import BoundingBox, FaceDetection, Frame, ModelInput
from irisflow.preprocessing.crops import crop_with_replicate_pad
from irisflow.preprocessing.normalize import (
    ChannelOrder,
    NormalizationScheme,
    normalize_into,
)
from irisflow.preprocessing.rect_vector import RECT_DIM, build_rect_vector
from irisflow.preprocessing.resize import Interpolation, resize_to

__all__ = ["ModelInputBuilder"]


class ModelInputBuilder:
    """Preprocess ``(Frame, FaceDetection)`` into a ``ModelInput`` in one call.

    Args:
        face_input_size: ``(H, W)`` the CNN's face tensor expects.
        eye_input_size: ``(H, W)`` for each eye tensor.
        input_channel_order: What the source frame delivers. OpenCV → ``"BGR"``.
        output_channel_order: What the CNN was trained on. Set by config.
        normalization: Which :mod:`irisflow.preprocessing.normalize` scheme
            to apply.
        interpolation: Interpolation for :mod:`irisflow.preprocessing.resize`.
    """

    __slots__ = (
        "_eye_input_size",
        "_face_input_size",
        "_face_uint8",
        "_input_channel_order",
        "_interpolation",
        "_left_eye_uint8",
        "_normalization",
        "_out_face",
        "_out_left_eye",
        "_out_rect",
        "_out_right_eye",
        "_output_channel_order",
        "_right_eye_uint8",
    )

    def __init__(
        self,
        *,
        face_input_size: tuple[int, int] = (224, 224),
        eye_input_size: tuple[int, int] = (112, 112),
        input_channel_order: ChannelOrder = "BGR",
        output_channel_order: ChannelOrder = "RGB",
        normalization: NormalizationScheme = "unit",
        interpolation: Interpolation = "auto",
    ) -> None:
        _check_size(face_input_size, "face_input_size")
        _check_size(eye_input_size, "eye_input_size")
        self._face_input_size = face_input_size
        self._eye_input_size = eye_input_size
        self._input_channel_order: ChannelOrder = input_channel_order
        self._output_channel_order: ChannelOrder = output_channel_order
        self._normalization: NormalizationScheme = normalization
        self._interpolation: Interpolation = interpolation

        # Preallocated uint8 intermediates (after resize, before normalize).
        fh, fw = face_input_size
        eh, ew = eye_input_size
        self._face_uint8: NDArray[np.uint8] = np.empty((fh, fw, 3), dtype=np.uint8)
        self._left_eye_uint8: NDArray[np.uint8] = np.empty((eh, ew, 3), dtype=np.uint8)
        self._right_eye_uint8: NDArray[np.uint8] = np.empty((eh, ew, 3), dtype=np.uint8)

        # Preallocated float32 CNN-ready tensors — the fields of every
        # ModelInput this builder produces reference these.
        self._out_face: NDArray[np.float32] = np.empty((fh, fw, 3), dtype=np.float32)
        self._out_left_eye: NDArray[np.float32] = np.empty((eh, ew, 3), dtype=np.float32)
        self._out_right_eye: NDArray[np.float32] = np.empty((eh, ew, 3), dtype=np.float32)
        self._out_rect: NDArray[np.float32] = np.empty(RECT_DIM, dtype=np.float32)

    # ------------------------------------------------------------------ API
    def build(self, frame: Frame, detection: FaceDetection) -> ModelInput:
        """Produce the ModelInput for one frame + detection pair."""
        self._fill_face(frame, detection)
        self._fill_eye(
            frame,
            detection.left_eye_bbox,
            uint8_buf=self._left_eye_uint8,
            out=self._out_left_eye,
        )
        self._fill_eye(
            frame,
            detection.right_eye_bbox,
            uint8_buf=self._right_eye_uint8,
            out=self._out_right_eye,
        )
        build_rect_vector(
            detection.face_bbox,
            detection.left_eye_bbox,
            detection.right_eye_bbox,
            frame_width=frame.width,
            frame_height=frame.height,
            out=self._out_rect,
        )
        return ModelInput(
            face=self._out_face,
            left_eye=self._out_left_eye,
            right_eye=self._out_right_eye,
            rect=self._out_rect,
        )

    # ------------------------------------------------------------------ internals
    def _fill_face(self, frame: Frame, detection: FaceDetection) -> None:
        cropped = crop_with_replicate_pad(frame.data, detection.face_bbox)
        resize_to(
            cropped,
            self._face_input_size,
            interpolation=self._interpolation,
            dst=self._face_uint8,
        )
        normalize_into(
            self._out_face,
            self._face_uint8,
            scheme=self._normalization,
            input_order=self._input_channel_order,
            output_order=self._output_channel_order,
        )

    def _fill_eye(
        self,
        frame: Frame,
        bbox: BoundingBox,
        *,
        uint8_buf: NDArray[np.uint8],
        out: NDArray[np.float32],
    ) -> None:
        cropped = crop_with_replicate_pad(frame.data, bbox)
        resize_to(
            cropped,
            self._eye_input_size,
            interpolation=self._interpolation,
            dst=uint8_buf,
        )
        normalize_into(
            out,
            uint8_buf,
            scheme=self._normalization,
            input_order=self._input_channel_order,
            output_order=self._output_channel_order,
        )


def _check_size(size: tuple[int, int], name: str) -> None:
    if len(size) != 2 or size[0] <= 0 or size[1] <= 0:
        raise ValueError(f"{name} must be positive (H, W), got {size!r}")
