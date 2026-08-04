"""Preprocessing layer — pure functions that turn frames + detections
into the exact tensors the CNN was trained on.

Public surface:

* :class:`ModelInputBuilder` — the one-call orchestrator; reuses buffers
  so no allocation happens in the hot loop.
* :func:`crop_with_replicate_pad`, :func:`resize_to`, :func:`normalize_into`,
  :func:`build_rect_vector` — individual stages, exposed so tests can
  probe them in isolation.
"""

from irisflow.preprocessing.builder import ModelInputBuilder
from irisflow.preprocessing.crops import crop_with_replicate_pad
from irisflow.preprocessing.normalize import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ChannelOrder,
    NormalizationScheme,
    normalize_into,
)
from irisflow.preprocessing.rect_vector import RECT_DIM, build_rect_vector
from irisflow.preprocessing.resize import Interpolation, resize_to

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "RECT_DIM",
    "ChannelOrder",
    "Interpolation",
    "ModelInputBuilder",
    "NormalizationScheme",
    "build_rect_vector",
    "crop_with_replicate_pad",
    "normalize_into",
    "resize_to",
]
