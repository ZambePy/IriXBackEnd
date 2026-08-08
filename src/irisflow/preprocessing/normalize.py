"""Pixel-value normalization schemes + channel-order swap.

The three schemes cover every pretrained-model convention we're likely to
encounter:

* ``unit`` — divide by 255, output in ``[0, 1]``. Most Keras Applications
  models trained ``from_scratch`` use this.
* ``signed`` — ``x/127.5 - 1``, output in ``[-1, 1]``. What MobileNet /
  Inception-family use out of the box.
* ``imagenet`` — subtract per-channel mean, divide by per-channel std.
  Standard for ResNet, VGG, EfficientNet — anything Torch-derived.

**Which one to use is not a choice; it is a measurement.** Sprint 6's
:file:`inspect_model.py` reads the training config and writes the answer
into :file:`MODEL_CARD.md`. Until then we default to ``unit`` because it
matches the config default (:mod:`irisflow.config.schema`).

**Channel order:** OpenCV frames are BGR; most CNNs want RGB. The swap
happens here so the crop and resize modules can stay ignorant of it.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "ChannelOrder",
    "NormalizationScheme",
    "normalize_into",
]


NormalizationScheme = Literal["unit", "signed", "imagenet"]
ChannelOrder = Literal["RGB", "BGR"]


IMAGENET_MEAN: NDArray[np.float32] = np.array([0.485, 0.456, 0.406], dtype=np.float32)
"""Per-channel mean assumed by ImageNet-pretrained models (in RGB order)."""

IMAGENET_STD: NDArray[np.float32] = np.array([0.229, 0.224, 0.225], dtype=np.float32)
"""Per-channel std assumed by ImageNet-pretrained models (in RGB order)."""


def normalize_into(
    dst: NDArray[np.float32],
    src: NDArray[np.uint8],
    *,
    scheme: NormalizationScheme,
    input_order: ChannelOrder,
    output_order: ChannelOrder,
) -> NDArray[np.float32]:
    """Fill ``dst`` with the normalized ``src`` pixels — no allocation.

    Args:
        dst: Preallocated ``(H, W, 3)`` float32 output buffer. Must have
            the same spatial shape as ``src``.
        src: ``(H, W, 3)`` uint8 input.
        scheme: Which normalization to apply (see module docstring).
        input_order: Channel order of ``src`` (usually ``"BGR"`` from OpenCV).
        output_order: Channel order the CNN expects.

    Returns:
        The same ``dst`` buffer (returned for chaining).

    Raises:
        ValueError: Shape/dtype mismatch, or an unknown ``scheme``.
    """
    if src.ndim != 3 or src.shape[2] != 3 or src.dtype != np.uint8:
        raise ValueError(f"src must be (H, W, 3) uint8, got shape={src.shape!r} dtype={src.dtype}")
    if dst.shape != src.shape or dst.dtype != np.float32:
        raise ValueError(
            f"dst must match src shape as float32, got dst shape={dst.shape!r} "
            f"dtype={dst.dtype}, src shape={src.shape!r}"
        )

    view = src if input_order == output_order else src[..., ::-1]
    # np.copyto with 'unsafe' casts uint8 -> float32 into the preallocated dst.
    np.copyto(dst, view, casting="unsafe")

    if scheme == "unit":
        dst *= np.float32(1.0 / 255.0)
    elif scheme == "signed":
        dst *= np.float32(1.0 / 127.5)
        dst -= np.float32(1.0)
    elif scheme == "imagenet":
        dst *= np.float32(1.0 / 255.0)
        dst -= IMAGENET_MEAN
        dst /= IMAGENET_STD
    else:  # pragma: no cover — Literal keeps this unreachable
        raise ValueError(f"unknown normalization scheme: {scheme!r}")
    return dst
