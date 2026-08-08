"""Sprint 5 — end-to-end preprocessing builder + golden-hash regression."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from irisflow.core.types import BoundingBox, FaceDetection, Frame
from irisflow.preprocessing.builder import ModelInputBuilder


def _synthetic_frame(width: int = 640, height: int = 480) -> Frame:
    """Deterministic BGR frame with a diagonal gradient — every input tensor
    the builder produces from this frame is a function of a known signal."""
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    gx, gy = np.meshgrid(x, y)
    b = gx
    g = gy
    r = ((gx.astype(np.uint16) + gy.astype(np.uint16)) // 2).astype(np.uint8)
    data = np.stack([b, g, r], axis=-1)
    return Frame(data=data, frame_id=0, timestamp=0.0)


def _detection() -> FaceDetection:
    return FaceDetection(
        face_bbox=BoundingBox(x=120, y=80, w=200, h=200),
        left_eye_bbox=BoundingBox(x=160, y=140, w=40, h=25),
        right_eye_bbox=BoundingBox(x=240, y=140, w=40, h=25),
        landmarks=np.zeros((478, 3), dtype=np.float32),
        confidence=1.0,
    )


# ---------------------------------------------------------------------------
# Shape / dtype invariants — Sprint 5 DoD.
# ---------------------------------------------------------------------------
def test_builder_produces_exact_shapes_and_dtypes() -> None:
    builder = ModelInputBuilder()
    result = builder.build(_synthetic_frame(), _detection())
    assert result.face.shape == (224, 224, 3)
    assert result.face.dtype == np.float32
    assert result.left_eye.shape == (112, 112, 3)
    assert result.left_eye.dtype == np.float32
    assert result.right_eye.shape == (112, 112, 3)
    assert result.right_eye.dtype == np.float32
    assert result.rect.shape == (12,)
    assert result.rect.dtype == np.float32


def test_builder_reuses_output_buffers_across_calls() -> None:
    """Zero-alloc contract: the same numpy buffer must back every call's
    output tensors — that's what makes preprocessing loop-safe."""
    builder = ModelInputBuilder()
    a = builder.build(_synthetic_frame(), _detection())
    b = builder.build(_synthetic_frame(), _detection())
    # Same underlying memory for every field.
    assert a.face is b.face
    assert a.left_eye is b.left_eye
    assert a.right_eye is b.right_eye
    assert a.rect is b.rect


def test_builder_rect_values_stay_in_unit_range() -> None:
    builder = ModelInputBuilder()
    result = builder.build(_synthetic_frame(), _detection())
    assert result.rect.min() >= 0.0
    assert result.rect.max() <= 1.0


def test_builder_signed_normalization_maps_to_minus_one_to_one() -> None:
    builder = ModelInputBuilder(normalization="signed")
    result = builder.build(_synthetic_frame(), _detection())
    assert result.face.min() >= -1.0001
    assert result.face.max() <= 1.0001


def test_builder_channel_swap_produces_different_pixels_than_no_swap() -> None:
    swap = ModelInputBuilder(output_channel_order="RGB")
    no_swap = ModelInputBuilder(output_channel_order="BGR")
    swap_face = swap.build(_synthetic_frame(), _detection()).face.copy()
    no_swap_face = no_swap.build(_synthetic_frame(), _detection()).face
    assert not np.array_equal(swap_face, no_swap_face)


def test_builder_rejects_bad_face_input_size() -> None:
    with pytest.raises(ValueError, match="face_input_size"):
        ModelInputBuilder(face_input_size=(0, 224))


def test_builder_rejects_bad_eye_input_size() -> None:
    with pytest.raises(ValueError, match="eye_input_size"):
        ModelInputBuilder(eye_input_size=(112, 0))


# ---------------------------------------------------------------------------
# Golden hash regression — freezes the exact output for a deterministic
# input so any change to preprocessing (interpolation, ordering, scheme) is
# caught by CI before it silently degrades the model.
# ---------------------------------------------------------------------------
_EXPECTED_HASHES = {
    "face": "dcbe7e253394ce0c0fabd5786d384c4ba20169c34ad29f74b503c88830cbc959",
    "left_eye": "59372131ca5eea55554e7c48cf1f7ccc9fcf56065510c3f0a97dbaaeb875165e",
    "right_eye": "e784fdc6c7c8e6baa5d021b7078b40a8808b5436a894a5b35e18116657caf4e0",
    "rect": "2b922267a8e37d15abb695653c6f383339c08654951eaa97f86ec4b8e9311d19",
}


def _hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def test_golden_hashes_have_not_changed() -> None:
    """If this fails after a legitimate preprocessing change, regenerate
    the constants above by running:

        >>> from tests.unit.test_preprocessing_builder import _hash, ...

    and pasting the new hashes into _EXPECTED_HASHES."""
    builder = ModelInputBuilder(
        face_input_size=(224, 224),
        eye_input_size=(112, 112),
        input_channel_order="BGR",
        output_channel_order="RGB",
        normalization="unit",
        interpolation="auto",
    )
    result = builder.build(_synthetic_frame(), _detection())
    hashes = {
        "face": _hash(result.face),
        "left_eye": _hash(result.left_eye),
        "right_eye": _hash(result.right_eye),
        "rect": _hash(result.rect),
    }
    assert hashes == _EXPECTED_HASHES, (
        f"Preprocessing output changed. If intentional, update _EXPECTED_HASHES with:\n{hashes}"
    )
