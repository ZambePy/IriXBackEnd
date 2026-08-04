"""Sprint 5 — crop with edge-replication padding."""

from __future__ import annotations

import numpy as np
import pytest

from irisflow.core.types import BoundingBox
from irisflow.preprocessing.crops import crop_with_replicate_pad


def _grad_image(h: int = 40, w: int = 60) -> np.ndarray:
    """Horizontal gradient — useful to prove that replicated pixels come
    from the correct edge (not e.g. from the other side)."""
    row = np.linspace(0, 255, w, dtype=np.uint8)
    img = np.repeat(row[np.newaxis, :], h, axis=0)
    return np.stack([img, img, img], axis=-1)


def test_full_inside_crop_matches_direct_slice() -> None:
    img = _grad_image()
    bbox = BoundingBox(x=5, y=10, w=20, h=15)
    out = crop_with_replicate_pad(img, bbox)
    assert out.shape == (15, 20, 3)
    np.testing.assert_array_equal(out, img[10:25, 5:25])


def test_crop_partially_off_left_replicates_left_edge() -> None:
    img = _grad_image()
    bbox = BoundingBox(x=-5, y=0, w=10, h=5)
    out = crop_with_replicate_pad(img, bbox)
    assert out.shape == (5, 10, 3)
    # The first 5 columns of the crop should be replicas of img[:, 0].
    np.testing.assert_array_equal(out[:, 0, 0], np.full(5, img[0, 0, 0]))
    np.testing.assert_array_equal(out[:, 4, 0], np.full(5, img[0, 0, 0]))


def test_crop_partially_off_right_replicates_right_edge() -> None:
    img = _grad_image(h=20, w=30)
    bbox = BoundingBox(x=25, y=0, w=10, h=5)
    out = crop_with_replicate_pad(img, bbox)
    assert out.shape == (5, 10, 3)
    # Last pixel of the source should be replicated on the right side.
    assert out[0, -1, 0] == img[0, -1, 0]


def test_crop_out_of_frame_top_and_left_stays_valid() -> None:
    img = _grad_image(h=30, w=30)
    bbox = BoundingBox(x=-3, y=-2, w=10, h=10)
    out = crop_with_replicate_pad(img, bbox)
    assert out.shape == (10, 10, 3)
    assert out.dtype == np.uint8


def test_crop_fully_outside_frame_raises() -> None:
    img = _grad_image()
    with pytest.raises(ValueError, match="no intersection"):
        crop_with_replicate_pad(img, BoundingBox(x=100, y=100, w=10, h=10))


def test_crop_zero_size_bbox_raises() -> None:
    img = _grad_image()
    with pytest.raises(ValueError, match="non-positive"):
        crop_with_replicate_pad(img, BoundingBox(x=0, y=0, w=0, h=5))


def test_crop_rejects_bad_image_shape() -> None:
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        crop_with_replicate_pad(np.zeros((10, 10), dtype=np.uint8),
                                BoundingBox(x=0, y=0, w=1, h=1))


def test_crop_returns_contiguous_array() -> None:
    img = _grad_image()
    bbox = BoundingBox(x=5, y=5, w=10, h=10)
    out = crop_with_replicate_pad(img, bbox)
    assert out.flags["C_CONTIGUOUS"]
