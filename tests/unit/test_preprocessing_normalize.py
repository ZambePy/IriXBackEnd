"""Sprint 5 — pixel normalization + channel-order swap, zero-alloc."""

from __future__ import annotations

import numpy as np
import pytest

from irisflow.preprocessing.normalize import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    normalize_into,
)


def _rgb_img() -> np.ndarray:
    # Distinct per-channel content so a channel swap is detectable.
    r = np.full((4, 4), 255, dtype=np.uint8)
    g = np.full((4, 4), 128, dtype=np.uint8)
    b = np.full((4, 4), 0, dtype=np.uint8)
    return np.stack([b, g, r], axis=-1)  # BGR


def test_unit_scheme_maps_255_to_1() -> None:
    src = _rgb_img()
    dst = np.empty(src.shape, dtype=np.float32)
    normalize_into(
        dst, src, scheme="unit", input_order="BGR", output_order="BGR"
    )
    assert dst.max() == pytest.approx(1.0)
    assert dst.min() == pytest.approx(0.0)


def test_signed_scheme_maps_0_and_255_to_minus_one_and_one() -> None:
    src = _rgb_img()
    dst = np.empty(src.shape, dtype=np.float32)
    normalize_into(
        dst, src, scheme="signed", input_order="BGR", output_order="BGR"
    )
    assert dst.max() == pytest.approx(1.0)
    assert dst.min() == pytest.approx(-1.0)


def test_imagenet_scheme_subtracts_mean_and_divides_std() -> None:
    src = _rgb_img()
    dst = np.empty(src.shape, dtype=np.float32)
    # Input BGR, but the imagenet mean/std are defined for RGB, so we ask
    # normalize_into to swap before applying.
    normalize_into(
        dst, src, scheme="imagenet", input_order="BGR", output_order="RGB"
    )
    expected_r = (255 / 255 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    expected_g = (128 / 255 - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
    expected_b = (0 / 255 - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
    assert dst[0, 0, 0] == pytest.approx(expected_r, abs=1e-4)
    assert dst[0, 0, 1] == pytest.approx(expected_g, abs=1e-4)
    assert dst[0, 0, 2] == pytest.approx(expected_b, abs=1e-4)


def test_channel_swap_flips_bgr_to_rgb() -> None:
    src = _rgb_img()  # B=0 G=128 R=255
    dst = np.empty(src.shape, dtype=np.float32)
    normalize_into(
        dst, src, scheme="unit", input_order="BGR", output_order="RGB"
    )
    # After swap the first channel should be R (255→1.0), last should be B (0→0.0).
    assert dst[0, 0, 0] == pytest.approx(1.0)
    assert dst[0, 0, 2] == pytest.approx(0.0)


def test_no_swap_when_orders_match() -> None:
    src = _rgb_img()
    dst = np.empty(src.shape, dtype=np.float32)
    normalize_into(
        dst, src, scheme="unit", input_order="BGR", output_order="BGR"
    )
    assert dst[0, 0, 0] == pytest.approx(0.0)   # B channel first
    assert dst[0, 0, 2] == pytest.approx(1.0)   # R channel last


def test_normalize_into_reuses_dst_buffer_identity() -> None:
    src = _rgb_img()
    dst = np.empty(src.shape, dtype=np.float32)
    result = normalize_into(
        dst, src, scheme="unit", input_order="BGR", output_order="BGR"
    )
    assert result is dst


def test_reject_wrong_src_dtype() -> None:
    src = np.zeros((4, 4, 3), dtype=np.float32)
    dst = np.zeros((4, 4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="src must"):
        normalize_into(
            dst, src,  # type: ignore[arg-type]
            scheme="unit", input_order="BGR", output_order="BGR",
        )


def test_reject_wrong_dst_shape() -> None:
    src = _rgb_img()
    dst = np.zeros((10, 10, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="dst must match"):
        normalize_into(
            dst, src, scheme="unit", input_order="BGR", output_order="BGR"
        )
