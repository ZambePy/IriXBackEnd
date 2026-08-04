"""Sprint 5 — resize with auto/explicit interpolation and preallocated dst."""

from __future__ import annotations

import numpy as np
import pytest

from irisflow.preprocessing.resize import resize_to


def _img(h: int, w: int) -> np.ndarray:
    return (np.arange(h * w * 3, dtype=np.uint8) % 255).reshape(h, w, 3)


def test_resize_downscale_returns_target_shape() -> None:
    out = resize_to(_img(200, 300), (100, 150))
    assert out.shape == (100, 150, 3)
    assert out.dtype == np.uint8


def test_resize_upscale_returns_target_shape() -> None:
    out = resize_to(_img(60, 40), (120, 80))
    assert out.shape == (120, 80, 3)


def test_resize_into_preallocated_dst_writes_and_returns_dst() -> None:
    src = _img(100, 100)
    dst = np.empty((50, 50, 3), dtype=np.uint8)
    result = resize_to(src, (50, 50), dst=dst)
    assert result is dst
    # And it was actually populated (not left as garbage).
    assert result.max() > 0


@pytest.mark.parametrize("interp", ["linear", "area", "cubic", "nearest"])
def test_explicit_interpolations_all_produce_valid_output(interp: str) -> None:
    out = resize_to(_img(50, 50), (25, 25), interpolation=interp)  # type: ignore[arg-type]
    assert out.shape == (25, 25, 3)


def test_dst_with_wrong_shape_raises() -> None:
    src = _img(100, 100)
    dst = np.empty((40, 50, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="dst must"):
        resize_to(src, (50, 50), dst=dst)


def test_dst_with_wrong_dtype_raises() -> None:
    src = _img(100, 100)
    dst = np.empty((50, 50, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="dst must"):
        resize_to(src, (50, 50), dst=dst)  # type: ignore[arg-type]


def test_reject_non_positive_target() -> None:
    with pytest.raises(ValueError, match="positive"):
        resize_to(_img(50, 50), (0, 50))


def test_reject_bad_source_shape() -> None:
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        resize_to(np.zeros((50, 50), dtype=np.uint8), (25, 25))
