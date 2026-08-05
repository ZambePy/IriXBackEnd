"""Sprint 9 -- ScreenInfo / MultiScreenLayout."""

from __future__ import annotations

import pytest

from irisflow.mapping.screen_info import MultiScreenLayout, ScreenInfo


def test_screen_rejects_zero_or_negative_dims() -> None:
    with pytest.raises(ValueError, match="positive dims"):
        ScreenInfo(width_px=0, height_px=1080)
    with pytest.raises(ValueError, match="positive dims"):
        ScreenInfo(width_px=1920, height_px=-1)


def test_screen_rejects_non_positive_dpi() -> None:
    with pytest.raises(ValueError, match="dpi"):
        ScreenInfo(width_px=1920, height_px=1080, dpi=0.0)


def test_screen_contains_pixel_within_rect() -> None:
    s = ScreenInfo(width_px=1920, height_px=1080, origin_x_px=10, origin_y_px=20)
    assert s.contains(10, 20)
    assert s.contains(1929, 1099)
    assert not s.contains(9, 20)
    assert not s.contains(10, 1100)


def test_screen_x2_y2_use_origin() -> None:
    s = ScreenInfo(width_px=100, height_px=50, origin_x_px=5, origin_y_px=10)
    assert s.x2 == 105
    assert s.y2 == 60


def test_layout_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        MultiScreenLayout(screens=())


def test_layout_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="duplicate screen_id"):
        MultiScreenLayout(
            screens=(
                ScreenInfo(width_px=100, height_px=100, screen_id=0),
                ScreenInfo(width_px=200, height_px=200, screen_id=0),
            )
        )


def test_layout_get_returns_by_id() -> None:
    layout = MultiScreenLayout(
        screens=(
            ScreenInfo(width_px=100, height_px=100, screen_id=0),
            ScreenInfo(width_px=200, height_px=200, screen_id=1),
        )
    )
    assert layout.get(1).width_px == 200
    with pytest.raises(KeyError):
        layout.get(2)


def test_layout_primary_is_lowest_id() -> None:
    layout = MultiScreenLayout(
        screens=(
            ScreenInfo(width_px=100, height_px=100, screen_id=2),
            ScreenInfo(width_px=200, height_px=200, screen_id=0),
        )
    )
    assert layout.primary().screen_id == 0
