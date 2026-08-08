"""Sprint 9 -- ScreenMapper: normalized gaze -> pixel with border clamp.

Property-based coverage on the clamp criterion: for any (nx, ny) --
including values outside [0, 1] -- the returned pixel is always inside
the safe rectangle. This is the SPRINTS §9 DoD "point always within
screen limits".
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from irisflow.core.types import CalibratedGaze
from irisflow.mapping.screen import ScreenMapper
from irisflow.mapping.screen_info import ScreenInfo


def _mapper(margin: int = 8) -> ScreenMapper:
    return ScreenMapper(
        screen=ScreenInfo(width_px=1920, height_px=1080),
        clamp_margin_px=margin,
    )


def test_center_maps_to_center_pixel() -> None:
    result = _mapper().to_screen(CalibratedGaze(x=0.5, y=0.5, profile_id="p"))
    assert result.px == 960
    assert result.py == 540
    assert result.screen_id == 0


def test_corners_clamp_to_margin_edges() -> None:
    m = _mapper(margin=8)
    tl = m.to_screen(CalibratedGaze(x=0.0, y=0.0, profile_id="p"))
    assert tl.px == 8
    assert tl.py == 8
    br = m.to_screen(CalibratedGaze(x=1.0, y=1.0, profile_id="p"))
    assert br.px == 1911  # 1920 - 1 - 8
    assert br.py == 1071


def test_out_of_range_gaze_is_clamped() -> None:
    m = _mapper(margin=8)
    out = m.to_screen(CalibratedGaze(x=-0.5, y=1.5, profile_id="p"))
    assert out.px == 8
    assert out.py == 1071


def test_multi_screen_origin_offset_applied() -> None:
    mapper = ScreenMapper(
        screen=ScreenInfo(width_px=1920, height_px=1080, origin_x_px=1920, screen_id=1)
    )
    out = mapper.to_screen(CalibratedGaze(x=0.5, y=0.5, profile_id="p"))
    assert out.px == 1920 + 960
    assert out.screen_id == 1


def test_negative_margin_rejected() -> None:
    with pytest.raises(ValueError, match="clamp_margin_px must be"):
        ScreenMapper(
            screen=ScreenInfo(width_px=1920, height_px=1080),
            clamp_margin_px=-1,
        )


def test_excessive_margin_rejected() -> None:
    with pytest.raises(ValueError, match="no usable area"):
        ScreenMapper(
            screen=ScreenInfo(width_px=100, height_px=100),
            clamp_margin_px=50,
        )


@given(
    nx=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    ny=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
)
def test_output_always_inside_safe_rectangle(nx: float, ny: float) -> None:
    m = _mapper(margin=8)
    out = m.to_screen(CalibratedGaze(x=nx, y=ny, profile_id="p"))
    assert 8 <= out.px <= 1911
    assert 8 <= out.py <= 1071
