"""Sprint 8 -- calibration/points.py grid generator."""

from __future__ import annotations

import pytest

from irisflow.calibration.points import CalibrationTarget, generate_targets


def test_generate_9_targets_layout_is_3x3_grid() -> None:
    targets = generate_targets(9, margin=0.1)
    assert len(targets) == 9
    xs = sorted({t.nx for t in targets})
    ys = sorted({t.ny for t in targets})
    assert len(xs) == 3
    assert len(ys) == 3
    assert xs[0] == pytest.approx(0.1)
    assert xs[-1] == pytest.approx(0.9)
    assert xs[1] == pytest.approx(0.5)


def test_generate_5_targets_center_present() -> None:
    targets = generate_targets(5, margin=0.1)
    assert len(targets) == 5
    assert any(t.nx == pytest.approx(0.5) and t.ny == pytest.approx(0.5) for t in targets)


def test_generate_13_targets_superset_of_9() -> None:
    nine = {(t.nx, t.ny) for t in generate_targets(9, margin=0.1)}
    thirteen = {(t.nx, t.ny) for t in generate_targets(13, margin=0.1)}
    assert nine.issubset(thirteen)
    assert len(thirteen) == 13


def test_generate_indices_are_sequential_and_unique() -> None:
    targets = generate_targets(9)
    assert [t.index for t in targets] == list(range(9))


def test_generate_targets_stay_inside_margin() -> None:
    for count in (5, 9, 13):
        targets = generate_targets(count, margin=0.08)  # type: ignore[arg-type]
        for t in targets:
            assert 0.08 - 1e-9 <= t.nx <= 0.92 + 1e-9
            assert 0.08 - 1e-9 <= t.ny <= 0.92 + 1e-9


def test_generate_rejects_negative_or_too_large_margin() -> None:
    with pytest.raises(ValueError, match="margin"):
        generate_targets(9, margin=-0.1)
    with pytest.raises(ValueError, match="margin"):
        generate_targets(9, margin=0.5)


def test_target_rejects_out_of_range_coords() -> None:
    with pytest.raises(ValueError, match="target coordinates"):
        CalibrationTarget(index=0, nx=1.2, ny=0.5)


def test_target_rejects_negative_index() -> None:
    with pytest.raises(ValueError, match="target index"):
        CalibrationTarget(index=-1, nx=0.5, ny=0.5)


def test_target_as_tuple_matches_fields() -> None:
    t = CalibrationTarget(index=3, nx=0.25, ny=0.75)
    assert t.as_tuple() == (0.25, 0.75)
