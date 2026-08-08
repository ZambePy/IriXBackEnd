"""Sprint 8 -- calibration store: JSON persistence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from irisflow.calibration.models import AffineCalibration, PolynomialCalibration
from irisflow.calibration.quality import CalibrationQualityReport, TargetError
from irisflow.calibration.store import (
    CalibrationProfile,
    CalibrationStore,
    build_profile_from_report,
    load_profile_dict,
    save_profile_dict,
)
from irisflow.core.exceptions import CalibrationError


def _fitted_polynomial() -> PolynomialCalibration:
    from irisflow.core.types import RawGaze

    model = PolynomialCalibration(profile_id="maria")
    raw = [
        RawGaze(x=0.1, y=0.1, confidence=1.0, inference_ms=1.0),
        RawGaze(x=0.5, y=0.5, confidence=1.0, inference_ms=1.0),
        RawGaze(x=0.9, y=0.9, confidence=1.0, inference_ms=1.0),
        RawGaze(x=0.1, y=0.9, confidence=1.0, inference_ms=1.0),
        RawGaze(x=0.9, y=0.1, confidence=1.0, inference_ms=1.0),
        RawGaze(x=0.5, y=0.1, confidence=1.0, inference_ms=1.0),
        RawGaze(x=0.5, y=0.9, confidence=1.0, inference_ms=1.0),
    ]
    tgt = [(0.1, 0.1), (0.5, 0.5), (0.9, 0.9), (0.1, 0.9), (0.9, 0.1), (0.5, 0.1), (0.5, 0.9)]
    model.fit(raw, tgt)
    return model


def _make_profile(profile_id: str = "maria") -> CalibrationProfile:
    return CalibrationProfile(
        profile_id=profile_id,
        model=_fitted_polynomial(),
        screen_width=1920,
        screen_height=1080,
        created_at_wall_s=1730000000.0,
        mean_error_px=42.0,
        p95_error_px=55.5,
        max_error_px=70.1,
        n_samples=100,
    )


def test_save_and_load_profile_round_trips(tmp_path: Path) -> None:
    store = CalibrationStore(tmp_path)
    profile = _make_profile()
    path = store.save(profile)
    assert path.exists()

    loaded = store.load("maria")
    assert loaded.profile_id == profile.profile_id
    assert loaded.screen_width == profile.screen_width
    assert loaded.screen_height == profile.screen_height
    assert loaded.mean_error_px == profile.mean_error_px
    assert isinstance(loaded.model, PolynomialCalibration)
    # numerically identical weights -- this is a DoD criterion
    assert profile.model.weights is not None
    assert loaded.model.weights is not None
    np.testing.assert_array_equal(loaded.model.weights, profile.model.weights)


def test_list_profiles_returns_saved_ids(tmp_path: Path) -> None:
    store = CalibrationStore(tmp_path)
    store.save(_make_profile("alice"))
    store.save(_make_profile("bob"))
    assert store.list_profiles() == ["alice", "bob"]


def test_load_missing_profile_raises(tmp_path: Path) -> None:
    store = CalibrationStore(tmp_path)
    with pytest.raises(CalibrationError, match="not found"):
        store.load("missing")


def test_delete_returns_true_when_present(tmp_path: Path) -> None:
    store = CalibrationStore(tmp_path)
    store.save(_make_profile("alice"))
    assert store.delete("alice") is True
    assert store.delete("alice") is False


def test_profile_id_rejects_path_separators(tmp_path: Path) -> None:
    store = CalibrationStore(tmp_path)
    with pytest.raises(CalibrationError, match="path separators"):
        store.save(_make_profile("../evil"))


def test_load_rejects_unknown_schema_version(tmp_path: Path) -> None:
    payload = save_profile_dict(_make_profile())
    payload["schema_version"] = 999
    with pytest.raises(CalibrationError, match="schema_version"):
        load_profile_dict(payload)


def test_load_rejects_unknown_model_kind(tmp_path: Path) -> None:
    payload = save_profile_dict(_make_profile())
    payload["model"]["kind"] = "cuboid"
    with pytest.raises(CalibrationError, match="unknown model kind"):
        load_profile_dict(payload)


def test_atomic_save_never_leaves_tmp_around(tmp_path: Path) -> None:
    store = CalibrationStore(tmp_path)
    store.save(_make_profile("alice"))
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []


def test_save_produces_sorted_json_for_deterministic_diffs(tmp_path: Path) -> None:
    store = CalibrationStore(tmp_path)
    path = store.save(_make_profile())
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    # sort_keys=True was used -- verify by re-serialising in same shape
    assert json.dumps(data, indent=2, sort_keys=True) == text


def test_build_profile_from_report_populates_targets() -> None:
    report = CalibrationQualityReport(
        mean_error_px=10.0,
        max_error_px=30.0,
        p95_error_px=25.0,
        screen_width=1920,
        screen_height=1080,
        n_samples=20,
        targets=[TargetError(0, 0.1, 0.1, 0.12, 0.11, 5.0)],
    )
    from irisflow.core.types import RawGaze

    model = AffineCalibration(profile_id="alice")
    model.fit(
        [RawGaze(x=0.1, y=0.1, confidence=1.0, inference_ms=1.0)] * 3,
        [(0.1, 0.1)] * 3,
    )
    profile = build_profile_from_report(
        profile_id="alice",
        model=model,
        report=report,
        created_at_wall_s=123.0,
    )
    assert profile.quality_targets[0]["error_px"] == 5.0
    assert profile.mean_error_px == 10.0
