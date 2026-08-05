"""Versioned JSON persistence of calibration profiles.

Profile format::

    {
        "schema_version": 1,
        "profile_id": "maria",
        "created_at_wall_s": 1730000000.123,
        "screen": {"width": 1920, "height": 1080},
        "quality": {
            "mean_error_px": 47.2, "p95_error_px": 63.9, "max_error_px": 78.1,
            "n_samples": 270,
        },
        "model": { ...output of CalibrationModel.to_dict()... }
    }

Both save and load are lossless round-trips: reload → :meth:`fit` output
is bit-identical to what was saved (a requirement of the Sprint 8 DoD).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from irisflow.calibration.models import (
    AffineCalibration,
    PolynomialCalibration,
    RidgeCalibration,
)
from irisflow.calibration.quality import CalibrationQualityReport
from irisflow.core.exceptions import CalibrationError

__all__ = [
    "CalibrationProfile",
    "CalibrationStore",
    "load_profile_dict",
    "save_profile_dict",
]


SCHEMA_VERSION = 1


CalibrationModelImpl = AffineCalibration | PolynomialCalibration | RidgeCalibration


@dataclass
class CalibrationProfile:
    """In-memory bundle: the calibration model plus session metadata.

    Kept as a plain dataclass so tests can construct one without hitting
    the filesystem, and so ``asdict()`` gets us most of the JSON payload
    for free.
    """

    profile_id: str
    model: CalibrationModelImpl
    screen_width: int
    screen_height: int
    created_at_wall_s: float
    mean_error_px: float
    p95_error_px: float
    max_error_px: float
    n_samples: int
    schema_version: int = SCHEMA_VERSION
    quality_targets: list[dict[str, Any]] = field(default_factory=list)


def save_profile_dict(profile: CalibrationProfile) -> dict[str, Any]:
    """Serialise ``profile`` to the on-disk dict — pure, no I/O."""
    return {
        "schema_version": profile.schema_version,
        "profile_id": profile.profile_id,
        "created_at_wall_s": profile.created_at_wall_s,
        "screen": {"width": profile.screen_width, "height": profile.screen_height},
        "quality": {
            "mean_error_px": profile.mean_error_px,
            "p95_error_px": profile.p95_error_px,
            "max_error_px": profile.max_error_px,
            "n_samples": profile.n_samples,
            "targets": list(profile.quality_targets),
        },
        "model": profile.model.to_dict(),
    }


def load_profile_dict(data: dict[str, Any]) -> CalibrationProfile:
    """Parse a saved dict back into a :class:`CalibrationProfile`."""
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise CalibrationError(
            f"unsupported profile schema_version={version!r}, "
            f"expected {SCHEMA_VERSION}"
        )
    try:
        screen = data["screen"]
        quality = data["quality"]
        model_data = data["model"]
    except KeyError as exc:
        raise CalibrationError(
            f"profile missing required field: {exc.args[0]!r}"
        ) from exc

    kind = model_data.get("kind")
    model: CalibrationModelImpl
    if kind == "affine":
        model = AffineCalibration.from_dict(model_data)
    elif kind == "polynomial":
        model = PolynomialCalibration.from_dict(model_data)
    elif kind == "ridge":
        model = RidgeCalibration.from_dict(model_data)
    else:
        raise CalibrationError(f"unknown model kind {kind!r} in profile")

    return CalibrationProfile(
        profile_id=str(data["profile_id"]),
        model=model,
        screen_width=int(screen["width"]),
        screen_height=int(screen["height"]),
        created_at_wall_s=float(data["created_at_wall_s"]),
        mean_error_px=float(quality["mean_error_px"]),
        p95_error_px=float(quality["p95_error_px"]),
        max_error_px=float(quality["max_error_px"]),
        n_samples=int(quality["n_samples"]),
        quality_targets=list(quality.get("targets", [])),
        schema_version=SCHEMA_VERSION,
    )


def build_profile_from_report(
    *,
    profile_id: str,
    model: CalibrationModelImpl,
    report: CalibrationQualityReport,
    created_at_wall_s: float,
) -> CalibrationProfile:
    """Assemble a :class:`CalibrationProfile` from evaluate output.

    Kept here so :mod:`session` doesn't have to know the profile shape
    beyond "ask the store to build one".
    """
    return CalibrationProfile(
        profile_id=profile_id,
        model=model,
        screen_width=report.screen_width,
        screen_height=report.screen_height,
        created_at_wall_s=created_at_wall_s,
        mean_error_px=report.mean_error_px,
        p95_error_px=report.p95_error_px,
        max_error_px=report.max_error_px,
        n_samples=report.n_samples,
        quality_targets=[asdict(t) for t in report.targets],
    )


class CalibrationStore:
    """Reads and writes :class:`CalibrationProfile` under ``root_dir``.

    File layout: ``<root_dir>/<profile_id>.json``. The store never
    creates a partially-written file: it writes to a sibling ``.tmp``
    and atomically renames it, so a crash mid-save cannot corrupt an
    existing profile.
    """

    __slots__ = ("_root",)

    def __init__(self, root_dir: str | Path) -> None:
        self._root = Path(root_dir)

    @property
    def root_dir(self) -> Path:
        return self._root

    def path_for(self, profile_id: str) -> Path:
        _validate_profile_id(profile_id)
        return self._root / f"{profile_id}.json"

    def save(self, profile: CalibrationProfile) -> Path:
        _validate_profile_id(profile.profile_id)
        self._root.mkdir(parents=True, exist_ok=True)
        payload = save_profile_dict(profile)
        target = self.path_for(profile.profile_id)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp.replace(target)
        return target

    def load(self, profile_id: str) -> CalibrationProfile:
        target = self.path_for(profile_id)
        if not target.exists():
            raise CalibrationError(f"profile not found: {target}")
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CalibrationError(f"profile file is not valid JSON: {target}") from exc
        return load_profile_dict(data)

    def list_profiles(self) -> list[str]:
        if not self._root.exists():
            return []
        return sorted(p.stem for p in self._root.glob("*.json"))

    def delete(self, profile_id: str) -> bool:
        target = self.path_for(profile_id)
        if not target.exists():
            return False
        target.unlink()
        return True


def _validate_profile_id(profile_id: str) -> None:
    """Reject profile IDs that would let a user escape ``root_dir``."""
    if not profile_id:
        raise CalibrationError("profile_id must be non-empty")
    if "/" in profile_id or "\\" in profile_id or profile_id in (".", ".."):
        raise CalibrationError(
            f"profile_id must not contain path separators, got {profile_id!r}"
        )
