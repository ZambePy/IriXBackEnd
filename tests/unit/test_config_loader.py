"""Sprint 1 — configuration loading, validation, and override precedence."""

from __future__ import annotations

from pathlib import Path

import pytest

from irisflow.config import AppConfig, load_config
from irisflow.core.exceptions import ConfigError


def _write_yaml(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Defaults / round-trip
# ---------------------------------------------------------------------------
def test_defaults_only_produces_valid_appconfig() -> None:
    config = load_config(yaml_path=None, env={})
    assert isinstance(config, AppConfig)
    assert config.camera.device_id == 0
    assert config.model.channel_order == "RGB"
    assert config.filtering.chain == ["outlier", "one_euro", "fixation"]


def test_yaml_values_override_field_defaults(tmp_path: Path) -> None:
    yaml_file = _write_yaml(
        tmp_path / "cfg.yaml",
        "camera:\n  device_id: 3\n  width: 640\n",
    )

    config = load_config(yaml_path=yaml_file, env={})

    assert config.camera.device_id == 3
    assert config.camera.width == 640
    assert config.camera.height == 720  # untouched default


# ---------------------------------------------------------------------------
# Override precedence — required by Sprint 1
# ---------------------------------------------------------------------------
def test_env_var_overrides_yaml_value(tmp_path: Path) -> None:
    yaml_file = _write_yaml(
        tmp_path / "cfg.yaml",
        "camera:\n  device_id: 0\n",
    )

    config = load_config(
        yaml_path=yaml_file,
        env={"IRISFLOW_CAMERA__DEVICE_ID": "1"},
    )

    assert config.camera.device_id == 1


def test_cli_overrides_beat_env_and_yaml(tmp_path: Path) -> None:
    yaml_file = _write_yaml(tmp_path / "cfg.yaml", "camera:\n  device_id: 0\n")

    config = load_config(
        yaml_path=yaml_file,
        env={"IRISFLOW_CAMERA__DEVICE_ID": "1"},
        cli_overrides={"camera": {"device_id": 2}},
    )

    assert config.camera.device_id == 2


def test_env_ignores_unrelated_prefixes() -> None:
    config = load_config(
        yaml_path=None,
        env={"PATH": "/usr/bin", "OTHER__NESTED": "x"},
    )
    assert config.camera.device_id == 0


def test_env_coerces_json_scalars() -> None:
    config = load_config(
        yaml_path=None,
        env={
            "IRISFLOW_CAMERA__WIDTH": "800",
            "IRISFLOW_CONTROL__ENABLED": "true",
        },
    )
    assert config.camera.width == 800
    assert config.control.enabled is True


# ---------------------------------------------------------------------------
# Validation errors — required by Sprint 1
# ---------------------------------------------------------------------------
def test_invalid_value_raises_configerror_naming_the_field(tmp_path: Path) -> None:
    yaml_file = _write_yaml(
        tmp_path / "cfg.yaml",
        "camera:\n  device_id: -5\n",
    )

    with pytest.raises(ConfigError) as excinfo:
        load_config(yaml_path=yaml_file, env={})

    message = str(excinfo.value)
    assert "camera" in message
    assert "device_id" in message


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    yaml_file = _write_yaml(
        tmp_path / "cfg.yaml",
        "camera:\n  device_id: 0\n  bogus: 1\n",
    )

    with pytest.raises(ConfigError) as excinfo:
        load_config(yaml_path=yaml_file, env={})

    assert "bogus" in str(excinfo.value)


def test_missing_yaml_file_is_configerror(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(yaml_path=tmp_path / "does_not_exist.yaml", env={})


def test_malformed_yaml_is_configerror(tmp_path: Path) -> None:
    bad = _write_yaml(tmp_path / "cfg.yaml", "camera: [unterminated\n")

    with pytest.raises(ConfigError):
        load_config(yaml_path=bad, env={})


def test_non_mapping_top_level_is_configerror(tmp_path: Path) -> None:
    bad = _write_yaml(tmp_path / "cfg.yaml", "- item1\n- item2\n")

    with pytest.raises(ConfigError):
        load_config(yaml_path=bad, env={})


# ---------------------------------------------------------------------------
# Default YAML shipped in the repo must remain loadable
# ---------------------------------------------------------------------------
def test_repository_default_yaml_loads_without_error() -> None:
    repo_default = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    assert repo_default.exists(), "configs/default.yaml is missing from the repo"

    config = load_config(yaml_path=repo_default, env={})
    assert isinstance(config, AppConfig)
