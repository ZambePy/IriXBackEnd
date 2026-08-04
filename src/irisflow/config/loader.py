"""Load and merge configuration from every source in priority order.

Precedence, from highest to lowest:

1. **CLI overrides** — an already-built nested ``dict`` handed to
   :func:`load_config` by the CLI layer.
2. **Environment variables** prefixed with ``IRISFLOW_``, nested via ``__``.
   Example: ``IRISFLOW_CAMERA__DEVICE_ID=1`` maps to ``camera.device_id``.
3. **YAML file** (``configs/default.yaml`` by default).
4. **Field defaults** declared in :mod:`irisflow.config.schema`.

Any :class:`pydantic.ValidationError` is re-raised as
:class:`irisflow.core.exceptions.ConfigError` so the outer layer can catch a
single domain exception and print a message that points at the failing field.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from irisflow.config.schema import AppConfig
from irisflow.core.exceptions import ConfigError

ENV_PREFIX = "IRISFLOW_"
ENV_NESTED_DELIMITER = "__"
DEFAULT_YAML_PATH = Path("configs/default.yaml")


def load_config(
    yaml_path: Path | None = DEFAULT_YAML_PATH,
    cli_overrides: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> AppConfig:
    """Compose the effective :class:`AppConfig` from all sources.

    Args:
        yaml_path: Path to the YAML file. ``None`` skips YAML loading; a
            non-existent path is a hard error.
        cli_overrides: Nested ``dict`` shaped like the schema (e.g.
            ``{"camera": {"device_id": 2}}``).
        env: Environment mapping to read. Defaults to :data:`os.environ` — pass
            explicitly for tests.

    Raises:
        ConfigError: YAML is missing, malformed, or produces an invalid config.
    """
    yaml_data = _load_yaml(yaml_path) if yaml_path is not None else {}
    env_data = _parse_env_overrides(env if env is not None else os.environ)
    cli_data = dict(cli_overrides) if cli_overrides else {}

    merged: dict[str, Any] = {}
    _deep_merge(merged, yaml_data)
    _deep_merge(merged, env_data)
    _deep_merge(merged, cli_data)

    try:
        return AppConfig.model_validate(merged)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc)) from exc


# ---------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------
def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Malformed YAML in {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Top level of {path} must be a mapping, got {type(data).__name__}")
    return data


# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
def _parse_env_overrides(env: Mapping[str, str]) -> dict[str, Any]:
    """Translate ``IRISFLOW_SECTION__FIELD=value`` into a nested dict."""
    result: dict[str, Any] = {}
    for key, raw_value in env.items():
        if not key.startswith(ENV_PREFIX):
            continue
        path = key[len(ENV_PREFIX) :].lower().split(ENV_NESTED_DELIMITER)
        if not path or not all(path):
            continue
        cursor: dict[str, Any] = result
        for segment in path[:-1]:
            existing = cursor.get(segment)
            if not isinstance(existing, dict):
                existing = {}
                cursor[segment] = existing
            cursor = existing
        cursor[path[-1]] = _coerce_env_value(raw_value)
    return result


def _coerce_env_value(raw: str) -> Any:
    """Best-effort coercion for env-var strings.

    Strings that look like JSON (``[...]``, ``{...}``, ``null``, ``true``,
    ``false``, or a bare number) are parsed as JSON; everything else is kept
    as a plain string and Pydantic will coerce it against the target field.
    """
    stripped = raw.strip()
    if stripped == "":
        return ""
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return raw


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------
def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> None:
    """Merge ``overlay`` into ``base`` in place; overlay wins on conflict."""
    for key, value in overlay.items():
        existing = base.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            _deep_merge(existing, value)
        else:
            base[key] = value


# ---------------------------------------------------------------------------
# Error formatting
# ---------------------------------------------------------------------------
def _format_validation_error(exc: ValidationError) -> str:
    lines = ["Invalid configuration:"]
    for err in exc.errors():
        location = ".".join(str(part) for part in err["loc"]) or "<root>"
        lines.append(f"  - {location}: {err['msg']}")
    return "\n".join(lines)
