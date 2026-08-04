"""Sprint 1 — structlog wiring: human sink + JSON sink + frame_id context."""

from __future__ import annotations

import io
import json
import logging as stdlib_logging
from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog

from irisflow.logging import (
    bind_frame,
    bound,
    clear_context,
    configure_logging,
    get_logger,
)


@pytest.fixture(autouse=True)
def _reset_structlog() -> Iterator[None]:
    """Ensure each test starts from a clean logging state."""
    clear_context()
    yield
    clear_context()
    structlog.reset_defaults()
    root = stdlib_logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# ---------------------------------------------------------------------------
# JSON sink — event structure and context binding
# ---------------------------------------------------------------------------
def test_json_sink_writes_one_object_per_event(tmp_path: Path) -> None:
    log_file = tmp_path / "irisflow.jsonl"
    configure_logging(level="DEBUG", human_console=False, json_file=log_file)

    log = get_logger("test")
    log.info("gaze.raw", x=0.5, y=0.5)
    log.warning("face.lost")

    records = _read_jsonl(log_file)
    assert len(records) == 2
    assert records[0]["event"] == "gaze.raw"
    assert records[0]["x"] == 0.5
    assert records[0]["level"] == "info"
    assert records[1]["event"] == "face.lost"
    assert records[1]["level"] == "warning"


def test_bind_frame_puts_frame_id_into_every_event(tmp_path: Path) -> None:
    log_file = tmp_path / "irisflow.jsonl"
    configure_logging(level="DEBUG", human_console=False, json_file=log_file)
    log = get_logger("test")

    bind_frame(42, stage="inference")
    log.info("gaze.raw", x=0.1, y=0.2)

    (record,) = _read_jsonl(log_file)
    assert record["frame_id"] == 42
    assert record["stage"] == "inference"


def test_bound_context_unbinds_after_the_block(tmp_path: Path) -> None:
    log_file = tmp_path / "irisflow.jsonl"
    configure_logging(level="DEBUG", human_console=False, json_file=log_file)
    log = get_logger("test")

    with bound(frame_id=7):
        log.info("inside")
    log.info("outside")

    records = _read_jsonl(log_file)
    assert records[0]["frame_id"] == 7
    assert "frame_id" not in records[1]


def test_json_records_carry_monotonic_and_wall_timestamps(tmp_path: Path) -> None:
    log_file = tmp_path / "irisflow.jsonl"
    configure_logging(level="DEBUG", human_console=False, json_file=log_file)

    get_logger("test").info("tick")

    (record,) = _read_jsonl(log_file)
    assert isinstance(record["ts_mono"], (int, float))
    assert isinstance(record["timestamp"], str)


# ---------------------------------------------------------------------------
# Human console sink
# ---------------------------------------------------------------------------
def test_human_console_writes_readable_line() -> None:
    stream = io.StringIO()
    configure_logging(level="INFO", human_console=True, json_file=None, console_stream=stream)

    get_logger("test").info("hello.world", answer=42)

    output = stream.getvalue()
    assert "hello.world" in output
    assert "42" in output


# ---------------------------------------------------------------------------
# Level filtering
# ---------------------------------------------------------------------------
def test_level_filters_lower_severity_events(tmp_path: Path) -> None:
    log_file = tmp_path / "irisflow.jsonl"
    configure_logging(level="WARNING", human_console=False, json_file=log_file)
    log = get_logger("test")

    log.debug("shhh")
    log.info("also.shhh")
    log.warning("this.one.stays")

    records = _read_jsonl(log_file)
    assert [r["event"] for r in records] == ["this.one.stays"]


def test_unknown_level_raises() -> None:
    with pytest.raises(ValueError, match="Unknown log level"):
        configure_logging(level="TRACE", human_console=False, json_file=None)


# ---------------------------------------------------------------------------
# Reconfiguration replaces sinks (does not accumulate)
# ---------------------------------------------------------------------------
def test_reconfiguration_replaces_previous_handlers(tmp_path: Path) -> None:
    first = tmp_path / "one.jsonl"
    second = tmp_path / "two.jsonl"

    configure_logging(level="INFO", human_console=False, json_file=first)
    configure_logging(level="INFO", human_console=False, json_file=second)

    get_logger("test").info("only.in.second")

    assert not first.exists() or first.read_text(encoding="utf-8") == ""
    assert _read_jsonl(second)[0]["event"] == "only.in.second"


def test_json_file_parent_directory_is_created(tmp_path: Path) -> None:
    log_file = tmp_path / "nested" / "dir" / "irisflow.jsonl"
    configure_logging(level="INFO", human_console=False, json_file=log_file)

    get_logger("test").info("boot")

    assert log_file.exists()
    assert _read_jsonl(log_file)[0]["event"] == "boot"
