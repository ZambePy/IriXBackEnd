"""Cursor control layer (Sprint 10).

Public surface:

* :class:`~irisflow.control.cursor.PynputCursor` — the production adapter
  that actually moves the OS cursor via :mod:`pynput`, rate-limited.
* :class:`~irisflow.control.noop.NoOpCursor` — production no-op used by
  ``irisflow run --no-cursor`` (the default) and by tests.
* :class:`~irisflow.control.dwell.DwellClicker` — pure dwell-click state
  machine.
* :class:`~irisflow.control.safety.SafetyGate` /
  :class:`~irisflow.control.safety.Watchdog` — the guarantees that make
  cursor control safe for a user with ALS.
* :class:`~irisflow.control.kill_switch.KillSwitchListener` — global
  hotkey listener that releases the mouse in ≤ 100 ms.
"""

from irisflow.control.cursor import PynputCursor
from irisflow.control.dwell import DwellClicker, DwellDecision, DwellParams
from irisflow.control.kill_switch import KillSwitchListener, parse_hotkey
from irisflow.control.noop import NoOpCursor
from irisflow.control.safety import (
    PauseReason,
    RestZone,
    SafetyGate,
    SafetySnapshot,
    Watchdog,
)

__all__ = [
    "DwellClicker",
    "DwellDecision",
    "DwellParams",
    "KillSwitchListener",
    "NoOpCursor",
    "PauseReason",
    "PynputCursor",
    "RestZone",
    "SafetyGate",
    "SafetySnapshot",
    "Watchdog",
    "parse_hotkey",
]
