"""Test-suite bootstrap.

Two things happen at import time:

1. Force Typer's Rich console to render at width 200 by both setting
   ``TERMINAL_WIDTH`` and monkey-patching ``typer.rich_utils.MAX_WIDTH``.
   This *should* stop ``--help`` output from wrapping option names.
2. Add a ``flat_stdout`` property to ``click.testing.Result`` that
   returns the captured stdout with ANSI escapes stripped and all
   whitespace collapsed. CLI ``--help`` tests use this so option-name
   assertions survive any Rich formatting quirks that (1) fails to
   suppress on the CI runner.
"""

from __future__ import annotations

import os
import re

os.environ.setdefault("TERMINAL_WIDTH", "200")

import typer.rich_utils
import typer.testing

typer.rich_utils.MAX_WIDTH = 200

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_WS_RE = re.compile(r"\s+")


def _flat_stdout(self: typer.testing.Result) -> str:
    return _WS_RE.sub("", _ANSI_RE.sub("", self.stdout))


typer.testing.Result.flat_stdout = property(_flat_stdout)  # type: ignore[attr-defined]
