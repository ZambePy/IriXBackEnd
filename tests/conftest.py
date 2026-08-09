"""Test-suite bootstrap.

Force Typer's Rich console to render ``--help`` panels wide enough that
option names (e.g. ``--screen-width``) never wrap and remain
substring-searchable in ``result.stdout``. Setting ``TERMINAL_WIDTH`` via
the CI ``env:`` block was not propagating through ``uv run`` in
GitHub Actions, so we also patch the module attribute directly — that
override wins regardless of import order or environment quirks.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("TERMINAL_WIDTH", "200")

import typer.rich_utils

typer.rich_utils.MAX_WIDTH = 200

print(
    f"[conftest] TERMINAL_WIDTH env={os.environ.get('TERMINAL_WIDTH')!r} "
    f"typer.rich_utils.MAX_WIDTH={typer.rich_utils.MAX_WIDTH!r}",
    file=sys.stderr,
)
