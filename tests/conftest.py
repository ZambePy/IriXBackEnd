"""Test-suite bootstrap.

Set ``TERMINAL_WIDTH`` before any test module imports Typer so its Rich
console renders ``--help`` panels wide enough that option names (e.g.
``--screen-width``) never wrap and remain substring-searchable in
``result.stdout``. Read at module import time by
``typer.rich_utils`` (``_TERMINAL_WIDTH = getenv("TERMINAL_WIDTH")``).
"""

from __future__ import annotations

import os

os.environ.setdefault("TERMINAL_WIDTH", "200")
