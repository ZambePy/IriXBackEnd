"""Print coverage totals per top-level ``src/irisflow/<layer>`` package.

Feed with ``coverage.json`` produced by:

    pytest --cov=irisflow --cov-report=json

Domain vs. global targets (SPRINTS §13 DoD) are printed at the bottom.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DOMAIN_LAYERS = {"core", "preprocessing", "calibration", "filtering", "mapping"}


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "coverage.json")
    data = json.loads(path.read_text(encoding="utf-8"))

    per_layer: dict[str, dict[str, int]] = {}
    for file_path, info in data["files"].items():
        p = file_path.replace("\\", "/")
        if "src/irisflow/" not in p:
            continue
        rest = p.split("src/irisflow/")[1]
        layer = rest.split("/")[0]
        entry = per_layer.setdefault(layer, {"stmts": 0, "covered": 0})
        entry["stmts"] += info["summary"]["num_statements"]
        entry["covered"] += info["summary"]["covered_lines"]

    print(f"{'layer':<16} {'stmts':>8} {'covered':>8} {'pct':>7}")
    print("-" * 46)
    total_s = total_c = 0
    domain_s = domain_c = 0
    for layer, entry in sorted(per_layer.items()):
        pct = 100.0 * entry["covered"] / max(entry["stmts"], 1)
        print(f"{layer:<16} {entry['stmts']:>8} {entry['covered']:>8} {pct:>6.1f}%")
        total_s += entry["stmts"]
        total_c += entry["covered"]
        if layer in DOMAIN_LAYERS:
            domain_s += entry["stmts"]
            domain_c += entry["covered"]
    print("-" * 46)
    print(
        f"{'DOMAIN':<16} {domain_s:>8} {domain_c:>8} "
        f"{100.0 * domain_c / max(domain_s, 1):>6.1f}%  (target >= 90%)"
    )
    print(
        f"{'GLOBAL':<16} {total_s:>8} {total_c:>8} "
        f"{100.0 * total_c / max(total_s, 1):>6.1f}%  (target >= 70%)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
