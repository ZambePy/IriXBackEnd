"""Record raw gaze axes from the live pipeline to a CSV.

Usage:
    uv run python scripts/axis_probe.py --duration 30 --label "olhando_centro"
    uv run python scripts/axis_probe.py --duration 30 --output data/axis_probe.csv

Output CSV columns: timestamp, raw_x, raw_y, label
"""
from __future__ import annotations

import csv
import pathlib
import sys
import threading
import time
from typing import Annotated

import typer

# Add src/ to path for development use.
_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

app = typer.Typer(add_completion=False)


@app.command()
def main(
    duration: Annotated[float, typer.Option("--duration", min=1.0)] = 30.0,
    label: Annotated[str, typer.Option("--label")] = "probe",
    output: Annotated[pathlib.Path, typer.Option("--output")] = pathlib.Path(
        "data/axis_probe.csv"
    ),
    config: Annotated[pathlib.Path | None, typer.Option("--config", "-c")] = None,
) -> None:
    """Record raw gaze axes to CSV for axis health analysis."""
    from irisflow.config.loader import load_config
    from irisflow.core.events import RawGazeReady
    from irisflow.pipeline.orchestrator import build_pipeline
    from irisflow.pipeline.runner import PipelineRunner

    cfg = load_config(yaml_path=config) if config is not None else load_config()
    components = build_pipeline(cfg)

    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    def _on_raw(event: object) -> None:
        if not isinstance(event, RawGazeReady):
            return
        rows.append(
            {
                "timestamp": event.timestamp,
                "raw_x": event.x,
                "raw_y": event.y,
                "label": label,
            }
        )

    components.bus.subscribe(RawGazeReady, _on_raw)
    runner = PipelineRunner(components)
    bg = threading.Thread(target=runner.run, daemon=True)
    bg.start()
    typer.echo(f"Recording for {duration:.0f}s — label='{label}' → {output}")
    time.sleep(duration)
    runner.stop()
    bg.join(timeout=2.0)

    with output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["timestamp", "raw_x", "raw_y", "label"])
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        xs = [r["raw_x"] for r in rows]
        ys = [r["raw_y"] for r in rows]
        amp_x = max(xs) - min(xs)  # type: ignore[type-var]
        amp_y = max(ys) - min(ys)  # type: ignore[type-var]
        typer.echo(f"Recorded {len(rows)} samples.")
        typer.echo(f"  raw_x: min={min(xs):.4f} max={max(xs):.4f} amp={amp_x:.4f}")
        typer.echo(f"  raw_y: min={min(ys):.4f} max={max(ys):.4f} amp={amp_y:.4f}")
        typer.echo(f"Saved to {output}")
    else:
        typer.echo("[warn] No samples recorded — is the camera working?", err=True)


if __name__ == "__main__":
    app()
