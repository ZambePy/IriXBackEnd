"""``irisflow doctor`` — inspect capture devices and measure real capture rate.

Prints:

* One row per probed device index, whether it opens and what it reports.
* If any device opens: an empirical measurement of FPS and read latency,
  taken from the configured camera device (``camera.device_id``).

Works with no camera attached — that case is reported explicitly rather
than crashed on, because a user with ALS may have unplugged the camera and
we still want ``doctor`` to help them find that out.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from irisflow.capture import (
    CameraInfo,
    CaptureMeasurement,
    WebcamSource,
    enumerate_cameras,
    measure_capture,
)
from irisflow.config.loader import load_config
from irisflow.core.exceptions import IrisFlowError

__all__ = ["doctor"]


def doctor(
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to the YAML config. Defaults to configs/default.yaml.",
        ),
    ] = None,
    max_index: Annotated[
        int,
        typer.Option(
            "--max-index",
            min=1,
            help="Probe device indices [0, max_index).",
        ),
    ] = 3,
    duration: Annotated[
        float,
        typer.Option(
            "--duration",
            min=0.1,
            help="Seconds to sample the configured camera for FPS measurement.",
        ),
    ] = 2.0,
) -> None:
    """Report capture devices and measure real FPS/latency of the configured camera."""
    try:
        cfg = load_config(yaml_path=config) if config is not None else load_config()
    except IrisFlowError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from exc

    typer.echo("IrisFlow capture diagnostics")
    typer.echo("=" * 60)

    cameras = enumerate_cameras(max_index=max_index)
    _print_camera_table(cameras)

    available = [c for c in cameras if c.available]
    if not available:
        typer.echo("")
        typer.echo("No camera devices detected.")
        typer.echo("  - Check that a webcam is connected and not in use by another app.")
        typer.echo("  - On Linux, verify that your user is in the `video` group.")
        raise typer.Exit(code=1)

    typer.echo("")
    typer.echo(f"Measuring device_id={cfg.camera.device_id} for {duration:.1f}s...")

    source = WebcamSource(
        device_id=cfg.camera.device_id,
        width=cfg.camera.width,
        height=cfg.camera.height,
        fps=cfg.camera.fps,
        reconnect_backoff_ms=cfg.camera.reconnect_backoff_ms,
    )
    try:
        source.open()
        measurement = measure_capture(source, duration_s=duration)
    finally:
        source.close()

    _print_measurement(measurement, requested_fps=cfg.camera.fps)

    if measurement.frames_captured == 0:
        typer.echo("")
        typer.echo(f"[warn] Device {cfg.camera.device_id} enumerates but produced no frames.")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def _print_camera_table(cameras: list[CameraInfo]) -> None:
    typer.echo("")
    typer.echo(f"{'id':>3} {'ok':>3} {'width':>6} {'height':>6} {'fps':>6}  backend")
    typer.echo("-" * 60)
    for cam in cameras:
        ok = "yes" if cam.available else "no"
        typer.echo(
            f"{cam.device_id:>3} {ok:>3} {cam.width:>6} {cam.height:>6} "
            f"{cam.fps_reported:>6.1f}  {cam.backend}"
        )


def _print_measurement(measurement: CaptureMeasurement, *, requested_fps: int) -> None:
    typer.echo("")
    typer.echo("Measured capture:")
    typer.echo(f"  frames captured   : {measurement.frames_captured}")
    typer.echo(f"  duration          : {measurement.duration_s:.2f} s")
    typer.echo(f"  measured FPS      : {measurement.fps_measured:.2f}  (requested {requested_fps})")
    typer.echo(f"  read latency      : {measurement.mean_read_latency_ms:.2f} ms (mean)")
