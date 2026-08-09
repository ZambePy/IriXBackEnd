"""Sweep 12 preprocessing combinations on a recorded video to find working config.

Usage:
    uv run python scripts/config_sweep.py --video axis_probe.mp4

This tests all combinations of:
    channel_order ∈ {RGB, BGR}
    normalization ∈ {unit, signed, imagenet}
    swap_eye     ∈ {false, true}

For each combination it reports: amplitude of raw_x, amplitude of raw_y,
NaN count, and zero-sample count. The combination with highest sum amplitude
is likely the correct preprocessing.

NOTE: swap_eye here swaps which eye patch goes to left_eye vs right_eye in
the model input. It does NOT modify preprocessing/builder.py — this script
injects the swap directly.
"""
from __future__ import annotations

import pathlib
import sys
import threading
from dataclasses import dataclass, field
from typing import Annotated

import numpy as np
import typer

_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

app = typer.Typer(add_completion=False)


@dataclass
class _SweepResult:
    channel_order: str
    normalization: str
    swap_eye: bool
    n_samples: int = 0
    n_nan: int = 0
    n_zero: int = 0
    xs: list[float] = field(default_factory=list)
    ys: list[float] = field(default_factory=list)

    @property
    def amp_x(self) -> float:
        return (max(self.xs) - min(self.xs)) if self.xs else 0.0

    @property
    def amp_y(self) -> float:
        return (max(self.ys) - min(self.ys)) if self.ys else 0.0


def _make_preprocessor_cls(
    cfg_face_size: tuple[int, int],
    cfg_eye_size: tuple[int, int],
    channel_order: str,
    normalization: str,
    swap_eye: bool,
) -> type:
    """Return a preprocessor class bound to the given combination.

    Using a factory avoids the loop-variable-capture bug that ruff B023
    would otherwise flag when defining the class inline in the loop body.
    """
    from irisflow.core.types import FaceDetection, Frame, ModelInput
    from irisflow.preprocessing.builder import ModelInputBuilder

    class _SwappingPreprocessor:
        def __init__(self) -> None:
            self._inner = ModelInputBuilder(
                face_input_size=cfg_face_size,
                eye_input_size=cfg_eye_size,
                input_channel_order="BGR",
                output_channel_order=channel_order,  # type: ignore[arg-type]
                normalization=normalization,  # type: ignore[arg-type]
            )

        def build(self, frame: Frame, detection: FaceDetection) -> ModelInput:
            inp = self._inner.build(frame, detection)
            if swap_eye:
                return ModelInput(
                    face=inp.face,
                    left_eye=inp.right_eye,
                    right_eye=inp.left_eye,
                    rect=inp.rect,
                )
            return inp

    return _SwappingPreprocessor


@app.command()
def main(
    video: Annotated[pathlib.Path, typer.Option("--video", help="Path to recorded video file.")],
    config: Annotated[pathlib.Path | None, typer.Option("--config", "-c")] = None,
    max_frames: Annotated[
        int, typer.Option("--max-frames", help="Max frames per combination.")
    ] = 300,
) -> None:
    """Sweep all 12 preprocessing combinations and report gaze amplitudes."""
    from irisflow.capture.video_file import VideoFileSource
    from irisflow.config.loader import load_config
    from irisflow.core.events import RawGazeReady
    from irisflow.pipeline.orchestrator import build_pipeline
    from irisflow.pipeline.runner import PipelineRunner

    if not video.exists():
        typer.echo(f"[error] video not found: {video}", err=True)
        raise typer.Exit(1)

    cfg = load_config(yaml_path=config) if config is not None else load_config()
    results: list[_SweepResult] = []

    combinations = [
        (ch, norm, swap)
        for ch in ("RGB", "BGR")
        for norm in ("unit", "signed", "imagenet")
        for swap in (False, True)
    ]

    for channel_order, normalization, swap_eye in combinations:
        result = _SweepResult(
            channel_order=channel_order, normalization=normalization, swap_eye=swap_eye
        )

        preprocessor_cls = _make_preprocessor_cls(
            cfg_face_size=cfg.model.face_input_size,
            cfg_eye_size=cfg.model.eye_input_size,
            channel_order=channel_order,
            normalization=normalization,
            swap_eye=swap_eye,
        )

        source = VideoFileSource(path=video, loop=False)
        components = build_pipeline(cfg, source=source, preprocessor=preprocessor_cls())
        frames_seen = {"n": 0}

        def _on_raw(event: object, _r: _SweepResult = result, _fs: dict = frames_seen) -> None:
            if not isinstance(event, RawGazeReady):
                return
            _fs["n"] += 1
            _r.n_samples += 1
            x, y = event.x, event.y
            if np.isnan(x) or np.isnan(y):
                _r.n_nan += 1
            elif x == 0.0 and y == 0.0:
                _r.n_zero += 1
            else:
                _r.xs.append(x)
                _r.ys.append(y)

        components.bus.subscribe(RawGazeReady, _on_raw)
        runner = PipelineRunner(components)
        stop = threading.Event()

        def _run(_runner=runner, _stop=stop, _max=max_frames) -> None:
            _runner.run(max_ticks=_max * 3)
            _stop.set()

        bg = threading.Thread(target=_run, daemon=True)
        bg.start()
        stop.wait(timeout=60.0)
        runner.stop()
        bg.join(timeout=3.0)
        results.append(result)

    # Print table
    typer.echo("")
    typer.echo(
        f"{'channel_order':<14} {'norm':<10} {'swap':<6} "
        f"{'n':<6} {'nan':<5} {'amp_x':<8} {'amp_y':<8}"
    )
    typer.echo("-" * 65)
    for r in sorted(results, key=lambda r: r.amp_x + r.amp_y, reverse=True):
        typer.echo(
            f"{r.channel_order:<14} {r.normalization:<10} {r.swap_eye!s:<6} "
            f"{r.n_samples:<6} {r.n_nan:<5} {r.amp_x:<8.4f} {r.amp_y:<8.4f}"
        )

    best = max(results, key=lambda r: r.amp_x + r.amp_y)
    typer.echo("")
    typer.echo(
        f"Best: channel_order={best.channel_order} normalization={best.normalization} "
        f"swap_eye={best.swap_eye}  (amp_x={best.amp_x:.4f} amp_y={best.amp_y:.4f})"
    )
    if best.amp_x + best.amp_y < 0.15:
        typer.echo(
            "[warn] No combination yields amp > 0.15. "
            "See T6b: compare pipeline with eval harness."
        )


if __name__ == "__main__":
    app()
