"""``irisflow bench`` — inference latency benchmark + parity + sanity check.

Three modes, mutually orthogonal so the developer can pick one per run:

* ``--latency`` — measure per-frame inference time (p50/p95/max over N frames).
* ``--compare-backends`` — parity check between Keras and ONNX. Falls
  back to the Keras-embedding-vs-ONNX-encoder comparison when the ONNX
  artifact is encoder-only (current release).
* ``--sanity-check`` — feed the Keras backend a series of synthetic
  rect-vector positions (centre and four corners) and print the raw
  gaze output. **This is the manual check the Sprint 6 gate depends on**;
  compare the printout against MODEL_CARD.md §6 "Sanity check log".
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from irisflow.config.loader import load_config
from irisflow.core.exceptions import IrisFlowError
from irisflow.core.types import ModelInput

__all__ = ["bench"]


def bench(
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to the YAML config. Defaults to configs/default.yaml.",
        ),
    ] = None,
    latency: Annotated[
        bool,
        typer.Option("--latency", help="Measure Keras inference latency."),
    ] = False,
    compare_backends: Annotated[
        bool,
        typer.Option(
            "--compare-backends",
            help="Parity check between Keras and ONNX artifacts.",
        ),
    ] = False,
    sanity_check: Annotated[
        bool,
        typer.Option(
            "--sanity-check",
            help="Print gaze output at (centre, TL, TR, BL, BR) rect positions.",
        ),
    ] = False,
    iterations: Annotated[
        int,
        typer.Option("--iterations", min=1, help="Frames per latency run."),
    ] = 100,
) -> None:
    """Run one or more inference-layer diagnostics."""
    if not (latency or compare_backends or sanity_check):
        typer.echo(
            "[error] Pick at least one of --latency / --compare-backends / "
            "--sanity-check.",
            err=True,
        )
        raise typer.Exit(code=2)

    try:
        cfg = load_config(yaml_path=config) if config is not None else load_config()
    except IrisFlowError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from exc

    if latency:
        _run_latency(cfg, iterations=iterations)
    if sanity_check:
        _run_sanity_check(cfg)
    if compare_backends:
        _run_backend_comparison(cfg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _dummy_input(*, x_norm: float = 0.5, y_norm: float = 0.5) -> ModelInput:
    """Synthetic ModelInput centred at ``(x_norm, y_norm)`` in rect space.

    Pixels are mid-gray; only ``rect`` varies. Enough signal to exercise
    the graph but not enough to derive a meaningful gaze — the point of
    ``--sanity-check`` is that the model output should nevertheless
    respond monotonically to the rect changes.
    """
    face = np.full((224, 224, 3), 0.5, dtype=np.float32)
    eye = np.full((112, 112, 3), 0.5, dtype=np.float32)
    rect = np.zeros(12, dtype=np.float32)
    # Face box near (x_norm, y_norm), size ~0.4x0.4.
    rect[0] = max(0.0, x_norm - 0.2)
    rect[1] = max(0.0, y_norm - 0.2)
    rect[2] = 0.4
    rect[3] = 0.4
    # Left eye
    rect[4] = max(0.0, x_norm - 0.10)
    rect[5] = max(0.0, y_norm - 0.05)
    rect[6] = 0.06
    rect[7] = 0.03
    # Right eye
    rect[8] = min(1.0, x_norm + 0.04)
    rect[9] = max(0.0, y_norm - 0.05)
    rect[10] = 0.06
    rect[11] = 0.03
    return ModelInput(face=face, left_eye=eye, right_eye=eye, rect=rect)


def _build_keras(cfg) -> object:  # type: ignore[no-untyped-def]
    from irisflow.inference.registry import build_gaze_estimator
    from irisflow.inference.warmup import warm_up_backend

    est = build_gaze_estimator(
        backend="keras",
        model_path=cfg.model.path,
        channel_order=cfg.model.channel_order,
        normalization=cfg.model.normalization,
    )
    warm_up_backend(est, iterations=cfg.model.warmup_iterations)
    return est


def _run_latency(cfg, *, iterations: int) -> None:  # type: ignore[no-untyped-def]
    typer.echo(f"Loading Keras backend ({cfg.model.path})...")
    est = _build_keras(cfg)
    mi = _dummy_input()
    durations_ms: list[float] = []
    typer.echo(f"Running {iterations} inferences...")
    for _ in range(iterations):
        raw = est.predict(mi)  # type: ignore[attr-defined]
        durations_ms.append(raw.inference_ms)
    arr = np.asarray(durations_ms)
    typer.echo("")
    typer.echo("Keras inference latency (ms):")
    typer.echo(f"  samples : {iterations}")
    typer.echo(f"  p50     : {np.percentile(arr, 50):.2f}")
    typer.echo(f"  p95     : {np.percentile(arr, 95):.2f}")
    typer.echo(f"  max     : {arr.max():.2f}")
    typer.echo(f"  mean    : {arr.mean():.2f}")


def _run_sanity_check(cfg) -> None:  # type: ignore[no-untyped-def]
    typer.echo(f"Loading Keras backend ({cfg.model.path})...")
    est = _build_keras(cfg)
    positions = [
        ("centre",       0.50, 0.50),
        ("top-left",     0.20, 0.20),
        ("top-right",    0.80, 0.20),
        ("bottom-left",  0.20, 0.80),
        ("bottom-right", 0.80, 0.80),
    ]
    typer.echo("")
    typer.echo("Sanity check — raw gaze at synthetic rect positions:")
    typer.echo(f"  {'position':<14} {'gaze_x':>8} {'gaze_y':>8}")
    typer.echo(f"  {'-'*14} {'-'*8} {'-'*8}")
    seen: list[tuple[str, float, float]] = []
    for name, xn, yn in positions:
        raw = est.predict(_dummy_input(x_norm=xn, y_norm=yn))  # type: ignore[attr-defined]
        typer.echo(f"  {name:<14} {raw.x:>8.4f} {raw.y:>8.4f}")
        seen.append((name, raw.x, raw.y))
    typer.echo("")
    typer.echo(
        "Interpretation: with pixels held constant and only the rect vector "
        "moving, we expect a monotonic response along each axis. Compare the "
        "printout to models/MODEL_CARD.md §6 and update that log."
    )


def _run_backend_comparison(cfg) -> None:  # type: ignore[no-untyped-def]
    from irisflow.inference.onnx_backend import _load_session, onnx_output_kind
    from irisflow.inference.parity import (
        compare_gaze_backends,
        compare_keras_embedding_vs_onnx,
    )
    from irisflow.inference.registry import build_gaze_estimator

    onnx_path = Path("models/gaze_encoder.onnx")
    if not onnx_path.exists():
        typer.echo(f"[error] ONNX artifact not found at {onnx_path}", err=True)
        raise typer.Exit(code=2)

    session = _load_session(onnx_path)
    kind = onnx_output_kind(session)
    inputs = [_dummy_input(x_norm=x, y_norm=y)
              for x in (0.25, 0.5, 0.75) for y in (0.25, 0.5, 0.75)]

    if kind == "gaze_xy":
        typer.echo("ONNX exports full gaze estimator — running head-to-head parity.")
        keras_est = build_gaze_estimator(
            backend="keras", model_path=cfg.model.path,
            channel_order=cfg.model.channel_order, normalization=cfg.model.normalization,
        )
        onnx_est = build_gaze_estimator(
            backend="onnx", model_path=onnx_path,
            channel_order=cfg.model.channel_order, normalization=cfg.model.normalization,
        )
        report = compare_gaze_backends(keras_est, onnx_est, inputs)
    else:
        typer.echo(
            f"ONNX is an {kind!r} (see MODEL_CARD.md §5). Comparing Keras' "
            "embedding layer against the ONNX encoder output."
        )
        report = compare_keras_embedding_vs_onnx(
            cfg.model.path, onnx_path, inputs, tolerance=1e-3
        )
    typer.echo("")
    typer.echo("Parity report:")
    typer.echo(f"  samples       : {report.samples}")
    typer.echo(f"  max_abs_diff  : {report.max_abs_diff:.6f}")
    typer.echo(f"  mean_abs_diff : {report.mean_abs_diff:.6f}")
    typer.echo(f"  tolerance     : {report.tolerance:.6f}")
    typer.echo(f"  passed        : {report.passed}")
    if not report.passed:
        raise typer.Exit(code=1)
