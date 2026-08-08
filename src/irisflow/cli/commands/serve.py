"""``irisflow serve`` — bring up the FastAPI + WebSocket surface (Sprint 12).

Thin wrapper around :func:`irisflow.api.create_app` + ``uvicorn.run``.
The pipeline itself boots inside the app's lifespan handler, so
``Ctrl+C`` stops both the HTTP server and the background pipeline thread.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from irisflow.config.loader import load_config
from irisflow.core.exceptions import IrisFlowError

__all__ = ["serve"]


def serve(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to the YAML config."),
    ] = None,
    host: Annotated[
        str,
        typer.Option("--host", help="Bind address. Default 127.0.0.1 (local only)."),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", min=1, max=65535, help="TCP port to listen on."),
    ] = 8000,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="uvicorn log level (debug, info, warning, error, critical).",
        ),
    ] = "info",
) -> None:
    """Serve ``/ws/gaze`` + ``/health`` + calibration endpoints."""
    try:
        cfg = load_config(yaml_path=config) if config is not None else load_config()
    except IrisFlowError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from exc

    # Import lazily so the CLI (and its tests) don't pay the FastAPI/uvicorn
    # import cost when running unrelated subcommands.
    try:
        import uvicorn

        from irisflow.api import create_app
    except ImportError as exc:
        typer.echo(
            "[error] API extras not installed. Run: uv sync --extra api",
            err=True,
        )
        raise typer.Exit(code=2) from exc

    app = create_app(config=cfg)
    typer.echo(f"IrisFlow API on http://{host}:{port} — Ctrl+C to stop.")
    uvicorn.run(app, host=host, port=port, log_level=log_level)
