# IrisFlow — Backend

Webcam-based eye tracking backend for accessibility, built for people
with ALS. Turns a standard webcam feed into a screen-cursor position via
a trained CNN (`gaze_cnn_best.keras`) plus per-user calibration and
temporal filtering.

The full development guide — architecture, sprint roadmap, contracts,
Definition of Done — lives in [`SPRINTS.MD`](SPRINTS.MD). This README is
the *user*-facing entry point: install, first run, calibrate, serve.

For deeper reading:

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — how the layers fit.
- [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) — dev setup + quality
  gates.
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — closed ADRs (`rect`,
  channels, calibration, cursor, click).
- [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) — the fix list.
- [`configs/schema.md`](configs/schema.md) — every config field.
- [`models/MODEL_CARD.md`](models/MODEL_CARD.md) — the gaze CNN contract.

## 1. Requirements

- Python **3.11+**
- [`uv`](https://docs.astral.sh/uv/) for dependency management (fast,
  reproducible envs)
- A webcam **only** to actually run the pipeline — the test suite never
  touches one.

## 2. Install (zero-to-run)

```bash
# 1. clone + enter
git clone <your-fork-url> irisflow-backend
cd irisflow-backend

# 2. base install (dev tools + testing)
uv sync --extra dev

# 3. add the extras you need
uv sync --extra dev --extra inference-keras       # to actually infer gaze
uv sync --extra dev --extra api                   # to serve WebSocket
uv sync --extra dev --extra inference-keras --extra api   # both
```

Model artifacts are **not** in the repo (Git LFS or out-of-repo). Drop
them in `models/`:

- `models/gaze_cnn_best.keras` — required for `run`, `bench --latency`,
  `serve`.
- `models/gaze_encoder.onnx` — optional; only used by `bench
  --compare-backends`.
- `models/face_landmarker.task` — required for detection (MediaPipe 1.0
  asset).

## 3. Sanity check

Before pointing the cursor at anything:

```bash
# List cameras and measure real FPS — never lies.
uv run irisflow doctor

# Run the model on synthetic input to confirm it loaded and outputs
# stay in [0, 1]². Exit code 0 = safe to proceed.
uv run irisflow bench --sanity-check

# Live overlay of face + eye ROIs. Iterate framing / lighting here.
uv run irisflow preview
```

## 4. First run

Cursor control is **off by default** — the process only prints gaze to
the terminal:

```bash
uv run irisflow run
```

Enable the OS cursor once you're comfortable with the kill switch
(default `Ctrl+Alt+Esc`):

```bash
uv run irisflow run --cursor
```

Record a session for later replay (deterministic, `FakeClock` based):

```bash
uv run irisflow run --record --session-id demo-01
uv run irisflow replay data/recordings/demo-01.jsonl
```

## 5. Calibrate

Fresh gaze is roughly correct but biased per user — calibration corrects
that.

```bash
uv run irisflow calibrate --profile maria
```

The 9-point session takes about a minute. Profile is saved to
`configs/profiles/maria.json`. Then:

```bash
uv run irisflow run --cursor --profile maria
```

If quality is below `calibration.max_residual_px`, the CLI names the
problematic targets — recalibrate just those.

## 6. Serve the WebSocket API (Sprint 12)

For frontend integration:

```bash
uv run irisflow serve --host 127.0.0.1 --port 8000
```

Endpoints:

- `GET /health` — pipeline state, FPS, backend.
- `GET /config` / `PATCH /config` — read + narrow runtime tweaks.
- `GET /profiles` — enumerate calibration profiles.
- `GET /calibration/status` / `POST /calibration/abort`.
- `WS /ws/gaze` — bidirectional. Message schemas in
  `src/irisflow/api/schemas.py`.

Default bind is `127.0.0.1` — never expose without explicit consent.

## 7. Quality gates (same as CI)

```bash
.venv/Scripts/pytest.exe
.venv/Scripts/ruff.exe check src tests
.venv/Scripts/ruff.exe format --check src tests
.venv/Scripts/mypy.exe src/irisflow
.venv/Scripts/lint-imports.exe
```

Current status: 572 tests, 82% coverage globally (≥ 90% on the pure
domain layers).

## 8. Layout

Top-level tree — full description in `SPRINTS.MD` §3.

```
configs/     YAML + per-user profiles (profiles gitignored)
data/        logs, recordings, calibration state (all gitignored)
docs/        ARCHITECTURE, CONTRIBUTING, DECISIONS, TROUBLESHOOTING
models/      trained artifacts (gitignored)
src/irisflow/
  core/          pure domain: types, protocols, geometry, clock, events
  config/        Pydantic settings + YAML loader
  capture/       webcam / video-file / synthetic frame sources
  detection/     MediaPipe face+eye landmarks, ROI tracking
  preprocessing/ crops, resize, normalization, rect vector
  inference/     Keras / ONNX gaze backends
  calibration/   per-user correction models + persistence
  mapping/       normalized gaze -> screen pixel
  filtering/     outlier, One Euro, fixation classifier
  control/       cursor + dwell click + safety (kill switch)
  pipeline/      orchestration, state machine, event bus, replay
  telemetry/     metrics, recording, report
  logging/       structlog setup
  api/           FastAPI + WebSocket (Sprint 12)
  cli/           Typer entry points
tests/       unit / integration / e2e + shared stubs
scripts/     inspect_model.py, benchmark.py
```

## 9. Non-negotiable premise

The trained model (`gaze_cnn_best.keras`) is the **only** gaze estimator
in production. No sprint retrains it, replaces it with a heuristic, or
ships a temporary estimator to unblock work. See `SPRINTS.MD §1.1`.

If gaze looks wrong: the bug is in preprocessing (channels,
normalization, `rect`), calibration or filtering — pause and diagnose
before touching the model.
