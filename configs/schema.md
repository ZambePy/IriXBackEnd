# Configuration schema

Reference for every field in `configs/default.yaml`. Fully populated on Sprint 1
when `config/schema.py` (Pydantic) becomes the source of truth. Until then this
file describes the intended shape — no runtime is enforcing it yet.

Override precedence: **CLI > env (`IRISFLOW_*`) > YAML > code defaults**.
Nested env keys use double underscore: `IRISFLOW_CAMERA__DEVICE_ID=1`.

## Sections

- `camera` — capture parameters (Sprint 3)
- `detection` — MediaPipe / ROI tracking (Sprint 4)
- `model` — inference backend + I/O contract (Sprint 6)
- `calibration` — per-user correction (Sprint 8)
- `mapping` — normalized gaze → screen pixels (Sprint 9)
- `filtering` — temporal chain (Sprint 9)
- `control` — cursor + dwell + safety (Sprint 10)
- `telemetry` — metrics + recording (Sprint 11)
- `logging` — structlog sinks (Sprint 1)

Detailed per-field docs are added Sprint by Sprint as each module lands.
