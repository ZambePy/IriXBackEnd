# Configuration schema

Reference for every field in `configs/default.yaml`. `src/irisflow/config/schema.py`
(Pydantic) is the source of truth — this document mirrors it. If they disagree,
the code wins and this file should be updated in the same commit.

## Override precedence

**CLI > env (`IRISFLOW_*`) > YAML > code defaults.**

Nested env keys use double underscore as the separator. Values that look like
JSON (`true`, `false`, `null`, numbers, `[..]`, `{..}`) are parsed as JSON;
everything else is treated as a string and coerced by Pydantic.

```
IRISFLOW_CAMERA__DEVICE_ID=1        # → camera.device_id = 1
IRISFLOW_CONTROL__ENABLED=true      # → control.enabled = true
IRISFLOW_LOGGING__LEVEL=DEBUG       # → logging.level = "DEBUG"
```

Any invalid value fails initialization with `ConfigError` naming the exact
path of the offending field.

## Sections

### `camera` — Sprint 3
| Field | Type | Default | Notes |
|---|---|---|---|
| `device_id` | `int ≥ 0` | `0` | OpenCV `VideoCapture` index |
| `width` | `int > 0` | `1280` | Requested capture width (px) |
| `height` | `int > 0` | `720` | Requested capture height (px) |
| `fps` | `int 1..240` | `30` | Requested frame rate |
| `reconnect_backoff_ms` | `int ≥ 0` | `500` | Delay between reconnect attempts |

### `detection` — Sprint 4
| Field | Type | Default | Notes |
|---|---|---|---|
| `refine_landmarks` | `bool` | `true` | Required for iris landmarks |
| `roi_margin` | `float 0..1` | `0.20` | Extra padding around raw ROI |
| `smoothing_alpha` | `float 0..1` | `0.4` | EMA over ROI boxes |
| `lost_hysteresis_frames` | `int ≥ 1` | `5` | Frames without face before LOST |

### `model` — Sprint 6
| Field | Type | Default | Notes |
|---|---|---|---|
| `backend` | `keras \| onnx` | `keras` | Backend selected in `inference/registry.py` |
| `path` | `Path` | `models/gaze_cnn_best.keras` | Artifact path |
| `warmup_iterations` | `int ≥ 0` | `3` | Dummy inferences during boot |
| `channel_order` | `RGB \| BGR` | `RGB` | Confirm on S6 sanity check |
| `normalization` | `unit \| signed \| imagenet` | `unit` | Must mirror the training scheme |
| `face_input_size` | `(int, int)` | `(224, 224)` | Face crop size |
| `eye_input_size` | `(int, int)` | `(112, 112)` | Eye crop size |

### `calibration` — Sprint 8
| Field | Type | Default | Notes |
|---|---|---|---|
| `points` | `5 \| 9 \| 13` | `9` | Grid density |
| `samples_per_point` | `int ≥ 1` | `30` | Samples collected per target |
| `stabilization_ms` | `int ≥ 0` | `600` | Wait time before sampling |
| `model_kind` | `affine \| polynomial \| ridge` | `polynomial` | Correction model family |
| `polynomial_degree` | `int 1..4` | `2` | Only used when `model_kind = polynomial` |
| `max_residual_px` | `float > 0` | `60.0` | Fail-loud threshold on hold-out error |
| `profiles_dir` | `Path` | `configs/profiles` | Persistence directory |

### `mapping` — Sprint 9
| Field | Type | Default | Notes |
|---|---|---|---|
| `screen_id` | `int ≥ 0` | `0` | Monitor index |
| `clamp_margin_px` | `int ≥ 0` | `8` | Border margin the cursor cannot cross |

### `filtering` — Sprint 9
| Field | Type | Default | Notes |
|---|---|---|---|
| `chain` | `list[filter name]` | `[outlier, one_euro, fixation]` | Execution order |
| `outlier.max_velocity_px_per_s` | `float > 0` | `6000.0` | Physically implausible speed threshold |
| `ema.alpha` | `0 < float ≤ 1` | `0.4` | Baseline smoother (only used when `ema` in `chain`) |
| `one_euro.min_cutoff` | `float > 0` | `1.0` | One Euro base cutoff (Hz) |
| `one_euro.beta` | `float ≥ 0` | `0.007` | Speed-driven cutoff coefficient |
| `one_euro.d_cutoff` | `float > 0` | `1.0` | Cutoff for derivative signal |
| `fixation.velocity_threshold_px_per_s` | `float > 0` | `40.0` | Below this → fixation |

### `control` — Sprint 10
| Field | Type | Default | Notes |
|---|---|---|---|
| `enabled` | `bool` | `false` | Cursor is **off** by default |
| `rate_limit_hz` | `int > 0` | `60` | Cap on mouse move rate |
| `dwell.radius_px` | `int > 0` | `40` | Dwell cluster radius |
| `dwell.duration_ms` | `int > 0` | `800` | Time-in-radius for a click |
| `dwell.refractory_ms` | `int ≥ 0` | `400` | Post-click cooldown |
| `safety.kill_switch` | `str` | `ctrl+alt+esc` | Global hotkey (Sprint 10) |
| `safety.pause_on_face_lost_ms` | `int ≥ 0` | `2000` | Auto-pause threshold |
| `safety.rest_zone_px` | `(int, int, int, int)` | `(0, 0, 0, 0)` | `x, y, w, h`; `(0,0,0,0)` disables |
| `safety.watchdog_timeout_ms` | `int ≥ 0` | `500` | Cursor is disabled when no gaze tick lands in this window |

### `telemetry` — Sprint 11
| Field | Type | Default | Notes |
|---|---|---|---|
| `metrics_enabled` | `bool` | `true` | Per-stage latency histograms |
| `record_sessions` | `bool` | `false` | Save frames + intermediate data for replay |
| `recordings_dir` | `Path` | `data/recordings` | Destination directory |

### `logging` — Sprint 1
| Field | Type | Default | Notes |
|---|---|---|---|
| `level` | `DEBUG \| INFO \| WARNING \| ERROR \| CRITICAL` | `INFO` | Applies to all sinks |
| `json_file` | `Path \| null` | `data/logs/irisflow.jsonl` | `null` disables the JSON sink |
| `human_console` | `bool` | `true` | Colored one-line events on stderr |

The pipeline stages bind `frame_id` (and often `stage`) into the ambient
context via `irisflow.logging.bind_frame` / `bound(...)` so every event
downstream is traceable back to the frame that produced it.
