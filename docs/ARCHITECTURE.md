# Arquitetura do IrisFlow

Documento de referência para desenvolvedores que precisam navegar no
código. Complementa (não substitui) `SPRINTS.MD §2` — este arquivo é
focado em "por onde começar a ler" e "onde vive cada responsabilidade".

## 1. Camadas

```
┌──────────────────────────────────────────────────────────────────────┐
│  APLICAÇÃO                                                           │
│  cli/ (Typer)                            api/ (FastAPI + WebSocket)  │
└───────────────┬──────────────────────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  ORQUESTRAÇÃO                                                        │
│  pipeline/  — monta estágios, roda o loop, publica no bus            │
└───────────────┬──────────────────────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  DOMÍNIO (puro, sem I/O, 100% testável offline)                      │
│  preprocessing/   calibration/   mapping/   filtering/               │
└───────────────┬──────────────────────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  ADAPTADORES (I/O, bibliotecas externas, SO)                         │
│  capture/  detection/  inference/  control/                          │
└───────────────┬──────────────────────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  INFRAESTRUTURA                                                      │
│  config/  telemetry/  logging/                                       │
└───────────────┬──────────────────────────────────────────────────────┘
                ▼
                          core/  (tipos, protocols, geometria, clock)
```

**Regra dura (verificada por `import-linter` em CI):** setas apontam
sempre para baixo. `core/` não importa ninguém. Domínio nunca importa
adapter concreto — usa `Protocol`s de `core/interfaces.py`.

O contrato completo vive em `pyproject.toml [tool.importlinter]`. Layers
mesmos siblings (dentro de `|`) não podem se importar entre si — foi por
isso que, na Sprint 11, `SessionReplayer` migrou de `telemetry/` para
`pipeline/`: precisa usar `control/` (dwell + safety), e `telemetry` é
sibling de `control`.

## 2. Fluxo de um frame (regime TRACKING)

```
┌────────┐  ┌────────┐  ┌───────────┐  ┌────────┐  ┌──────────┐  ┌────────┐  ┌────────┐  ┌──────┐
│capture ├─►│detect  ├─►│preprocess ├─►│infer   ├─►│calibrate ├─►│  map   ├─►│ filter ├─►│ bus  │
└────────┘  └────────┘  └───────────┘  └────────┘  └──────────┘  └────────┘  └────────┘  └──┬───┘
                              (RGB, 224/112, rect)                                              │
                                                                                                ▼
                                                                          ControlSink  MetricsSink  WSSink
```

- **capture** (`capture/webcam.py`) — thread dedicada + fila `maxsize=1`
  com descarte *drop-oldest*. Latência importa mais que completude.
- **detect** (`detection/mediapipe_detector.py`) — FaceMesh 478 landmarks
  + `TrackedFaceDetector` (histerese anti-tremor, EMA sobre ROIs).
- **preprocess** (`preprocessing/builder.py`) — crops + resize + normalize
  + `build_rect_vector`. Zero alocação nova no loop quente.
- **infer** (`inference/keras_backend.py`) — o *único* estimador de gaze
  do sistema (`gaze_cnn_best.keras`, SPRINTS §1.1). ONNX existe como
  arquitetura mas só serve para futuros exports de grafo completo.
- **calibrate** (`calibration/models.py`) — polinomial 2ª ordem por perfil
  de usuário; passthrough + warning quando não há perfil carregado.
- **map** (`mapping/screen.py`) — normalizado → pixel + clamp por margem.
- **filter** (`filtering/chain.py`) — outlier → One Euro → fixation. Puro,
  testado com sinais sintéticos e `FakeClock`.

O runner (`pipeline/runner.py`) instrumenta cada etapa via
`MetricsRecorder.time()`, publica `RawGazeReady` + `GazeUpdated` no bus,
alimenta o watchdog e sobrevive a exceções por estágio (frame perdido
não derruba o pipeline).

## 3. Modelo de concorrência

Duas threads + um asyncio loop opcional:

```
Thread A: capture    (webcam.py — leitura contínua, fila descarta velhos)
Thread B: pipeline   (runner.py — consome fila, emite bus, ~30 Hz)
Loop asyncio (opcional): api/ws.py — bridge para WebSocket via
                        loop.call_soon_threadsafe (ver api/session.py).
```

- O bus (`pipeline/bus.py`) é **síncrono e single-thread** — publica da
  thread do runner, consumidores rodam inline. Sinks pesados (WS) fazem
  o hop thread → asyncio internamente.
- O `Clock` (`core/clock.py`) é injetável: `SystemClock` em produção,
  `FakeClock` / `AutoAdvanceClock` em testes. Todo estágio que precisa de
  tempo (dwell, filtros, safety, replayer) recebe o clock por
  construção.

## 4. Máquina de estados

```
IDLE ─start─► CALIBRATING ─fit ok─► TRACKING ◄─face_lost─► LOST
                                       │
                                       ▼ pause/safety
                                     PAUSED
```

Em `LOST` e `PAUSED` o cursor **não se move** — política de
acessibilidade não-negociável (SPRINTS §1.2). Toda transição inválida
levanta `ValueError` — falha ruidosa é sempre preferível a comportamento
silenciosamente errado.

Fonte: `pipeline/state.py::PipelineStateMachine`.

## 5. Event bus

`pipeline/bus.py::EventBus` — pub/sub síncrono tipado por classe do
evento. Consumidor levanta exceção? é logado como `bus.handler_raised` e
os demais sinks continuam recebendo (SPRINTS §11.3 regra 4: bus nunca
derruba o pipeline).

Eventos publicados (source of truth: `core/events.py`):

| Evento | Publisher | Quando |
|---|---|---|
| `RawGazeReady` | runner (após infer) | todo frame com detecção |
| `GazeUpdated` | runner (após filter) | todo frame com pipeline completo |
| `FaceLost` / `FaceAcquired` | runner | histerese do detector |
| `StateChanged` | state machine | toda transição |
| `CalibrationProgress` | calibration/session | por sample/target |
| `DwellProgress` / `DwellClick` | control_sink | por sample dentro do raio |
| `SafetyPaused` / `SafetyResumed` | control_sink | kill switch / watchdog / face lost |

Wire contracts para WS (`api/schemas.py`) são derivados destes eventos —
nunca o contrário.

## 6. Testabilidade

**Regra ouro:** nenhum teste da suite CI toca a webcam, o modelo real
ou o mouse do desenvolvedor. Dublês em `tests/fixtures/stubs.py`:

- `SyntheticFrameSource` — frames determinísticos.
- `StubFaceDetector` — ROIs fixas ou roteirizadas.
- `StubGazeEstimator` — trajetória programada (círculo, degrau, ruído).
  **Nunca importado por `src/`** — verificado por `import-linter`.
- `NoOpCursorController` — grava chamadas, não move o cursor real.
- `FakeClock` — tempo controlado para filtros determinísticos.

Testes marcados `@pytest.mark.model` rodam contra os artefatos reais em
`models/`. Ficam de fora do CI normal, mas rodam ad-hoc quando o modelo
muda.

## 7. Convenções que não estão em código

- **Frozen dataclasses** com `slots=True` para tipos do domínio — imutáveis,
  hash-friendly, footprint mínimo.
- **Pydantic v2** com plugin mypy ativo (`[tool.pydantic-mypy]`). Todo
  campo default composto usa `Field(default_factory=Cls)` — necessário
  para `frozen=True` e para o strict-mypy.
- Timestamps são sempre `float` em segundos monotônicos. Frame IDs são
  inteiros que só crescem.
- **Layer sibling? Passe primitivos.** Nunca importe Pydantic settings
  numa camada de baixo — o padrão está em
  `pipeline/orchestrator.py::filter_config_from_app` (traduz
  `FilterConfig` → `ChainConfig`).
