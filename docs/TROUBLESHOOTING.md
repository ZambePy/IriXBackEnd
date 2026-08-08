# Troubleshooting

Erros mais comuns e o que fazer. Toda mensagem começando com `[error]`
que o CLI emite é uma `IrisFlowError` — a hierarquia inteira está em
`src/irisflow/core/exceptions.py`.

## Instalação

### `ModuleNotFoundError: No module named 'fastapi'`

Ao rodar `irisflow serve` sem o extra `api`:

```
[error] API extras not installed. Run: uv sync --extra api
```

**Ação:** `uv sync --extra api`.

### `ModuleNotFoundError: No module named 'tensorflow'`

Ao rodar `irisflow run` ou `irisflow bench --latency` sem o extra
`inference-keras`:

**Ação:** `uv sync --extra inference-keras`. TensorFlow puxa ~700 MB —
não instalar por engano em CI.

### `uv sync` remove pacotes que eu tinha

`uv sync --extra X` mantém apenas os extras passados. Passe todos que
você usa:

```bash
uv sync --extra dev --extra api --extra inference-keras --extra inference-onnx
```

## Captura de câmera

### `CaptureError: could not open camera at device_id=0`

**Sintomas:** `irisflow doctor` ou `irisflow run` falham no boot.

**Ações:**

1. Rode `irisflow doctor` — enumera câmeras disponíveis e mede FPS
   real (o declarado costuma mentir).
2. Confirme que outro processo não está segurando a câmera (Zoom, Meet,
   OBS...).
3. Se você tem várias câmeras, ajuste `camera.device_id` no YAML ou
   passe `IRISFLOW_CAMERA__DEVICE_ID=1`.
4. Em Linux: `ls /dev/video*` mostra os devices ativos.

### Câmera desconecta durante execução

O runner tenta reconectar com backoff (`camera.reconnect_backoff_ms`).
Se reconectar, a suíte de segurança pausa o cursor via `face_lost` —
comportamento esperado. Nada a fazer.

## Detecção facial

### `MediaPipe: legacy face_mesh removed in 1.0`

MediaPipe 1.0 aposentou `mp.solutions.face_mesh`. O IrisFlow usa a nova
API `mediapipe.tasks.python.vision.FaceLandmarker` com o asset
`models/face_landmarker.task`.

**Ação:** baixe o asset (uma vez) e confirme o path em
`configs/default.yaml` (`detection.face_model_path`).

### `DetectionError: no face detected` por segundos seguidos

O pipeline transita para `LOST` após `detection.lost_hysteresis_frames`
(default 5). Isso é intencional — a UI deve mostrar o estado, não crashar.

**Ações:**

1. Iluminação frontal, sem contra-luz.
2. Câmera aproximadamente na altura dos olhos.
3. Se persistir: `irisflow preview` mostra os landmarks em tempo real e
   ajuda a diagnosticar (câmera muito acima, óculos com reflexo, etc.).

## Inferência

### `InferenceError: gaze_encoder.onnx exports embedding (not gaze)`

O artefato ONNX enviado é um encoder, não estimador completo. O ONNX
backend recusa carregar — comportamento correto.

**Ação:** use `model.backend: keras` (default). Ver `docs/DECISIONS.md
§D3`.

### Latência p95 > 25 ms

Alvo do SPRINTS §1.2 é 25 ms; a máquina de desenvolvimento atingiu
p95=49 ms (`models/MODEL_CARD.md`). Estratégias:

1. Reduzir resolução de captura (`camera.width/height`).
2. Rodar em hardware alvo (a máquina do usuário costuma ser mais
   dedicada que a de desenvolvimento).
3. Sprint 13 hardening: quantização ONNX quando houver um export do
   grafo completo.

## Calibração

### `CalibrationError: only N samples for a K-parameter model`

O usuário completou a coleta com menos amostras do que o mínimo (`affine
= 3`, `polynomial = 6`). A UI deve permitir recoleta parcial.

**Ação:** aumentar `calibration.samples_per_point` ou usar
`model_kind = affine` (menos parâmetros, mais robusto com poucas
amostras).

### `holdout mean error > max_residual_px`

A calibração terminou mas ficou acima do threshold. O `outcome.accepted`
é `False` e `outcome.bad_targets` lista os alvos problemáticos.

**Ação:** repita os alvos ruins (a `CalibrationSession` suporta
`retry_targets([idx, ...])`).

### Sem calibração o cursor está deslocado

Passthrough intencional (mensagem `[warn] not calibrated`). Rode
`irisflow calibrate --profile <nome>` e depois `irisflow run --profile
<nome>` (ou o WS `set_profile`).

## Controle do cursor

### O cursor não se move

**Causas comuns:**

1. `control.enabled = false` (default). Passe `--cursor` no CLI.
2. Kill switch acionado — a mensagem `[safety] paused reason=kill_switch`
   deve ter aparecido. Rode de novo.
3. Face perdida por > `pause_on_face_lost_ms` (default 2 s) —
   reposicione-se na frente da câmera.
4. Watchdog disparou (pipeline travou por > `watchdog_timeout_ms`, default
   500 ms). Logs em `data/logs/irisflow.jsonl` mostram o motivo.

### Cursor está "sequestrado" após crash

**Nunca** deveria acontecer — o runner desabilita o cursor no `_shutdown`
sob qualquer circunstância (SPRINTS §10 DoD). Se ocorrer:

1. Pressione o kill switch (default `Ctrl+Alt+Esc`).
2. Reporte um bug com o traceback — é regressão do DoD.

## API / WebSocket

### `503 pipeline not initialized`

O lifespan do FastAPI ainda não terminou de subir o pipeline. Espere
alguns segundos ou olhe `/health`.

### Frontend recebe `type: error, code: bad_message`

Payload não bate com nenhum `ClientMessage` do `api/schemas.py`. Cheque
o campo `type` e o schema — Pydantic valida com discriminator.

### Frontend recebe menos que 30 Hz

Backpressure por descarte: `SessionHub` (fila `gaze_queue_size=4`) prefere
descartar frames velhos a bloquear o pipeline. Se o frontend renderiza
lento, aumentar `gaze_queue_size` **não** ajuda — melhora a taxa
efetiva melhorando o consumer.

## Testes / desenvolvimento

### Testes travam sem output

Provavelmente há um `await queue.get()` sem timeout num teste asyncio. A
suite CI roda sem `--timeout` — problemas assim aparecem como hang. O
padrão `next_outgoing` em `api/session.py` é um exemplo de como cuidar
de tasks pendentes.

### `import-linter` reprova imports que "sempre funcionaram"

Verifique se os módulos envolvidos estão no mesmo layer
(`pyproject.toml [tool.importlinter]`). Siblings dentro do mesmo layer
não podem se importar — mova a implementação para o layer certo ou
converta para primitivos.
