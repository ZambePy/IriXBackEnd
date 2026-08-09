# IrisFlow — Achados e Medições

> Criado em 2026-08-08 conforme regra 0.3 do plano de execução.
> Registrar aqui: medições, decisões, anomalias. Sem isso, os problemas viram mistério.

---

## T1 — Artefatos do modelo

- `models/gaze_cnn_best.keras`: presente (2.5 MB)
- `models/face_landmarker.task`: baixado de mediapipe-models/float16/1 (3.6 MB)
- `configs/default.yaml`: caminhos corretos (`model.path`, `detection.face_model_path`)
- `.gitignore`: adicionado `models/*.task`

---

## T2 — Ambiente Python/Node

- Python via `.venv/Scripts/python.exe` (user memory: uv não funciona direto).
- `pytest -q`: **575 passed** em 22.57s, 1 warning (deprecation StarletteDeprecation).
- `ruff check`: **All checks passed** (após corrigir 7 erros em `scripts/config_sweep.py`).
- `mypy src/`: **Success: no issues found in 95 source files**.
- `lint-imports`: **4 contracts kept, 0 broken**.
- Node/`npm run verify`: _(pendente — o humano tem Node local; não bloqueia)._

---

## T3 — Limpeza mínima

- `frontend/python_scripts/virtual_mouse.py`: removido
- `configs/demo-visual.yaml`: criado (`control.enabled: false`)
- `configs/demo-control.yaml`: criado (`control.enabled: true`, `dwell.duration_ms: 999999`)

---

## T4 — Instrumentação de latência

Já implementada. `_print_snapshot` em `src/irisflow/cli/commands/run.py:355`
imprime a cada `--metrics-every N` segundos:

- **linha 1**: `fps`, `ok`, `dropped`, `face_lost` (contadores agregados)
- **linhas por estágio**: `capture`, `detection`, `preprocess`, `inference`,
  `calibrate`, `map`, `filter`, `tick_wall` — cada um com `n`, `p50_ms`,
  `p95_ms`, `max_ms`
- **última linha**: `capture_dropped (queue): N` — frames descartados pela
  fila do `WebcamSource` (pipeline não conseguiu acompanhar).

Métrica `tick_wall` é o **tempo total do loop**. Se ela for muito maior que
a soma dos estágios, o resíduo está fora dos estágios (logging, GIL, fila).

_(números reais dependem de rodar T5 com hardware — pendente 🧑)_

---

## T5 — FPS real e gargalo

**Handoff pronto.** Ver `docs/ACHADOS.md#handoff` no final para os
comandos exatos. Após rodar, preencher tabela:

| Etapa | Comando | FPS | p50/p95 relevantes |
|---|---|---|---|
| 1. Só captura | `doctor --duration 10` | | |
| 2. Cap + detecção | `preview` | | |
| 3. Só inferência | `bench --latency --iterations 50` | | |
| 4. Pipeline completo | `run --no-cursor --metrics-every 1 --quiet-gaze` | | |

_(requer humano)_

---

## T6 — Portão dos eixos Y

**Handoff pronto.** Scripts `scripts/axis_probe.py` e
`scripts/config_sweep.py` prontos. Ver seção "Handoff" abaixo.

_(requer humano + vídeo `axis_probe.mp4` gravado)_

---

## T14 — Dwell no frontend, backend só move cursor

- Config `demo-control.yaml` já usa `dwell.duration_ms: 999999` (≈ 16 min).
  Passa `Field(gt=0)` do `DwellConfig`; nenhuma alteração de schema necessária.
- Não existe flag `enabled` em `DwellConfig` — decisão de não adicionar
  para evitar escopo. O truque do `duration_ms=999999` é reversível e
  não altera assinatura pública.
- Verificações **🧑** ainda pendentes: kill switch cronometrado, matar
  processo não sequestra mouse, sem clique duplicado. Sem rosto real
  na câmera, não posso conferir.

---

## T17 — Rotas REST inexistentes no backend

Backend expõe **apenas** (confirmado via `app.openapi()` em runtime + `ws.py`):

| Método | Endpoint |
|---|---|
| GET   | `/health` |
| GET   | `/config` |
| PATCH | `/config` |
| GET   | `/profiles` |
| GET   | `/calibration/status` |
| POST  | `/calibration/abort` |
| WS    | `/ws/gaze` |

Frontend chama (via `frontend/src/utils/api.ts`) — **nenhuma existe**:

| Método | Endpoint | Tela que chama (grep confirmado) |
|---|---|---|
| POST | `/voice/clone` | `SettingsScreen.tsx:70` |
| GET  | `/voice/status/{id}` | `SettingsScreen.tsx:72` |
| POST | `/voice/synthesize` | (definida em `api.ts` mas não referenciada por tela) |
| POST | `/alerts/help` | `output/EmergencyEscalation.tsx:46` |
| POST | `/alerts/iamok` | `caregiver/IAmOkScreen.tsx:19` |
| POST | `/smart-home/action` | (definida em `api.ts` mas não referenciada por tela) |
| POST | `/chatbot/message` | `ai/ChatbotScreen.tsx:36` |

Sem `CORSMiddleware` no `create_app`. Se o frontend ficar em `:5173`
falando com backend em `:8000`, chamadas REST vão bater em CORS antes
mesmo de 404 — mas o WS não é bloqueado por CORS.

**Telas seguras para a demo** (só gaze + TTS do browser, sem REST inexistente):
- `FollowTarget` (games/FollowTarget)
- `BubblePopGame`
- `QuickPhrasesScreen`
- `PictogramScreen` (core/PictogramScreen)

**Grade recomendada** (com precisão atual: mediana ~3,1 cm; p90 ~6,2 cm):
- 4×3 em tela cheia funciona
- 6×4 fica no limite
- Alvos abaixo de ~250 px de lado frustram

**Evitar no roteiro**:
- `KeyboardScreen` (alvos muito pequenos)
- `SettingsScreen` (chama `/voice/*` — vai falhar)
- `ChatbotScreen` (chama `/chatbot/message` — vai falhar)
- `EmergencyEscalation` (chama `/alerts/help` — vai falhar)
- `IAmOkScreen` (chama `/alerts/iamok` — vai falhar)
- `VirtualMouseScreen` (rastreador antigo removido; pode quebrar)

**Nota:** `SmartHomeScreen.tsx` não existe. `api.smartHomeAction` está
definida mas nunca é chamada — sem risco na demo.

---

## Anomalias

_(registrar aqui qualquer achado fora do escopo)_

---

## Handoff — Comandos para o humano rodar

Todas essas tarefas exigem rosto real diante da webcam. Prepare o
ambiente, execute na ordem, e cole a saída de volta aqui.

### T5 — Medir FPS por etapa

Terminal, um comando por vez:

```bash
# 1. Só captura (sem MediaPipe, sem CNN)
.venv/Scripts/python.exe -m irisflow doctor --duration 10

# 2. Captura + detecção (com MediaPipe, sem CNN)
.venv/Scripts/python.exe -m irisflow preview

# 3. Só inferência isolada (sem câmera, sem MediaPipe)
.venv/Scripts/python.exe -m irisflow bench --latency --iterations 50

# 4. Pipeline completo — imprime métricas a cada 1s por 30s
.venv/Scripts/python.exe -m irisflow run --no-cursor --metrics-every 1 --quiet-gaze
# (Ctrl+C para parar após ~30s de rosto na câmera olhando pra tela)
```

Cole aqui: **FPS de cada etapa** e **p50/p95 dos estágios** da etapa 4.

### T6 — Portão dos eixos Y

1. Grave um vídeo curto olhando (3s por posição):
   `centro → esquerda → direita → topo → base → centro`
   Salve como `axis_probe.mp4` na raiz do repo.

2. Rode a varredura das 12 combinações:

```bash
.venv/Scripts/python.exe scripts/config_sweep.py --video axis_probe.mp4
```

Cole aqui a tabela impressa (channel_order × norm × swap → amp_x, amp_y).

**Alternativa "ao vivo" sem vídeo** (se a webcam já estiver funcionando):

```bash
# Rode com rosto na câmera, olhe pra cima e pra baixo por 30s.
# Aceite as prints — raw_x/raw_y devem variar.
.venv/Scripts/python.exe scripts/axis_probe.py --duration 30 --label "olhar_livre"
```

### T10 — Calibração

Depois que T5 e T6 estiverem OK:

```bash
.venv/Scripts/python.exe -m irisflow calibrate --profile demo \
    --screen-width 1920 --screen-height 1080
```

Cole: **erro residual final** + se **aceito ou rejeitado**.

Depois, verificar:

```bash
ls configs/profiles/                   # deve ter demo.json
.venv/Scripts/python.exe -m irisflow run --profile demo --no-cursor
```

### T13 — WebSocket isolado

```bash
# Terminal 1:
.venv/Scripts/python.exe -m irisflow serve --config configs/demo-visual.yaml
```

No console do navegador em qualquer aba:

```js
const ws = new WebSocket('ws://localhost:8000/ws/gaze');
let n = 0;
ws.onmessage = (e) => { n++; if (n <= 3) console.log(JSON.parse(e.data)); };
setTimeout(() => console.log('mensagens em 5s:', n), 5000);
```

Cole: **3 primeiras mensagens** e **contagem em 5s** (deve bater com FPS de T5).

### T15 — Integração completa

```bash
# Terminal 1
.venv/Scripts/python.exe -m irisflow serve --config configs/demo-visual.yaml

# Terminal 2
cd frontend && npm run dev
```

Abra `http://localhost:5173` em tela cheia (F11).

- [ ] Ponto de gaze aparece e acompanha o olhar
- [ ] Sem espelhamento (olhar à esquerda move à esquerda)
- [ ] Alinhamento correto em tela cheia

### T16 — Roteiro de validação

Ver tabela em `tarefas.md#T16`. Reportar cada item, não resumir.

O **teste 4** (perda de rosto → ponto **congela**, não voa) é o mais
importante — código: `SafetyGate.pause_on_face_lost_ms=2000` em
`configs/demo-visual.yaml`.
