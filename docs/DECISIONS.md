# Decisões de arquitetura (ADRs)

Fecha as decisões abertas de `SPRINTS.MD §13`. Cada entrada aponta para o
código-verdade quando existe — este arquivo é apenas o resumo executivo.

---

## D1 — Semântica do vetor `rect` (Sprint 5)

**Decisão:** convenção iTracker/GazeCapture, 12 dimensões, coordenadas do
canto superior-esquerdo relativas ao frame inteiro, normalização por
`W`/`H`. "Esquerdo" = **do sujeito** (não da imagem).

| Índice | Campo | Divisor |
|---|---|---|
| 0-3 | `face.x, face.y, face.w, face.h` | `/W, /H, /W, /H` |
| 4-7 | `left_eye.{x,y,w,h}` (do sujeito) | idem |
| 8-11 | `right_eye.{x,y,w,h}` (do sujeito) | idem |

**Motivação:** o modelo de treino não foi acompanhado do preprocessing
script — a convenção iTracker é a mais comum em CNNs do tipo GazeCapture
que aceitam quatro tensores `(face, left_eye, right_eye, rect)`. O sanity
check da S6 (probe sintético em `bench --sanity-check`) confirmou que a
CNN produz saídas distintas para rects distintos, e a Sprint 7 (pipeline
fim-a-fim) não expôs inversões visíveis. Fica como candidata a revisitar
se a Sprint 13/uso real evidenciar desalinhamento espacial.

**Verdade no código:** `src/irisflow/preprocessing/rect_vector.py::build_rect_vector`.

**Fonte:** `models/MODEL_CARD.md §4`.

---

## D2 — Ordem de canais e normalização (Sprint 5)

**Decisão:** `channel_order = RGB` (`configs/default.yaml`) e
`normalization = unit` (dividir por 255 → floats em `[0, 1]`).

**Motivação:** OpenCV entrega BGR mas o grafo Keras não contém camada de
preprocessing (verificado por `scripts/inspect_model.py`). Sem código de
treino disponível, `RGB` + `unit` é o padrão mais frequente. A inversão
`RGB↔BGR` produz gaze "quase certo" que passa em testes automáticos mas
falha na diagonal — o sanity check da S6 é a proteção. Alternativas
suportadas via config: `signed` (`x/127.5 − 1`) e `imagenet`
(padronização por média/desvio do ImageNet).

**Verdade no código:** `src/irisflow/preprocessing/normalize.py`.

**Fonte:** `models/MODEL_CARD.md §§2-3`.

---

## D3 — Backend de produção: Keras (Sprint 6)

**Decisão:** `model.backend = keras`. O ONNX segue disponível na
arquitetura mas **não é usável** como estimador completo — o artefato
`gaze_encoder.onnx` publica apenas o `embedding` de 256 dimensões, não
`(gaze_x, gaze_y)`. `OnnxBackend` levanta `InferenceError` explícito ao
carregar um encoder-only.

**Motivação:** o Sprint 6 gate confirmou paridade Keras vs ONNX apenas no
nível do embedding (`max_abs_diff = 6.7e-06`, `bench --compare-backends`).
Trocar o backend em runtime custa uma linha de YAML no dia em que houver
um export completo do grafo.

**Verdade no código:** `src/irisflow/inference/onnx_backend.py`,
`src/irisflow/inference/registry.py`.

**Fonte:** `models/MODEL_CARD.md §5`.

---

## D4 — Modelo de calibração padrão: polinomial de 2ª ordem (Sprint 8)

**Decisão:** `calibration.model_kind = polynomial`,
`polynomial_degree = 2`.

**Motivação:** o polinomial de 6 termos (`[1, x, y, x², xy, y²]` por eixo)
corrige o abaulamento típico de webcams off-center e cabeças levemente
inclinadas sem precisar de regularização. `ridge` (mesma base + L2) fica
como fallback quando o usuário coleta pouquíssimas amostras por alvo — a
config já o suporta. `affine` é o baseline rápido para testes.

**Verdade no código:** `src/irisflow/calibration/models.py`.

**Nota:** o critério empírico permanece — `SPRINTS.MD §8` exige "reduzir
o erro em ≥ 30% frente ao passthrough". `polynomial` cumpre em datasets
sintéticos com viés injetado (testes unitários em `test_calibration_models.py`).

---

## D5 — Biblioteca de controle de cursor: `pynput` (Sprint 10)

**Decisão:** `pynput`.

**Motivação:** mais leve que `pyautogui`, não depende de captura de tela
(`pyautogui` importa Pillow para funcionar em algumas plataformas) e
oferece hotkeys globais para o kill switch — que é o *safety-critical
must-have* para o usuário com ELA. O adapter fica isolado em
`src/irisflow/control/cursor.py`; trocar de biblioteca custa uma classe.

**Verdade no código:** `src/irisflow/control/cursor.py`,
`src/irisflow/control/kill_switch.py`.

---

## D6 — Estratégia de clique: apenas dwell (Sprint 10)

**Decisão:** dwell-click. Piscada **não** é implementada.

**Motivação:** ELA compromete musculatura orofacial — a piscada voluntária
é frágil ou impossível em estágios avançados. Um trigger baseado em
piscada correria o risco de:

1. bloquear o usuário quando ele perde a capacidade de piscar de forma
   controlada;
2. disparar cliques falsos por piscadas involuntárias.

O dwell (permanência ≥ N ms num raio) é totalmente parametrizável por
usuário (`config.control.dwell.{radius_px,duration_ms,refractory_ms}`),
publica `DwellProgress` no bus para o frontend desenhar um anel de
progresso, e respeita a zona de descanso + refratariedade para evitar
cliques acidentais.

**Verdade no código:** `src/irisflow/control/dwell.py`,
`src/irisflow/pipeline/control_sink.py`.

---

## D7 — Configuração de pré-processamento vencedora (janela demo 2026-08)

**Decisão:** _(a preencher após T6 rodar com rosto real)._ Placeholder
até a varredura de `scripts/config_sweep.py` produzir a tabela e a
combinação vencedora ser aplicada em `configs/default.yaml`.

**Motivação prevista:** o eixo Y ficou constante em `raw_y = 0.0` na
tentativa anterior, apesar da avaliação offline registrar
`per_axis_mae_cm: {x: 2.40, y: 2.07}`. A hipótese em `tarefas.md §T6`
é divergência de pré-processamento — não do modelo. A decisão será
tomada por medição, não por palpite.

**Verdade no código (após medir):** `configs/default.yaml`,
`src/irisflow/preprocessing/normalize.py`. Se `swap_eye_inputs` for
adicionado, também `src/irisflow/preprocessing/builder.py` e
`src/irisflow/config/schema.py::ModelConfig`.

**Fonte:** tabela T6 em `docs/ACHADOS.md`.

---

## D8 — Propriedade do dwell na janela demo 2026-08

**Decisão (temporária):** o **backend move o cursor**, o **frontend
detém o dwell** via `DwellButton`. Efetivada em `configs/demo-control.yaml`
com `control.dwell.duration_ms: 999999` (≈ 16 min) — o `DwellClicker`
do backend nunca dispara dentro do tempo humano, e o `DwellButton`
do React fica como único disparador de seleção.

**Motivação:** `DwellButton` usa `onMouseEnter`/`onMouseLeave`, que só
respondem ao cursor real do SO — o ponto de gaze desenhado tem
`pointer-events: none`. Se o backend também disparar dwell, haveria
clique duplo. Deixar o dwell no frontend permite reaproveitar a UX
existente (anel de progresso, feedback visual por componente).

**Alternativa considerada:** adicionar `control.dwell.enabled: bool` ao
schema. Rejeitada: `duration_ms=999999` já passa `Field(gt=0)` sem
alterar API/schema — mais reversível.

**Escopo:** só a janela desta demo. Pós-demo, migrar para dwell
autoritativo no backend com eventos `dwell_progress`/`click` no WS
já expostos pelo `SessionHub`, e remover o `DwellButton` (ou fazer
com que ele escute os eventos ao invés de acionar `mouseenter`).

**Verdade no código:** `configs/demo-control.yaml`,
`frontend/src/components/DwellButton.tsx`,
`src/irisflow/pipeline/control_sink.py`.
