# IrisFlow — Guia Oficial de Desenvolvimento do Backend

> **Documento de referência única (SSOT)** para a implementação do backend do sistema de rastreamento ocular por webcam da IrisFlow.
> Público-alvo: desenvolvedores humanos e agentes (Claude Code) executando as Sprints de forma incremental.
> Versão: 1.0 · Status: aprovado para execução

---

## Sumário

1. [Contexto e princípios](#1-contexto-e-princípios)
2. [Arquitetura proposta](#2-arquitetura-proposta)
3. [Estrutura de pastas](#3-estrutura-de-pastas)
4. [Fluxo completo do backend](#4-fluxo-completo-do-backend)
5. [Módulos e responsabilidades](#5-módulos-e-responsabilidades)
6. [Contratos de dados (tipos do domínio)](#6-contratos-de-dados-tipos-do-domínio)
7. [O modelo de IA e seu contrato](#7-o-modelo-de-ia-e-seu-contrato)
8. [Roadmap de Sprints](#8-roadmap-de-sprints)
9. [Grafo de dependências entre Sprints](#9-grafo-de-dependências-entre-sprints)
10. [Estratégia de testes](#10-estratégia-de-testes)
11. [Integração futura via WebSocket/FastAPI](#11-integração-futura-via-websocketfastapi)
12. [Cronograma de 10 dias](#12-cronograma-de-10-dias)
13. [Riscos e decisões em aberto](#13-riscos-e-decisões-em-aberto)
14. [Convenções de código e Definition of Done global](#14-convenções-de-código-e-definition-of-done-global)

---

## 1. Contexto e princípios

### 1.1 O que estamos construindo

Um backend em Python que transforma o vídeo de uma webcam comum em uma posição de cursor na tela, permitindo que pessoas com Esclerose Lateral Amiotrófica (ELA) operem um computador apenas com o olhar. O modelo de IA já existe e está treinado; **o trabalho aqui é de engenharia de sistema, não de pesquisa em visão computacional.**

> ## ⚠️ PREMISSA INEGOCIÁVEL DO PROJETO
>
> **O modelo já treinado (`gaze_cnn_best.keras`) é o motor de inferência do sistema.** Ele é o componente que produz a estimativa de olhar, e todo o restante do backend existe para alimentá-lo corretamente e tratar sua saída.
>
> **Não está no escopo, em nenhuma Sprint:**
> - treinar uma nova CNN;
> - fazer fine-tuning, retreino ou alteração de pesos do modelo existente;
> - substituir o modelo por heurísticas geométricas, por regressão sobre landmarks, ou por qualquer biblioteca de gaze de terceiros;
> - implementar um estimador de olhar "provisório" para destravar Sprints.
>
> **Se o pipeline não estiver produzindo bons resultados, a causa está no pré-processamento, na calibração ou na filtragem — não no modelo.** Corrija a cadeia, nunca troque o motor.
>
> Duas coisas no documento **não** contradizem isso, e não devem ser lidas como se contradissessem:
>
> | Aparente exceção | O que realmente significa |
> |---|---|
> | `StubGazeEstimator` (seções 10.2 e Sprints 2/7) | Dublê **exclusivo de teste automatizado**, para que a suite rode em CI sem carregar TensorFlow. Nunca é usado em execução real. Proibido em qualquer caminho de produção. |
> | `GazeEstimator` como Protocol (seção 7.5) | Garantia de **manutenibilidade futura**: se um dia você treinar um modelo melhor, ele entra trocando uma linha de YAML. É a capacidade de troca, não um plano de troca. Hoje há um único motor de produção: o seu. |



### 1.2 Requisitos não-funcionais (guiam todas as decisões)

| Requisito | Alvo | Por quê |
|---|---|---|
| Latência fim-a-fim (frame → cursor) | ≤ 80 ms p95 | Acima disso o cursor "arrasta" e a interação parece quebrada |
| Taxa de atualização | ≥ 20 FPS sustentados em CPU comum | Hardware do usuário final é modesto |
| Estabilidade do cursor em fixação | jitter ≤ 15 px RMS | Tremor visual causa fadiga ocular severa |
| Recuperação de perda de rosto | ≤ 300 ms, sem cursor "voando" | Usuário pode tossir, ser reposicionado na cadeira |
| Falha segura | Sistema nunca prende o usuário | Usuário com ELA pode não conseguir usar teclado para abortar |
| Consumo de CPU | ≤ 60% de um core moderno em regime | Deixar folga para o software assistivo que o usuário roda |

> **Princípio da acessibilidade:** cada decisão técnica deve ser avaliada perguntando "o que acontece se isso falhar enquanto o usuário está sozinho e não pode usar as mãos?". Nunca implemente um caminho que exija intervenção manual para sair.

### 1.3 Princípios de arquitetura

1. **Pipeline em estágios desacoplados.** Cada estágio recebe um objeto tipado, devolve outro objeto tipado, e não conhece seus vizinhos.
2. **Inversão de dependência nas bordas.** Webcam, runtime do modelo e controle do SO são *adaptadores* atrás de `Protocol`s. Trocar de MediaPipe para outro detector, ou de Keras para ONNX, não toca no núcleo.
3. **Núcleo puro e testável.** `core/`, `preprocessing/`, `calibration/`, `filtering/` e `mapping/` não fazem I/O e não importam OpenCV para captura, nem TensorFlow. São testáveis com arrays sintéticos.
4. **Configuração declarativa.** Nada de constantes mágicas espalhadas: tudo em YAML validado por Pydantic.
5. **O frontend é um consumidor tardio.** A camada `api/` é um adaptador de saída, não o centro do sistema. O backend deve ser 100% funcional via CLI antes de existir um WebSocket.
6. **Observabilidade desde o dia 1.** Sem métricas de latência e qualidade, o ajuste de filtros vira adivinhação.

### 1.4 Inspiração arquitetural (EyeTheia) — e o que adaptamos

O EyeTheia é a referência para o **padrão de organização**: pipeline linear de estágios nomeados, pastas por responsabilidade (captura / detecção / pré-processamento / inferência / calibração / controle), configuração centralizada e separação entre motor de estimativa e aplicação.

**O que a IrisFlow faz diferente, por decisão explícita:**

| Aspecto | Padrão de referência | IrisFlow |
|---|---|---|
| Motor de gaze | Acoplado ao pipeline | Atrás de `GazeEstimator` (Protocol), com backends Keras/ONNX intercambiáveis por config |
| Calibração | Integrada ao loop | Módulo isolado, com persistência por perfil de usuário e métricas de qualidade |
| Concorrência | Loop único | Captura e inferência em threads separadas com fila *drop-oldest* |
| Saída | Cursor direto | Barramento de eventos; cursor e WebSocket são dois *sinks* independentes |
| Público | Uso geral | ELA: dwell click, fail-safe, recuperação de perda de rosto, perfis persistentes |

---

## 2. Arquitetura proposta

### 2.1 Visão em camadas

```
┌──────────────────────────────────────────────────────────────────────┐
│  CAMADA DE APLICAÇÃO                                                 │
│  cli/ (Typer)            api/ (FastAPI + WebSocket — Sprints 11-12)  │
└───────────────┬──────────────────────────────────────────────────────┘
                │ comanda / observa
┌───────────────▼──────────────────────────────────────────────────────┐
│  CAMADA DE ORQUESTRAÇÃO                                              │
│  pipeline/  — monta os estágios, roda o loop, publica eventos,       │
│               mede latência, gerencia ciclo de vida e degradação     │
└───────────────┬──────────────────────────────────────────────────────┘
                │ compõe
┌───────────────▼──────────────────────────────────────────────────────┐
│  CAMADA DE DOMÍNIO (pura, sem I/O, 100% testável offline)            │
│  preprocessing/   calibration/   mapping/   filtering/   core/       │
└───────────────┬──────────────────────────────────────────────────────┘
                │ depende de abstrações (Protocols em core/interfaces.py)
┌───────────────▼──────────────────────────────────────────────────────┐
│  CAMADA DE ADAPTADORES (I/O, bibliotecas externas, SO)               │
│  capture/ (OpenCV)  detection/ (MediaPipe)                           │
│  inference/ (Keras, ONNX Runtime)  control/ (pynput)                 │
│  logging/ (structlog)  config/ (Pydantic Settings)                   │
└──────────────────────────────────────────────────────────────────────┘
```

**Regra de dependência (verificada em CI por `import-linter`):** as setas só apontam para baixo. `core/` não importa ninguém. Domínio nunca importa adaptador concreto.

### 2.2 Modelo de concorrência

```
Thread A (Captura)          Fila (maxsize=1, drop-oldest)      Thread B (Pipeline)
─────────────────           ─────────────────────────────      ───────────────────
cap.read()  ──────────────► [ Frame mais recente ]  ──────────► detect → preprocess
~30 FPS                     descarta frames antigos            → infer → calibrate
                                                               → map → filter
                                                                      │
                                                                      ▼
                                                               EventBus (assíncrono)
                                                                ├─► CursorSink
                                                                ├─► TelemetrySink
                                                                └─► WebSocketSink (S12)
```

**Por que *drop-oldest*:** se a inferência ficar mais lenta que a captura, queremos o frame **mais novo**, não uma fila de frames velhos. Latência importa mais que completude — um frame perdido é invisível; 200 ms de atraso acumulado não é.

---

## 3. Estrutura de pastas

```
irisflow-backend/
├── pyproject.toml                  # deps, ruff, mypy, pytest, build (uv/hatch)
├── README.md
├── SPRINTS.md                      # este documento
├── .env.example
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
│
├── configs/
│   ├── default.yaml                # configuração base versionada
│   ├── profiles/                   # perfis por usuário (não versionado)
│   │   └── .gitkeep
│   └── schema.md                   # documentação dos campos
│
├── models/                         # artefatos do modelo (Git LFS ou fora do repo)
│   ├── gaze_cnn_best.keras
│   ├── gaze_cnn_last.keras
│   ├── gaze_encoder.onnx
│   └── MODEL_CARD.md               # contrato de I/O, gerado na Sprint 6
│
├── data/
│   ├── calibration/                # perfis de calibração persistidos (JSON)
│   ├── recordings/                 # sessões gravadas para replay/teste
│   └── logs/
│
├── src/irisflow/
│   ├── __init__.py
│   ├── __main__.py                 # python -m irisflow
│   │
│   ├── core/                       # ← núcleo, zero dependências internas
│   │   ├── types.py                # Frame, FaceDetection, ModelInput, GazePoint...
│   │   ├── interfaces.py           # Protocols: FrameSource, FaceDetector, GazeEstimator...
│   │   ├── exceptions.py           # hierarquia de erros do domínio
│   │   ├── geometry.py             # BoundingBox, Point, normalização, clamp
│   │   ├── clock.py                # relógio monotônico injetável (testes determinísticos)
│   │   └── events.py               # definição de eventos do barramento
│   │
│   ├── config/
│   │   ├── schema.py               # modelos Pydantic de toda a configuração
│   │   ├── loader.py               # YAML + env vars + overrides de CLI
│   │   └── defaults.py
│   │
│   ├── capture/                    # ADAPTADOR
│   │   ├── base.py                 # implementa FrameSource
│   │   ├── webcam.py               # OpenCV VideoCapture + thread + drop-oldest
│   │   ├── video_file.py           # fonte a partir de arquivo (testes/replay)
│   │   ├── synthetic.py            # fonte sintética determinística (testes unitários)
│   │   └── diagnostics.py          # enumerar câmeras, resolução, FPS real
│   │
│   ├── detection/                  # ADAPTADOR
│   │   ├── base.py
│   │   ├── mediapipe_detector.py   # FaceMesh → landmarks
│   │   ├── landmarks.py            # índices nomeados (olhos, íris, contorno)
│   │   ├── roi.py                  # landmarks → bounding boxes de face e olhos
│   │   └── tracking.py             # suavização de ROI, histerese, perda/reaquisição
│   │
│   ├── preprocessing/              # DOMÍNIO (puro)
│   │   ├── crops.py                # recorte com margem, padding seguro na borda
│   │   ├── resize.py               # 224×224 face, 112×112 olhos
│   │   ├── normalize.py            # escala de pixels — DEVE espelhar o treino
│   │   ├── rect_vector.py          # construção do vetor rect de 12 dims
│   │   └── builder.py              # monta o ModelInput completo
│   │
│   ├── inference/                  # ADAPTADOR
│   │   ├── base.py                 # implementa GazeEstimator
│   │   ├── keras_backend.py
│   │   ├── onnx_backend.py
│   │   ├── registry.py             # seleção de backend por config
│   │   ├── warmup.py               # inferência dummy para evitar 1º frame lento
│   │   └── parity.py               # comparação Keras × ONNX
│   │
│   ├── calibration/                # DOMÍNIO (puro)
│   │   ├── points.py               # geração dos alvos na tela (5/9/13 pontos)
│   │   ├── session.py              # máquina de estados da sessão de calibração
│   │   ├── collector.py            # coleta amostras estáveis por alvo
│   │   ├── models.py               # afim, polinomial 2ª ordem, ridge
│   │   ├── quality.py              # erro residual, detecção de alvos ruins
│   │   └── store.py                # persistência JSON por perfil
│   │
│   ├── mapping/                    # DOMÍNIO (puro)
│   │   ├── screen.py               # gaze normalizado → pixel, com clamp e margens
│   │   └── screen_info.py          # resolução, múltiplos monitores, DPI
│   │
│   ├── filtering/                  # DOMÍNIO (puro)
│   │   ├── base.py                 # Protocol Filter
│   │   ├── ema.py                  # baseline
│   │   ├── one_euro.py             # filtro principal (adaptativo a velocidade)
│   │   ├── kalman.py               # opcional/experimental
│   │   ├── outlier.py              # rejeição de saltos improváveis
│   │   ├── fixation.py             # detecção de fixação × sacada
│   │   └── chain.py                # composição configurável de filtros
│   │
│   ├── control/                    # ADAPTADOR
│   │   ├── base.py                 # Protocol CursorController
│   │   ├── cursor.py               # pynput
│   │   ├── dwell.py                # clique por permanência
│   │   ├── safety.py               # kill switch, pausa, limites de taxa
│   │   └── noop.py                 # implementação nula (testes/headless)
│   │
│   ├── pipeline/
│   │   ├── stages.py               # cada estágio como unidade nomeada e medida
│   │   ├── orchestrator.py         # montagem por injeção de dependência
│   │   ├── runner.py               # loop principal, ciclo de vida, shutdown limpo
│   │   ├── state.py                # estados: IDLE, CALIBRATING, TRACKING, LOST, PAUSED
│   │   └── bus.py                  # publicação de eventos para múltiplos sinks
│   │
│   ├── telemetry/
│   │   ├── metrics.py              # latência por estágio, FPS, drops
│   │   ├── recorder.py             # gravação de sessão para replay
│   │   └── report.py               # relatório de benchmark
│   │
│   ├── logging/
│   │   └── setup.py                # structlog: console humano + JSON em arquivo
│   │
│   ├── api/                        # ← só a partir da Sprint 11
│   │   ├── app.py                  # factory FastAPI
│   │   ├── ws.py                   # endpoint /ws/gaze
│   │   ├── routes/                 # /health, /calibration, /config
│   │   ├── schemas.py              # contratos Pydantic das mensagens
│   │   └── session.py              # ponte pipeline ↔ conexões WS
│   │
│   └── cli/
│       ├── main.py                 # Typer: run, calibrate, doctor, bench, replay
│       └── commands/
│
├── tests/
│   ├── conftest.py
│   ├── unit/                       # espelha src/irisflow/
│   ├── integration/                # pipeline com fontes falsas
│   ├── e2e/                        # sessão completa sobre vídeo gravado
│   └── fixtures/
│       ├── frames/                 # frames golden versionados
│       ├── sessions/               # gravações curtas
│       └── stubs.py                # StubGazeEstimator, FakeCamera, FakeCursor
│
└── scripts/
    ├── inspect_model.py            # extrai assinatura real dos artefatos
    ├── benchmark.py
    └── record_session.py
```

---

## 4. Fluxo completo do backend

### 4.1 Fluxo de rastreamento (regime permanente)

```
 [1] CAPTURA
     webcam → Frame(ndarray BGR, timestamp_mono, frame_id)
        │
 [2] DETECÇÃO
     FaceMesh → 478 landmarks → ROIs (face, olho_esq, olho_dir)
     ├─ sem rosto? → estado LOST → congela cursor, emite evento, tenta reaquisição
     └─ com rosto → suavização de ROI (histerese anti-tremor)
        │
 [3] PRÉ-PROCESSAMENTO
     recorte + resize + normalização + construção do vetor rect
     → ModelInput(face 224×224×3, eye_l 112×112×3, eye_r 112×112×3, rect[12])
        │
 [4] INFERÊNCIA  ← seu modelo treinado (gaze_cnn_best.keras) roda AQUI
     GazeEstimator.predict(ModelInput) → RawGaze(x, y) ∈ [0,1]²
     este é o único ponto do sistema que estima olhar; todos os demais
     estágios apenas preparam a entrada ou tratam a saída deste
        │
 [5] CALIBRAÇÃO
     transformação aprendida por usuário → CalibratedGaze(x, y)
     (se não calibrado → passthrough + aviso de precisão degradada)
        │
 [6] MAPEAMENTO
     normalizado → pixel da tela, com clamp e margens de borda
     → ScreenPoint(px, py)
        │
 [7] FILTRAGEM TEMPORAL
     rejeição de outliers → One Euro → detecção de fixação
     → SmoothedPoint(px, py, is_fixation)
        │
 [8] SAÍDA (fan-out via EventBus)
     ├─► CursorController.move(px, py) + DwellClick
     ├─► Telemetria (latência, FPS, qualidade)
     └─► WebSocketSink (Sprint 12)
```

### 4.2 Fluxo de calibração

```
CLI/API dispara → estado CALIBRATING
  para cada alvo (9 pontos padrão, ordem embaralhada):
    exibir alvo → aguardar estabilização (~600 ms)
    coletar N amostras de RawGaze com o rosto estável
    descartar amostras com desvio alto ou rosto perdido
  ajustar modelo (ridge/polinomial) em (raw → alvo)
  validar com hold-out → erro residual médio em px e graus
  erro aceitável? → persistir perfil → estado TRACKING
  erro alto?      → apontar alvos problemáticos → oferecer recoleta parcial
```

### 4.3 Máquina de estados do pipeline

```
        ┌──────┐  start   ┌──────────────┐  perfil ok  ┌──────────┐
        │ IDLE ├─────────►│ CALIBRATING  ├────────────►│ TRACKING │
        └──▲───┘          └──────────────┘             └──┬───┬───┘
           │                                    rosto     │   │  pause
           │ stop                              perdido    │   │
           │                                   ┌──────────▼┐  ▼
           └───────────────────────────────────┤   LOST    │ PAUSED
                                               └───────────┘
```

Em `LOST` e `PAUSED`, o cursor **não se move** — nunca extrapolar posição sem dado válido.

---

## 5. Módulos e responsabilidades

| Módulo | Responsabilidade única | NÃO faz | Depende de |
|---|---|---|---|
| `core` | Tipos, protocolos, geometria, erros, relógio | Qualquer I/O ou lógica de negócio | nada |
| `config` | Carregar e validar configuração | Aplicar configuração | `core` |
| `capture` | Entregar o frame mais recente com timestamp | Detectar, redimensionar, interpretar | `core`, OpenCV |
| `detection` | Frame → landmarks → ROIs estáveis | Recortar, normalizar | `core`, MediaPipe |
| `preprocessing` | ROIs → tensores exatamente como no treino | Chamar o modelo | `core`, NumPy |
| `inference` | `ModelInput` → gaze bruto normalizado | Calibrar, suavizar, mapear | `core`, Keras/ORT |
| `calibration` | Corrigir viés individual do usuário | Suavizar, mover cursor | `core` |
| `mapping` | Normalizado → coordenada de tela | Filtrar | `core` |
| `filtering` | Estabilidade temporal do sinal | Corrigir viés espacial | `core` |
| `control` | Efeito no SO (mover, clicar) com segurança | Decidir *para onde* olhar | `core`, pynput |
| `pipeline` | Compor estágios, medir, gerenciar estado | Regra de negócio de qualquer estágio | todos os acima |
| `telemetry` | Métricas, gravação, relatórios | Alterar comportamento | `core` |
| `api` | Expor pipeline via HTTP/WS | Lógica de rastreamento | `pipeline` |
| `cli` | Ponto de entrada humano | Lógica de rastreamento | `pipeline` |

> **Teste de sanidade de responsabilidade:** se um arquivo precisa mudar por dois motivos diferentes (ex.: "mudou a webcam" e "mudou o filtro"), ele está no lugar errado.

---

## 6. Contratos de dados (tipos do domínio)

Definidos na Sprint 2, em `core/types.py`. São `@dataclass(frozen=True, slots=True)` — imutáveis, sem métodos de negócio.

```python
Frame            : data(ndarray HxWx3 BGR), frame_id(int), timestamp(float mono), width, height
BoundingBox      : x, y, w, h (int, pixels)  + métodos puros: expand(), clip(), normalized()
FaceDetection    : face_bbox, left_eye_bbox, right_eye_bbox, landmarks(ndarray), confidence
ModelInput       : face(224,224,3 f32), left_eye(112,112,3 f32), right_eye(112,112,3 f32), rect(12, f32)
RawGaze          : x, y (float ∈ [0,1]), confidence, inference_ms
CalibratedGaze   : x, y (float ∈ [0,1]), profile_id
ScreenPoint      : px, py (int), screen_id
SmoothedPoint    : px, py (int), is_fixation(bool), velocity(float)
PipelineTick     : frame_id, timestamp, state, stage_timings(dict[str,float]), point|None
```

**Regra:** nenhum estágio recebe ou devolve `dict` ou tupla anônima. Sempre um tipo nomeado. Isso é o que impede o "bug silencioso de ordem de argumentos" que assola pipelines de visão.

---

## 7. O modelo de IA e seu contrato

**Este projeto usa o modelo já treinado fornecido pela IrisFlow como motor de estimativa de olhar. Nenhum treinamento ocorre em nenhuma Sprint.** O papel do backend em relação ao modelo é triplo e apenas isto: (a) entregar entradas exatamente no formato em que ele foi treinado, (b) executá-lo com baixa latência, (c) corrigir e estabilizar sua saída via calibração e filtros.

### 7.1 Artefatos

| Arquivo | Papel | Observação |
|---|---|---|
| `gaze_cnn_best.keras` | **Modelo de produção padrão** | Melhor checkpoint por validação |
| `gaze_cnn_last.keras` | Fallback / comparação | Último estado do treino (`checkpoint_state.json`: época 82) |
| `gaze_encoder.onnx` | Backend alternativo / acelerado | **Nome sugere encoder** — pode emitir *embedding*, não `(gaze_x, gaze_y)` |
| `training_log.csv` | Referência de qualidade | Estabelece o piso de erro esperado em validação |

### 7.2 Contrato declarado

```
Entradas:  face      (224, 224, 3)
           eye_left  (112, 112, 3)
           eye_right (112, 112, 3)
           rect      (12,)
Saída:     (gaze_x, gaze_y) normalizados em [0, 1]
```

### 7.3 ⚠️ Gate obrigatório da Sprint 6 — verificação do contrato real

**Nada no pipeline pode ser construído sobre suposições aqui.** Antes de escrever o backend de inferência, `scripts/inspect_model.py` deve extrair e registrar em `models/MODEL_CARD.md`:

1. **Nomes e ordem exatos** dos tensores de entrada (Keras nomeia entradas; a ordem posicional pode não ser a esperada).
2. **Layout de canais**: RGB ou BGR? OpenCV entrega BGR; se o treino usou RGB, o modelo funciona "quase certo" — o pior tipo de bug.
3. **Faixa de normalização**: `[0,1]`, `[-1,1]`, ou padronização por média/desvio do ImageNet?
4. **Semântica das 12 dimensões do `rect`** — ver 7.4.
5. **Saída do `gaze_encoder.onnx`**: shape `(1,2)` (é um estimador completo) ou shape maior (é um encoder e **não** serve como backend direto)?
6. **Convenção de eixos**: `y` cresce para baixo (imagem) ou para cima (cartesiano)?

**Fonte da verdade = o código de treino.** Se ele estiver disponível, ele resolve os 6 pontos. Se não estiver, cada item vira uma hipótese a ser validada empiricamente (Sprint 6, tarefa de *sanity check*).

### 7.4 Definição canônica do vetor `rect` (proposta — confirmar na Sprint 6)

Convenção mais comum em arquiteturas do tipo iTracker/GazeCapture, normalizada pela largura/altura do frame:

| Índice | Campo | Normalização |
|---|---|---|
| 0–3 | `face_x, face_y, face_w, face_h` | `/W`, `/H`, `/W`, `/H` |
| 4–7 | `left_eye_x, left_eye_y, left_eye_w, left_eye_h` | idem |
| 8–11 | `right_eye_x, right_eye_y, right_eye_w, right_eye_h` | idem |

Ambiguidades a resolver explicitamente e documentar em `rect_vector.py`:
- "esquerdo" = esquerda **do usuário** ou esquerda **da imagem** (espelhada)?
- `x, y` = canto superior-esquerdo ou centro do box?
- coordenadas dos olhos relativas ao frame ou ao recorte da face?

> **Sanity check empírico:** com o rosto centralizado e o usuário olhando fixamente para o centro da tela, a saída deve ficar próxima de `(0.5, 0.5)`. Olhando para os quatro cantos, as saídas devem se separar de forma monotônica e coerente. Se olhar para a esquerda aumenta `gaze_x`, há espelhamento invertido em algum ponto da cadeia.

### 7.5 Substituibilidade (capacidade futura, não plano de troca)

> **Leia com atenção:** esta seção descreve uma propriedade de manutenibilidade, **não** uma intenção de substituir o modelo. O motor de produção é e continua sendo o `gaze_cnn_best.keras`. O objetivo aqui é apenas garantir que, se você treinar uma versão melhor daqui a seis meses, a troca custe uma linha de configuração em vez de uma refatoração.

Todo o sistema conhece apenas:

```python
class GazeEstimator(Protocol):
    def predict(self, model_input: ModelInput) -> RawGaze: ...
    def warmup(self) -> None: ...
    @property
    def metadata(self) -> ModelMetadata: ...
```

Uma eventual troca futura = adicionar uma classe em `inference/` e mudar uma linha do YAML. Nenhum outro arquivo é tocado. **Este é o requisito arquitetural mais importante do documento** — e, justamente por estar garantido, não há qualquer motivo para considerar outro motor durante a implementação atual.

---

## 8. Roadmap de Sprints

Cada Sprint é independente, entrega valor verificável e termina com testes verdes. **Nenhuma Sprint pode ser iniciada com a anterior "quase pronta".**

Legenda de esforço: `P` ≈ 2h · `M` ≈ 4h · `G` ≈ 6-8h

---

### Sprint 0 — Fundação do repositório · `P`
**Depende de:** nada

**Objetivo:** esqueleto executável com qualidade automatizada, para que nenhuma Sprint seguinte precise discutir tooling.

**Entregáveis**
- `pyproject.toml` (layout `src/`, Python 3.11+, gerenciado com `uv`)
- Dependências: `opencv-python`, `mediapipe`, `numpy`, `pydantic`, `pydantic-settings`, `pyyaml`, `structlog`, `typer`, `pynput`, `pytest`, `pytest-cov`, `ruff`, `mypy`, `import-linter`
- Árvore completa de pastas com `__init__.py`
- `ruff` + `mypy --strict` no núcleo + `pre-commit` + CI no GitHub Actions
- Contrato de camadas em `import-linter` (falha o build se o domínio importar adaptador)
- `python -m irisflow --version` funcionando

**Critérios de conclusão**
- [x] `uv sync && pytest && ruff check && mypy src/` passam em ambiente limpo
- [x] CI verde no primeiro push
- [x] `import-linter` já reprova uma violação proposital de teste

---

### Sprint 1 — Configuração e Logging · `P`
**Depende de:** S0

**Objetivo:** toda constante do sistema nasce em um só lugar, validada.

**Entregáveis**
- `config/schema.py`: `CameraConfig`, `DetectionConfig`, `ModelConfig`, `CalibrationConfig`, `FilterConfig`, `ControlConfig`, `TelemetryConfig`, `AppConfig`
- `config/loader.py`: precedência **CLI > env (`IRISFLOW_*`) > YAML > defaults**
- `configs/default.yaml` documentado campo a campo
- `logging/setup.py`: structlog com console legível + JSON em `data/logs/`, com `frame_id` no contexto
- `core/exceptions.py`: `IrisFlowError` → `CaptureError`, `DetectionError`, `InferenceError`, `CalibrationError`, `ConfigError`

**Critérios de conclusão**
- [x] Config inválida falha na inicialização com mensagem que aponta o campo exato
- [x] `IRISFLOW_CAMERA__DEVICE_ID=1` sobrescreve o YAML (teste automatizado)
- [x] Nenhum `print()` no `src/`

---

### Sprint 2 — Contratos do domínio · `M`
**Depende de:** S0

**Objetivo:** congelar as fronteiras entre estágios antes de existir qualquer implementação. **Sprint mais barata e mais valiosa do projeto.**

**Entregáveis**
- `core/types.py` com todos os tipos da seção 6
- `core/interfaces.py` com todos os `Protocol`s: `FrameSource`, `FaceDetector`, `Preprocessor`, `GazeEstimator`, `CalibrationModel`, `ScreenMapper`, `Filter`, `CursorController`
- `core/geometry.py`: `expand`, `clip_to_frame`, `to_square`, `normalize`, `denormalize`, `clamp`
- `core/clock.py`: `Clock` real + `FakeClock` para testes
- `core/events.py`: `GazeUpdated`, `FaceLost`, `FaceAcquired`, `StateChanged`, `CalibrationProgress`
- `tests/fixtures/stubs.py`: implementação stub de **cada** Protocol

**Critérios de conclusão**
- [x] `mypy --strict` limpo em `core/`
- [x] 100% de cobertura em `geometry.py` (inclui casos de borda: box saindo do frame, w/h zero, coordenadas negativas)
- [x] Cada Protocol tem um stub e um teste que verifica conformidade estrutural
- [x] `core/` não importa nada além de stdlib e numpy

---

### Sprint 3 — Captura de vídeo · `M`
**Depende de:** S1, S2

**Objetivo:** frame mais recente disponível sob demanda, sem acúmulo de latência.

**Entregáveis**
- `capture/webcam.py`: thread dedicada, fila `maxsize=1` com descarte, context manager, reconexão automática com backoff
- `capture/video_file.py` e `capture/synthetic.py`
- `capture/diagnostics.py`: lista dispositivos, resolução e **FPS real medido** (o declarado costuma mentir)
- CLI `irisflow doctor`: reporta câmera, resolução, FPS, latência de leitura

**Critérios de conclusão**
- [x] 60 s de captura contínua sem crescimento de memória
- [x] Timestamps monotônicos e sem gaps anômalos
- [x] Desconectar a webcam durante execução → erro tratado + tentativa de reconexão, sem travar o processo
- [x] `irisflow doctor` roda em máquina sem câmera e informa isso com clareza
- [x] Testes usam `synthetic`/`video_file` — **suite nunca depende de hardware**

---

### Sprint 4 — Detecção facial e ocular · `G`
**Depende de:** S3

**Objetivo:** ROIs de face e olhos estáveis o suficiente para alimentar a CNN.

**Entregáveis**
- `detection/mediapipe_detector.py` (FaceMesh, `refine_landmarks=True` para íris)
- `detection/landmarks.py`: índices nomeados como constantes, nunca literais no código
- `detection/roi.py`: landmarks → bounding boxes com margem configurável e conversão para quadrado
- `detection/tracking.py`: suavização de ROI (EMA sobre os boxes), histerese de perda (N frames antes de declarar `LOST`), reaquisição
- CLI `irisflow preview`: janela com overlay dos boxes (ferramenta de depuração essencial)

**Critérios de conclusão**
- [x] Detecta corretamente em ≥ 95% dos frames de um vídeo fixture de 30 s
- [x] ROIs não tremem visivelmente com o rosto parado (variação < 3 px)
- [x] Rosto parcialmente fora do frame → box recortado corretamente, sem índice negativo
- [x] Ausência de rosto → retorna `None`, nunca levanta exceção no loop
- [x] Detecção ≤ 15 ms por frame em CPU de referência

---

### Sprint 5 — Pré-processamento e vetor `rect` · `M`
**Depende de:** S4

**Objetivo:** produzir tensores **bit-a-bit equivalentes** aos do treino.

**Entregáveis**
- `preprocessing/crops.py`: recorte com padding de borda (replicar, não preencher de preto)
- `preprocessing/resize.py`: interpolação consistente com o treino (documentar `INTER_AREA` vs `INTER_LINEAR`)
- `preprocessing/normalize.py`: parametrizado por config (`[0,1]`, `[-1,1]`, ImageNet) e por ordem de canais
- `preprocessing/rect_vector.py`: construção documentada das 12 dimensões, com a convenção escolhida explícita em docstring
- `preprocessing/builder.py`: `FaceDetection + Frame → ModelInput`

**Critérios de conclusão**
- [x] Shapes e dtypes exatos: `(224,224,3) f32`, `(112,112,3) f32`, `(112,112,3) f32`, `(12,) f32`
- [x] Vetor `rect` sempre em `[0,1]` — teste com rosto em todos os cantos do frame
- [x] Frames golden versionados: alteração no pré-processamento quebra o teste (proteção contra regressão silenciosa)
- [x] Pré-processamento ≤ 5 ms por frame
- [x] Zero alocação de arrays dentro do loop quente (buffers reaproveitados)

---

### Sprint 6 — Camada de inferência ⭐ *sprint de maior risco* · `G`
**Depende de:** S5

**Objetivo:** transformar `ModelInput` em `RawGaze` confiável, com backend intercambiável.

**Entregáveis**
- `scripts/inspect_model.py` → `models/MODEL_CARD.md` respondendo aos 6 pontos da seção 7.3
- `inference/keras_backend.py` e `inference/onnx_backend.py`
- `inference/registry.py`: seleção por config (`model.backend: keras|onnx`)
- `inference/warmup.py`: 3 inferências dummy na inicialização
- `inference/parity.py` + CLI `irisflow bench --compare-backends`
- **Sanity check de orientação** (seção 7.4) documentado com resultados reais

**Critérios de conclusão**
- [x] `MODEL_CARD.md` preenchido — **nenhum campo "a confirmar"**
- [x] Saída sempre em `[0,1]²`, com clamp defensivo e log de warning se o modelo extrapolar
- [x] Paridade Keras × ONNX: diferença máxima < 1e-3 **ou** documentado por que o ONNX não é utilizável como estimador completo (caso ele seja apenas encoder)
- [x] Sanity check de orientação aprovado: centro ≈ `(0.5, 0.5)`, cantos separam de forma coerente e monotônica
- [x] Latência de inferência ≤ 25 ms p95 em CPU de referência
- [x] Trocar `keras` → `onnx` no YAML funciona sem alterar nenhuma outra linha

> **Gate de projeto:** se o sanity check falhar, **pare**. Não prossiga para calibração — a calibração vai "esconder" o erro de orientação atrás de uma transformação aprendida, e o bug reaparecerá de forma incompreensível na Sprint 10.

---

### Sprint 7 — Pipeline e orquestração · `G`
**Depende de:** S6

**Objetivo:** primeiro sistema fim-a-fim rodando: webcam → coordenada de gaze no terminal.

**Entregáveis**
- `pipeline/stages.py`: cada estágio encapsulado, com medição automática de duração
- `pipeline/orchestrator.py`: montagem por injeção de dependência a partir da config
- `pipeline/runner.py`: loop, tratamento de erro por estágio, shutdown limpo (SIGINT)
- `pipeline/state.py`: máquina de estados da seção 4.3
- `pipeline/bus.py`: publicação de eventos com múltiplos sinks
- `telemetry/metrics.py`: FPS, latência por estágio (p50/p95), frames descartados
- CLI `irisflow run --no-cursor`: imprime gaze e métricas ao vivo

**Critérios de conclusão**
- [x] 5 minutos contínuos sem crash e sem vazamento de memória
- [x] Exceção em um estágio → frame descartado com log, pipeline continua vivo
- [x] `Ctrl+C` libera câmera e threads em < 1 s
- [x] Latência fim-a-fim reportada, p95 ≤ 80 ms
- [x] Teste de integração completo com `SyntheticSource` + `StubGazeEstimator`, sem hardware

---

### Sprint 8 — Calibração · `G`
**Depende de:** S7

**Objetivo:** corrigir o viés individual do usuário — sem isso, a precisão é inaceitável na prática.

**Entregáveis**
- `calibration/points.py`: grades de 5, 9 e 13 pontos com margem de borda
- `calibration/session.py`: máquina de estados da coleta, com progresso publicado no bus
- `calibration/collector.py`: espera estabilização, coleta N amostras, descarta outliers e frames com rosto perdido
- `calibration/models.py`: `AffineCalibration`, `PolynomialCalibration` (2ª ordem), `RidgeCalibration` — todas sob o mesmo Protocol
- `calibration/quality.py`: erro residual por alvo, validação hold-out, identificação de alvos ruins
- `calibration/store.py`: persistência JSON versionada em `data/calibration/<profile>.json`
- CLI `irisflow calibrate --profile <nome>`

**Critérios de conclusão**
- [x] Sessão completa de 9 pontos em ≤ 60 s
- [x] Perfil salvo e recarregado com resultado numericamente idêntico
- [x] Modelo polinomial reduz o erro em ≥ 30% frente ao passthrough (medido em hold-out)
- [x] Alvos com erro alto são identificados e permitem recoleta parcial
- [x] Sem calibração → sistema funciona em modo degradado com warning explícito, nunca falha
- [x] Testes unitários com dados sintéticos de viés conhecido: a transformação recupera o viés injetado

---

### Sprint 9 — Mapeamento para tela e filtros temporais · `M`
**Depende de:** S8

**Objetivo:** um ponto de tela estável o bastante para ser usado por horas sem fadiga.

**Entregáveis**
- `mapping/screen.py` + `mapping/screen_info.py` (resolução, DPI, múltiplos monitores)
- `filtering/outlier.py`: rejeição por velocidade fisicamente implausível
- `filtering/one_euro.py`: implementação completa e parametrizada (`min_cutoff`, `beta`)
- `filtering/ema.py` como baseline comparativo
- `filtering/fixation.py`: classificação fixação × sacada por limiar de velocidade
- `filtering/chain.py`: composição declarada no YAML
- CLI `irisflow bench --filters`: compara jitter e latência entre configurações

**Critérios de conclusão**
- [ ] Jitter em fixação ≤ 15 px RMS
- [ ] Latência adicionada pela filtragem ≤ 20 ms durante sacadas (o One Euro não pode "borrar" movimentos rápidos)
- [ ] Ponto sempre dentro dos limites da tela (clamp verificado)
- [ ] Filtros são funções puras testadas com sinais sintéticos (degrau, rampa, ruído branco) e `FakeClock`
- [ ] Trocar a cadeia de filtros no YAML não exige mudança de código

---

### Sprint 10 — Controle do cursor e segurança · `M`
**Depende de:** S9

**Objetivo:** efeito real no sistema operacional, com garantias de segurança para o usuário com ELA.

**Entregáveis**
- `control/cursor.py` (pynput), com limitação de taxa de movimento
- `control/dwell.py`: clique por permanência (raio + duração configuráveis), com estado de progresso publicado no bus (para o frontend desenhar o anel de progresso)
- `control/safety.py`:
  - **kill switch** por atalho global (`Ctrl+Alt+Esc`)
  - **pausa automática** quando o rosto some por > N segundos
  - **zona de descanso** configurável que não gera cliques
  - **período de refratariedade** após cada clique (evita cliques repetidos acidentais)
  - **watchdog**: se o pipeline travar, o controle do mouse é liberado imediatamente
- `control/noop.py` para testes e modo headless
- CLI `irisflow run` completo

**Critérios de conclusão**
- [ ] Cursor acompanha o olhar de forma utilizável em teste manual
- [ ] Kill switch libera o mouse em ≤ 100 ms, testado sob carga
- [ ] Nenhum clique disparado em estado `LOST` ou `PAUSED`
- [ ] Crash forçado do processo não deixa o mouse "sequestrado"
- [ ] Toda a suite automatizada roda com `NoOpCursor` — **testes nunca movem o mouse do desenvolvedor**

> Marco: **backend funcional completo em modo standalone.** Da Sprint 11 em diante, o trabalho é de exposição e robustez.

---

### Sprint 11 — Telemetria, gravação e replay · `M`
**Depende de:** S10

**Objetivo:** tornar o sistema depurável e mensurável — pré-requisito para ajustar parâmetros com um usuário real sem tentativa e erro.

**Entregáveis**
- `telemetry/recorder.py`: grava sessão (frames opcionais + todos os dados intermediários) em `data/recordings/`
- CLI `irisflow replay <sessão>`: reexecuta o pipeline sobre a gravação, de forma determinística
- `telemetry/report.py`: relatório de sessão (FPS, latência por estágio, % de rosto perdido, jitter, cliques)
- Fixtures de sessão gravada para os testes E2E

**Critérios de conclusão**
- [ ] Replay produz saída **idêntica** à execução original (determinismo com `FakeClock`)
- [ ] Relatório gerado ao fim de cada sessão
- [ ] Overhead de telemetria ≤ 3% do tempo de frame
- [ ] Suite E2E roda sobre gravação, sem webcam, em CI

---

### Sprint 12 — Camada FastAPI + WebSocket · `G`
**Depende de:** S11

**Objetivo:** expor o pipeline ao frontend, **sem mover nenhuma lógica para dentro da API.**

**Entregáveis**
- `api/schemas.py`: contratos Pydantic de todas as mensagens (seção 11)
- `api/app.py`: factory FastAPI com `lifespan` gerenciando o pipeline
- `api/ws.py`: `/ws/gaze` — streaming de posição + eventos
- `api/routes/`: `GET /health`, `GET/POST /config`, `POST /calibration/start`, `GET /calibration/status`, `GET /profiles`
- `api/session.py`: ponte entre o `EventBus` e as conexões WS, com *backpressure* (descartar, nunca enfileirar indefinidamente)
- CLI `irisflow serve`

**Critérios de conclusão**
- [ ] Frontend recebe posição a ≥ 20 Hz de forma estável
- [ ] Cliente lento não degrada o pipeline (mensagens descartadas, medido)
- [ ] Desconexão abrupta não derruba o pipeline; múltiplos clientes simultâneos suportados
- [ ] Calibração pode ser conduzida inteiramente pelo WebSocket
- [ ] **`api/` não contém nenhuma regra de rastreamento** — verificado por `import-linter` e por revisão
- [ ] Testes com `httpx` + cliente WS de teste do FastAPI

---

### Sprint 13 — Hardening e documentação final · `M`
**Depende de:** S12

**Objetivo:** deixar o sistema pronto para uso real por alguém que não o escreveu.

**Entregáveis**
- Perfil de performance e otimização dos gargalos identificados
- Tratamento de todos os erros conhecidos com mensagens acionáveis
- `README.md`: instalação, primeiro uso, calibração, troubleshooting
- `configs/schema.md`: referência completa de configuração
- `docs/ARCHITECTURE.md` e `docs/CONTRIBUTING.md`
- Verificação da cobertura de testes por camada

**Critérios de conclusão**
- [ ] Sessão de 30 min sem degradação de FPS ou crescimento de memória
- [ ] Cobertura: ≥ 90% no domínio, ≥ 70% global
- [ ] Instalação a partir do zero seguindo apenas o README, em máquina limpa
- [ ] Todas as decisões em aberto da seção 13 resolvidas e documentadas

---

## 9. Grafo de dependências entre Sprints

```
S0 ──┬──► S1 ──┐
     │         ├──► S3 ──► S4 ──► S5 ──► S6 ──► S7 ──► S8 ──► S9 ──► S10 ──► S11 ──► S12 ──► S13
     └──► S2 ──┘         ▲                ▲       ▲
                         │                │       │
                    (paralelizável)   GATE ⭐   marco: fim-a-fim
```

**Caminho crítico:** `S3 → S4 → S5 → S6 → S7 → S8 → S9 → S10`. Atraso aqui atrasa tudo.

**Paralelizações possíveis** (se houver mais de um executor):
- S1 e S2 em paralelo após S0
- S9 (filtros — lógica pura, testável com sinais sintéticos) pode ser desenvolvida em paralelo a S6/S7
- S11 (telemetria) pode começar em paralelo a S10

**Regra inegociável:** S6 é um *gate*. Nada depois dela deve começar antes do `MODEL_CARD.md` estar completo e o sanity check aprovado.

---

## 10. Estratégia de testes

### 10.1 Pirâmide

| Nível | Alvo | Escopo | Roda em CI |
|---|---|---|---|
| **Unitário** (~70%) | Domínio puro: geometria, rect, calibração, filtros, mapeamento | Sem I/O, sem modelo, `FakeClock` | Sim, < 10 s |
| **Integração** (~25%) | Pipeline com fontes falsas e estimador stub | Sem webcam, sem modelo real | Sim, < 60 s |
| **E2E** (~5%) | Sessão gravada → pipeline completo → `NoOpCursor` | Modelo real, sem hardware | Sim (nightly) |
| **Manual** | Precisão percebida, conforto, fadiga | Usuário real | Não |

### 10.2 Dublês obrigatórios (`tests/fixtures/stubs.py`, Sprint 2)

| Dublê | Substitui | Comportamento |
|---|---|---|
| `SyntheticFrameSource` | Webcam | Frames determinísticos com padrão conhecido |
| `VideoFileSource` | Webcam | Vídeo fixture versionado |
| `StubFaceDetector` | MediaPipe | Retorna ROIs fixas ou sequência roteirizada |
| `StubGazeEstimator` | CNN | Retorna trajetória programada (círculo, degrau, ruído) |
| `NoOpCursorController` | pynput | Registra chamadas, não toca no SO |
| `FakeClock` | Tempo real | Avanço controlado — filtros testáveis deterministicamente |

### 10.3 Testes específicos por tipo de risco

- **Golden frames** (S5): tensores de referência versionados. Qualquer mudança no pré-processamento que altere a saída quebra o teste — é a única defesa contra "o modelo piorou e ninguém sabe por quê".
- **Property-based** (`hypothesis`) em `geometry.py` e `mapping/`: para qualquer box e frame, o resultado clipado está sempre dentro do frame.
- **Sinais sintéticos** para filtros: degrau (mede latência), sinal constante + ruído (mede jitter), sacada rápida (mede *overshoot*).
- **Viés injetado** para calibração: gerar dados com transformação conhecida, verificar que o ajuste a recupera dentro da tolerância.
- **Testes de resiliência**: câmera desconectada, frame corrompido, rosto ausente por 30 s, modelo retornando `NaN`.
- **Testes de segurança** (S10): kill switch sob carga, ausência de clique em estado inválido, liberação do mouse após crash.

### 10.4 Regras da suite

1. Nenhum teste exige webcam. Nenhum teste move o mouse real.
2. Suite unitária + integração completa em < 90 s.
3. Todo bug corrigido entra com um teste de regressão antes do fix.
4. Testes que dependem do modelo real são marcados `@pytest.mark.model` e podem ser pulados.

---

## 11. Integração futura via WebSocket/FastAPI

Implementada apenas na **Sprint 12**, mas prevista desde a Sprint 7 pelo `EventBus`. O WebSocket é apenas mais um *sink* — o pipeline não sabe que ele existe.

### 11.1 Superfície da API

| Método | Rota | Propósito |
|---|---|---|
| `GET` | `/health` | Estado do pipeline, FPS, backend do modelo |
| `GET` | `/config` | Configuração efetiva atual |
| `PATCH` | `/config` | Ajuste em runtime de parâmetros seguros (filtros, dwell) |
| `GET` | `/profiles` | Perfis de calibração disponíveis |
| `POST` | `/calibration/start` | Inicia sessão (nº de pontos, resolução da tela) |
| `POST` | `/calibration/abort` | Aborta sessão |
| `WS` | `/ws/gaze` | Canal bidirecional principal |

### 11.2 Mensagens do WebSocket

**Servidor → Cliente** (alta frequência, ~30 Hz):
```json
{ "type": "gaze", "x": 842, "y": 391, "nx": 0.438, "ny": 0.362,
  "fixation": true, "confidence": 0.91, "ts": 1730000000.123, "frame_id": 4821 }
```

**Servidor → Cliente** (eventos, baixa frequência):
```json
{ "type": "state",             "state": "TRACKING", "previous": "LOST" }
{ "type": "face_lost",         "duration_ms": 640 }
{ "type": "dwell_progress",    "progress": 0.62, "x": 842, "y": 391 }
{ "type": "click",             "x": 842, "y": 391, "button": "left" }
{ "type": "calibration_point", "index": 3, "total": 9, "x": 0.5, "y": 0.1, "phase": "collecting" }
{ "type": "calibration_result","error_px": 47.2, "worst_points": [2, 7], "accepted": true }
{ "type": "error",             "code": "CAMERA_LOST", "message": "..." }
```

**Cliente → Servidor:**
```json
{ "type": "start_tracking" }   { "type": "pause" }   { "type": "resume" }
{ "type": "start_calibration", "points": 9, "screen": {"w": 1920, "h": 1080} }
{ "type": "calibration_ready", "index": 3 }
{ "type": "set_profile", "profile": "maria" }
{ "type": "set_cursor_control", "enabled": false }
```

### 11.3 Regras de projeto da camada de API

1. **Backpressure por descarte.** Fila por cliente com `maxsize` pequeno; se encher, descarta a posição mais antiga. Nunca bloquear o pipeline por causa de um cliente lento.
2. **Separação de canais.** Posições são efêmeras e descartáveis; eventos de estado e calibração são críticos e devem ser entregues. Filas separadas com políticas distintas.
3. **Ponte thread → asyncio.** O pipeline roda em thread; usar `asyncio.run_coroutine_threadsafe` ou uma fila thread-safe consumida pelo loop de eventos. Nunca chamar `await` a partir da thread do pipeline.
4. **A API não decide nada.** Ela traduz mensagens em chamadas do orquestrador. Zero regra de rastreamento.
5. **Segurança.** Bind em `127.0.0.1` por padrão. Controle de cursor é ativado explicitamente pelo cliente, nunca por default.
6. **Versionamento.** Campo `protocol_version` no handshake, desde o primeiro dia.

---

## 12. Cronograma de 10 dias

Estimativa para **um desenvolvedor em tempo integral assistido por Claude Code**, ~7h produtivas/dia.

| Dia | Sprints | Entregável observável ao fim do dia |
|---|---|---|
| **1** | S0 + S1 + S2 | Repo com CI verde, config validada, tipos e protocolos congelados |
| **2** | S3 + início S4 | `irisflow doctor` funciona; captura estável em thread |
| **3** | S4 | `irisflow preview` mostra face e olhos com boxes estáveis |
| **4** | S5 + `inspect_model.py` | `ModelInput` correto; `MODEL_CARD.md` preenchido |
| **5** | S6 ⭐ | Gaze bruto saindo do modelo com sanity check aprovado |
| **6** | S7 | Pipeline fim-a-fim imprimindo coordenadas + métricas |
| **7** | S8 | Calibração de 9 pontos funcionando e persistida |
| **8** | S9 + S10 | **Cursor controlado pelo olhar, com kill switch** |
| **9** | S11 + início S12 | Replay determinístico; servidor FastAPI de pé |
| **10** | S12 + S13 | WebSocket em produção; docs e hardening completos |

### Marcos de verificação

| Fim do dia | Marco | Se não atingido |
|---|---|---|
| 3 | Detecção estável | Reduzir escopo: aceitar ROIs de qualidade menor e revisitar em S13 |
| 5 | **Modelo inferindo corretamente** | **Não avance.** Este é o risco #1 do projeto |
| 8 | Cursor funcional | Adiar S11 para depois de S12; telemetria é a folga do cronograma |
| 10 | Backend completo | S13 é a folga; documentação pode escorregar para o dia 11 |

**Onde está a folga:** S11 e S13 são compressíveis. S6, S7, S8 e S10 não são.

---

## 13. Riscos e decisões em aberto

| # | Risco | Impacto | Mitigação |
|---|---|---|---|
| R1 | **Contrato real do modelo diferente do documentado** (canais, normalização, semântica do `rect`) | Sistema "quase funciona" — pior cenário possível | Gate da S6; sanity check obrigatório; recuperar o código de treino |
| R2 | `gaze_encoder.onnx` ser encoder e não estimador | Perda do backend alternativo — **nenhum impacto sobre o motor**, já que o `.keras` é o padrão de produção | Verificar shape na S6; se confirmado, Keras é o único backend e o ONNX é reservado para exportação futura |
| R3 | Latência acima do alvo em CPU modesta | Experiência inutilizável | Medir desde a S6; opções: reduzir resolução de captura, inferir a cada 2 frames com interpolação, quantização ONNX |
| R4 | Precisão insuficiente após calibração | Alvos pequenos inalcançáveis | Calibração polinomial; aumentar nº de pontos; considerar zonas ampliadas no frontend |
| R5 | Sensibilidade a iluminação e posição da cabeça | Uso real degradado | `irisflow doctor` avalia iluminação; documentar setup recomendado |
| R6 | Fadiga ocular do usuário | Problema clínico, não técnico | Dwell configurável, zona de descanso, sessões curtas — validar com fonoaudiólogo/terapeuta ocupacional |

### Decisões a tomar (registrar em `docs/DECISIONS.md`)

- [ ] **D1** — Semântica exata do vetor `rect` *(bloqueia S5; resolver no dia 4)*
- [ ] **D2** — Ordem de canais e esquema de normalização *(bloqueia S5)*
- [ ] **D3** — Backend padrão de produção: Keras ou ONNX *(S6)*
- [ ] **D4** — Modelo de calibração padrão: polinomial ou ridge *(S8, decidir por medição)*
- [ ] **D5** — Biblioteca de controle de cursor: `pynput` vs `pyautogui` *(S10 — recomendação: `pynput`, mais leve e sem dependência de screenshot)*
- [ ] **D6** — Estratégia de clique: apenas dwell, ou também piscada *(S10 — piscada é arriscada com ELA; dwell é o padrão)*

---

## 14. Convenções de código e Definition of Done global

### Convenções

- Python 3.11+, type hints obrigatórios em todo código público
- `ruff` (lint + format), `mypy --strict` em `core/`, `preprocessing/`, `calibration/`, `filtering/`, `mapping/`
- Docstrings estilo Google em toda API pública; **unidades sempre explícitas** (`px`, `ms`, `graus`, `normalizado`)
- Nomes descritivos e sem abreviação: `left_eye_bbox`, não `leb`
- Nenhum número mágico fora de `config/`
- Commits em Conventional Commits; um commit por unidade lógica
- Uma branch por Sprint: `sprint/06-inference`

### Definition of Done — aplica-se a **toda** Sprint

Uma Sprint só está concluída quando **todos** os itens abaixo são verdadeiros:

- [ ] Todos os critérios de conclusão específicos da Sprint estão marcados
- [ ] `pytest`, `ruff check`, `mypy` e `import-linter` passam
- [ ] Cobertura da lógica nova ≥ 80% (≥ 90% se for domínio puro)
- [ ] Nenhum `TODO` sem issue associada; nenhum `print()`; nenhum `except: pass`
- [ ] Configurações novas documentadas em `configs/schema.md`
- [ ] Erros novos herdam de `IrisFlowError` e trazem mensagem acionável
- [ ] Existe uma forma de **observar o resultado** (comando de CLI, log ou métrica)
- [ ] `README.md` atualizado se a forma de uso mudou
- [ ] Branch mergeada com CI verde

### Instruções para o Claude Code

1. **Uma Sprint por vez.** Não antecipe código de Sprints futuras — é a principal fonte de retrabalho.
2. **Leia os Protocols antes de implementar.** Se a implementação exigir mudar um Protocol, pare e sinalize: isso significa que o contrato da Sprint 2 estava errado e a mudança afeta outros módulos.
3. **Testes junto com o código**, não depois.
4. **Se um critério de conclusão não puder ser atendido, reporte antes de contornar.** Não "adapte" o critério silenciosamente.
5. **Não invente o comportamento do modelo.** Qualquer suposição sobre entradas/saídas deve ser verificada empiricamente e registrada em `MODEL_CARD.md`.
6. **Nunca substitua o motor de inferência.** O modelo fornecido (`gaze_cnn_best.keras`) é o único estimador de olhar do sistema. É proibido implementar heurística geométrica, regressão sobre landmarks, biblioteca de gaze de terceiros ou "estimador temporário" para destravar uma Sprint, mesmo que o modelo esteja apresentando resultados ruins. Se o modelo parecer errado, o problema está no pré-processamento (canais, normalização, `rect`) — pare e reporte. O `StubGazeEstimator` existe **apenas** dentro de `tests/` e não pode ser importado por nenhum arquivo em `src/irisflow/`.
7. **Ao terminar uma Sprint, reporte:** o que foi entregue, quais critérios foram atendidos, o que ficou pendente e quais decisões foram tomadas.