# Contributing to IrisFlow

Guia curto para novos contribuidores. `SPRINTS.MD` é o SSOT do projeto e
o `docs/ARCHITECTURE.md` explica como o código está organizado — comece
por eles.

## Setup local

Requisitos: Python **3.11+**, [`uv`](https://docs.astral.sh/uv/) e uma
webcam apenas se você for rodar o pipeline (a suite de testes nunca
precisa de câmera).

```bash
uv sync --extra dev
uv run pre-commit install
```

Extras opcionais (só se você for tocar essas camadas):

```bash
uv sync --extra dev --extra inference-keras   # backend Keras real
uv sync --extra dev --extra inference-onnx    # backend ONNX real
uv sync --extra dev --extra api               # FastAPI + WebSocket
```

## Quality gates (mesmos do CI)

Todos precisam passar antes do PR:

```bash
.venv/Scripts/pytest.exe                     # ou uv run pytest
.venv/Scripts/ruff.exe check src tests
.venv/Scripts/ruff.exe format --check src tests
.venv/Scripts/mypy.exe src/irisflow
.venv/Scripts/lint-imports.exe               # NÃO python -m importlinter
```

> **Windows.** `uv run <cmd>` costuma re-sincronizar o env em cada
> execução; para iteração rápida, chame os binários direto do
> `.venv/Scripts/`.

## Cobertura por camada (baseline Sprint 13)

Alvo global ≥ 70%, domínio ≥ 90%. Estado atual (Sprint 13):

| Camada | Cobertura | Status |
|---|---:|---|
| `core/` | 100% | ✅ |
| `preprocessing/` | 100% | ✅ |
| `calibration/` | 87-100% | ✅ (domínio ≥ 90%) |
| `mapping/` | 100% | ✅ |
| `filtering/` | 99-100% | ✅ |
| `capture/` | 89-100% | ✅ |
| `detection/` | 73-100% | ✅ (paths com hardware ignorados) |
| `inference/` | 62-100% | ✅ (ONNX encoder-only path skipa) |
| `control/` | 87-100% | ✅ |
| `pipeline/` | 44-100% | ✅ (`orchestrator` tem paths com Keras/webcam reais) |
| `api/` | 54-100% | ✅ (`ws.py` conexões edge-cases; `state.py` calibração via WS) |
| `cli/` | 16-100% | ⚠️ CLI é smoke-tested no `test_cli_main.py`; comandos que iniciam pipeline real (`run`, `serve`) rodam sob o marker `@model` |
| **Global** | **82%** | ✅ |

Rodar coverage:

```bash
.venv/Scripts/pytest.exe --cov=irisflow --cov-report=term
```

## Regras não-negociáveis do projeto

1. **Não substituir o motor de gaze.** O modelo `gaze_cnn_best.keras` é
   o único estimador de produção. Se o resultado parece errado, o bug
   está em preprocessing/calibração/filtro. Ver SPRINTS §1.1.
2. **Nada de `StubGazeEstimator` fora de `tests/`.** `import-linter`
   reprova o build se um adaptador falso vazar para `src/`.
3. **Domínio importa Protocol, nunca implementação.** Se você precisa
   passar `Config` para um filtro, traduza para primitivos primeiro
   (padrão em `pipeline/orchestrator.filter_config_from_app`).
4. **`core/` importa apenas stdlib + numpy.** Se você precisou adicionar
   um import interno em `core/`, algo está no lugar errado.
5. **Uma Sprint por vez.** Não antecipar código de sprints futuras é a
   principal proteção contra retrabalho.

## Convenções de código

- Type hints em todo código público. `mypy --strict` para core, config,
  preprocessing, calibration, filtering, mapping.
- Docstrings estilo Google. **Unidades sempre explícitas**: `px`, `ms`,
  `graus`, `normalizado`.
- Sem números mágicos fora de `config/`.
- Frozen `@dataclass(frozen=True, slots=True)` para tipos do domínio.
- Erros do domínio herdam de `IrisFlowError` (`core/exceptions.py`) com
  mensagem que aponta o campo/ação afetados.
- Commits em Conventional Commits; um commit por unidade lógica.
- Uma branch por Sprint: `sprint/13-hardening`.

## Adicionando um estágio novo ao pipeline

1. Defina o `Protocol` em `core/interfaces.py` — assinatura mínima, sem
   detalhe de implementação.
2. Crie a implementação no layer certo (adapter se toca I/O; domínio se
   é lógica pura).
3. Wire em `pipeline/orchestrator.build_pipeline` com override opcional
   para testes.
4. Consuma no `pipeline/runner.py` dentro de `metrics.time("nome")`.
5. Se emite eventos novos, adicione em `core/events.py` e no `Event`
   union.
6. Teste em `tests/unit/` (implementação) + `tests/integration/`
   (fluxo com stubs).

## Adicionando um endpoint HTTP/WS

- HTTP: um arquivo em `api/routes/`, exporte `router`, registre em
  `api/routes/__init__.py`. Use `Annotated[AppState, Depends(...)]`
  em vez de `Depends(...)` no default (regra do ruff B008).
- WS: adicione o Pydantic message em `api/schemas.py` (dentro do union
  correto — `ServerMessage` ou `ClientMessage`) e case no dispatcher de
  `api/ws.py`.
- A camada API traduz, não decide. Se você precisou colocar `if state ==
  TRACKING` na rota, mova para `AppState` ou para o orchestrator.

## Pull requests

- Escopo pequeno (uma responsabilidade por PR).
- Descrição aponta a Sprint / DoD atendida.
- Testes junto do código, não em PR separado.
- CI verde (`pytest + ruff + mypy + lint-imports`) — não use `--no-verify`
  para pular hooks.
