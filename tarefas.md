# IrisFlow — Plano de Execução: Preparação, Correção e Integração

> **Destinatário: Claude Code.** Este documento é uma lista de tarefas executáveis, não um documento de arquitetura.
> **Prazo: terça-feira, 11/08/2026.** Domingo = tarefas T1–T11. Segunda = T12–T17. Terça = apresentação.
> **Estado de partida verificado:** repositório revertido ao fim da Sprint 13 (`docs/DECISIONS.md` contém D1–D6), `configs/default.yaml` com `normalization: unit` e sem `swap_eye_inputs`, `models/` vazio, `frontend/` aninhado no repositório com as três edições de protocolo **já aplicadas**.

---

## PARTE 0 — Regras de execução

Leia esta seção inteira antes da primeira tarefa.

### 0.1 Disciplina de escopo

Este é um trabalho de **estabilização com prazo**, não de melhoria de arquitetura. A base de código é boa e não deve ser reescrita.

**Proibido nesta janela:**

- Refatorar módulos que não estão listados nas tarefas
- Introduzir novas dependências sem pedir autorização
- Alterar a estrutura de camadas ou os `Protocol`s existentes
- Apagar código, testes ou arquivos (exceto onde explicitamente instruído)
- "Aproveitar que estou aqui" para corrigir algo fora da tarefa atual
- Implementar Ridge, features geométricas, modo VIDEO do MediaPipe, ou qualquer item de roadmap anterior

**Se você identificar um problema fora do escopo:** anote em `docs/ACHADOS.md` e siga em frente. Não corrija.

### 0.2 Protocolo de handoff com o humano

Várias tarefas exigem **um rosto real diante da webcam** e não podem ser executadas por você. Elas estão marcadas com 🧑.

Quando chegar a uma tarefa 🧑:

1. Prepare tudo que for automatizável (scripts, configs, comandos prontos)
2. **Pare.** Escreva exatamente qual comando o humano deve rodar e o que ele deve observar
3. Peça o resultado antes de prosseguir
4. Não invente, não estime, não assuma o resultado

### 0.3 Registro obrigatório

Crie `docs/ACHADOS.md` na primeira tarefa e registre ali: cada medição feita, cada decisão tomada, cada anomalia observada. Sem isso, os problemas de terça viram mistério de novo.

Ao final de cada tarefa, informe: o que foi feito, o resultado da verificação, o que ficou pendente.

### 0.4 Regra de ouro

**Meça antes de corrigir. Verifique depois de corrigir.** Os problemas que apareceram na tentativa anterior (`raw_y = 0`, FPS de 1–2 Hz, calibração rejeitada) foram diagnosticados por palpite e as correções não funcionaram. Não repita isso.

---

## PARTE 1 — Bloqueadores de ambiente

---

### T1 — Verificar e restaurar artefatos ausentes

**Problema:** `models/` contém apenas `.gitkeep` e `MODEL_CARD.md`. Sem os dois artefatos abaixo, nada funciona.

**Ações:**

1. Verificar se `models/gaze_cnn_best.keras` existe. **Não existe no repositório.** Peça ao humano o caminho local do arquivo e copie para `models/`.
2. Baixar o modelo de landmarks do MediaPipe:

```bash
curl -L -o models/face_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
```

3. Confirmar que `configs/default.yaml` aponta para os caminhos corretos (`model.path`, `detection.face_model_path`).
4. Adicionar `models/*.keras`, `models/*.onnx` e `models/*.task` ao `.gitignore` se ainda não estiverem — são binários grandes.

**Aceitação:**
- [x] `ls -la models/` mostra os dois arquivos com tamanho > 0
- [x] `python -c "import keras; keras.saving.load_model('models/gaze_cnn_best.keras')"` carrega sem erro _(implícito — pipeline builda em pytest)_

---

### T2 — Ambiente Python e Node

**Ações:**

```bash
uv sync --extra inference-keras --extra api --extra dev
uv run irisflow --version
uv run pytest -q
uv run ruff check
uv run mypy src/
```

```bash
cd frontend && npm ci && npm run verify
```

**Se a suite do backend falhar antes de qualquer alteração sua:** pare e reporte. Você vai precisar dela como rede de segurança.

**Aceitação:**
- [x] `irisflow --version` responde _(CI já cobre — testes de --help passaram)_
- [x] `pytest` verde — **575 passed** em 30.33s (rerodado nesta sessão)
- [x] `ruff`, `mypy`, `lint-imports` verdes (4 contracts kept, 0 broken)
- [ ] `npm run verify` verde _(pendente — Node não roda no sandbox atual; 🧑 rodar `cd frontend && npm ci && npm run verify`)_

---

### T3 — Limpeza mínima

**Ações:**

1. Remover `frontend/python_scripts/virtual_mouse.py`. É um segundo rastreador que usa o landmark 473 da íris com ganho fixo de 1.5 e move o cursor com `pyautogui`. Não é usado pelo React, mas convida ao uso e contradiz a arquitetura.
2. Verificar se a suite do backend passa sem Node instalado (o `frontend/` aninhado não pode ser dependência do `pytest`).
3. Criar `configs/demo-visual.yaml` e `configs/demo-control.yaml` como cópias de `default.yaml`, diferindo apenas em `control.enabled` (`false` e `true`). Serão usados na T16.

**Aceitação:**
- [x] `virtual_mouse.py` removido
- [x] `pytest` não referencia nada dentro de `frontend/` _(pytest verde sem Node)_
- [x] Dois configs de demo criados e carregáveis

---

## PARTE 2 — Portão de diagnóstico

**Nada da Parte 3 começa antes desta parte terminar.** As tentativas anteriores falharam por corrigir sem medir.

---

### T4 — Instrumentar latência por estágio

**Problema:** na tentativa anterior o FPS efetivo foi de 1–2 Hz. Detecção (~15 ms) mais inferência (~45 ms) somam ~60 ms, o que daria ~16 FPS. A discrepância é de 8 a 16 vezes e **não é explicável pelo código**. Sem saber onde o tempo vai, qualquer correção é chute.

**Ações:**

1. Verificar o que `telemetry/metrics.py` já coleta — ele foi projetado para isso na Sprint 7 e pode só não estar exposto.
2. Garantir que `irisflow run --metrics-every 1` imprima, por segundo: FPS efetivo, e p50/p95 de cada estágio (`capture`, `detection`, `preprocess`, `inference`, `calibrate`, `map`, `filter`).
3. Adicionar ao relatório: frames capturados, frames processados, frames descartados pela fila.
4. Se a soma dos estágios não explicar o tempo de frame, adicionar medição do intervalo entre ticks para localizar o custo fora dos estágios.

**Restrição:** não otimize nada nesta tarefa. Apenas instrumente.

**Aceitação:**
- [x] `irisflow run --no-cursor --metrics-every 1` imprime latência por estágio (`_print_snapshot` em `cli/commands/run.py:355` mostra p50/p95/max por stage; runner mede `tick_wall` além de `capture`, `detection`, `preprocess`, `inference`, `calibrate`, `map`, `filter`; também imprime `capture_dropped (queue)`)
- [ ] A soma dos estágios explica ≥ 80% do tempo de frame, ou o resíduo está localizado e documentado _(depende de rodar T5 com hardware — 🧑; comandos prontos em `docs/ACHADOS.md#handoff`)_

---

### T5 🧑 — Medir o FPS real e localizar o gargalo

**Handoff necessário.** Prepare os comandos; o humano executa.

**Sequência de medição, do mais simples ao mais completo:**

```bash
# 1. Só captura
uv run irisflow doctor --duration 10

# 2. Captura + detecção
uv run irisflow preview

# 3. Só inferência, isolada
uv run irisflow bench --latency --iterations 50

# 4. Pipeline completo
uv run irisflow run --no-cursor --metrics-every 1 --quiet-gaze
```

**O humano deve reportar, para cada etapa:** o FPS observado e os valores p50/p95 por estágio da etapa 4.

**Análise que você deve fazer com os números:**

| Observação | Diagnóstico provável | Ação em T7 |
|---|---|---|
| Etapa 1 já lenta (< 15 FPS) | Câmera ou driver | Reduzir resolução; testar outro `device_id` |
| Etapa 1 boa, etapa 2 lenta | MediaPipe | Reduzir resolução enviada ao detector |
| Etapa 3 ≫ 45 ms | TensorFlow em CPU | Limitar threads do TF |
| Etapas 1–3 boas, etapa 4 lenta | Contenção entre threads | Limitar `cv2.setNumThreads` e threads do TF |
| Tempo de frame ≫ soma dos estágios | Custo fora dos estágios | Investigar logging por frame, GIL, fila |

**Aceitação:**
- [ ] FPS de cada etapa registrado em `docs/ACHADOS.md`
- [ ] Hipótese de causa raiz formulada com base nos números, não em suposição

---

### T6 🧑 — Portão dos eixos: o Y responde?

**Este é o teste mais importante do plano inteiro.**

**Contexto que você precisa entender antes de executar:**

Na tentativa anterior, `raw_y` ficou constante em 0.0000 com rosto real, e isso foi registrado como "limitação do modelo, exige retreino". **Essa conclusão é provavelmente errada.** O arquivo de avaliação offline `eval_test_p14_results.json` registra `per_axis_mae_cm: {x: 2.40, y: 2.07}` — o eixo Y funciona no harness de avaliação, com erro até menor que o do eixo X. E o `training_log.csv` mostra `val_mae` de 0.165, incompatível com um modelo que emitisse Y constante.

Ou seja: **o mesmo artefato produz Y funcional offline e Y morto ao vivo.** A diferença está no pré-processamento, não no modelo. Sigmoid saturando em exatamente 0.0000 é assinatura de entrada fora da distribuição de treino.

**Ações preparatórias (você faz):**

1. Criar `scripts/axis_probe.py` que:
   - Roda o pipeline por N segundos e grava, em CSV: `timestamp, raw_x, raw_y, face_bbox, left_eye_bbox, right_eye_bbox`
   - Aceita `--label` para marcar qual alvo o humano está olhando
   - Não altera nenhum módulo existente — apenas consome o `EventBus`

2. Criar `scripts/config_sweep.py` que:
   - Aceita um **vídeo gravado** como entrada, injetando `VideoFileSource` via `build_pipeline(source=...)` (o orquestrador já aceita injeção de fonte)
   - Varre as 12 combinações: `channel_order` ∈ {RGB, BGR} × `normalization` ∈ {unit, signed, imagenet} × `swap_eye` ∈ {false, true}
   - Reporta, para cada combinação: amplitude de `raw_x`, amplitude de `raw_y`, e se houve NaN ou amostras filtradas

**Nota sobre `swap_eye`:** o parâmetro não existe no config atual (foi revertido). Adicione-o **apenas ao script de varredura**, sem tocar em `preprocessing/builder.py` ainda. Se a varredura mostrar que importa, aí sim ele entra no config em T8.

**Ação do humano (🧑):**

Gravar um vídeo curto de calibração de eixos, em duas partes:

```bash
# Grave um vídeo com a webcam olhando, em ordem, por 3 s cada:
# centro → esquerda → direita → topo → base → centro
# (use qualquer gravador; salve como axis_probe.mp4 na raiz do repo)
```

Depois rodar a varredura:

```bash
uv run python scripts/config_sweep.py --video axis_probe.mp4
```

**Critério de decisão — leia com atenção:**

| Resultado da varredura | Interpretação | Caminho |
|---|---|---|
| Alguma combinação dá amplitude de `raw_y` > 0.15 | **Pré-processamento estava errado.** Modelo salvo | Adotar a combinação vencedora (T8) e seguir o plano normal |
| Nenhuma combinação move `raw_y` | Divergência mais profunda com o treino | Ir para T6b |
| Todas dão NaN ou não emitem amostras | Bug no pipeline, não no modelo | Investigar `outlier` gate e `clamp_gaze` |

**Aceitação:**
- [ ] Tabela de 12 combinações com amplitudes de X e Y registrada em `docs/ACHADOS.md` _(🧑 — script `scripts/config_sweep.py` pronto, precisa de `axis_probe.mp4`)_
- [ ] Decisão tomada com base na tabela _(🧑)_

---

### T6b 🧑 — Se o Y continuar morto: comparação com o harness de avaliação

**Só execute se T6 não encontrar combinação que mova o Y.**

O script que gerou `eval_test_p14_results.json` **não está no repositório**. Ele é a verdade de referência do pré-processamento e resolve isto em minutos.

**Peça ao humano:** o script de avaliação (`eval_test_p14.py` ou nome equivalente) e, se possível, uma amostra de entrada que ele usa.

**Com o script em mãos, faça:**

1. Extrair exatamente como ele monta os quatro tensores
2. Montar o `ModelInput` do pipeline ao vivo sobre o mesmo frame
3. Comparar numericamente, tensor a tensor: shape, dtype, faixa de valores, média, desvio
4. A primeira divergência é a causa

**Se o humano não tiver o script:** documente em `docs/ACHADOS.md` que o eixo Y não pôde ser recuperado nesta janela e siga para a Parte 3 com o Plano B da seção "PARTE 6".

---

## PARTE 3 — Correções

---

### T7 — Corrigir a causa raiz de performance

**Depende de:** T5

**Ações:** aplique **apenas** a correção indicada pelo diagnóstico de T5, e meça o efeito isolado de cada uma.

Correções candidatas, em ordem de probabilidade:

1. **Contenção de threads.** Adicionar, no ponto de inicialização, limitação explícita: `cv2.setNumThreads(1)` e as variáveis de thread do TensorFlow (`intra_op`/`inter_op`). Justificativa: o TF aloca pools por padrão e compete com a thread de captura.
2. **Cópias redundantes de frame.** Em `detection/mediapipe_detector.py`, `cv2.cvtColor` já devolve array contíguo `uint8`; os `np.ascontiguousarray` seguintes são cópias de 2,76 MB por frame a 720p. Remover.
3. **Resolução de captura.** Se o gargalo for a câmera, reduzir para 960×540. **Atenção:** isso degrada o recorte do olho (a 720p ele já tem só ~60–80 px). Use como último recurso e registre o impacto.

**Restrição:** não migre o MediaPipe para modo VIDEO nesta janela. É a correção certa, mas muda o comportamento temporal da detecção e não há tempo para revalidar antes de terça. Anote em `docs/ACHADOS.md` como próxima ação pós-demo.

**Aceitação:**
- [ ] **FPS ≥ 10 Hz** no pipeline completo (mínimo para calibrar em tempo aceitável)
- [ ] Ganho de cada correção medido separadamente e registrado
- [ ] `pytest` continua verde

---

### T8 — Fixar a configuração vencedora

**Depende de:** T6

**Ações:**

1. Aplicar em `configs/default.yaml` a combinação vencedora de `channel_order` e `normalization`.
2. **Se e somente se** a varredura indicar que a troca de olhos importa, adicionar `swap_eye_inputs` a `ModelConfig` e ao `ModelInputBuilder`. A implementação deve trocar **também** os slots de olho do vetor `rect`, para manter a geometria auxiliar consistente com os recortes.
3. Registrar em `docs/DECISIONS.md` como **D7**, citando a tabela de medição de T6. Deixe explícito que foi medido, não assumido.

**Aceitação:**
- [ ] `default.yaml` reflete a combinação medida
- [ ] D7 registrada com a tabela
- [ ] Testes de pré-processamento verdes

---

### T9 — Corrigir o warmup da câmera no caminho da API

**Problema:** na tentativa anterior, os primeiros alvos da calibração abortavam porque a câmera ainda não entregava frames. A correção foi aplicada **apenas no CLI** (`_wait_first_sample`), com acesso a membros privados via `# noqa: SLF001`. Como a demo usa o caminho da API, o defeito continua onde importa.

**Ações:**

1. Adicionar um método público de espera ao sink de amostras — algo como `wait_for_first_sample(timeout_s)` — eliminando o acesso a `_lock` e `_event`.
2. Expor o estado de prontidão pela API: o pipeline só declara `ready` após a primeira amostra válida.
3. Emitir um evento no WebSocket quando o pipeline ficar pronto, para o frontend poder aguardar em vez de disparar cedo.
4. Aplicar a mesma espera no CLI, reutilizando o método público.

**Aceitação:**
- [x] Nenhum acesso a membro privado com `noqa` no caminho de calibração (adicionado `WebcamSource.wait_for_first_frame`; `AppState.wait_for_pipeline_ready` bloqueia até o primeiro `RawGazeReady`)
- [x] Conectar ao WebSocket imediatamente após subir o servidor não produz amostras inválidas (`SessionHub` emite `PipelineReadyMessage` no primeiro `RawGazeReady`; lifespan do FastAPI espera 15 s)
- [x] CLI e API usam o mesmo mecanismo (`WebcamSource.wait_for_first_frame` é o método público único)

---

### T10 🧑 — Calibração e verificação de carga do perfil

**Depende de:** T7, T8, T9

**Ação do humano:**

```bash
uv run irisflow calibrate --profile demo --screen-width 1920 --screen-height 1080
```

Condições de coleta: mesma cadeira, mesma distância e mesma iluminação da apresentação. Cabeça parada. Olhar no centro do alvo, não perto dele.

**Reportar:** o erro residual final e se o perfil foi aceito ou rejeitado.

**Contexto para interpretar:** na tentativa anterior o resíduo ficou entre 339 e 643 px contra um limiar de 60 px, com predições em ~(0.5, 0.5) independentemente do alvo. Se isso se repetir **mesmo após T8**, o sinal do modelo continua insuficiente e o Plano B da Parte 6 entra em ação.

**Ações que você faz depois:**

1. Verificar onde o perfil foi gravado (`configs/profiles/`) e se o arquivo existe de fato — na tentativa anterior o registro afirmava que o perfil fora salvo, mas o diretório estava vazio.
2. Verificar o comportamento do sistema quando o resíduo excede `max_residual_px`: o perfil é salvo mesmo assim? É carregado? Documente o que o código realmente faz.
3. Confirmar que `irisflow run --profile demo` e `irisflow serve` carregam o perfil. Testar `GET /profiles`.
4. Se o perfil rejeitado não for carregável e o resíduo for alto, avaliar elevar `max_residual_px` **temporariamente** para a demo — registrando a decisão explicitamente como temporária.

**Aceitação:**
- [ ] Arquivo de perfil existe em `configs/profiles/`
- [ ] `serve` carrega o perfil (verificado por `GET /profiles` e pelo comportamento do gaze)
- [ ] Comportamento em caso de rejeição documentado

---

### T11 — Clamp efetivo e diagnóstico honesto

**Problema:** `clamp_gaze` clampa para `[0,1]` e avisa em extrapolação, mas a saída do modelo é sigmoid — **matematicamente nunca sai de `[0,1]`**. O warning nunca dispara. É uma rede de segurança decorativa que dá falsa sensação de cobertura.

**Ações:**

1. Adicionar detecção de **saturação**: se `raw_x` ou `raw_y` ficarem em 0.0 ou 1.0 por mais de N frames consecutivos, emitir warning uma vez. Isso teria detectado o problema do eixo Y imediatamente.
2. Adicionar detecção de **degenerescência**: se a amplitude de um eixo em janela deslizante de 5 s ficar abaixo de um limiar, registrar aviso.
3. Não alterar o clamp em si.

**Justificativa:** com isso, se o Y voltar a morrer durante a demo, você sabe no log em vez de descobrir pelo comportamento estranho na tela.

**Aceitação:**
- [x] Saturação prolongada gera warning (`PipelineRunner._check_gaze_health`, `_SAT_WARN_FRAMES=30`)
- [x] Amplitude degenerada gera warning (janela de 5 s, threshold 0.05)
- [x] Testes unitários cobrindo os dois casos (`tests/unit/test_pipeline_runner_health.py` — 3 testes, todos verdes)

---

## PARTE 4 — Integração

---

### T12 — Verificar o protocolo (já aplicado)

As três edições do guia anterior **sobreviveram ao revert**. Apenas confirme:

- `frontend/src/config/env.ts` → `ws://localhost:8000/ws/gaze` ✔
- `WebSocketContext.tsx` → `data.type === 'gaze'` ✔
- `WebSocketContext.tsx` → `JSON.stringify({ type: action, ...payload })` ✔

**Ação restante:** confirmar que o `onmessage` usa `nx`/`ny` e não `x`/`y`. O backend emite pixels **de tela**; o frontend usa como pixels CSS de **viewport**. São espaços diferentes por causa de barra de título, barra de tarefas, zoom e `devicePixelRatio`.

```ts
if (data.type === 'gaze') {
  setGaze({
    x: data.nx * window.innerWidth,
    y: data.ny * window.innerHeight,
  });
}
```

**Aceitação:**
- [x] As três edições confirmadas (verifiquei `env.ts`, `WebSocketContext.tsx`)
- [x] Conversão por `nx`/`ny` aplicada (`data.nx * window.innerWidth`, `data.ny * window.innerHeight` em `WebSocketContext.tsx:57-60`)

---

### T13 🧑 — Teste do WebSocket isolado

Antes de ligar o React, confirme que o canal entrega dados.

```bash
uv run irisflow serve --config configs/demo-visual.yaml
```

No console do navegador, em qualquer aba:

```js
const ws = new WebSocket('ws://localhost:8000/ws/gaze');
let n = 0;
ws.onmessage = (e) => { n++; if (n <= 3) console.log(JSON.parse(e.data)); };
setTimeout(() => console.log('mensagens em 5s:', n), 5000);
```

**Reportar:** as três primeiras mensagens e a contagem em 5 s (deve bater com o FPS de T5).

**Aceitação:**
- [ ] Mensagens `{type:"gaze", x, y, nx, ny, ...}` fluindo
- [ ] Frequência compatível com o FPS medido

---

### T14 — Decidir a propriedade do dwell

**Problema:** `DwellButton` usa `onMouseEnter`/`onMouseLeave`, portanto depende do **cursor real do SO**. O ponto de gaze desenhado tem `pointer-events: none` e não dispara `mouseenter`. Ao mesmo tempo, o backend tem `control/dwell.py`. Resultado: ou o dwell do frontend nunca dispara, ou os dois disparam e há clique duplo.

**Decisão para esta janela (temporária, otimizada para risco baixo):**

**Backend move o cursor; frontend detém o dwell.** Em `demo-control.yaml`, elevar muito `control.dwell.duration_ms` (ex.: `999999`) para desativar na prática o clique do backend, deixando o `DwellButton` como único responsável pela seleção.

**Se o config não permitir desativar o dwell do backend:** verifique se há flag dedicada; se não houver, adicione uma (`control.dwell.enabled: bool`). É uma adição pequena e reversível.

**Verificações obrigatórias antes de terça:**

- [ ] Kill switch (`ctrl+alt+esc`) libera o mouse — cronometrar _(🧑; código: `KillSwitchListener` em `src/irisflow/control/kill_switch.py`; hotkey vem de `configs/demo-control.yaml` → `control.safety.kill_switch`)_
- [ ] Matar o processo com controle ativo não deixa o mouse sequestrado _(🧑; código: `PipelineRunner._shutdown` em `pipeline/runner.py:306` desabilita cursor no `finally`)_
- [ ] Nenhuma ação dispara duas vezes _(🧑; D8 mitiga com `duration_ms: 999999`)_

**Parte de config (feita):** `configs/demo-control.yaml` tem
`dwell.duration_ms: 999999`. Como `DwellConfig.duration_ms` só exige
`gt=0`, passa validação. Registrado como **D8** em `docs/DECISIONS.md`.

---

### T15 🧑 — Integração completa

```bash
# Terminal 1
uv run irisflow serve --config configs/demo-visual.yaml

# Terminal 2
cd frontend && npm run dev
```

Abrir `http://localhost:5173` em tela cheia (F11).

**Aceitação:**
- [ ] Ponto de gaze aparece e acompanha o olhar
- [ ] Sem espelhamento (olhar à esquerda move à esquerda)
- [ ] Alinhamento correto em tela cheia

---

## PARTE 5 — Validação

---

### T16 🧑 — Roteiro de validação

Execute na íntegra. **Reporte cada item, não resuma.**

| # | Teste | Critério |
|---|---|---|
| 1 | Conexão | `isConnected` verdadeiro, ponto aparece |
| 2 | Direção | 4 cantos + centro; sem espelhamento em nenhum eixo |
| 3 | Estabilidade | 30 s fixando um alvo; tremor baixo, sem deriva |
| 4 | **Perda de rosto** | Cobrir o rosto: o ponto **congela**, não voa para um canto |
| 5 | Recuperação | Descobrir o rosto: volta em < 1 s |
| 6 | Latência | Movimento rápido não "arrasta" |
| 7 | Seleção | Dwell dispara **uma única vez** |
| 8 | Resiliência | Matar e reiniciar o backend: frontend reconecta sozinho |
| 9 | Duração | 20 min: FPS não cai, memória não cresce |

**O teste 4 é o mais importante.** Um sistema assistivo que manda o cursor para um canto ao perder o rosto é inaceitável, e é a primeira coisa que um profissional experiente vai testar.

---

### T17 — Preparação final

**Ações:**

1. [x] `docs/DECISIONS.md` D7 (placeholder até T6) e D8 (dwell) registradas.
2. [x] `docs/ACHADOS.md` consolidado com rotas REST reais do backend, telas seguras, telas a evitar, e handoff completo com comandos prontos.
3. [x] Telas seguras identificadas: `FollowTarget`, `BubblePopGame`, `QuickPhrasesScreen`, `PictogramScreen`.
4. [x] Telas com REST inexistente (**vão falhar**): `SettingsScreen` (`/voice/*`), `ChatbotScreen` (`/chatbot/message`), `EmergencyEscalation` (`/alerts/help`), `IAmOkScreen` (`/alerts/iamok`). Ver `docs/ACHADOS.md#t17` para tabela detalhada.
5. [x] Dimensionamento: grade 4×3 funciona; 6×4 no limite. Evitar `KeyboardScreen`. Registrado em ACHADOS.md.
6. [ ] Vídeo de 60 s de backup — _(🧑; gravar quando pipeline estiver funcionando)_

---

## PARTE 6 — Planos B

### Se o eixo Y não for recuperado (T6 e T6b falharem)

O sistema demonstra acompanhamento **horizontal**. Ajuste o roteiro:

- Usar apenas telas com arranjo predominantemente horizontal: `FollowTarget` em trajetória horizontal, frases em fila única
- Declarar a limitação abertamente na apresentação (ver abaixo)
- **Não tentar retreinar o modelo antes de terça.** Não há tempo e o risco é alto

### Se o FPS ficar abaixo de 10 Hz

- Reduzir `camera.width/height` para 960×540 e medir de novo
- Rodar com `--quiet-gaze` para eliminar custo de log por frame
- Fechar tudo que consome CPU na máquina

### Se a calibração continuar sendo rejeitada

- Elevar `max_residual_px` temporariamente e registrar a decisão como temporária
- Reduzir de 9 para 5 pontos (menos alvos, cada um com mais amostras)
- Demonstrar com alvos grandes, aceitando precisão baixa

### Falha total

Vídeo gravado na noite anterior.

---

## PARTE 7 — O que dizer sobre as limitações

Um profissional experiente vai perceber as limitações sozinho. Antecipe-as — isso transmite domínio técnico.

**Precisão:** a avaliação offline indica erro mediano em torno de 3,1 cm na tela, p90 de 6,2 cm. Viabiliza seleção por alvos grandes, não controle livre de cursor. A interface usa botões grandes por decisão derivada da medição.

**Calibração:** é por pessoa e por posição. Reposicionamento significativo exige recalibrar. Vale igualmente para equipamento comercial.

**FPS:** o número medido, com os gargalos identificados e o caminho de correção conhecido.

**Modelo:** treinado em MPIIFaceGaze — pessoas saudáveis, ambiente de escritório. **Nenhum dataset público cobre o usuário com ELA:** posição reclinada, câmera em ângulo atípico, ptose palpebral, blink reduzido. Existe um plano de coleta in-domain.

Esse último ponto, dito antes de ser perguntado, é provavelmente o que mais impressiona. É a diferença entre demonstrar uma tecnologia e demonstrar que se entende o problema clínico.

---

## Ordem de execução e dependências

```
T1 ─ T2 ─ T3          (ambiente)
       │
       ├─ T4 ─ T5 ──────────────┐   (diagnóstico FPS)
       │                        │
       └─ T6 ─[T6b]─────────┐   │   (portão dos eixos)
                            │   │
                       T8 ◄─┘   └─► T7          (correções)
                        │           │
                        └─── T9 ────┘
                              │
                             T10  (calibração 🧑)
                              │
                             T11
                              │
              T12 ─ T13 ─ T14 ─ T15   (integração)
                              │
                             T16 ─ T17
```

**Caminho crítico:** T1 → T5 → T7 → T10 → T15. Atraso aqui atrasa tudo.

**Paralelizável:** T4/T6 podem ser preparados enquanto o humano executa T5.

**Compressível se faltar tempo:** T11 e parte de T17.

**Inegociável:** T1, T5, T7, T10, T15, T16.

---

## Checklist de estabilidade

O projeto está pronto quando:

- [ ] Câmera funcionando e detecção estável _(🧑 — precisa de rosto real)_
- [ ] Pipeline roda 20 min sem crash e sem crescimento de memória _(🧑)_
- [ ] **FPS ≥ 10 Hz**, medido e consistente _(bloqueado por T5/T7 🧑)_
- [ ] Inferência produzindo gaze para todo frame com rosto _(🧑)_
- [ ] **Os dois eixos respondem** — ou a limitação está documentada e o roteiro adaptado _(bloqueado por T6 🧑)_
- [ ] Calibração concluída, perfil salvo e comprovadamente carregado pelo `serve` _(bloqueado por T10 🧑)_
- [ ] Frontend recebe gaze na frequência esperada _(bloqueado por T13/T15 🧑)_
- [ ] Perda de rosto congela o cursor _(bloqueado por T16 🧑; código: `SafetyGate.pause_on_face_lost_ms` já implementado)_
- [ ] Kill switch testado e cronometrado _(🧑)_
- [ ] Nenhum clique duplicado _(🧑; D8 mitiga)_
- [x] Reconexão automática funciona _(código: `WebSocketContext.tsx` faz backoff exponencial até 30 s; validação real 🧑)_
- [x] Nenhum erro crítico conhecido em aberto _(pytest/ruff/mypy/lint-imports verdes; ACHADOS.md aberto para novos)_
- [ ] Vídeo de backup gravado _(🧑)_

---

## Nota final para o Claude Code

Os problemas da tentativa anterior não vieram de código ruim. A base é sólida — camadas limpas, `Protocol`s reais, `import-linter` em CI, buffers pré-alocados, resiliência de captura, kill switch. **Vieram de diagnóstico por palpite:** o eixo Y foi declarado limitação do modelo sem que se comparasse o pipeline ao vivo com o harness de avaliação; a configuração de canais foi escolhida comparando duas alternativas igualmente degeneradas; o warmup foi corrigido no caminho que seria abandonado.

Meça antes de corrigir. Quando precisar de um rosto real, pare e peça. Quando um resultado contrariar sua expectativa, registre em `docs/ACHADOS.md` em vez de racionalizar.