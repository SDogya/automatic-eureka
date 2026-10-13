# Независимый аудит TraFL → Qwen3 BD3LM

Дата: 2026-10-13. Объект: локальный `TraFL_bd3lm`, соответствующий серверной ветке `bd3lm`, исходный коммит `c6d57b1`. Аудит статический: код не изменялся; тесты, модели, smoke, training и evaluation не запускались.

## Вердикт

**CORE MATCH / BD3LM ADAPTATION UNVALIDATED / CHECKPOINT COMPATIBILITY REJECT.**

Алгоритмическое ядро соответствует опубликованному practical TraFL. В текущих defaults не найдено подтверждённого нарушения формулы residual, порядка усреднений, нормировки loss, centering или числа optimizer updates. Замена LLaDA на BD3LM разрешена пользователем и сама по себе не является ошибкой.

Однако это **не подтверждённая экспериментальная репликация статьи**. BD3LM меняет условные распределения и доступный контекст; сохранение текстовой формулы Eq. (4) не доказывает пригодность нового surrogate. Найдены два опасных пути несовместимой загрузки checkpoints и один подтверждённый дефект маски для другого допустимого attention backend. Старые результаты MDLM не подтверждают работоспособность BD3LM.

Процент «соответствия статье» не приводится: ошибки загрузки, математический score и отсутствующие эксперименты нельзя честно свернуть в одну такую величину.

## Источники и границы доказательства

- Самостоятельно прочитан [оригинальный PDF](docs/32236_Beyond_Mode_Seeking_RL_T.pdf): основная часть, Algorithm 1, derivation Appendix C, теория Appendix D и экспериментальные приложения. Страницы далее — печатная нумерация PDF, начиная с 1.
- Прочитаны все исполняемые модули `trafl/`, `run.py`, `config.yaml`, README, supplied tests и adaptation metadata.
- Проверены первичные [BD3LM model card, pinned revision](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1/blob/60f77bba67ba94231c16f5fb29bc66c500b038f1/README.md) и [custom model code](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1/blob/60f77bba67ba94231c16f5fb29bc66c500b038f1/modeling_qwen3.py).
- Custom model code, config/tokenizer metadata и Transformers SDPA implementation прочитаны из cached серверных файлов без выполнения; identity snapshot фиксирован revision `60f77bba67ba94231c16f5fb29bc66c500b038f1`.
- Для backend-specific дефекта отдельно проверен [Qwen3 implementation Transformers 4.57.0](https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/qwen3/modeling_qwen3.py).
- Исторические APPROVED и 24 пройденных теста MDLM не принимались как доказательство корректности новой ветки.

Классы замечаний: **BUG** — конкретный программный дефект; **ADAPTATION** — разрешённая замена модели с научной границей; **OWN/GAP** — выбранное правило при неполной спецификации; **SCOPE** — отсутствующая часть экспериментальной репликации; **UNVERIFIED** — необходимое утверждение пока не проверено. Severity отражает последствия конкретного механизма, а не факт отличия от LLaDA.

## Что действительно совпадает

| Требование | Место | Результат статического чтения |
|---|---|---|
| Fresh current-policy rollout, G=5 | `training.py:82–98`, YAML `41` | В каждом update новые outputs; optimizer не обновляется между prompts одного batch. |
| Binary terminal reward, centering только внутри prompt group | `training.py:94–98`, `rewards.py:39–67` | Default reward 0/1, без деления на std или удаления all-equal groups. |
| Uniform nonempty mask level + uniform subset | `masking.py:24–32` | Domain — eligible completion positions; prompt не маскируется. |
| K=4, shared policy/reference masks | YAML `42`, `training.py:106–129` | Одни и те же corruption tensors передаются обоим моделям. |
| Mean log-probability по masked positions | `scoring.py:21` | FP32 log-softmax, original target IDs, divisor — количество выбранных positions. |
| Mean K до square residual | `scoring.py:64`, `training.py:130–135` | Нет ошибочного среднего квадратов четырёх отдельных residual. |
| Residual и mean по actual B·G | `objective.py:37–39`, `training.py:79,134` | `policy − reference − beta·centered_reward + log_z`; sequential backward с правильным divisor. |
| Один joint optimizer update | `training.py:77,155`, `runner.py:242–254` | Joint policy/head optimizer, один step после полного batch. |
| Frozen reference | `runner.py:110–114`, `training.py:119` | Отдельная base model, requires_grad=False, no_grad score. |
| LoRA rank/alpha 128/128 | YAML `26–27`, `runner.py:116–123` | Опубликованные значения сохранены. |
| Двухслойная mean-pooling head, detached features | `model.py:26–46,66–74` | Нет gradients головы к backbone через feature input; output — один scalar на prompt. |
| Cosine peak/min LR | YAML `61–62`, `objective.py:79–82` | 6e-5 / 6e-6; точная фаза — OWN default. |
| Math prompt Appendix E, no system message | YAML `93–103`, `data.py:63–69` | User template совпадает; Qwen serialization — разрешённая адаптация. |
| Published evaluation lengths/top-p/block-size и Pass@k | YAML `76–82`, `evaluation.py:28–30` | 256/512, .91, 32, n16; estimator эквивалентен статье. |

Опора: §3.2 / Eq. (4), с. 4–5; §4.1–4.2, с. 5–6; Algorithm 1, с. 14; Appendix E, с. 21–22. Эта таблица подтверждает структуру вычислений, не результаты runtime.

## Подтверждённые программные проблемы

### B01 — HIGH / BUG: resume MDLM checkpoint незаметно меняет attention protocol

**Место:** `trafl/runner.py:59–68,99–114,124–145,267–276`; unconditional staircase в `scoring.py:47–61`, `sampling.py:87–100`, `model.py:66–71`; schema `checkpoint.py:95–114`.

**Механизм:** full resume берёт `config` старого checkpoint и восстанавливает его `model_id`/revision. Но текущие functions всегда создают staircase attention. У MDLM checkpoint, созданного предыдущей веткой, backbone был bidirectional. MDLM и BD3LM имеют совместимые размеры LoRA/head, поэтому tensor shape сам по себе не обнаруживает ошибку: загрузятся прежние moments, weights и RNG, а последующие rollout/score/features будут вычисляться с другим attention. Это не точное продолжение ни прежнего MDLM run, ни нового BD3LM run. Saved config не содержит обязательной версии attention protocol/реализации.

**Опора PDF:** frozen-reference и current-policy terms в §3.2, Eq. (1)–(4); Algorithm 1, с. 14. Resume не опубликован, поэтому это BUG нашей дополнительной функции, а не нарушение раскрытого авторского resume.

**Исправление:** сохранять и проверять model ID/revision, diffusion/attention protocol и версию training schema; BD3LM branch должна явно отвергать incompatible MDLM resume или dispatch на исходную MDLM реализацию. README `60–62` предупреждает об этом, но не предотвращает фактическую ошибочную загрузку.

### B02 — HIGH / BUG: evaluation может загрузить adapter на неправильные base weights или с неправильным LoRA scaling

**Место:** `trafl/runner.py:99–145` и `146–150`.

**Механизм:** при `--mode eval` base, reference, tokenizer, `LoraConfig` и head создаются из текущего requested YAML. Если найден `training_state.pt`, берутся только policy/head/step; сохранённый `model_state['config']` не участвует в выборе base и LoRA. Старый MDLM checkpoint может попасть на BD3LM base без shape mismatch. Даже BD3LM checkpoint можно оценить с другим `lora_alpha`, тем же rank и теми же tensor shapes: загрузка удастся, но эффективная policy изменится. Несовпадение block size также меняет geometry. Legacy export path аналогично использует заранее загруженную base из текущего YAML; adapter metadata не служит проверкой её identity.

**Опора PDF:** §4.2, с. 6, требует оценки trained policy под согласованным protocol; reference anchoring §3.2 привязан к конкретной модели. Загрузка checkpoint — собственный engineering path.

**Исправление:** читать checkpoint identity до создания model; использовать сохранённые base revision / LoRA / head / attention settings, разрешая отдельно явные evaluation overrides. Несовместимые checkpoints отвергать. Default YAML + собственный BD3LM checkpoint совпадают, поэтому это дефект конкретных допустимых CLI combinations, а не утверждение, что любой default eval уже неправильный.

### B03 — HIGH, conditional / BUG: `attention_implementation: eager` разрушает staircase

**Место:** `trafl/attention.py:26–29`, YAML `12`, `runner.py:104–109`; custom model forward передаёт 4D mask напрямую.

**Механизм:** helper возвращает boolean 4D mask. В default SDPA boolean True/False корректно означает allowed/blocked. В pinned Transformers Qwen3 eager path выполняется `attn_weights + causal_mask`: bool превращается в +1/+0, а не 0/−∞. Будущие блоки получают ненулевую attention mass; появляется доступ к запрещённому контексту в rollout, score и features. Это подтверждено первичным [Transformers 4.57.0 code](https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/qwen3/modeling_qwen3.py#L122-L143). Custom backbone не конвертирует уже 4D mask.

**Опора PDF:** разрешённая модельная адаптация, не требование оригинального LLaDA; нарушается собственное заявленное BD3LM staircase condition. Для Eq. (4) меняется conditional model.

**Исправление:** зафиксировать поддерживаемый backend как SDPA и явно отвергать остальные, либо давать backend-compatible additive mask 0/−∞. Поддержку FlashAttention/flex нельзя объявлять на основании этого bool helper. **В текущем default `sdpa` этот дефект не активен.**

## Научные границы BD3LM-адаптации

### A01 — HIGH / ADAPTATION: global corruption ≠ block-diffusion clean-prefix likelihood

**Место:** `masking.py:24–38`, `scoring.py:47–63`, `sampling.py:89–114`, YAML `43–48,64–80`.

**Механизм:** corruption выбирается по всей completion; для target в позднем блоке предыдущие completion blocks могут тоже содержать [mask]. Во время decoder generation предыдущие блоки уже clean. Например, при target во втором блоке score может условиться на corrupted первый блок, хотя decoder не приходит в такое состояние при генерации второго блока. Early-block targets не видят clean later blocks, потому что staircase запрещает такой доступ. Поэтому доступный контекст отличается и от fully bidirectional LLaDA experiment, и от blockwise clean-prefix objective новой модели.

**Опора PDF:** Eq. (4), с. 5; Appendix C, с. 15–16, определяет uniform subset **всей completion**. Указанное global masking действительно сохраняет этот q_l и reductions. **Это не подтверждённое нарушение Eq. (4)**. Менять masking на «только текущий блок, предыдущие clean» без новой формулировки значило бы изменить опубликованный q_l.

**Граница:** это TraFL surrogate, вычисленный другой conditional model. Нельзя называть его BD3LM exact likelihood или blockwise ELBO и нельзя считать, что качество policy-reference difference автоматически сохранилось. Нужна отдельная проверка пригодности; возможный blockwise вариант должен называться модификацией TraFL, иметь собственную формулу weighting и сравнение с исходным global-mask baseline.

### A02 — MEDIUM / ADAPTATION: сохранена uniform-reveal интерпретация, но не вероятность block sampler

**Место:** `scoring.py:25–64`, `sampling.py:89–114`, `objective.py:21–39`.

**Опора PDF:** Appendix C, Eq. (6), с. 15–16, прямо предупреждает, что surrogate не unbiased/consistent/order-preserving likelihood actual low-confidence sampler. Appendix D.5, с. 20, ограничивает практическое утверждение path invariance.

Staircase сама по себе не уничтожает комбинаторное равенство uniform masks ↔ uniform reveal order: conditional может просто игнорировать запрещённые keys. Но соответствующий auxiliary model допускает arbitrary position reveal, а используемый decoder завершает блоки последовательно. Их нельзя отождествлять. Current code корректно называет score surrogate; нужен такой же осторожный язык для выводов об anchored probability distribution.

**Исправление/граница:** сохранять это различие в отчётах; не заменять surrogate на true log p и не заявлять exact Eq. (1) distribution matching по факту малой training loss.

### A03 — MEDIUM / ADAPTATION: partial prompt block пересчитывается и зависит от completion

**Место:** `sampling.py:86–100`, `attention.py:26–29`, `model.py:66–74`.

Если P не кратно 32, последние prompt positions находятся в том же bidirectional block, что первые completion positions. В полном no-cache forward эти prompt hidden states видят текущую corruption этого остатка блока; последующие layers могут передавать этот контекст обратно. Это соответствует выбранной полной staircase geometry. Но cached-prefix decoder не обязан быть эквивалентен: model-card example строит prefix cache отдельно. Для partial block cached prompt states не пересчитываются с suffix. Поэтому «выключили cache, вычисляем ту же policy» без дополнительной оговорки неверно.

Clean-prompt features головы рассчитываются отдельно и **не** зависят от конкретного rollout — это правильно для prompt-dependent log Z. Их нельзя брать из полного score forward ради экономии: тогда head стала бы completion-dependent.

**Опора PDF:** §4.1, с. 5, prompt-dependent head; exact sampler/cache не раскрыт в §4.2. [Pinned BD3LM example](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1/blob/60f77bba67ba94231c16f5fb29bc66c500b038f1/README.md) использует cache; это источник сравнения, а не обязательный TraFL protocol.

**Граница:** prompt alignment выглядит обоснованным; не найден off-by-one bug. Но runtime equivalence cache/no-cache и его последствия для reward/score не проверены. Не менять geometry только ради сходства с MDLM.

### A04 — MEDIUM / OWN/GAP: sampler удовлетворяет high-level требованиям, exact decoder авторов не восстановлен

**Место:** `sampling.py:8–33,93–113`, YAML `67–80`.

Temperature-scaled categorical nucleus sampling реализован; crossing-threshold token сохраняется. Confidence берётся из raw softmax; threshold set дополняется forced maximum. Декодирование monotonic: раскрытые токены не remask. Нет параметра общего denoising-step budget; до одного шага на позицию, меньше при threshold-parallel reveal. Подавляется proposal [MASK].

**Опора PDF:** §4.2, с. 6: fastdLLM, confidence-based unmasking, stochastic nucleus .91 и block32. PDF не фиксирует threshold branch, threshold=.9, raw/scaled confidence, cache, mask suppression, force rule или точную версию decoder. §4.1 не раскрывает training generation settings.

**Граница:** ни threshold=.9, ни no-cache, ни force maximum нельзя назвать установленным нарушением PDF. Но фраза «sampler один в один как у авторов» не доказана. В model-card example используется другой proposal/transfer/termination protocol; идентичность ему тоже не заявлять. Сохранять OWN annotations и получить авторский decoder/config для более сильного утверждения.

### A05 — MEDIUM / OWN/GAP: EOS trimming меняет scoring context в последнем блоке

**Место:** `sampling.py:79–115`, `training.py:30–33,104–105`, `rewards.py:70–83`, `scoring.py:47–57`.

Decoder денойзит весь фиксированный canvas и не прекращает работу по EOS. Затем всё после первого EOS удаляется, а сам EOS видим и не mask. Prefix tokens до EOS могут генерироваться с masked/visible post-EOS positions своего блока; score этих suffix positions уже не содержит. Полученная terminal object — trimmed completion, не полный canvas. В staircase эффект ограничен текущим EOS block; будущие blocks на prefix не влияют при корректном SDPA.

**Опора PDF:** §4.1, с. 5, требует non-EOS masks, но не задаёт scoring length, trim/stop и видимость EOS. Appendix C использует fixed L.

**Граница:** допустимый явно отмеченный GAP convention; не объявлять score вероятностью полного sampler trajectory. Переход на early-stop или другой EOS context меняет protocol и требует отдельной документации.

### A06 — LOW / OWN/GAP с literal boundary mismatch: full-mask pair не complementary

**Место:** `masking.py:29–38`, YAML `43–45`.

При l=N два masks оба full; при N=1 это происходит всегда. Следовательно, буквальное complementary-mask condition §4.1, с. 5, на endpoint не выполнено. Но literal complementary endpoint дал бы empty mask, вне domain l∈{1,…,L} Eq. (4). PDF не разрешает это противоречие.

Текущая convention сохраняет uniform nonempty marginal distribution обеих replicates: complement maps l↔N−l для внутренних levels, endpoint maps N→N. Поэтому ожидаемый score не получает дополнительного смещения; variance/correlation scheme отличается. Утверждать «всегда exact complementary» нельзя, а менять endpoint на zero-score silently добавило бы другую проблему.

**Исправление/граница:** сохранить OWN label, уточнить термин «complementary except full endpoint», запросить авторское coupling для точной копии. Это известный GAP, не новый high-severity дефект BD3LM.

### A07 — MEDIUM / ADAPTATION: special-token semantics расходятся с no-padding обещанием

**Место:** `sampling.py:102–105`, `training.py:30–33,105`, `evaluation.py:106–108`, `attention.py:21–29`; YAML `46`.

Pinned tokenizer различает PAD/BOS 151643 и EOS 151645. Proposal suppression исключает только MASK. Если PAD/BOS будет сгенерирован, код считает его обычной valid position, eligibility исключает только EOS, а batch_decode(skip_special_tokens=True) скрывает этот token. Дополнительные chat specials тоже не ограничены. Поэтому «не создаём padding через collate» истинно, но из этого не следует «scoring никогда не маскирует PAD token». Строка mask_domain=completion_non_eos_nonpadding читается как более сильное обещание, чем реализация.

**Опора PDF:** специальное исключение EOS есть в §4.1; обработка остальных special tokens не раскрыта. Этот случай статически допустим; частота генерации PAD не измерялась.

**Исправление/граница:** явно определить, означает ли sampled PAD content token или invalid/padding position; либо подавлять его, либо исключать по согласованной geometry/reward convention. Это собственный protocol, не восстановленный авторский default. Не выдавать вероятность возникновения за измеренный факт.

### A08 — INFO / theory scope: learned log Z не доказывает true partition или сохранение path diversity

**Место:** `model.py:9–46,50–74`, `training.py:97–99,130–135`.

Head архитектурно соответствует статье. Но текущие policy features меняются с LoRA, а scalar обучается по surrogate и random group-centered rewards. Appendix D.5, с. 20, прямо говорит, что group centering нельзя точно поглотить в deterministic Z(x), а path-invariant residual не сохраняет p_theta(trajectory|completion,prompt) тождественно: shared parameters могут изменить его косвенно.

Следовательно, для BD3LM не доказаны true partition-function estimation, exact Eq. (1) matching или reference-level path diversity. Не требуется новый loss term «для исправления»; нужна ограниченная интерпретация и отдельное измерение, как в §5.1.

## Данные, эксперименты и непроверенные инженерные утверждения

### E01 — HIGH / SCOPE: масштаб и coverage репликации существенно сокращены

**Место:** YAML `84–90,117–120`, `runner.py:90–95`.

Train128/test8 (smoke test2) вместо GSM8K 7473/1319; нет самостоятельного MATH run, coding mixture и прочих benchmarks. PDF §4.1–4.2, с. 5–6, и Appendix F, с. 22–23, задаёт эти размеры/наборы. Этот scope ранее сознательно сокращён, поэтому не BUG. Полученные мини-метрики нельзя сравнивать с headline results статьи как сопоставимую репликацию.

**Исправление/граница:** после проверки механики full GSM8K limits=null; остальные задачи — отдельные этапы. 100 updates не опубликованный horizon и не гарантирует dataset coverage.

### E02 — HIGH / SCOPE: основные conclusions статьи пока не измеряются

**Место:** `evaluation.py:72–161`, YAML `76–82`.

Имеется T=.6, Pass@k и validation surrogate loss. Не реализованы sweep T∈{.3,.6,.9}, JustGRPO/ESPO comparison, uniform per-temperature/per-length aggregate, path reveal traces/Jaccard preservation, LLM judge, Z ablations, held-out transfer, uncertainty/error bars и decoding-cost accounting. Дополнительный k=5 не нарушает основной k-set; неусреднённые length-specific metrics не являются ошибкой, но не равны авторской сводке.

**Опора PDF:** §5.1–5.5, с. 6–9; Appendix B, H/I. Path diversity особенно нельзя переносить с bidirectional LLaDA на модель с заранее ограниченным межблочным порядком: baseline должен быть BD3LM под тем же decoder и той же geometry.

### E03 — MEDIUM / OWN/GAP: `dataset_revision: null` не фиксирует одинаковые данные между новыми runs

**Место:** YAML `86`, `data.py:43–45`, `runner.py:180–194`.

Новый run каждый раз разрешает latest dataset SHA через HfApi. Effective revision сохраняется и resume её использует — это правильно. Но одинаковый requested YAML не гарантирует один snapshot между отдельными launches. Кроме того, pipeline читает HF `openai/gsm8k`; скачанная Kaggle папка автоматически не используется.

**Опора PDF:** GSM8K source/scale §4.1–4.2; dataset snapshot сам не опубликован — GAP. **Исправление:** фиксировать известный revision в experiment YAML; если требуется Kaggle, сделать явный data-source выбор и доказать совпадение rows, не подменять источник незаметно.

### E04 — MEDIUM / OWN: validation loss не повторяет training rollout distribution

**Место:** `evaluation.py:79–103,105–138`, `training.py:83–99`.

Validation residual использует первые G из eval_samples и eval sampler settings/length. При отличающихся train/eval block size, length или temperature это не held-out loss training distribution; geometry/features головы тоже изменяются с eval_block_size. Сейчас T и block совпадают, но L512 отличается от training L256. Validation loss дополнительно зависит от stochastic reward-group composition.

**Опора PDF:** loss/validation protocol отсутствует; Algorithm 1 описывает training group. **Граница:** корректная собственная diagnostic метрика, не опубликованная validation target. Не интерпретировать её как likelihood или универсально сравнивать между lengths/protocols.

### E05 — MEDIUM / interface: часть YAML keys описательная, не исполняемая

**Место:** YAML `14,38–39,46,48,73–74,91,105,113–115`; callers `sampling.py:135–148`, `runner.py`, `data.py`, `rewards.py`.

`compile_scope`, `head_feature_source`, `head_pooling`, `mask_domain`, `completion_eos`, `generation_cache`, `generation_stop_on_eos`, `data_sampling`, `reward_mode`, `checkpoint_selection`, `tensorboard_host/port` не dispatch-параметры данного CLI. Изменение строки/boolean не включает другой алгоритм. Например, generation_cache=true всё равно даёт use_cache=False, generation_stop_on_eos=true не меняет sampler; tensorboard_host/port не запускает сервер. Это документированные fixed conventions старой версии, но flat YAML визуально предлагает больше runtime controls, чем имеется.

**Опора PDF:** engineering configuration не задана. **Исправление:** отделить metadata labels от active knobs или явно отметить fixed values возле каждого ключа; не обещать, что произвольная настройка YAML меняет этот behavior.

### E06 — MEDIUM / UNVERIFIED: новые BD3LM tests не проверяют действие attention в реальной модели

**Место:** `tests/test_bd3lm.py:17–75`; `tests/test_trafl.py:108–144`.

Known-matrix test проверяет helper; RecordingDenoiser записывает kwargs, но logits от mask не зависят. Старый synthetic scorer также не моделирует реальный staircase context. Такие tests не установят bool/additive backend semantics, отсутствие future-token leakage, gradient flow через реальную PEFT model, mask broadcast при n16 или отличие partial-prefix cached/no-cache. Это недостаток доказательства, не утверждение, что эти runtime paths сломаны.

**Опора PDF:** conditional prediction Eq. (4); всё корректное действие attention — prerequisite выбранной ADAPTATION. Позже нужны independent perturbation checks: изменения строго будущего блока не меняют ранние logits; permitted same/past block изменения доступны; actual mask honored under supported backend. В рамках этого аудита ничего не запускалось.

### E07 — HIGH / UNVERIFIED: нет BD3LM smoke и новых численных результатов

**Место:** `README.md:78–85`, `evidence/bd3lm_adaptation.json`.

Все исторические metrics/checkpoints/events относятся к MDLM. Даже исправленный resume был перенесён статически; supplied tests не содержат uninterrupted-vs-resumed real-model equivalence проверки. Нет доказательства, что новая ветка помещается в GPU память, проходит score backward и full checkpoint roundtrip. Syntax/import verification не заменяет это.

**Опора PDF:** practical experiment §4–5. **Граница:** непроверенный working candidate; нельзя выдавать старый APPROVED за runtime admission новой ветки. Отсутствие запусков соблюдает ограничение пользователя, поэтому это не скрытый провал тестов.

### E08 — MEDIUM / reproducibility: full resume сохраняет state, но exact execution identity не фиксирует

**Место:** `checkpoint.py:95–114`, `runner.py:195–207,267–276`; dataset materialization `data.py:63–73`.

Сохраняются adapter/head/optimizer, CPU/CUDA/NumPy/Python RNG, выделенные generators, batch permutation/cursor, next-update/horizon, selected IDs, running stats и JSONL position. Для deterministic pretokenized rows DataLoader prefetch не должен продвигать saved cursor. Нет очевидного пропуска optimizer или scheduler state в новой схеме.

Однако schema не содержит source commit/protocol version, tokenizer-rendered input hash/IDs, dependency/runtime identity или CUDA determinism policy. Versions JSON лежит рядом и при resume перезаписывается текущими versions. Восстановление на другом коде/Transformers или attention backend не гарантирует тот же процесс обучения, даже если RNG сохранён. Это отдельная проблема от B01/B02, не основание утверждать, что default same-version resume уже неисправен.

**Опора PDF:** exact resume авторами не описан. **Исправление:** минимальный checkpoint execution identity + immutable provenance, осторожное обещание «stateful continuation при одинаковой реализации/среде». Bitwise equivalence проверять отдельно; не приписывать статье требование такой детерминированности.

### E09 — LOW / reproducibility: JSONL содержит тексты, но не точные scorer inputs

**Место:** `training.py:139–149`, `evaluation.py:139–145`, `checkpoint.py:105–106`.

Artifacts сохраняют question/text/reward, но не generated token IDs, actual masks, per-replicate policy/reference scores или terminal length. `skip_special_tokens=True` скрывает разные token sequences за одним text. Из этих JSON нельзя точно восстановить Eq. (4) residual и независимо проверить special-token case. Full checkpoint полезен для продолжения, но не является журналом уже сделанных score computations.

**Опора PDF:** Eq. (4) и path-diversity §5.1 используют token positions. **Исправление/граница:** сохранить хотя бы небольшой diagnostic batch token/mask/scores и исходные source hashes, когда будет разрешён проверочный запуск. Полный trace logging каждого update статьёй не предписан.

## OWN defaults, которые не следует ошибочно объявлять нарушениями

| Решение | Место | Классификация |
|---|---|---|
| Qwen3 BD3LM вместо LLaDA-8B-Instruct | YAML `3–4` | Разрешённая пользователем замена; меняется backbone и доступный контекст. |
| beta=1, batch1, 100 updates | YAML `50–52` | Числа авторами не раскрыты; это не восстановленные hyperparameters. |
| AdamW betas/eps/decay/fused | YAML `53–57` | OWN optimizer choice. |
| LoRA target set, init, dropout0, biasnone | YAML `28–31` | OWN implementation; rank/alpha опубликованы. |
| Head256/SiLU/bias/zero-output и current clean final states | YAML `34–39`, `model.py` | OWN; два слоя, pooling и detach опубликованы. |
| Warmup0, first peak / last minimum | YAML `58–62`, `objective.py` | Cosine/endpoints опубликованы, phase/horizon нет. |
| Training L256/T.6/top-p.91, block32 | YAML `67–71` | OWN training sampler; evaluation значения нельзя выдавать за авторские training. |
| BF16, FP32 reductions, one GPU, kernel-only compile | YAML `8,11–19` | OWN numerics/hardware; никаких extra loss terms не добавлено. |
| Exact Fraction parser, last boxed, invalid→0 | `rewards.py:21–67`, YAML `105–106` | OWN extraction details; binary/extracted boxed accuracy опубликованы. |
| EOS trim, full-endpoint repeat, empty_score0 | YAML `45–48` | Explicit GAP convention, последствия разобраны выше. |
| Shuffle epochs/seeds/log cadence/checkpoint selection | YAML `20–24,91,108–115` | OWN orchestration, не авторское описание. |

Numeric parser принимает поддержанное рациональное представление, но не arbitrary mathematical expression, единицы или любые LaTeX wrappers. Сравнение `1/2` и `.5` через Fraction — собственная canonicalization; identical author extraction behavior не доказано. `invalid_reward=0` сохраняет binary default; произвольное другое значение меняло бы reward protocol.

## Приоритет действий

1. Устранить B01/B02: checkpoint/model/attention identity должна предотвращать cross-MDLM/BD3LM загрузку и неверный adapter scaling.
2. Устранить B03 или чётко ограничить backend SDPA.
3. Сохранить исходный global Eq. (4) как baseline; отдельно зафиксировать A01–A05 как границы новой модельной адаптации. Не заменять формулу на blockwise surrogate под видом исправления опубликованного TraFL.
4. При разрешении запусков проверить реальные BD3LM attention/gradient/checkpoint paths; старые MDLM evidence не использовать как замену.
5. Только после этого расширять GSM8K coverage и делать matched base/trained evaluation. Для conclusions о diversity нужны отдельные measurements; Pass@k этого не доказывает.

**Итог аудита:** формулы и online training loop сохранены аккуратно. Самая существенная научная неопределённость — global masked score для block-causal conditional; самые конкретные программные дефекты — checkpoint identity и boolean mask в eager. Остальные перечисленные отличия либо собственные GAP conventions, либо сознательно неполный экспериментальный scope, либо ещё не проверенные runtime утверждения. Выдуманного «фатального нарушения теоремы из-за самого BD3LM» аудит не устанавливает.
