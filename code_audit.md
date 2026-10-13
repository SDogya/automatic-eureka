# Боба: независимый аудит реализации TraFL

**Обновление 2026-10-13:** после этого аудита в `run.py` и YAML добавлен full resume
(optimizer, RNG, DataLoader permutation/cursor, scheduler position, logging context)
и atomic `training_state.pt`. Приведённые ниже APPROVED, hashes и runtime evidence
относятся к предыдущей версии. Новое изменение не проходило повторный независимый
аудит, тесты и smoke по прямому указанию пользователя; проверен только синтаксис.

## 1. Audit Verdict

- **Status:** `APPROVED (Ready for Implementation)` — реализация соответствует PDF в пределах утверждённого GSM8K MVP и разрешённых OWN choices; проверка исполнения завершена.
- **Discrepancy Severity Score:** `Zero`; подтверждённых расхождений ядра алгоритма: **0**.
- **Открытых замечаний:** **0**; C01 исправлено и перепроверено.
- **Scope:** GSM8K MVP, утверждённый пользователем план, Algorithm 1 / Eq. (4) / §4.1–4.2 / Appendices C, D.5, E. Полное воспроизведение coding, всех datasets, ablations, judge и trajectory diagnostics не входит в заявленный scope README.
- **Проверено математических блоков:** 12 опубликованных/раскрытых операций, перечисленных в разделе 3; дополнительные OWN-default операции проверены отдельно.
- **Тесты:** Боба непосредственно прочитал [tests.log](evidence/tests.log): **24 passed in 11.23 s**. Независимо прочитаны тесты, проверены численные ожидания и production-kernel eager/compiled parity test. Самостоятельный повтор pytest Боба не запускал.
- **Реальный GPU smoke:** **PASSED**, run `runs/20261012T232032.630275Z-smoke`: три updates и before/after evaluation двух problems, n16/L256. Координатор подтвердил primary process exit0; Боба непосредственно проверил [smoke.log](evidence/smoke.log), [verification.json](evidence/verification.json) и [scalars.json](evidence/scalars.json).
- **Checkpoint reload:** отдельный public CLI `--mode eval` завершился с exit0 по подтверждению координатора; [reload.log](evidence/reload.log) содержит step3 и Completed. Это диагностическая проверка 1 problem/n5/L32 на GPU1, не второй paper benchmark.

Источник — [PDF](../32236_Beyond_Mode_Seeking_RL_T.pdf), [утверждённая спецификация](../trafl_algorithm_spec.md), [её audit](../trafl_algorithm_audit.md), [lean-code требования](../code.md). Разрешённая пользователем точная база: `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1`, revision `c8d24a3f4adaeef46881b450e1bf7d1005203bd7`. Координатор подтвердил, что exact HF ID прямо назван пользователем в последующем запросе и включён в утверждённый план; первоначальное описание размера «0.8B» не является причиной отклонить этот checkpoint.

Ревьюер не редактирует pipeline. Рабочий server directory: `/home/sirius03/test/trafl`. Смена занятого GPU1 на GPU2 и привязка по UUID — явно документированная аппаратная настройка, не изменение алгоритма.

### Проверенные снимки файлов

| Файл | SHA-256 после закрытия C01 и проверки compile-scope change |
|---|---|
| run.py | `c2648ef84d3a5af770984cbb2c08ec054952d16d7956347819bb8c5f9668b346` |
| trafl.py | `1cccc651ff4bc868e2b6adb8f1c7a5a5211e693ef42cca440821a9603e47c630` |
| config.yaml | `483a64b32c7bcba80e0008a63fd4e445557060376ba55639148eae71197ff305` |
| test_trafl.py | `1d9c89689e161660cb8c2e3a83c3e54d8b19e9356e0b37fbf8ca28a8483a0a7d` |
| README.md | `a16655e3c3cf93127ece5d56d1092068dcc2bfe078356a72c3f8673a5aefeeb7` |

## 2. Discrepancy Matrix — история REJECT, открытых нет

Текущая active discrepancy matrix пуста. Единственное замечание первой формальной итерации сохранено ниже как история. Повторная проверка подтверждает: config.yaml:59 теперь относит PAPER только к peak/min и cosine, а строка 60 явно помечает scheduler placement/phase/horizon как OWN G05. Code formulas и optimizer order не изменены. Позиция C01 ниже — историческая, до добавления compile-scope fields.

### C01 — собственная фаза/порядок cosine scheduler ошибочно помечены PAPER

- **Location in Code:** [config.yaml](config.yaml), строка 57: `# PAPER §4.1. Schedule applied before each update, first peak / last minimum.` Фактическая собственная формула — [trafl.py](trafl.py), строки 170–174; применение перед update — [run.py](run.py), строки 254–258.
- **Reference in Paper:** §4.1, с. 5, строки PDF 264–265 задают peak LR 6×10⁻⁵, cosine decay и minimum LR 6×10⁻⁶. Они не задают scheduler placement, cosine horizon/phase и правило first update peak / last update minimum. Спецификация G05 прямо фиксирует отсутствие этих деталей.
- **The Issue:** code выполняет разрешённый OWN default правильно, но comment относит scheduler placement и endpoints к опубликованным авторским правилам. Это нарушает обязательное различие PAPER / OWN Gxx.
- **Required Fix:** оставить PAPER только у peak/min/cosine, а before-update placement и first/last endpoints явно пометить `OWN G05`. Код формулы/цикла менять не требуется.

## 3. Omission & Ambiguity Check

### Соответствие математике и порядку вычислений

| № | Проверенная операция | Code locations | Источник |
|---|---|---|---|
| 1 | Свежий current-policy group G=5, policy без промежуточной update | run.py:78–83, 110; config.yaml:41 | Algorithm 1, с. 14; §4.1, с. 5 |
| 2 | Binary exact-answer reward для math | trafl.py:11–41; run.py:63–68 | §4.1–4.2, с. 6; extraction details — OWN G15 |
| 3 | Group mean r̄ и r̃=r−r̄; без std scaling и skip | run.py:84–85; trafl.py:65–66 | Algorithm 1 |
| 4 | Uniform mask level и subset без replacement; K=4 | trafl.py:49–62; config.yaml:40 | Eq. (4), §4.1; complementary endpoint — OWN G11 |
| 5 | Corrupt completion positions, исходный target token, unchanged prompt | trafl.py:83–96 | Eq. (4), с. 5 |
| 6 | 1/l mean masked log probabilities | trafl.py:79–80, 94–96 | Eq. (4); ненормированный ELBO не подставлен |
| 7 | Mean по четырём score replicates до squared loss | trafl.py:97; run.py:100–102 | Algorithm 1: estimate scores → residual → square |
| 8 | Одинаковые current/reference masks | run.py:93–98 | Eq. (4); Appendix C, с. 16 |
| 9 | Mean-pooled prompt features, detached 2-layer scalar logZ | trafl.py:100–119; run.py:86, 99, 214–215 | §4.1, с. 5 |
| 10 | Residual Sθ−Sref−βr̃+logZ | trafl.py:69–70; run.py:100 | Algorithm 1, с. 14 |
| 11 | Mean squared residual по B×G и ровно один joint update | run.py:75–77, 101–110, 245–262 | Algorithm 1, online setup |
| 12 | Pass@k unbiased estimator, problem-level benchmark mean | trafl.py:177–178; run.py:129–132, 152 | §4.2, с. 6 |

Суммирование backward по rollout с divisor B×G эквивалентно gradient общего loss: между ними optimizer.step не вызывается. Reference frozen и вычисляется под no_grad (run.py:96–98, 207–208). Head/features detach не отключает trainable policy masked scores. Head forward повторён для каждого rollout одного prompt, но features и параметры неизменны до единственного joint update; это тот же математический loss.

No extra SFT, replay, group-softmax normalization, explicit backward model, PPO ratios/clipping, KL/entropy terms или REINFORCE-through-sampling correction в pipeline не добавлены. Raw target Eq. (1) не выдаётся за exact objective практического centered surrogate; approximation bias из Appendix C не устранён общими masks и не объявлен устранённым.

### Разрешённые OWN choices

Приняты как собственные явно согласованные решения: β1; B1/M100 и smoke M3; AdamW betas .9/.999, eps1e−8, wd0; no warmup; семь projection targets LoRA, rank/alpha128; current final clean-prompt features; MLP hidden256/SiLU/zero-final-layer; два antithetic pairs; uniform eligible l1…N; endpoint empty-complement→full; visible unscored first EOS, suffix truncation; empty-eligible score0; generation temp.6/top-p.91/block32, confidence .9, no cache; no system prompt и enable_thinking=false; last-balanced-boxed numeric rational exact reward; выбранные GSM8K subsets и seeds.

Endpoint rule сохраняет formal marginal law: если первый level l<N, второй имеет N−l; при l=N второй имеет N. Поэтому оба уровня uniform 1…N; uniform subset сохраняется complement mapping, а full endpoint является явным исключением complementary pairing. Это разрешённый OWN G11, не подмена авторского нераскрытого endpoint.

EOS не маскируется и не участвует в denominator: run.py:65, 91–95 / trafl.py:44–46. Empty-eligible case возвращает score0, но group rewards и BG divisor сохраняются; skip отсутствует. API `logits_to_keep` выбирает только соответствующие positions; code targets использует исходные completion tokens. Реальный smoke успешно выполнил generation, selected-position scores, backward и joint updates, подтвердив совместимость с конкретным remote checkpoint.

Nucleus sampling использует temperature до nucleus selection, сохраняет crossing token и вызывает categorical multinomial (trafl.py:122–133). Confidence берётся из исходных logits до suppression mask proposal (trafl.py:157–161). Sampler назван в README адаптированной no-cache confidence-threshold схемой Fast-dLLM v1, а не побитно идентичной авторской реализацией. Mask suppression не применяется к training likelihood. Прогресс при низком confidence обеспечивает one highest-confidence transfer (trafl.py:163–166).

### Проверка качества тестов и требований code.md

Тесты содержат независимые численные ожидания, а не только сравнение функции с её переписанным телом:

- В `test_paper_loss_and_analytic_gradients` заранее заданы residuals, loss2.485 и analytic gradients, включая B=2/G=5.
- `test_exhaustive_antithetic_uniform_marginals` исчерпывает l/subsets для N1…4 и проверяет law 1/[N·C(N,l)] для обоих marginals.
- Actual masked scoring сверяется с вручную вычисленным log normalizer и различием square(mean) / mean(square).
- Actual `update_batch` проверяет один optimizer.step, unchanged frozen reference, самостоятельный analytic policy gradient **−0.48** и measurable parameter change >1e−4. Diverse targets устраняют прежнюю возможность принять floating-point roundoff за полезный gradient.
- Detached features/head gradients, streamed/full loss gradients, nucleus distribution/crossing token/temperature, eager/compiled values и gradients, sampler termination, EOS/empty handling, reward edge cases и LR endpoints проверены отдельными тестами.

Основной pipeline состоит из двух простых файлов и flat YAML. Используются BF16 autocast, FP32 log-softmax/residual, torch.compile production score kernel с backward, pinned multiworker DataLoader, nonblocking copies, zero_grad(set_to_none=True), локальный SummaryWriter. Neural forwards остаются eager/SDPA. Try/except, Hydra/Lightning и matplotlib отсутствуют. GPU→CPU text transfer после group generation нужен reward extraction; metric transfers происходят по log interval. Necessary generation control sync отдельно обозначен. Фиксированные строковые YAML conventions не выдаются за переключаемые backends (README:83–88).

**Повторный review compile scope:** прежняя полная Inductor-компиляция PEFT forward не завершила baseline за более чем 10 минут и была прервана; этот run не считается успешным smoke. Изменение run.py:221–224 компилирует только `equations.masked_token_mean`. Function rebinding действует на globals `masked_score`; mathematical Eq. (4), LoRA parameter identities, checkpoint keys, head detach и optimizer order остаются прежними. `compile=True/backend=inductor/dynamic=True`, два compiler workers — явно OWN G14. README:92–103 честно описывает numerical-kernel scope. Новый test_trafl.py:201–209 сверяет фактический production kernel eager/compiled value и logits gradients. Все 24 теста текущего снимка прошли согласно remote log.

До готовности полного снимка были исправлены два предварительных замечания: `.format` разрушал literal boxed braces в prompt → заменено точечным `.replace`; identical fake completions в update test → diverse targets и независимый expected gradient. Это предварительные finding, не отдельные финальные REJECT-вердикты. Дополнительно проверены raw-confidence-before-mask-suppression и GPU aggregation validation loss.

### Ограничения проверки

Core соответствует опубликованным формулам с разрешёнными GAP defaults; это не полное paper-scale experiment reproduction. Основной конфиг использует train128/eval8, smoke eval2/L256; полный GSM8K download содержит train7473/test1319, а selected manifest фиксирует выбранные128/2. Полный training run M100, evaluation L512, coding/ablations/judge/path-diversity и прирост точности не подтверждены smoke-тестом. Paper reported performance для LLaDA не является acceptance threshold другой базы.

Успешные 24 теста подтверждают численное ядро, production-kernel compilation и CPU/toy integration. Завершённый real GPU smoke дополнительно подтверждает работу конкретного checkpoint, diffusion generation, CUDA scoring/backward, optimizer, evaluation, serialization и logging. Его результаты подтверждают работоспособность, а не улучшение качества модели.

### Проверенные runtime evidence и фактические результаты

Боба напрямую прочитал все пять локальных evidence files после завершения runs: tests.log, smoke.log, reload.log, verification.json и scalars.json. Они согласованы по update counts, step indices, метрикам и artifact metadata. Server exit codes получены от координатора, который наблюдал завершение процессов; тексты primary/reload logs независимо содержат финальные Completed.

| Проверка | Фактическое подтверждение |
|---|---|
| Test suite | 24 passed in 11.23 s |
| Primary run | `runs/20261012T232032.630275Z-smoke`, exit0; модель/revision закреплены |
| Data | dataset revision `740312add88f781978c0658806c59bc2815b9866`; selected128 training /2 evaluation |
| Hardware | primary GPU2, UUID `GPU-fe2d8dfd-06f2-a5c4-a7fd-4a5f23947005` |
| Updates | steps1/2/3; groups имеют 0/5,3/5,1/5 correct |
| LR | 6×10⁻⁵,3.3×10⁻⁵,6×10⁻⁶; совпадает с разрешённым OWN G05 scheduler |
| Train loss | 4.4408921×10⁻¹⁶,0.2396116406,0.1602068990; все конечны |
| Model/head activity | step2/3 mixed groups; score difference ≈2.10708×10⁻⁴/1.57750×10⁻⁴; logZ ≈0.00372571/0.00120040 |
| Primary evaluation | до/после: 2 problems, n16/L256; обоим answers files проверены counts согласно verification |
| Checkpoint | saved step3; adapters309 MiB/head1.1 MiB и finite head tensors подтверждены coordinator artifact inspection |
| Reload | `runs/reload-check/20261012T233425.246402Z-eval`, exit0; public CLI, saved step3, n5/L32/1 problem, без обучения |
| TensorBoard | 15 finite scalar tags; train steps1/2/3, val steps0/3; scalars.json прочитан непосредственно |

Первый all-fail group не был пропущен: маленький loss и score difference порядка10⁻⁸ согласуются с centered zero reward и numerical policy/reference difference. Следующие mixed groups действительно исполнили ненулевой reward balance; это не smoke, основанный только на нулевых градиентах. Однако реальные counterfactual gradients этого remote model отдельно не пересчитывались: численные gradients проверены независимыми golden/toy tests и production-kernel parity.

| Метрика primary smoke | До | После |
|---|---:|---:|
| Pass@1 | 0.40625 | 0.3125 |
| Pass@5 | 0.7707188645 | 0.6538461538 |
| Pass@16 | 1.0 | 1.0 |
| Validation surrogate loss | 0.2400000095 | 0.1220386028 |

На двух problems sampled accuracy после трёх updates снизилась. Снижение validation surrogate loss не доказывает improvement correctness. README:127–129 отражает эти значения и прямо не заявляет прирост. Никакой statistical significance или paper-scale conclusion из такого smoke не выводится. Короткий reload diagnostic L32/n5 не сравнивается с primary metrics L256/n16.

## 4. Final Directive to Writer

C01 исправлено и принято. Статические code/spec/math требования и качество независимых численных тестов проходят аудит. Pipeline files ревьюер не менял.

История:

1. Полный audit 1 — **REJECT (Requires Revision)**: 1 attribution замечание, 0 algorithm deviations.
2. Повторный audit 2 — **APPROVED (Ready for Implementation)**: C01 закрыто; открытых замечаний 0; SHA-256 текущих пяти файлов закреплены выше.
3. Повторный review execution scope — **APPROVED**, новых замечаний 0: full-model compilation заменена на разрешённый OWN G14 numerical-kernel compilation; после этого 24 теста прошли, hashes обновлены. Interrupted прежний runtime не выдан за success.
4. Final runtime evidence audit — **APPROVED**, открытых замечаний 0: primary smoke и отдельный CLI reload завершены; evidence files прочитаны непосредственно; README согласован с фактами; его итоговый SHA обновлён.

Неуказанные авторские настройки заменены только явно разрешёнными OWN choices; они не являются причиной REJECT. Допуск не означает побитное повторение нераскрытой авторской implementation или paper-scale результатов.

В пределах утверждённого MVP необходимых исправлений или ожидаемых runtime results не осталось. Source files и report готовы к синхронизации на сервер; последующая сверка server/local source hashes и TensorBoard HTTP выполняется координатором как проверка доставки. Это не заменяет и не изменяет выводы математического аудита. Любое изменение pipeline после закреплённых SHA требует проверки соответствующего diff.
