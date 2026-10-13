# Боба: независимый аудит спецификации TraFL

## 1. Audit Verdict

- **Status:** `APPROVED (Ready for Implementation)`.
- **Discrepancy Severity Score:** `Zero`.
- **Ревизия:** второй аудит исправленной `trafl_algorithm_spec.md`, 667 строк. Исторические позиции D1–D5 относятся к первому снимку в 662 строки; отдельная таблица ниже даёт позиции исправлений в окончательном документе.
- **SHA-256 проверенного окончательного документа:** `fb7ca5b25adf4e3e5b27a8a7ab26581f57d953465fc972efb3860a7b11a855a7`.
- **Проверено формул:** 46 формульных единиц: все 15 нумерованных формул статьи и 31 вспомогательный блок, перечисленный ниже. Несколько тесно связанных определений в одном блоке считаются одной единицей; этот счёт не означает число отдельных знаков равенства.
- **Найдено расхождений:** 5 в первой итерации, все исправлены; **открытых расхождений: 0**. Во второй итерации новых расхождений не найдено.

Источник: [исходный PDF](32236_Beyond_Mode_Seeking_RL_T.pdf), полностью прочитаны 25 страниц, основной текст и приложения A–K. Проверяемый результат: [спецификация Бибы](trafl_algorithm_spec.md). Сложные записи дополнительно сверены визуально на страницах 14, 15 и 20 PDF. Это отдельная проверка источника; Боба не редактирует спецификацию.

Нумерованные формулы: (1)–(3), с. 4 → строки 94–98 спецификации; (4), с. 5 → 116; (5)–(6), с. 15 → 122–124; (7)–(11), с. 18 → 150–154; (12)–(15), с. 19 → 160–163. Во всех 15 совпадают знаки, коэффициенты и индексы; явно разведены теоретическая цель, практический surrogate и вспомогательная теория.

Вспомогательные формульные блоки, сверенные отдельно:

| № | Блок статьи | Место в спецификации первого снимка |
|---|---|---|
| U01 | Terminal-reward objective J, §3.1 / D.1, с. 3 / 16 | 136 |
| U02 | Marginal completion как сумма paths, §3.1 / D.1 | 136 |
| U03 | Support/entropy inequalities Theorem 1, с. 4 / 16 | 138 |
| U04 | Entropy chain proof, D.1, с. 16 | 138 |
| U05 | Euclidean logit gradient Proposition 2, D.2, с. 17 | 142 |
| U06 | Global reward mean, conditional q и advantage definitions, D.2 | 140 |
| U07 | Within-completion logit flow, D.2 | 142 |
| U08 | Derivative log(q_i/q_j), D.2 | 143 |
| U09 | Restricted q dynamics, D.2 | 143 |
| U10 | Entropy derivative/covariance, D.2 | 144 |
| U11 | Conditional softmax parametrization, D.3, с. 18 | 151 |
| U12 | Exponential relative mass, D.3, с. 19 | 156 |
| U13 | Cubic Euclidean dynamics, D.3, с. 19 | 156 |
| U14 | Reference-tilted completion marginal и conditional paths, D.5, с. 20 | 102, 165 |
| U15 | Practical path-invariant residual, D.5 | 167 |
| U16 | DMPO importance log-score и softmax weights, D.6, с. 20 | 464 |
| U17 | Weighted DCE loss со stop-gradient weights, D.6 | 464 |
| U18 | Additive buffer-logit dynamics, D.6 | 464 |
| U19 | Normalized conditional buffer probabilities, D.6, с. 21 | 464 |
| U20 | Buffer replicator equation, D.6 | 464 |
| U21 | Buffer exponential solution, D.6 | 464 |
| U22 | Uniform corruption probability, C, с. 15 | 118 |
| U23 | Masked reconstruction product, C | 118 |
| U24 | One-step reconstruction marginal, C | 120 |
| U25 | Reveal-order product, ELBO expectation/bound и mixture, C | 126–128 |
| U26 | Prompt-group reward mean и centering, Algorithm 1, с. 14 | 286 |
| U27 | Practical centered residual, Algorithm 1 | 349 |
| U28 | Batch/rollout mean squared loss, Algorithm 1 | 363 |
| U29 | Unbiased Pass@k estimator, §4.2, с. 6 | 409–411 |
| U30 | Same-step Jaccard и pair/step average, §5.1, с. 7 | 430–432 |
| U31 | Preserved = J_ref / J_model × 100%, §5.1 | 432 |

Дополнительные математические раскрытия спецификации — partition normalization, decomposition residual, fixed-batch gradients, mean-pooling graph и sign-test upper tail — проверены отдельно как следствия, а не объявлены дополнительными авторскими настройками. В частности, DMPO flow приведён в явно указанном assumed additive regime; он не подменяет optimizer TraFL.

## 2. Discrepancy Matrix — архив REJECT, все замечания закрыты

Для текущего APPROVED активная discrepancy matrix пуста. Ниже сохранён отчёт первой итерации, чтобы допуск имел проверяемую историю. Повторно проверены все пять разделов, критические формулы и порядок online update; замечания закрыты по содержанию исправленного файла, не по обещанию автора.

| Замечание | Итог повторной проверки | Место исправления в окончательной спецификации |
|---|---|---|
| D1 | Closed: K=4 total replicates; неизвестность количества удалена из GAP | 308, 632 |
| D2 | Closed: initial θ₁/φ₁ и final θ_{M+1}/φ_{M+1} определены | 236, 395 |
| D3 | Closed: empty side исключает comparison; GAP сужен до отсутствия retained comparisons/non-ties | 472, 613–614 |
| D4 | Closed: independent означает от realized path; зависимость antithetic masks названа явно | 296, 308 |
| D5 | Closed: contradiction источника обозначено EXPLICIT GAP; числа Table 5 сохранены | 506–521 |

Далее — первоначальные доказательные замечания; они не являются текущими ошибками окончательной спецификации.

### D1 — известное количество replicates ошибочно объявлено неизвестным

- **Location in Spec:** строка 308: «означает ли K=4 четыре total masks или иной counting»; строка 627, G11: «exact antithetic coupling/count/weights отсутствуют». При этом строки 304–306 и 321 уже требуют четыре replicates.
- **Reference in Paper:** §4.1, с. 5, строки PDF 266–267: «4 antithetic replicates, i.e., complementary mask pairs». Algorithm 1, с. 14, строка PDF 734: «using K mask samples».
- **The Issue:** статья задаёт четыре replicates; спецификация превращает опубликованное количество в GAP и создаёт внутреннее противоречие между fixed-four execution и неопределённым counting.
- **Required Fix:** фиксировать K=4 total mask replicates. Оставить GAP для joint sampling/coupling, выбора levels, EOS domain, weights и zero-sized complement; удалить предположение о неуказанном числе replicates.

### D2 — индекс начального состояния policy/head не определён

- **Location in Spec:** строки 17 и 236: step s=1…M; строка 246: current θ_s; строка 335: φ_s; строка 381: после update «новое состояние (θ_{s+1}, φ_{s+1})». Начальная θ_1/θ_0 и φ_1/φ_0 явно не связаны с initialization.
- **Reference in Paper:** Algorithm 1, с. 14, строки PDF 728–729: p_θ ← p_{θ_init}, initialize Z_φ, затем for step=1 to M; результат — trained policy после M updates.
- **The Issue:** авторский loop понятен, но добавленная state-index notation не фиксирует, какой snapshot считается начальным и какой возвращается после M-й update. Для механической спецификации это неоднозначность.
- **Required Fix:** явно объявить θ_1=θ_init и φ_1=инициализированный head, с final snapshot (θ_{M+1},φ_{M+1}), либо последовательно использовать pre-update (θ_{s−1},φ_{s−1}) и post-update (θ_s,φ_s). Это выбор нотации, не новый гиперпараметр.

### D3 — правило исключения пустого correct-solution сравнения уже задано

- **Location in Spec:** строка 609: «пустые correct-solution judge sets … не имеют полного recovery protocol». Строка 472 одновременно правильно требует retain comparison только при хотя бы одном correct solution с каждой стороны.
- **Reference in Paper:** App. H, с. 24, строки PDF 1271–1273: filter each side to correct samples, «keep a comparison only when both sides contain at least one correct solution»; §5.5, с. 9, строки 463–464 подтверждают то же.
- **The Issue:** для пустого correct set одной из сторон авторский результат уже определён: problem comparison исключается. Общий список GAP размывает это конкретное правило.
- **Required Fix:** убрать empty-side comparison из неизвестных правил и явно сослаться на prescribed exclusion. Сохранить GAP для случая отсутствия retained comparisons вообще, отсутствия non-ties и malformed judge outputs.

### D4 — название corruption step допускает неверную независимость replicates

- **Location in Spec:** строка 296: «сэмплировать independent completion corruptions». Строка 302 задаёт antithetic complementary pairs.
- **Reference in Paper:** §4.1, с. 5: antithetic replicates/complementary pairs; App. D.5, с. 20, строки PDF 1049–1050: masking pattern independent of policy decoding order.
- **The Issue:** слово independent в названии может означать взаимную независимость mask replicates, несовместимую с complementary pairing. Источник требует независимость от realized path, а не независимость пары между собой.
- **Required Fix:** назвать шаг «Сэмплировать corruptions независимо от realized decoding path» и явно различить независимость от path и зависимость replicates внутри antithetic pair.

### D5 — внутренняя статистическая несогласованность статьи не отмечена

- **Location in Spec:** строка 506 задаёт one-sided exact upper-tail formula p=Σ_{j=f}^{u} C(u,j)2^{−u}; строки 511 и 513 воспроизводят p=0.39 при For<Against; строка 517 объясняет сохранение значений только округлением.
- **Reference in Paper:** App. H, с. 23, строки PDF 1240–1241: prespecified one-sided directional hypotheses; Tab. 5, с. 24: MATH-500, ESPO>Base, correct solutions — For 7.5%, Against 10.0%, p=0.39; LiveCodeBench, TraFL>Base, all samples — For 20.9%, Against 23.0%, p=0.39.
- **The Issue:** при f<a one-sided upper-tail sign test для For>Against имеет p>0.5. Указанные p=0.39 не могут получиться из заявленного теста; порядок For<Against не объясняется rounding. Это противоречие самого paper, а не повод «исправлять» напечатанные цифры. Спецификация пока не предупреждает, что одна stated calculation formula не воспроизводит все printed p.
- **Required Fix:** сохранить Table 5 без изменений; добавить рядом EXPLICIT GAP IN PAPER о несовпадении stated directional tests и некоторых published p, отсутствии actual calculation protocol/raw counts. Явно назвать upper-tail formula раскрытием заявленного теста, без гарантии совпадения всех published p.

## 3. Omission & Ambiguity Check

Core algorithm проходит сверку: raw rewards Eqs. (1)–(3) отделены от centered rewards Algorithm 1; centering только внутри prompt group, без standard-deviation scaling; score имеет 1/l; четыре masks current/reference совпадают; masked-token average берётся до squared residual; normalization predictor prompt-dependent; features detached; reference frozen; ровно один gradient update на свежем online batch. Дополнительные KL/PPO/entropy/replay/SFT компоненты не подставлены.

Теоретические оговорки сохранены: practical score не exact sampler likelihood; его Monte Carlo unbiasedness относится к expected surrogate, а не true completion likelihood; shared parameter updates могут косвенно менять conditional paths; centering не является exact absorption into deterministic Z; replicator Eq. (11) требует additive flow и отличается от Euclidean dynamics; conceptual backward TB Eq. (14) отделена от reference-anchored target.

Сверены опубликованные константы §4.1–4.2 и приложения A–K: LoRA rank/alpha 128; peak/min LR 6×10⁻⁵/6×10⁻⁶ и cosine; G=5, K=4; 2-layer MLP; no extra SFT; три отдельных training runs и размеры subsets; eval lengths 256/512, block32, top-p0.91, temperatures 0.3/0.6/0.9, n=16, Pass@5 и sampling budgets; N16/L256/steps256 trajectory diagnostic; judge format, filtering, blindness/randomization, significance threshold; compute и release ограничения. Tables 1–5 и numeric annotations Figures 2–3 перенесены верно. Неполный training sampler не заполнен evaluation defaults.

Источниковые gaps сохраняются как ограничения будущей реализации: β, M, prompt batch, optimizer, warmup/cosine timing, LoRA targets/init, head width/activation/init и hidden-state source, reference construction/revisions, training generation settings, antithetic coupling/EOS domain, empty/numerical conventions, exact rewards/evaluator versions, filtering thresholds/retained IDs, checkpoint selection/seeds, fastdLLM settings, trajectory aggregation, judge snapshot/protocol и raw statistics. Их отсутствие нельзя исправить выдумыванием авторских дефолтов.

Пользовательская замена pretrained на **Qwen 0.8B dLLM** разрешена пользователем и не является основанием REJECT. Точный checkpoint ID, hidden size и token conventions не опубликованы в пользовательском уточнении; их нельзя выводить из имени. В окончательном документе строки 7 и 663 правильно принимают diffusion-природу пользовательской модели, сохраняют авторскую LLaDA-8B-Instruct как отдельную экспериментальную базу и не выдумывают её checkpoint ID или внутренние размеры.

При повторной проверке дополнительно подтверждено: GAP для K-count удалён без устранения реального GAP antithetic coupling/EOS endpoints; инструкция исключения пустого correct-solution comparison не противоречит edge-case section; sign-test formula и напечатанные Table 5 results разведены с явным источниковым противоречием; изменения не затронули совпадавшие Eqs. (1)–(15), mean-centered training loss или one-update loop.

## 4. Final Directive to Writer

Все пять замечаний исправлены и приняты. Дополнительная переработка данного снимка не требуется. Спецификация допускается как точное описание опубликованной механики TraFL и экспериментов с явно разрешённой заменой модели и честно обозначенными авторскими пробелами; код и implementation files не создавались.

История проверки:

1. **Итерация 1 — REJECT (Requires Revision), Medium:** 5 замечаний, проверены все 15 numbered equations и 31 auxiliary block; все отправлены Бибе и координатору.
2. **Итерация 2 — APPROVED (Ready for Implementation), Zero:** 5 замечаний закрыты по исправленному документу; новых и открытых расхождений 0; пользовательская Qwen 0.8B dLLM configuration отражена корректно.

**Граница допуска:** APPROVED означает верность спецификации этому PDF. Это не подтверждение наличия уникальной полной авторской реализации: до её буквального воспроизведения остаются обозначенные G01–G24 и внутреннее несоответствие stated sign-test procedure некоторым опубликованным p-values. β, M, optimizer/batch, детали head/masks/EOS, filtering IDs и sampler settings нельзя получать из допуска ревьюера. Замена модели также не позволяет требовать повторения авторских численных результатов.
