# TraFL: Qwen diffusion + GSM8K

Рабочий каталог сервера: `/home/sirius03/test/trafl`. Два файла пайплайна,
один плоский YAML. Стек: PyTorch + Transformers + PEFT + datasets + TensorBoard.
Модель `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1` уже в HF cache,
revision `c8d24a3f4adaeef46881b450e1bf7d1005203bd7` зафиксирована;
повторное скачивание не требуется.
GSM8K скачивается автоматически, dataset SHA сохраняется в фактический конфиг.

## Запуск на сервере

```bash
cd /home/sirius03/test/trafl
../.venv/bin/python -m pytest -q test_trafl.py
../.venv/bin/python run.py --config config.yaml --mode smoke
../.venv/bin/python run.py --config config.yaml --mode train
../.venv/bin/python run.py --config runs/ИМЯ/checkpoint/config.yaml --mode eval --checkpoint runs/ИМЯ/checkpoint
../.venv/bin/tensorboard --logdir runs --host 127.0.0.1 --port 6006
```

Доступ к TensorBoard с локальной машины:

```bash
ssh -F /dev/null -N -L 6006:127.0.0.1:6006 sirius03@imladris.velkerr.ru
```

Открыть `http://127.0.0.1:6006`. Новые зависимости на **сервере** устанавливаются
через `/home/sirius03/.local/bin/uv pip install --python /home/sirius03/test/.venv/bin/python tensorboard`.
Никакая локальная установка не требуется. GPU задаётся полем `gpu` (physical index)
и привязывается по UUID, чтобы CUDA performance ordering не изменил выбор карты.
Перед запуском проверяется его доступность, чужие процессы не останавливаются.

## Строго опубликованная механика

Источник: `/home/sdogya/Documents/sirius_sota/polygon/32236_Beyond_Mode_Seeking_RL_T.pdf`,
Algorithm 1 (p14), Eq.(4) (p5), §4.1–4.2, Appendices C/D.5/E.

Свежие current-policy rollout: `G=5` на prompt. Binary reward центрируется только
внутри этой группы, без std scaling или пропуска all-equal groups. Для каждого
completion применяются `K=4` masks, одинаковые у policy и frozen reference.
Каждый score — mean исходных target-token log probabilities по masked positions;
затем mean по K. Residual: `S_policy - S_reference - beta * centered_reward + logZ`.
Loss: mean квадрата residual по B×G. Это square of mean scores, а не mean squares
per mask. Head — 2-layer MLP над detached mean-pooled prompt hidden states.

LoRA `rank=128, alpha=128`, cosine LR peak/min `6e-5/6e-6`, без дополнительной
SFT. До и после обучения используется общий evaluation protocol: stochastic
nucleus sampling, top-p `.91`, temperature `.6`, block32, lengths256/512, n16.
Pass@k — combinatorial form авторского unbiased estimator.

Веса обновляются **один раз** после полного online batch. Последовательные backward
по completion имеют divisor B×G и математически дают gradient этого же batch loss.
Current-policy scores trainable, reference no-grad/frozen. Rollout и reward
дискретные; REINFORCE correction к theoretical expectation не добавляется.

## Собственные решения для пробелов PDF

YAML comments отмечают PAPER / USER / OWN Gxx. Qwen — разрешённая замена LLaDA.
β, B, M, optimizer, targets/init LoRA, head width/init/feature source, seeds,
generation thresholds, EOS convention, subset selection и logging не выдаются за
опубликованные авторские defaults. Численные результаты статьи не являются
критерием приёмки smoke на другой модели.

- `G11`: две antithetic пары. Первый mask имеет uniform level 1..N и uniform
  subset без replacement; второй — complement. Full-mask endpoint даёт повтор
  full mask вместо empty complement. Обе marginal distributions остаются uniform
  по levels1..N; endpoint exception явно не назван авторской implementation.
- `G12`: N — количество non-EOS positions до/включая первого EOS; EOS видимый,
  unscored, suffix после EOS удалён. Empty-eligible score0 остаётся в B×G loss.
- `G09`: head features — final states current policy на **чистом prompt**, отдельный
  forward без completion. Нет head gradient к backbone.
- `G10/G20`: no-cache confidence-threshold Fast-dLLM v1 scheme, адаптированная
  к Qwen. Proposals — настоящий categorical nucleus после temperature scaling;
  confidence — исходная model probability. При низких confidence гарантирован
  один highest-confidence transfer, максимум length denoising calls. Mask token
  подавляется только в proposal distribution; scoring likelihood не фильтруется.
  EOS не завершает denoising досрочно: completion обрезается после denoising.
- `G15`: last balanced boxed expression, exact rational comparison (decimal,
  scientific notation, fraction), без numeric tolerance. Malformed/no-box→0.
- `G19`: последний checkpoint, без early stopping, checkpoint selection по reward
  и автоматического resume. Evaluation checkpoint содержит adapters/head/config.

Строковые поля `head_feature_source`, `head_pooling`, `mask_domain`,
`completion_eos`, `data_sampling`, `reward_mode`, `checkpoint_selection` и
`generation_cache/stop_on_eos` документируют фиксированные conventions первой
версии. Их изменение требует изменения соответствующей небольшой функции;
это не альтернативные backends, автоматически выбираемые по YAML. Численные
настройки и поддержанные bool knobs читаются кодом непосредственно.

## Производительность и артефакты

BF16 autocast + штатные model/SDPA kernels; `torch.compile` для FP32
log-softmax/gather/mean score kernel (включая backward), два compile workers;
pinned DataLoader с двумя workers и nonblocking copies. Для accuracy evaluation
score не заменяется training surrogate: correctness считается по text reward.
В generation остаётся одна необходимая GPU control sync для block completion;
нет per-token `.item()`/`.cpu()` и per-forward metric logging. CPU decoding
происходит один раз после каждого group generation и требуется reward extraction.

Полная Inductor-компиляция PEFT model forward на этом сервере затянулась более
10 минут до завершения первой evaluation. Поэтому `compile_scope` явно равен
`masked_token_mean`: numerical kernel компилируется, neural forward остаётся eager.
Это OWN G14 execution choice, а не изменение TraFL или обещание full-model compile.

TensorBoard scalars: `loss/train`, `loss/val` (также по length), `reward/train`,
`reward/val`, `reward/mixed_groups`, `score/difference`, `log_z/mean`, `lr`,
`eval/L/pass@k`. Writer создаётся один раз, flush по log interval, close в конце.

В `runs/ИМЯ`: requested/effective YAML, package versions, dataset SHA и выбранные
example IDs, train JSONL, evaluation answers/metrics, checkpoint и events.
Smoke сохраняет фактические overrides в config; основные параметры статьи не
меняются. Val loss использует отдельную G5 группу (первые пять из n16 draws).
Train/val subset взяты из разных официальных splits. `*_limit: null` включает
полный split. Paper-scale coding/ablations/judge/path diagnostic сюда не входят.

## Выполненная проверка

На сервере: **24 tests passed**. Smoke `runs/20261012T232032.630275Z-smoke`
завершился с exit0: 3 updates, G5/K4/rank128, 2 evaluation problems, n16, L256.
Training группы дали 0/5, 3/5 и 1/5 correct; все logged values конечны.
Checkpoint adapters (309 MiB) и head сохранены. TensorBoard events прочитаны:
train steps1/2/3, val steps0/3 и все требуемые scalar tags присутствуют.
Отдельный `--mode eval` успешно загрузил checkpoint и выполнил короткую
diagnostic evaluation: 1 problem, n5, L32, на GPU1. Это проверка CLI/reload,
не дополнительный paper benchmark.

На **двух** smoke problems Pass@1 до/после = **0.40625 / 0.3125**,
Pass@5 = **0.77072 / 0.65385**. Прирост качества не заявляется; smoke проверяет
работоспособность. Полный M100 run и L512 evaluation не запускались.
Локальные подтверждения: `/home/sdogya/Documents/sirius_sota/polygon/trafl/evidence/`;
независимый отчёт: `/home/sdogya/Documents/sirius_sota/polygon/trafl/code_audit.md`.
