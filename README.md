# TraFL с Qwen3 BD3LM

Серверный проект: `/home/sirius03/test/trafl`, Git branch `bd3lm`, созданная
от `modular`. Локальные исходники: `/home/sdogya/Documents/sirius_sota/TraFL_bd3lm`.
Предыдущие версии сохранены в `main` (монолит) и `modular` (MDLM).

Основная policy и frozen reference:
`dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1`, revision
`60f77bba67ba94231c16f5fb29bc66c500b038f1`. Модель и tokenizer скачаны
в HF cache сервера. YAML использует `local_files_only: true` и
`output_dir: runs/bd3lm`.

## Адаптация block diffusion

Checkpoint использует тот же Qwen attention interface, что MDLM. Без явной
4D mask custom backbone остаётся полностью bidirectional; название модели
не включает block causality автоматически. Поэтому добавлен модуль
`trafl/attention.py`: query block видит свой блок и предыдущие, будущие
блоки скрыты. Position IDs — абсолютные и непрерывные, начиная с нуля.
Loader использует unpadded rows, EOS остаётся видимым context token.

Staircase mask передаётся во все forward paths:

- generation текущей policy;
- masked score текущей policy и reference;
- отдельный clean-prompt forward для detached log Z features.

Block size в scoring/head соответствует `train_block_size` или `eval_block_size`
текущего protocol. Значение 32 совпадает с model-card default и evaluation
статьи TraFL. В generation grid отсчитывается от начала всего prompt:
при P % block_size != 0 сначала заполняется остаток prompt block, затем
полные blocks и последний partial block. Завершённые blocks кэшируются; незавершённый suffix prompt пересчитывается
вместе с текущим completion block. Перед каждым новым rollout prefix cache
строится заново из текущих весов policy.

Nucleus top-p .91, temperature, confidence threshold/forced maximum,
mask suppression и EOS trimming остаются явными собственными conventions.
При первом EOS завершается текущий block, последующие blocks для этой строки
не генерируются. Для score/reward completion обрезается после первого EOS;
EOS видимый, исключён из масок. Дополнительный хвост не входит в score.
Никакого logits shift нет: predictions относятся к тем же masked positions.

Ядро TraFL сохранено: G5, K4, group reward centering, shared corruptions,
mean token score и mean K до square, BG normalization, frozen reference,
LoRA 128/128, detached two-layer log Z head и один joint optimizer update.
`objective.py`, reward/mask rules и full-resume schema не менялись.
Staircase architecture — адаптация разрешённой пользователем модели, а не
описание backbone LLaDA из оригинального эксперимента авторов.

## Запуск на сервере

```bash
cd /home/sirius03/test/trafl
git switch bd3lm
source ../a100-package-check/.venv/bin/activate
python run.py --config config.yaml --mode smoke
python run.py --config config.yaml --mode train
python run.py --config config.yaml --mode eval --checkpoint runs/bd3lm/ИМЯ/checkpoint
python run.py --config config.yaml --mode train --resume runs/bd3lm/ИМЯ/checkpoint
tensorboard --logdir runs --host 127.0.0.1 --port 6006
```

Локальный запуск использует то же CLI после подготовки окружения и HF cache;
физический `gpu` задаётся YAML. Checkpoints этой модели нужно обучать заново:
MDLM adapters/resume checkpoints относятся к другой base revision и attention
protocol. Full resume продолжается из checkpoint того же `training_protocol` с saved
config/RNG. Старый no-cache checkpoint продолжать на новом sampler запрещено:
для него сохранён коммит до оптимизации `1c8a70c`. Seeds новой nucleus реализации
дают другой поток draws, хотя categorical distribution остаётся прежней.

## Модули и источники

Карта исходной модулизации и описание full resume сохранены в
[предыдущем README](docs/mdlm_modular_README.md).
BD3LM additions: `attention.py`, geometry kwargs в `scoring.py` и `model.py`,
cached decoder/CUDA graphs в `cached_sampling.py`, legacy oracle в `sampling.py`,
batched scoring в `scoring.py`, block-size arguments у callers
в `training.py` и `evaluation.py`. Плоский YAML задаёт model ID/revision,
train/eval block size и остальные прежние настройки.

- [Model card BD3LM](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1)
- [Model code, pinned revision](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1/blob/60f77bba67ba94231c16f5fb29bc66c500b038f1/modeling_qwen3.py)
- [TraFL PDF](docs/32236_Beyond_Mode_Seeking_RL_T.pdf)
- [Adaptation record](evidence/bd3lm_adaptation.json)

## Производительность и проверка

В [config.yaml](config.yaml) настройки оптимизаций отделены от G/K:
`score_completion_batch_size: 5`, `score_microbatch: 20` позволяют A100 считать
все 5×4 corruptions батчем. На небольших GPU уменьшить до 1/1 или 1/4.
Completion chunk ограничивает память autograd; microbatch внутри chunk ограничивает
forward workspace, но графы всех его masks сохраняются до backward. KV cache
не применяется к differentiable scoring. Padding скрыт attention mask и не
участвует в нормировке. Среднее K вычисляется до square, denominator остаётся B×G.

`generation_compile` компилирует nucleus/confidence transition;
`generation_cuda_graph` убирает Python dispatch из модельных forwards.
Графы читают текущие LoRA веса после optimizer update; cache не переносится
между rollout. Первый захват/компиляция медленнее тёплого запуска. Лимит 32
shape entries предотвращает постоянный перезахват типичных длин 256/512;
при нехватке памяти уменьшить его или отключить graphs. Reference остаётся
отдельной замороженной моделью. TensorBoard дополнительно пишет
`perf/update_seconds` (время последнего логируемого update) и
`perf/peak_allocated_gib`.

Для предыдущего commit `20eaad5` прошли 38 проверок в серверном изолированном окружении: Eq.(4), BG normalization,
one optimizer step, reference freezing, ragged batches/empty masks, scalar/batched
scores и gradients, compiled/eager reduction, EOS stopping, partial prompt blocks,
cold sampler replay сохранённого RNG и CUDA graph после изменения LoRA весов.
Окружение: PyTorch 2.8.0+cu126, Transformers 4.57.0, PEFT 0.17.1,
dllm commit `ca176752fbceec49c6b4777a2c18ae88e4eb10ed`.
Существующее `../.venv` и работающие benchmarks не изменены.

[Реальный замер](evidence/cached_training_probe.json): Quadro RTX 4000, FP16,
prompt 15 tokens, completion 32, G5/K4, rank128. При temperature .6/top-p .91:
rollout 3.755 → 1.471 с после прогрева (**2.55×**); score forward
2.271 → .546 с (**4.16×**, no_grad, microbatch4). Greedy tokens совпадают;
максимальная разница score .000909 из-за FP16 batching.
Отдельный полный joint update с ненулевым log Z, длиной 8 и chunks1/1 прошёл,
reference не изменяется, LoRA gradients ненулевые; peak allocated 3.83 GiB.
Время первого такого update 13.44 с включает compilation.
Это короткая проверка, не замер полного GSM8K throughput. A100/BF16 ещё не
проверены. A100 YAML оставляет BF16; runner отвергает GPU без нативной поддержки.

Повторить probe на доступной карте (FP16 используется только для старого сервера):

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/performance_probe.py --dtype float16
python -m pytest tests -q
```

Полный Train/Eval одной policy сравнивать только с base evaluation того же
cached-nucleus protocol. Предыдущий fast pretrained benchmark без top-p cutoff
не является сопоставимым baseline. Threshold .9/cache и EOS stopping остаются
помеченными choices/adaptation: полного training decoder authors не публикуют.

## Одна общая policy на нескольких GPU

Активировать окружение из команд выше, выбрать карты и запустить:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
  run.py --config config.yaml --mode train
tensorboard --logdir runs/bd3lm --host 127.0.0.1 --port 6006
```

Для одного GPU остаётся `python run.py --config config.yaml --mode train`.
В `torchrun` выбор карт задаёт `CUDA_VISIBLE_DEVICES`, YAML `gpu` относится
к обычному запуску через Python. Каждая выбранная карта должна поддерживать
заданный dtype; BF16 требует нативной поддержки. Существующие процессы не
останавливаются. Этот запуск вручную выбирает GPU; автоматического поиска
свободных карт и Slurm нет.

Это synchronous data parallel на `torch.distributed`, без DDP wrapper:
каждый процесс генерирует G5 для своих prompts, считает K4 shared masks,
накапливает gradients локально и **одним flattened all-reduce** усредняет только
LoRA/head gradients. Затем clipping и общий optimizer update; frozen reference
не синхронизируется и не обучается. Такой прямой путь сохраняет streamed backward,
не вызывает DDP reducer на генерации и detached features, и обрабатывает даже
rank с только empty scores. Инициализация trainable weights broadcast один раз;
после каждого update все policy/head веса одинаковые. Параметры не обновляются
между rollout и loss одного update.

`prompt_batch_size` — **на одну GPU**: при 1 и четырёх ranks глобальный B=4,
20 fresh rollouts за update. Centering по пять ответов каждого prompt; локальный
loss нормируется на local B×G, mean gradients по ranks даёт global B×G.
LR/horizon автоматически не масштабируются. G/K, LoRA/head, decoder и reward
сохраняются. Это новый эксперимент с увеличенным B; batch авторов не опубликован.
Effective config записывает `distributed_world_size` и `global_prompt_batch_size`.

Общая seeded permutation разделяется на непересекающиеся local batches.
Неполный distributed batch отбрасывается (`distributed_drop_last: true`),
никакие examples не повторяются для padding. В следующей epoch новая permutation;
seeded shuffle общий, rollout/mask RNG независимые: base seed + rank×1000003.
DataLoader остаётся pinned, два workers на процесс. AMP/compile, nucleus transition,
completed-prefix cache и CUDA graphs из предыдущей версии сохранены.

Validation prompts тоже распределяются по ranks, без padding/повторов;
loss/Pass@k суммируются и делятся на фактическое общее число задач, включая
неравные evaluation shards. Rank zero пишет общий TensorBoard и объединённые
`answers-step*.json`; train JSONL и evaluation shards имеют suffix `rankN`.
Evaluation RNG зависит от rank, поэтому matched before/after evaluation требует
одинаковое число ranks и порядок задач.

Теги TensorBoard: `loss/train`, `loss/val` (также по длинам), `reward/train`,
`reward/val`, `reward/mixed_groups`, `score/difference`, `log_z/mean`, `lr`,
`eval/<length>/pass@k`, `perf/update_seconds`, `perf/peak_allocated_gib`,
`perf/prompts_per_second`. Train statistics усредняются по ranks; время и память
логируемого update — максимум по ranks, throughput использует глобальный B.
Синхронизация для timing выполняется только на logging boundary.

Resume выполняется той же командой `torchrun`, с тем же world size:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
  run.py --config config.yaml --mode train --resume runs/bd3lm/ИМЯ/checkpoint
```

Checkpoint сохраняет одну общую модель/head/optimizer/scheduler и runtime каждого
rank: RNG, epoch permutation, cursor DataLoader, позиции JSONL и running metrics.
Rank zero атомарно публикует state, остальные ждут. Resume обрезает каждый JSONL
до своего saved cursor; смена world size при full resume отклоняется. Старые
single-GPU checkpoints продолжаются на одном GPU. Multi-node/shared-storage
настройки и restart manager в этот MVP не входят.

Для этого изменения запускается только один интеграционный тест:
`tests/test_parallel.py::test_parallel_update_matches_full_batch`.
Он прошёл: **1 passed in 11.71s**. Использует два CPU/Gloo ranks и synthetic denoiser: сравнивает gradients,
веса и metrics с обычным B=2, проверяет непересекающиеся shards, empty-score rank,
checkpoint/RNG/optimizer replay и weighted evaluation на неравных shards.
Это проверка математики/координации, а не проверка NCCL или скорости нескольких GPU.
Полное обучение и дополнительные замеры в этой задаче не запускаются.

## Установка, скачивание и обучение одной командой

```bash
bash run.sh --gpus 0,1,2,3 --output "$HOME/trafl_bd3lm"
```

`run.sh` переиспользует bootstrap прежнего pretrained launcher и его
`requirements.lock`: Python 3.12, Torch 2.8.0+cu126, Transformers 4.57.0,
PEFT 0.17.1, TensorBoard, pinned dllm commit. При необходимости uv ставится
в выбранную папку. Без sudo и изменения системного Python.

В `--output` создаются `.venv/`, `models/` (HF cache), `datasets/`, `vendor/dllm/`,
`launches/` (логи, requested/effective YAML, список пакетов) и `runs/`
(checkpoints, ответы, TensorBoard). Модель указанной revision и GSM8K из текущего
YAML скачиваются **до** torchrun; существующие cache используются повторно.
Это тренировочный HF dataset, Kaggle benchmark не подключается к training.

По умолчанию установка и обучение идут под nohup, сохраняются после отключения
SSH. PID означает старт launcher, а не успешное завершение установки;
ход установки и реальный run directory видны в напечатанном `launcher.log`.
Для вывода в терминал добавить `--foreground`. GPU список определяет число ranks;
BF16 configuration требует нативной поддержки BF16 на всех выбранных GPU.

```bash
bash run.sh --gpus 0,1,2,3 --mode smoke --foreground
bash run.sh --gpus 0,1,2,3 --config my-config.yaml --output "$HOME/trafl_experiment"
bash run.sh --gpus 0,1,2,3 --resume "$HOME/trafl_bd3lm/runs/ИМЯ/checkpoint"
tail -f "$HOME/trafl_bd3lm/launches/ИМЯ/launcher.log"
"$HOME/trafl_bd3lm/.venv/bin/tensorboard" --logdir "$HOME/trafl_bd3lm/runs"
```

Resume использует saved model/dataset/config, сохраняет прежний training horizon
и требует то же число ranks. Скрипт блокирует одновременные запуски с одним
output root, чтобы pip sync не менял окружение работающего обучения.
Зависимости dllm namespace включают сторонние trainer/eval utilities;
они устанавливаются для импорта dllm, обучение остаётся нашим чистым PyTorch.
Для launcher выполнена только статическая проверка `bash -n`; установку,
скачивание и обучение при подготовке этого скрипта не запускали.
