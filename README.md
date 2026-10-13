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
полные blocks и последний partial block. Prefix пересчитывается без cache,
в том числе prompt positions, которые входят в первый completion block.

Nucleus top-p .91, temperature, confidence threshold/forced maximum,
mask suppression и EOS trimming сохраняют предыдущие собственные conventions.
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
source ../.venv/bin/activate
python run.py --config config.yaml --mode smoke
python run.py --config config.yaml --mode train
python run.py --config config.yaml --mode eval --checkpoint runs/bd3lm/ИМЯ/checkpoint
python run.py --config config.yaml --mode train --resume runs/bd3lm/ИМЯ/checkpoint
tensorboard --logdir runs --host 127.0.0.1 --port 6006
```

Локальный запуск использует то же CLI после подготовки окружения и HF cache;
физический `gpu` задаётся YAML. Checkpoints этой модели нужно обучать заново:
MDLM adapters/resume checkpoints относятся к другой base revision и attention
protocol. Full resume продолжается из BD3LM checkpoint с его saved config/RNG.

## Модули и источники

Карта исходной модулизации и описание full resume сохранены в
[предыдущем README](docs/mdlm_modular_README.md).
BD3LM additions: `attention.py`, geometry kwargs в `scoring.py` и `model.py`,
absolute-block boundaries в `sampling.py`, block-size arguments у callers
в `training.py` и `evaluation.py`. Плоский YAML задаёт model ID/revision,
train/eval block size и остальные прежние настройки.

- [Model card BD3LM](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1)
- [Model code, pinned revision](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1/blob/60f77bba67ba94231c16f5fb29bc66c500b038f1/modeling_qwen3.py)
- [TraFL PDF](docs/32236_Beyond_Mode_Seeking_RL_T.pdf)
- [Adaptation record](evidence/bd3lm_adaptation.json)

## Статус проверки

Python/YAML syntax и import/call targets проверены статически.
Новые checks в `tests/test_bd3lm.py` описывают known staircase matrix и
partial/aligned prompt boundaries; старые synthetic denoisers принимают
новые attention kwargs. Тесты, real smoke, training и evaluation для BD3LM
не запускались по ранее указанному пользователем ограничению на тесты.
Исторический audit и numerical results в docs/evidence относятся к MDLM.
