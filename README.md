# TraFL на ДНК длины 8: точная проверка гипотез

Строки длины 8 над алфавитом `A,C,G,T` — всего 4^8 = 65 536, поэтому все вероятности
считаются точно, перебором. Генератор — маскированная диффузия: MLP по частично
раскрытой строке выдаёт распределения букв, на каждом шаге раскрывается одна случайная
скрытая позиция. Изучается дообучение TraFL (Ahmadi et al., arXiv 2605.13935) и то,
насколько выполняются допущения статьи.

## Задачи

- **Поттс, beta = 2** (`configs/potts.json`): log R(x) = beta · sum_{i<j} J_ij(x_i, x_j),
  J_ij(a, b) ~ N(0, 1), seed 42. На ней обучен претрейн.
- **Поттс + TFBind8** (`configs/potts_tfbind8.json`): обе части стандартизованы по всем
  строкам (среднее 0, дисперсия 1) и сложены, log R = 3.5 · (z_Potts + z_TFBind8).
  Цель TraFL с референсом p_ref = стартовая модель: p*(tau | x) ~ p_ref(tau | x) · R(y).

## Модель и обучение

- MLP `40 → 2^k → 2^k → 32`, GELU, вход — one-hot по `A,C,G,T,MASK`.
- **Претрейн** (`src/train.py`, `src/ablation.py`): обучение с учителем на 10 000 строках
  MH из Поттса, masked NLL, k = 1..7. Точный KL до цели Поттса на лучшей эпохе:
  1.09, 0.55, 0.14, 0.072, 0.056, 0.041, 0.031.
- **TraFL** (`src/rl/`): лосс статьи — суррогат log p через случайные маски с весом 1/l,
  референс на тех же масках, beta / u, голова log Z; контексты — свои выборки модели с
  маской 0.5, 256 контекстов по 5 достраиваний, 4 маски, Adam. Обучение до плато лосса.
  Есть и точный траекторный лосс (`src/rl/losses/trajectory_balance.py`).

## Измерения (`src/metrics/`)

- `trajectories.py` — точные p(tau | x), p(y | x), p(tau | x, y): перебор всех u! порядков
  раскрытия или динамика по 5^u частичным строкам.
- `trajectory_fit.py` — разложение KL(p*(tau|x) || p(tau|x)) = KL по финалам + KL по путям
  при данном y.
- `bias.py` — смещение градиента суррогата относительно точного (тождество Фишера).

## Структура

```
configs/        potts.json, potts_tfbind8.json
data/potts/     10 000 строк MH (data8.parquet), J (couplings.npy), диагностика MH
data/tfbind8/   score TFBind8 для всех строк, SOURCE.md — откуда и как получен
models/pretrain/mlp_k/   best.npz (старт и референс finetune), training.json
models/trafl/seed_s/{finetune,random_init}_k/   (не в git)
                step_XXXXX.npz, log_z_step_XXXXX.npz — каждые 250 шагов
                steps.csv — каждый шаг; evals.csv — точные метрики на чекпоинтах
                curves.png, settings.json
scripts/        trafl_sweep.sh, trajectory_fit.sh — параллельный запуск
src/  tests/
```

Сейчас в `models/trafl` прогоны k = 3, 5, 7, сиды 1–2: finetune до шага 20 000,
random_init до шага 22 500 (с 20 000 продолжен с постоянным learning rate до плато).

## Запуск

```bash
uv sync
uv run python -m pytest
uv run python -m src run --config configs/potts.json
uv run python -m src ablation --config configs/potts.json
scripts/trafl_sweep.sh
scripts/trajectory_fit.sh step_20000.npz
```

`run` проверяет существующий датасет и не перезаписывает результаты без `--overwrite`.
