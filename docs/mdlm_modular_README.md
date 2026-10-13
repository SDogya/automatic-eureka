# TraFL: версия с логическими модулями

Рабочая копия в `/home/sdogya/Documents/sirius_sota/TraFL_modular`, выделенная
из `TraFL_torch`. Вычисления, signatures, checkpoint format, YAML и порядок
действий исходного пайплайна сохранены. Изменены расположение функций, импорты
и docstrings: описаны формы tensors, device/RNG, gradients, side effects,
опубликованные правила и собственные conventions.

## Карта файлов

```text
TraFL_modular/
├── run.py                    # CLI entry point
├── config.yaml               # исходный плоский YAML
├── trafl/
│   ├── __init__.py
│   ├── runner.py             # исходная orchestration и training loop
│   ├── data.py               # dataset, chat tokenization, resume-aware batches
│   ├── sampling.py           # nucleus, block diffusion, train/eval wrapper
│   ├── rewards.py            # boxed numeric reward и EOS trimming
│   ├── masking.py            # antithetic masks
│   ├── scoring.py            # Eq. (4), masked-token mean, compile target
│   ├── objective.py          # centering, residual, loss, LR schedule
│   ├── model.py              # log Z head и prompt features
│   ├── training.py           # completion rewards и один optimizer update
│   ├── evaluation.py         # Pass@k, validation loss, eval artifacts
│   └── checkpoint.py         # JSON, atomic checkpoint, RNG restore
├── tests/test_trafl.py        # исходные численные проверки
├── docs/                     # статья, спецификация, историческая документация
├── evidence/                 # сохранённые подтверждения и проверка переноса
├── code_audit.md             # исторический аудит с оговоркой о версии
└── runs/                     # копии имеющихся результатов и checkpoint
```

`runner.main()` сохранён целиком: model/reference initialization, LoRA/head,
logging, evaluation и resume следуют прежнему порядку. Отдельных framework,
factory, новых configuration layers или дополнительных runtime abstractions нет.

## Запуск

Из корня проекта, с уже подготовленным окружением:

```bash
python run.py --config config.yaml --mode smoke
python run.py --config config.yaml --mode train
python run.py --config config.yaml --mode eval --checkpoint runs/ИМЯ/checkpoint
python run.py --config config.yaml --mode train --resume runs/ИМЯ/checkpoint
tensorboard --logdir runs --host 127.0.0.1 --port 6006
```

Для ручного запуска перенесённых тестов: `python -m pytest -q tests/`.
В этой работе тесты, training и evaluation не запускались.
Зависимости и базовая модель нужны те же: PyTorch, Transformers, PEFT, datasets,
huggingface_hub, PyYAML, TensorBoard; pytest для тестов. GPU index и доступность
HF cache задаются в исходном `config.yaml`, как в предыдущей версии.
Подробности алгоритма, окружения и full resume сохранены в
[исходном README](docs/monolithic_README.md).

## Проверка сохранения кода

Для всех 26 top-level functions/classes сравнили исходный и перенесённый AST
после удаления docstrings. Единственная правка внутри исполняемого тела —
`import trafl as equations` заменён на `from . import scoring as equations`:
`torch.compile` продолжает заменять kernel в том модуле, где `masked_score`
ищет `masked_token_mean`. Тела методов `LogZ` также входят в сравнение.

В тестах изменены только import paths, docstrings и цель `import run`:
теперь это `trafl.training`, чтобы monkeypatch `sample_group` действовал
на реальные globals `update_batch`. Тела тестов и synthetic policy совпадают.
YAML скопирован побайтово. Статически проверены синтаксис, имена module globals
и существование относительных import targets. Runtime equivalence отдельно
не проверялась. Карта всех перенесённых symbols и source hashes лежит в
[modularization.json](evidence/modularization.json).

Существующие checkpoints используют state_dict и tensors, поэтому перенос
Python-модулей не меняет их schema. Скопированный старый smoke checkpoint
содержит evaluation weights; его недостающий full optimizer/RNG context
от этого переноса не появляется.

Исторические reports и результаты в `docs/`, `evidence/`, `runs/` относятся
к исходным версиям. Они не являются новым runtime audit модульной версии.
