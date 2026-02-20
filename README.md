# Credit Default Risk Scoring

---

## Постановка задачи

Мой интерес к этой теме связан с тем, что недавно я подробно разбирался в причинах кризиса 2008 года и увидел, насколько важно корректно оценивать кредитный риск. Цель проекта – построить модель кредитного скоринга, которая по заявке клиента оценивает риск дефолта и выдаёт вероятность невозврата. И хотя тот мировой кризис был связан прежде всего с ипотечными кредитами, сам подход к управлению риском универсален: решения должны опираться на данные и количественные критерии, а качество модели – быть воспроизводимо проверяемым.

### Формат входных и выходных данных

**Входные данные:**

- Заявка на кредит, представленная в виде вектора признаков после препроцессинга.
- Формат: tensor<float>[1, D], где D – число признаков после преобразований.

**Выходные данные:**

- Вероятность дефолта для этой заявки (число от 0 до 1).
- Формат: tensor<float>[1, 1].
  Порог для получения класса из вероятности не задан и будет выбираться на валидационной выборке в зависимости от целевой метрики.

### Метрики

- **ROC-AUC** – основная метрика. Она показывает, насколько хорошо модель ранжирует заявки по уровню риска и отделяет более рискованные заявки от менее рискованных, не привязываясь к фиксированному порогу.
  Ожидаем получить значение заметно выше уровня случайного угадывания 0.5. Для базовой модели на табличных данных реалистичный ориентир – около 0.70–0.75. Точный целевой уровень уточним после первого бейзлайн-запуска.
- **LogLoss** – вспомогательная метрика. Она оценивает качество вероятностных предсказаний и сильнее штрафует уверенные ошибки, поэтому подходит для задачи, где на выходе требуется вероятность дефолта.
  Ожидаем значение ниже, чем у наивного бейзлайна “предсказывать одну и ту же вероятность для всех заявок” (т.е. константа, равная среднему TARGET на train). Конкретное целевое значение также уточним после вычисления этого бейзлайна.

### Валидация и тест

Используем стратифицированное разбиение исходного датасета, чтобы сохранить долю классов во всех выборках. Для воспроизводимости результатов зафиксируем random_seed, а также будем использовать одинаковое разбиение при каждом запуске. Данные разделим на train/val/test в пропорции 70/15/15.

### Датасет

- **Источник**: Kaggle, соревнование Home Credit Default Risk.
- **Состав данных**: основной датасет содержит главную таблицу заявок (train и test). Также есть дополнительные таблицы, но в базовой версии проекта будем использовать только основную таблицу заявок.
- **Количество сэмплов**: train – 307 511 заявок, test – 48 744 заявок.
- **Количество признаков**: 121 (без TARGET).
- **Объём**: около 180 МБ (train + test).
- **Возможные проблемы**: много категориальных признаков и пропусков (нужен аккуратный препроцессинг), возможен дисбаланс классов.

---

## Моделирование

### Бейзлайн

В качестве baseline решения реализована минимальная нейросеть для табличных данных – однослойный перцептрон.

**Архитектура**: один линейный слой (Linear) и выход на одно значение (вероятность дефолта). Это позволит максимально быстро получить стартовые метрики, с которыми будет сравниваться основная модель.

### Основная модель

Простая MLP для табличных данных на PyTorch Lightning.

**Архитектура**: два скрытых слоя (Linear + ReLU) и выходной слой на одно значение (вероятность дефолта). Для снижения переобучения добавлена базовая регуляризация (dropout или weight decay). Обучение будет проводиться с ранней остановкой по валидации.

---

## Внедрение

Модель оформлена в виде Python-пакета, который можно установить и использовать как библиотеку в других проектах. Пакет предоставляет собой API для загрузки сохранённых артефактов (веса модели и параметры препроцессинга) и получения предсказания по одной заявке:

- функция **predict_proba(features)** возвращает вероятность дефолта,
- функция **predict(features, threshold)** возвращает класс по порогу.

---

## Setup

### Установка зависимостей

```bash
poetry install
```

### Настройка pre-commit

```bash
poetry run pre-commit install
poetry run pre-commit run -a
```

### Данные

Данные не хранятся в git и управляются через DVC.

При желании можно скачать заранее вручную:

```bash
poetry run dvc pull data/application_train.csv
poetry run dvc pull data/application_test.csv
```

### MLflow (локальный запуск сервера)

Пример команды для поднятия MLflow сервера на `127.0.0.1:8080`:

```bash
poetry run mlflow server \
  --host 127.0.0.1 \
  --port 8080 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts
```

---

## Train

Обучение запускается через Hydra-конфиги из папки `configs/`.

### Baseline

```bash
poetry run python -m credit_scoring.train model=baseline_perceptron
```

### Основная модель

```bash
poetry run python -m credit_scoring.train model=mlp
```

### Примеры переопределения гиперпараметров

```bash
poetry run python -m credit_scoring.train model=mlp trainer.max_epochs=5 trainer.batch_size=512 trainer.lr=0.0005
```

### Результаты обучения

- Метрики, параметры и артефакты логируются в MLflow (tracking URI задаётся в `configs/train.yaml`).
- Графики сохраняются в папку `plots/` и также прикрепляются к MLflow-run.
- Артефакты для инференса сохраняются в `artifacts/`.

---

## Production preparation

Подготовка модели к использованию после обучения в этом проекте включает сохранение минимального набора артефактов, необходимых для повторяемого инференса.

Папка `artifacts/` после `train` содержит:

- `baseline_perceptron.pt` или `mlp.pt` — веса модели
- `preprocess.json` — параметры препроцессинга (список фич, медианы для NaN, параметры стандартизации)
- `model_config.json` — конфиг модели (имя, hidden_sizes, dropout, num_features)

Этого набора достаточно, чтобы:

- запускать CLI-инференс (`python -m credit_scoring.infer`)
- вызывать Python API (`credit_scoring.api.predict_proba / predict`)

---

## Infer

### 1) Command-line interface

Перед инференсом нужно, чтобы были артефакты в `artifacts/` (то есть сначала запустить `train`).

Запуск (по умолчанию берётся `request_id` из `configs/infer_params/default.yaml`):

```bash
poetry run python -m credit_scoring.infer model=mlp
```

Указать конкретный `request_id` и порог:

```bash
poetry run python -m credit_scoring.infer model=mlp infer_params.request_id=100005 infer_params.threshold=0.5
```

Что делает команда:

- проверяет наличие `artifacts/preprocess.json` и весов модели
- при необходимости скачивает
- берёт одну строку по `SK_ID_CURR == request_id`
- выводит `prob_default` и `pred_class` (с учётом `threshold`)

### 2) Python API

API находится в `credit_scoring/api.py`:

- `predict_proba(features, ...) -> float`
- `predict(features, threshold, ...) -> int`

Пример:

```python
from credit_scoring.api import predict_proba, predict

features = {
    "AMT_INCOME_TOTAL": 180000,
    "AMT_CREDIT": 600000,
    "AMT_ANNUITY": 25000,
}

p = predict_proba(features, model_name="mlp", artifacts_dir="artifacts")
y = predict(features, threshold=0.5, model_name="mlp", artifacts_dir="artifacts")

print("proba:", p)
print("class:", y)
```

---

## Repo structure

Примерная структура репозитория (часть папок появляется после запуска `train`/`infer`):

```text
.
├── credit_scoring/                 # python-пакет проекта
│   ├── api.py                      # predict_proba / predict
│   ├── train.py                    # входная точка обучения
│   ├── infer.py                    # входная точка инференса
│   ├── lightning_module.py         # шаги train/val/test, логирование метрик
│   ├── model.py                    # модели MLP и Perceptron
│   ├── dataset.py                  # Dataset / DataLoader-логика
│   ├── data.py                     # утилиты данных (DVC pull / fallback download, чтение CSV)
│   └── __init__.py
├── configs/                        # Hydra-конфиги (train/infer и группы параметров)
│   ├── train.yaml
│   ├── infer.yaml
│   ├── model/
│   │   ├── baseline_perceptron.yaml
│   │   └── mlp.yaml
│   ├── data/
│   │   └── home_credit.yaml
│   ├── preprocess/
│   │   └── numeric_only.yaml
│   ├── trainer/
│   │   └── default.yaml
│   └── infer_params/
│       └── default.yaml
├── data/                           # данные (в git лежат только *.dvc)
│   ├── application_train.csv.dvc
│   └── application_test.csv.dvc
├── artifacts/                      # артефакты после обучения
├── plots/                          # графики обучения
├── outputs/                        # выходы Hydra
├── mlruns/                         # локальные MLflow runs
├── mlartifacts/                    # локальные MLflow artifacts
├── mlflow.db                       # локальный backend-store MLflow
├── .pre-commit-config.yaml
├── pyproject.toml
├── poetry.lock
└── README.md
```
