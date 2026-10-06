# Emotion Classification of Vietnamese E-commerce Reviews (Tiki)

Многоклассовая классификация эмоций (anger, disgust, fear, happiness, sadness, surprise)
в отзывах покупателей вьетнамского маркетплейса Tiki. Сравниваются четыре нейросетевые
архитектуры (MLP, CNN, BiLSTM, CNN+BiLSTM), обученные с нуля, и трансформерный подход
(замороженный PhoBERT + MLP-голова).

## Результаты (тестовая выборка, 5 236 отзывов)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| **CNN+BiLSTM (Keras)** | **0.9339** | **0.9312** | **0.9289** | **0.9289** |
| MLP (Keras) | 0.9326 | 0.9267 | 0.9283 | 0.9268 |
| PhoBERT (frozen) + MLP | 0.9328 | 0.9293 | 0.9253 | 0.9266 |
| CNN (Keras) | 0.9322 | 0.9286 | 0.9264 | 0.9265 |
| BiLSTM (Keras) | 0.9305 | 0.9299 | 0.9247 | 0.9256 |

> Лучшая модель: CNN+BiLSTM (macro F1 0.9289).

## Структура репозитория

```
.
├── data/
│   ├── raw/                  # data.csv (26 382 отзыва)
├── notebooks/
│   ├── 01_deep_learning_models.ipynb       # EDA + обучение MLP/CNN/BiLSTM/CNN+BiLSTM
│   ├── 02_results_comparison.ipynb         # сравнение и визуализация ВСЕХ моделей
│   └── 03_phobert_feature_extraction.ipynb # frozen PhoBERT + MLP-голова
├── models/                   # сохранённые модели (.keras / .pt)
├── results/                  # метрики: CSV/JSON со всех экспериментов
└── report/                   # отчёт (LaTeX + pdf)
```

## Воспроизведение

1. Положите `data.csv` (колонки `content`, `lable`) в `data/raw/`.
2. Установите зависимости: `pip install underthesea tensorflow transformers torch scikit-learn seaborn ...`
3. Запустите ноутбуки по порядку: `01` → `03` (обучение; требуют GPU,
   пути внутри настроены на Kaggle — при локальном запуске поправьте
   константы `DATA_PATH`/`OUTPUT_DIR` в первых ячейках), затем `02`
   (сравнение; читает только `results/`, GPU не нужен).
4. Сводные метрики всех моделей — в `results/model_comparison.csv`.

## Протокол эксперимента

- Одинаковый сплит: 80/20, стратификация по классам, `random_state=42` — для всех моделей.
- Одинаковая предобработка: очистка → teencode → сегментация underthesea → стоп-слова.
- После предобработки 26 176 отзывов; распределение классов почти сбалансировано
  (дисбаланс 1.7:1, самый маленький класс — disgust, 2 762).
- Основная метрика — macro F1 (равный вес всех шести классов).

