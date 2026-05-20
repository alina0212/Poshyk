# SemanticUA

[![Tests](https://github.com/alina0212/Poshyk/actions/workflows/tests.yml/badge.svg)](https://github.com/alina0212/Poshyk/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Веб-застосунок для **семантичного пошуку** по текстових документах (PDF, DOCX, TXT, MD).
Працює локально, без зовнішніх API. Українська мова — пріоритетна.

> Дипломна робота, КНУ ім. Тараса Шевченка, ФІТ, кафедра ПСТ. 2026.

---

## ✨ Що вміє

- 🔍 **Семантичний пошук** — знаходить документи за змістом, а не тільки за словами
- ⚡ **BM25 fallback** — класичний пошук за ключовими словами, працює навіть без ML
- 👥 **Multi-user** — кожен користувач має ізольований документ + індекс
- 📄 **Підтримка форматів:** PDF, DOCX, TXT, MD (до 32 MB)
- 🎨 **Темна UI** з візуалізацією пайплайну і підсвічуванням збігів
- 🚀 **Production-ready:** Gunicorn, logging, тести, CI

---

## 🏗 Архітектура

```
Документ ──► chunks ──► E5 Bi-Encoder ──► FAISS ANN ──► top-60 кандидатів
                                                              │
                                                              ▼
                                                       BGE Cross-Encoder
                                                              │
                                                              ▼
                                                    Адаптивний CE-gate
                                                              │
                                                              ▼
                                                       Сортування + UI
```

**Двостадійний пайплайн** (industry-standard):

| Стадія | Модель | Швидкість | Точність |
|--------|--------|-----------|----------|
| **Retrieval** | `intfloat/multilingual-e5-base` (768-dim) | швидко (мс) | висока recall |
| **Reranking** | `BAAI/bge-reranker-v2-m3` | повільно (50мс/пара) | висока precision |
| **Fallback** | BM25 (TF-IDF, k1=1.5, b=0.75) | миттєво | baseline |

---

## 🚀 Швидкий старт

### Встановлення

```bash
git clone https://github.com/alina0212/Poshyk.git
cd Poshyk
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

При першому запуску завантажаться ML-моделі (~850 MB, кешуються у `~/.cache/huggingface`).

### Запуск

```bash
# Development
python app.py
# → http://localhost:5000

# Production (Gunicorn, 4 воркери × 2 потоки)
./gunicorn.sh
```

---

## 🧪 Тести

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

27 тестів покривають:
- `tokenize()` — українська/англійська токенізація, стоп-слова
- `BM25` — пошук, IDF, морфологічні варіанти
- `cleanup_idle` — видалення неактивних сесій (multi-user)
- `semantic_search` — контракт API, fallback при падінні ML

CI запускається автоматично на кожен push.

---

## 📊 Як використовувати

1. Відкрий http://localhost:5000
2. Перетягни файл (PDF/DOCX/TXT/MD) або клацни щоб обрати
3. Дочекайся індексації (~3-5 сек для 200 абзаців)
4. Введи запит у пошук
5. Переключай між **Семантичний** і **За словами** (BM25)
6. Подивись пайплайн зверху — bi/ce час і скільки кандидатів пройшло фільтр

---

## ⚙️ Конфігурація

Всі параметри з коментарями — у [`config.py`](config.py).

**Ключові:**
- `SEARCH_TOP_K_BI = 60` — скільки кандидатів іде у CE reranking
- `SEMANTIC_THRESHOLD = 0.503` — мінімальний CE-сигнал
- `QUERY_RELEVANCE_GATE = 50.4` — нижче цього → 0 результатів (нерелевантний запит)
- `CHUNK_MIN_LEN / MAX_LEN = 60 / 600` — розбиття документа

**Environment variables:**
- `SECRET_KEY` — для Flask cookie (production)
- `SESSION_TTL` — час життя idle-сесії (секунди, дефолт 3600)
- `LOG_LEVEL` — `DEBUG`/`INFO`/`WARNING`/`ERROR`
- `WORKERS` / `THREADS` — для Gunicorn

---

## 🏛 Структура проєкту

```
Poshyk/
├── app.py              # Flask entry point + cleanup-thread + logging
├── config.py           # параметри з блочними коментарями
├── core/
│   ├── state.py        # per-session state (RLock) + shared ML
│   ├── document.py     # extract_text + sliding-window chunking
│   ├── models.py       # load_models + FAISS build + per-session cache
│   ├── bm25.py         # BM25 клас, tokenize, IDF
│   └── search.py       # semantic_search + bm25_search + highlight
├── routes/
│   └── api.py          # /upload /search /status /clear — cookie sid
├── templates/
│   └── index.html      # темна UI з pipeline visualization
├── tests/              # pytest (27 тестів)
├── cache/{sid}/        # per-user FAISS-індекс на диск
├── gunicorn.sh         # production-запуск
└── .github/workflows/  # CI
```

---

## 🌐 Підтримувані мови

- 🇺🇦 **Українська** — основна, повна підтримка
- 🇬🇧 **English** — повна підтримка
- 🌍 Інші мови — частково (ML-моделі мультимовні, але BM25-токенізатор обмежений UA/EN-алфавітами)

---

## ⚠️ Відомі обмеження

1. **Слабка синонімія на українській** — BGE-reranker дає ±0.5% сигналу для пар типу "фото ↔ зображення". Жоден публічний мультимовний reranker станом на 2026 не вирішує це надійно. Зменшується якщо запит з основного терміна документа.
2. **In-memory FAISS** — кожен воркер тримає свої індекси в RAM. Для тисяч одночасних користувачів треба векторну БД (Qdrant/Weaviate).
3. **CE на CPU** — ~3 сек/запит. На GPU прискорилось би у 30×.
4. **PDF з картинками** — обробляється тільки витягнутий текст. Сканований PDF (без OCR-шару) даватиме порожній результат.

---

## 🛠 Технологічний стек

- **Backend:** Python 3.12+, Flask 3, Gunicorn
- **ML:** sentence-transformers, FAISS (CPU)
- **Models:** E5-base + BGE-reranker-v2-m3 (мультимовні)
- **Frontend:** vanilla JS, dark theme (Inter font, glassmorphism)
- **Testing:** pytest, GitHub Actions CI

---

## 📜 Ліцензія

MIT — використовуй вільно для навчальних і комерційних цілей.

---

## 👤 Автор

**Коврига Аліна** · ІПЗ-42 · 2026
Науковий керівник: к.ф.-м.н., доцент Духновська К. К.
