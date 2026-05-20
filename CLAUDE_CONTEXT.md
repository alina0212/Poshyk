# SemanticUA — контекст проєкту

Flask-застосунок семантичного пошуку по документах (дипломна робота, 2026).
Multi-user. Pipeline: **E5 Bi-Encoder + FAISS → BGE Cross-Encoder → адаптивний gate** + BM25 fallback.

**Запуск:**
- Dev: `python app.py` → http://localhost:5000
- Prod: `./gunicorn.sh` (4 воркери × 2 потоки = 8 паралельних запитів)

---

## Структура

```
Diploma1/
├── app.py              # Flask + secret_key + ML-thread + cleanup-thread
├── config.py           # параметри з блочними коментарями (що/чому/тонко)
├── core/
│   ├── state.py        # ml (shared) + _sessions[sid] (per-user, RLock)
│   ├── document.py     # extract_text() + split_into_chunks() з sliding window
│   ├── models.py       # load_models() + build_faiss_index(sid, state) + per-session disk cache
│   ├── bm25.py         # BM25 клас з .idf(), rebuild_bm25(state)
│   └── search.py       # semantic_search(state, q) + bm25_search(state, q) + highlight() + global LRU query cache
├── routes/
│   └── api.py          # /upload /search /status /clear — cookie-based sid через flask.session
├── templates/
│   └── index.html      # темна тема, pipeline visualization
├── cache/{sid}/        # per-user faiss.index + session.json
├── uploads/            # тимчасові файли при upload
├── gunicorn.sh         # production-команда
└── requirements.txt    # + gunicorn>=22.0.0
```

---

## Поточна конфігурація (config.py)

| Параметр | Значення | Призначення |
|----------|----------|-------------|
| `BI_ENCODER_MODEL` | `intfloat/multilingual-e5-base` | 768-dim, потребує "query: "/"passage: " префіксів |
| `CE_MODEL` | `BAAI/bge-reranker-v2-m3` | мультимовний, sigmoid 50-57% на укр. |
| `SEARCH_TOP_K_BI` | 60 | кандидатів FAISS → CE |
| `SEMANTIC_THRESHOLD` | 0.503 | абсолютний floor CE |
| `QUERY_RELEVANCE_GATE` | 50.4 | top_ce < gate → 0 результатів |
| `ADAPTIVE_CE_WINDOW` | 7.0 | беремо чанки в межах від top_ce |
| `BI_MIN_PCT` | 0 | вимкнено (E5 baseline ~80% для всього) |
| `CHUNK_MIN_LEN`/`MAX_LEN` | 60 / 600 | sliding window 80 при overflow |
| `_WORD_SIM_THRESHOLD` | 0.68 (search.py) | sim_words відсіч |
| `_MAX_SIM_WORDS_PER_RESULT` | 5 (search.py) | не флудити UI |
| `MODEL_VERSION` | f"{BI_ENCODER_MODEL}::v1" | інвалідація FAISS-кешу при зміні моделі |

---

## Ключові архітектурні рішення

### Per-user ізоляція (core/state.py)
- `ml` — shared dict з Bi/CE моделями (loaded once)
- `_sessions[sid]` — per-user: chunks, meta, index, embs, bm25, last_active
- `get_state(sid)` створює state ліниво, оновлює `last_active`
- `cleanup_idle(ttl)` видаляє сесії без активності > 1 год (запускається кожні 10 хв з app.py)

### Cookie-based sid (routes/api.py)
- `flask_session["sid"] = uuid.uuid4().hex` (signed cookie, 7 днів)
- `_current_state()` повертає `(sid, state)` для кожного route
- На рестарт: state в RAM губиться, але `cache/{sid}/` лишається → `_current_state()` робить `load_cache()` якщо `state["chunks"]` порожні

### Per-session disk cache (core/models.py)
- `cache/{sid}/faiss.index` (бінарний FAISS)
- `cache/{sid}/session.json` (chunks + meta + model_version)
- Інвалідація: якщо `model_version` не співпадає → видаляємо

### Адаптивний CE-gate (core/search.py)
- BGE на укр. видає скори в вузькому діапазоні 50-57%
- `top_ce = max(ce_pct)`. Якщо `top_ce < QUERY_RELEVANCE_GATE` → `[]`
- Інакше: `adaptive_min = max(SEMANTIC_THRESHOLD*100, top_ce - ADAPTIVE_CE_WINDOW)`
- Сортуємо за `bi_pct` (E5 розрізняє краще ніж BGE у вузькому діапазоні)

### TF×IDF для sim_words (core/search.py)
- Замість частоти у результатах → `tf_in_results × bm25.idf(word)`
- Виділяє характерні (рідкісні в корпусі) замість бойлерплейту
- Fallback на most_common якщо BM25 ще не побудовано

### Smart snippet (core/search.py:highlight)
- Якщо exact_token знайдено на позиції > 80 у тексті → центруємо вікно 300 символів навколо
- Інакше — показуємо з початку
- Усуває "BM25 знайшов, але користувач не бачить"

---

## Що працює зараз (поведінка)

| Запит | Результат | Чому |
|-------|-----------|------|
| "зображення" | 23 чанки ✓ | top_ce=57.1, BGE впевнено |
| "програма" | 8 чанків ✓ | top_ce=56.5 |
| "фото" | 3 чанки (синонім слабкий) | BGE слабко на укр. синонімах, top_ce=50.6 |
| "двері" | 0 ✓ | top_ce=50.0 < gate 50.4 |
| "стратегія" | 0 ✓ | top_ce=50.1 < gate 50.4 |

---

## Відомі обмеження (для дипломної праці)

1. **Тільки UA/EN** — BM25 regex `[а-яіїєґa-z']+`, інші алфавіти не токенізуються
2. **Слабка синонімія на укр.** — BGE-reranker-v2-m3 на українській синонімії ("фото"="зображення") дає лише ±0.5% сигналу. Жоден публічний мультимовний reranker станом на 2026 не вирішує це надійно.
3. **In-memory FAISS** — Не масштабується горизонтально, кожен воркер тримає свої сесії
4. **CE на CPU** — ~50мс/пара × 60 пар = ~3 сек/запит, обмежує throughput

---

## Виправлені баги (не повертати)
1. BM25 завжди 0 — фікс через `import core.bm25 as bm25_module` (історично, тепер per-state)
2. PatternError Python 3.14 у `highlight()` — заздалегідь фільтруємо sim_words від exact_tokens
3. Race condition FAISS при upload до завантаження моделей
4. Повільний пошук — `_assign_sim_words_batch()` одним батчем
5. Single-user state — тепер per-session з RLock
6. Snippet обрізав слово у середині чанку — `highlight()` центрує навколо матча

---

## Корисне для розуміння алгоритму

- **Bi-Encoder**: запит і чанки → 768-dim вектори (нормалізовані) → FAISS IndexFlatIP = cosine similarity
- **Cross-Encoder**: пара [query, chunk] → logit → sigmoid → %
- **E5 особливість**: anisotropic — baseline cos ~0.80 для будь-яких укр. пар. Тому `BI_MIN_PCT = 0`.
- **BGE особливість на укр.**: плоскі скори. Тому фактично використовуємо як binary gate.
- **Final ranking**: за `bi_pct`, не за `ce_pct` (BGE надто стискає скори).
- **BM25**: k1=1.5, b=0.75, IDF використовується ще для tf×idf зважування sim_words
