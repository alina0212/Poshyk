"""
Пошукові функції:
  - semantic_search   — Bi-Encoder + FAISS + Cross-Encoder reranking
  - bm25_search       — пошук через BM25
  - highlight         — підсвічування токенів у сніпеті

Усі функції приймають state (per-user) як аргумент.
Query embedding cache — глобальний (ембединги нейтральні, не приватні).
"""

import math
import re
import time
from collections import Counter, OrderedDict, defaultdict

import numpy as np

from config import (
    SEARCH_TOP_K_BI, SEMANTIC_THRESHOLD, BI_MIN_PCT, E5_QUERY_PREFIX,
    QUERY_RELEVANCE_GATE, ADAPTIVE_CE_WINDOW,
)
from core.state import ml
from core.bm25 import tokenize, _STOP_WORDS

# Мінімальна схожість слова з запитом для підсвічування (0–1)
_WORD_SIM_THRESHOLD = 0.68
# Максимум унікальних слів для батч-кодування (кандидатів)
_MAX_WORDS_BATCH    = 40
# Скільки sim-слів максимум показати в одному результаті
_MAX_SIM_WORDS_PER_RESULT = 5

# ── Кеш ембедингів запитів (глобальний — ембединги не приватні) ──
_QUERY_CACHE_SIZE = 256
_query_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()


def _cached_query_embedding(query: str, bi) -> np.ndarray:
    """Повертає ембединг запиту з кешу або кодує і кешує."""
    key = query.strip().lower()
    if key in _query_cache:
        _query_cache.move_to_end(key)
        return _query_cache[key]
    # E5 вимагає "query: " префікс
    emb = bi.encode([E5_QUERY_PREFIX + query], normalize_embeddings=True).astype("float32")
    _query_cache[key] = emb
    if len(_query_cache) > _QUERY_CACHE_SIZE:
        _query_cache.popitem(last=False)
    return emb


def clear_query_cache() -> None:
    """Викликається на server start. Не за сесією — кеш глобальний."""
    _query_cache.clear()


# ── Підсвічування ─────────────────────────────────────

def highlight(text: str, exact_tokens: list[str], sim_words: list[str] | None = None) -> str:
    """
    Будує snippet ~300 символів навколо першого збігу.
    - exact_tokens → <mark>слово</mark>              (жовтий — точний збіг)
    - sim_words    → <mark class="sem">слово</mark>  (зелений — семантично схоже)
    """
    SNIPPET_LEN = 300
    PREFIX      = 80

    match_pos = -1
    valid_tokens = [t for t in exact_tokens if t]
    if valid_tokens:
        pattern = "|".join(re.escape(t) for t in valid_tokens)
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            match_pos = m.start()

    if match_pos > PREFIX:
        start = match_pos - PREFIX
        end   = start + SNIPPET_LEN
        prefix_dots = "…" if start > 0 else ""
        suffix_dots = "…" if end < len(text) else ""
        snippet = prefix_dots + text[start:end] + suffix_dots
    else:
        snippet = text[:SNIPPET_LEN] + ("…" if len(text) > SNIPPET_LEN else "")

    exact_lower = {t.lower() for t in exact_tokens}

    if sim_words:
        extra = [w for w in sim_words if w.lower() not in exact_lower]
        for w in sorted(extra, key=len, reverse=True):
            snippet = re.sub(
                f"({re.escape(w)})", r'<mark class="sem">\1</mark>',
                snippet, flags=re.IGNORECASE,
            )

    for t in sorted(exact_tokens, key=len, reverse=True):
        snippet = re.sub(
            f"({re.escape(t)})", r"<mark>\1</mark>",
            snippet, flags=re.IGNORECASE,
        )

    return snippet


# ── Конвертація скорів ────────────────────────────────

def ce_to_percent(raw_score: float) -> float:
    return round(1.0 / (1.0 + math.exp(-raw_score)) * 100, 1)


def bi_to_percent(cosine: float) -> float:
    return round(max(cosine, 0.0) * 100, 1)


# ── Семантично схожі слова ────────────────────────────

def _assign_sim_words_batch(
    q_emb,
    results: list[dict],
    bi,
    bm25,
    threshold: float = _WORD_SIM_THRESHOLD,
) -> None:
    """Знаходить семантично схожі слова для всіх результатів за один батч."""
    word_to_idxs: dict[str, list[int]] = defaultdict(list)
    word_counter: Counter = Counter()
    for idx, r in enumerate(results):
        tokens = re.findall(r"[а-яіїєґa-zA-Z]{3,}", r["text"].lower())
        word_counter.update(t for t in tokens if t not in _STOP_WORDS)
        for w in set(tokens):
            if w not in _STOP_WORDS:
                word_to_idxs[w].append(idx)

    # Топ-N за tf*idf — характерні (рідкісні в корпусі) важливіші за бойлерплейт.
    if bm25 is not None:
        scored = [(w, tf * bm25.idf(w)) for w, tf in word_counter.items()]
        scored.sort(key=lambda x: -x[1])
        all_words = [w for w, score in scored[:_MAX_WORDS_BATCH] if score > 0]
    else:
        all_words = [w for w, _ in word_counter.most_common(_MAX_WORDS_BATCH)]

    for r in results:
        r["sim_words"] = []

    if not all_words:
        return

    try:
        prefixed = [E5_QUERY_PREFIX + w for w in all_words]
        word_embs = bi.encode(prefixed, normalize_embeddings=True)
    except Exception:
        return
    sims = (word_embs @ q_emb.T).flatten()

    per_result: list[list[tuple[str, float]]] = [[] for _ in results]
    for word, sim, idxs in zip(all_words, sims, [word_to_idxs[w] for w in all_words]):
        if float(sim) >= threshold:
            for idx in idxs:
                per_result[idx].append((word, float(sim)))

    for idx, items in enumerate(per_result):
        items.sort(key=lambda x: -x[1])
        results[idx]["sim_words"] = [w for w, _ in items[:_MAX_SIM_WORDS_PER_RESULT]]


# ── Семантичний пошук ─────────────────────────────────

def semantic_search(
    state: dict,
    query: str,
    top_k_bi: int = SEARCH_TOP_K_BI,
    threshold: float = SEMANTIC_THRESHOLD,
) -> tuple[list[dict], dict]:
    """Повертає (результати, метрики_пайплайну) для сесії state."""
    chunks = state["chunks"]
    index  = state["index"]
    bi     = ml["bi"]
    ce     = ml["ce"]

    if index is None or bi is None:
        return [], {}

    # ① Bi-Encoder + FAISS
    t0 = time.time()
    try:
        q_emb = _cached_query_embedding(query, bi)
        scores, idxs = index.search(q_emb, min(top_k_bi, len(chunks)))
    except Exception as exc:
        raise RuntimeError(f"Bi-Encoder/FAISS: {exc}") from exc
    t_bi = (time.time() - t0) * 1000

    cands = [
        {
            "id":       int(i),
            "text":     chunks[i],
            "bi_score": float(s),
            "bi_pct":   bi_to_percent(float(s)),
            "bi_rank":  rank + 1,
        }
        for rank, (i, s) in enumerate(zip(idxs[0], scores[0]))
        if i >= 0
    ]
    cands_before = len(cands)
    cands = [c for c in cands if c["bi_pct"] >= BI_MIN_PCT]
    cands_after_bi = len(cands)

    # ② Cross-Encoder
    t1    = time.time()
    pairs = [[query, c["text"][:400]] for c in cands]
    try:
        ce_scores = ce.predict(pairs)
        t_ce = (time.time() - t1) * 1000
        for c, cs in zip(cands, ce_scores):
            c["ce_score"] = float(cs)
            c["ce_pct"]   = ce_to_percent(float(cs))
    except Exception:
        t_ce = 0.0
        for c in cands:
            c["ce_score"] = c["bi_score"]
            c["ce_pct"]   = c["bi_pct"]

    # ③ Адаптивний CE-gate
    min_pct = threshold * 100
    if not cands:
        filtered = []
        adaptive_min = min_pct
    else:
        top_ce = max(c["ce_pct"] for c in cands)
        if top_ce < QUERY_RELEVANCE_GATE:
            filtered = []
            adaptive_min = QUERY_RELEVANCE_GATE
        else:
            adaptive_min = max(min_pct, top_ce - ADAPTIVE_CE_WINDOW)
            filtered = [c for c in cands if c["ce_pct"] >= adaptive_min]

    reranked = sorted(filtered, key=lambda x: -x["bi_pct"])

    # ④ Sim-слова (з per-session BM25 для IDF)
    _assign_sim_words_batch(q_emb, reranked, bi, state.get("bm25"))

    pipeline = {
        "bi_ms":        round(t_bi, 1),
        "ce_ms":        round(t_ce, 1),
        "candidates":   cands_before,
        "after_bi":     cands_after_bi,
        "bi_min_pct":   BI_MIN_PCT,
        "passed":       len(reranked),
        "threshold":    round(adaptive_min, 1),
    }
    return reranked, pipeline


# ── BM25-пошук ────────────────────────────────────────

def bm25_search(state: dict, query: str, top_k: int | None = None) -> tuple[list[dict], list[str]]:
    """Повертає (результати, токени_запиту) для сесії state."""
    chunks = state["chunks"]
    bm25   = state["bm25"]
    k      = top_k if top_k is not None else len(chunks)
    toks   = tokenize(query)

    if bm25 is None:
        return [], toks

    raw = bm25.search(query, top_k=k)
    results = [
        {"id": i, "text": chunks[i], "bm25": round(score, 4)}
        for i, score in raw
    ]
    return results, toks
