"""
Тести структури результатів semantic_search.

Реальні моделі (E5 + BGE) ~850 МБ — у unit-тестах мокаємо.
Перевіряємо контракт: правильні ключі, типи, fallback при відсутності індексу.
"""
import numpy as np
import pytest

from core.bm25 import BM25
from core.search import bm25_search, semantic_search
from core.state import get_state, ml


# ── Mock-моделі ──────────────────────────────────────

class FakeBiEncoder:
    """Імітує SentenceTransformer.encode() — повертає випадкові вектори."""
    def encode(self, texts, normalize_embeddings=False, show_progress_bar=False):
        rng = np.random.default_rng(seed=42)
        return rng.random((len(texts), 8), dtype=np.float32)


class FakeCrossEncoder:
    """Імітує CrossEncoder.predict() — повертає logits."""
    def predict(self, pairs):
        # Перша пара отримує високий logit (релевантна), решта — біля нуля
        return np.array([2.5] + [0.0] * (len(pairs) - 1), dtype=np.float32)


class FakeFaissIndex:
    """Імітує faiss.IndexFlatIP.search() — повертає top-N індексів."""
    def __init__(self, n):
        self.n = n

    def search(self, query_emb, k):
        k = min(k, self.n)
        scores = np.array([[0.9 - i * 0.01 for i in range(k)]], dtype=np.float32)
        ids    = np.array([[i for i in range(k)]], dtype=np.int64)
        return scores, ids


# ── semantic_search ──────────────────────────────────

def test_semantic_search_no_index_returns_empty():
    """Без FAISS-індексу — порожній результат, без винятків."""
    state = get_state("test_no_index")
    state["chunks"] = ["doc1", "doc2"]
    # state["index"] лишається None

    results, pipeline = semantic_search(state, "тест")
    assert results == []
    assert pipeline == {}


def test_semantic_search_returns_correct_structure(monkeypatch):
    """Перевіряє контракт результату."""
    state = get_state("test_struct")
    state["chunks"] = [
        "Зображення обробляється системою.",
        "BM25 алгоритм пошуку.",
        "Cross-Encoder reranking.",
    ]
    state["index"] = FakeFaissIndex(n=3)

    monkeypatch.setitem(ml, "bi", FakeBiEncoder())
    monkeypatch.setitem(ml, "ce", FakeCrossEncoder())

    results, pipeline = semantic_search(state, "зображення")

    # Pipeline-метрики
    assert "bi_ms" in pipeline
    assert "ce_ms" in pipeline
    assert "candidates" in pipeline
    assert "passed" in pipeline
    assert "threshold" in pipeline
    assert isinstance(pipeline["candidates"], int)

    # Результати — список dict-ів з очікуваними полями
    if results:
        r = results[0]
        for key in ("id", "text", "bi_score", "bi_pct", "ce_score", "ce_pct", "bi_rank", "sim_words"):
            assert key in r, f"Missing key: {key}"
        assert isinstance(r["id"], int)
        assert isinstance(r["text"], str)
        assert isinstance(r["bi_pct"], (int, float))
        assert 0 <= r["bi_pct"] <= 100
        assert isinstance(r["sim_words"], list)


def test_semantic_search_raises_on_bi_failure(monkeypatch):
    """Якщо bi-encoder падає — RuntimeError для перехоплення в api.py."""
    state = get_state("test_bi_fail")
    state["chunks"] = ["a", "b"]
    state["index"] = FakeFaissIndex(n=2)

    class BrokenBi:
        def encode(self, *a, **kw):
            raise RuntimeError("GPU out of memory")

    monkeypatch.setitem(ml, "bi", BrokenBi())
    monkeypatch.setitem(ml, "ce", FakeCrossEncoder())

    with pytest.raises(RuntimeError, match="Bi-Encoder/FAISS"):
        semantic_search(state, "запит")


def test_semantic_search_degrades_on_ce_failure(monkeypatch):
    """Якщо ce-encoder падає — використовуємо bi-скори, пошук триває."""
    state = get_state("test_ce_fail")
    state["chunks"] = ["doc1", "doc2", "doc3"]
    state["index"] = FakeFaissIndex(n=3)

    class BrokenCE:
        def predict(self, pairs):
            raise RuntimeError("CE crashed")

    monkeypatch.setitem(ml, "bi", FakeBiEncoder())
    monkeypatch.setitem(ml, "ce", BrokenCE())

    # Не падає, повертає валідну структуру
    results, pipeline = semantic_search(state, "запит")
    assert isinstance(pipeline, dict)
    assert pipeline["ce_ms"] == 0.0  # CE час = 0 при падінні


# ── bm25_search ──────────────────────────────────────

def test_bm25_search_returns_results():
    state = get_state("test_bm25")
    state["chunks"] = ["alpha bravo", "charlie delta", "alpha echo"]
    state["bm25"] = BM25(state["chunks"])

    results, tokens = bm25_search(state, "alpha")
    assert "alpha" in tokens
    assert len(results) >= 2  # два чанки містять "alpha"
    # Кожен результат має ключі id, text, bm25
    for r in results:
        assert "id" in r
        assert "text" in r
        assert "bm25" in r


def test_bm25_search_without_index():
    state = get_state("test_no_bm25")
    state["chunks"] = ["doc"]
    state["bm25"] = None
    results, tokens = bm25_search(state, "doc")
    assert results == []
