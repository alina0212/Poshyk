"""
BM25 — класичний пошук за ключовими словами.

Використовується як основний алгоритм (якщо обрано вручну)
або як fallback, коли ML-моделі ще не завантажені.

Двигун зберігається у per-session state, не глобально.
"""

import logging
import math
import re
from collections import Counter, defaultdict

log = logging.getLogger(__name__)


# ── Стоп-слова ────────────────────────────────────────
_STOP_WORDS: set[str] = {
    "та", "в", "на", "з", "до", "і", "а", "або", "що", "як", "для",
    "це", "у", "є", "за", "не", "the", "a", "an", "is", "are", "of",
    "to", "and", "or", "with", "по", "від", "які", "який", "яка",
}


def tokenize(text: str) -> list[str]:
    """Токенізація: літери (укр + лат), без стоп-слів, мінімум 2 символи."""
    return [
        t for t in re.findall(r"[а-яіїєґa-z']+", text.lower())
        if t not in _STOP_WORDS and len(t) > 1
    ]


# ── Клас BM25 ─────────────────────────────────────────

class BM25:
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(corpus)
        self.tok_corpus = [tokenize(c) for c in corpus]
        self.lengths = [len(t) for t in self.tok_corpus]
        self.avgdl = sum(self.lengths) / max(self.N, 1)
        self.counts = [Counter(t) for t in self.tok_corpus]

        # Інвертований індекс: термін → список id документів
        self.iidx: dict[str, list[int]] = defaultdict(list)
        for i, cnt in enumerate(self.counts):
            for term in cnt:
                self.iidx[term].append(i)

    def _score(self, tokens: list[str], doc_id: int) -> float:
        cnt = self.counts[doc_id]
        dl  = self.lengths[doc_id]
        s   = 0.0
        for t in tokens:
            tf = cnt.get(t, 0)
            df = len(self.iidx.get(t, []))
            if tf == 0 or df == 0:
                continue
            idf  = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
            tf_n = tf * (self.k1 + 1) / (
                tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            )
            s += idf * tf_n
        return s

    def idf(self, term: str) -> float:
        """
        IDF як міра "характерності" слова: рідкісні в корпусі → високі.
        Використовується для відбору тематичних sim_words (а не бойлерплейту).
        """
        df = len(self.iidx.get(term, []))
        if df == 0:
            return 0.0
        return math.log(self.N / df)

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        toks  = tokenize(query)
        cands = {i for t in toks for i in self.iidx.get(t, [])}
        if not cands:
            cands = set(range(self.N))
        scored = [(i, self._score(toks, i)) for i in cands]
        scored = [(i, s) for i, s in scored if s > 0]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]


# ── Перебудова для конкретної сесії ───────────────────

def rebuild_bm25(state: dict) -> None:
    """Будує BM25-індекс для документа конкретної сесії."""
    state["bm25"] = BM25(state["chunks"])
    log.info("BM25 index built (%d paragraphs)", len(state["chunks"]))
