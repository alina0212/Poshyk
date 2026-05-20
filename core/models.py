"""
Завантаження shared ML-моделей і побудова per-session FAISS-індексу.

Bi-Encoder + Cross-Encoder завантажуються один раз і використовуються
усіма користувачами (read-only).
FAISS-індекс і чанки — окремі для кожної сесії, кешуються на диск
у cache/{session_id}/.
"""

import json
import logging
import os
import time

from config import (
    BI_ENCODER_MODEL, CE_MODEL, CE_MAX_LENGTH,
    CACHE_DIR, E5_PASSAGE_PREFIX, MODEL_VERSION,
)
from core.state import ml

log = logging.getLogger(__name__)


# ── Шляхи кешу для сесії ─────────────────────────────

def _session_cache_dir(sid: str) -> str:
    path = os.path.join(CACHE_DIR, sid)
    os.makedirs(path, exist_ok=True)
    return path


def _faiss_path(sid: str) -> str:
    return os.path.join(_session_cache_dir(sid), "faiss.index")


def _meta_path(sid: str) -> str:
    return os.path.join(_session_cache_dir(sid), "session.json")


# ── Збереження / відновлення per-session ─────────────

def save_cache(sid: str, state: dict) -> None:
    """Серіалізує FAISS-індекс та чанки сесії на диск."""
    import faiss
    faiss.write_index(state["index"], _faiss_path(sid))
    with open(_meta_path(sid), "w", encoding="utf-8") as f:
        json.dump({
            "chunks":        state["chunks"],
            "meta":          state["meta"],
            "filename":      state["filename"],
            "model_version": MODEL_VERSION,
        }, f, ensure_ascii=False)
    log.info("[%s] cache saved (FAISS + session.json)", sid[:8])


def load_cache(sid: str, state: dict) -> bool:
    """Повертає True якщо кеш сумісний з поточною bi-моделлю."""
    import faiss
    faiss_p = _faiss_path(sid)
    meta_p  = _meta_path(sid)
    if not (os.path.exists(faiss_p) and os.path.exists(meta_p)):
        return False
    try:
        with open(meta_p, encoding="utf-8") as f:
            data = json.load(f)

        if data.get("model_version") != MODEL_VERSION:
            log.warning("[%s] bi-model changed, dropping stale cache", sid[:8])
            for p in (faiss_p, meta_p):
                try: os.remove(p)
                except FileNotFoundError: pass
            return False

        state["index"]    = faiss.read_index(faiss_p)
        state["chunks"]   = data["chunks"]
        state["meta"]     = data.get("meta", {})
        state["filename"] = data.get("filename")
        log.info("[%s] cache restored (%d chunks)", sid[:8], len(data["chunks"]))
        return True
    except Exception as e:
        log.warning("[%s] cache load failed: %s", sid[:8], e)
        return False


def delete_cache(sid: str) -> None:
    """Видаляє кеш конкретної сесії (при /clear або cleanup)."""
    for path in (_faiss_path(sid), _meta_path(sid)):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
    cache_dir = os.path.join(CACHE_DIR, sid)
    try:
        os.rmdir(cache_dir)
    except OSError:
        pass


# ── Завантаження shared ML-моделей ───────────────────

def load_models() -> None:
    """
    Завантажує Bi-Encoder і Cross-Encoder у фоновому потоці.
    Це shared resources — ОДИН екземпляр на весь сервер.
    """
    try:
        from sentence_transformers import SentenceTransformer, CrossEncoder

        log.info("Loading Bi-Encoder (%s)…", BI_ENCODER_MODEL)
        ml["bi"] = SentenceTransformer(BI_ENCODER_MODEL)

        log.info("Loading Cross-Encoder (%s)…", CE_MODEL)
        ml["ce"] = CrossEncoder(CE_MODEL, max_length=CE_MAX_LENGTH)

        ml["status"] = "ready"
        log.info("ML models ready")

    except Exception as e:
        ml["status"] = "error"
        ml["error"]  = str(e)
        log.error("ML load failed: %s — BM25 fallback active", e)


# ── Побудова FAISS для конкретної сесії ──────────────

def build_faiss_index(sid: str, state: dict) -> None:
    """
    Кодує чанки користувача через Bi-Encoder і будує FAISS IndexFlatIP.
    Зберігає індекс у state та на диск.
    """
    import faiss

    chunks = state["chunks"]
    bi     = ml["bi"]
    if bi is None or not chunks:
        return

    log.info("[%s] FAISS encoding %d paragraphs…", sid[:8], len(chunks))
    t0 = time.time()

    # E5 вимагає "passage: " префікс
    prefixed = [E5_PASSAGE_PREFIX + c for c in chunks]
    embs = bi.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
    dim  = embs.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embs.astype("float32"))

    state["index"] = index
    state["embs"]  = embs
    log.info("[%s] FAISS ready in %.0f ms", sid[:8], (time.time() - t0) * 1000)

    save_cache(sid, state)
