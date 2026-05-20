"""
Стан застосунку: розділено на shared ML-моделі та per-user сесії.

Архітектура:
  ml         — словник зі shared об'єктами (Bi-Encoder, Cross-Encoder).
               Завантажуються один раз, read-only після завантаження.
  _sessions  — словник {session_id: state}. Кожен користувач має свій
               документ, FAISS-індекс, BM25-двигун. Захищено RLock.
"""

import threading
import time


# ── Shared ML-моделі (read-only після старту) ─────────
ml: dict = {
    "bi":     None,   # SentenceTransformer
    "ce":     None,   # CrossEncoder
    "status": "loading",
    "error":  "",
}


# ── Per-user стан (приватний) ────────────────────────
_sessions: dict[str, dict] = {}
_lock = threading.RLock()


def _new_state() -> dict:
    return {
        "filename":    None,
        "chunks":      [],
        "meta":        {},
        "index":       None,   # faiss.IndexFlatIP
        "embs":        None,   # numpy array
        "bm25":        None,   # BM25 engine instance
        "last_active": time.time(),
    }


def get_state(sid: str) -> dict:
    """Повертає state поточної сесії. Створює якщо відсутній."""
    with _lock:
        if sid not in _sessions:
            _sessions[sid] = _new_state()
        _sessions[sid]["last_active"] = time.time()
        return _sessions[sid]


def clear_state(sid: str) -> None:
    with _lock:
        _sessions.pop(sid, None)


def cleanup_idle(ttl_seconds: int = 3600) -> int:
    """
    Видаляє сесії без активності понад ttl_seconds.
    Викликається періодично з фонового потоку.
    """
    cutoff = time.time() - ttl_seconds
    with _lock:
        idle = [sid for sid, s in _sessions.items() if s["last_active"] < cutoff]
        for sid in idle:
            del _sessions[sid]
        return len(idle)


def session_count() -> int:
    """Кількість активних сесій — для моніторингу."""
    with _lock:
        return len(_sessions)


def all_sids() -> list[str]:
    """Список усіх активних session_id — для cleanup-операцій."""
    with _lock:
        return list(_sessions.keys())
