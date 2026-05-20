"""
SemanticUA — точка входу
========================
Multi-user семантичний пошук по документах.

Архітектура:
  • ML-моделі (E5 + BGE-reranker)  — shared, завантажуються один раз
  • Документ + FAISS + BM25         — per-user, ізольовані за cookie sid

Підтримує: TXT, PDF, DOCX, MD
Pipeline:   Bi-Encoder + FAISS + Cross-Encoder reranking
Fallback:   BM25 якщо моделі не завантажені

Локальний запуск:   python app.py
Production:         ./gunicorn.sh  (Gunicorn + воркери)
"""

import logging
import os
import secrets
import sys
import threading
import time
from datetime import timedelta

from flask import Flask

from config import MAX_CONTENT_LENGTH
from core.models import load_models
from core.state import cleanup_idle, session_count
from routes.api import api


# ── Логування ─────────────────────────────────────────
# Stdout-формат сумісний з gunicorn та системними journal/docker logs.
# Можна налаштувати рівень: LOG_LEVEL=DEBUG python app.py
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("semanticua")


# ── Ініціалізація Flask ───────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"]      = MAX_CONTENT_LENGTH
app.config["SECRET_KEY"]              = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=7)
app.config["SESSION_COOKIE_HTTPONLY"]  = True
app.config["SESSION_COOKIE_SAMESITE"]  = "Lax"

app.register_blueprint(api)


# ── Фонове завантаження ML-моделей ───────────────────
threading.Thread(target=load_models, daemon=True).start()


# ── Cleanup idle-сесій ───────────────────────────────
SESSION_TTL_SECONDS  = int(os.environ.get("SESSION_TTL", 3600))   # 1 година
CLEANUP_INTERVAL_SEC = 600  # кожні 10 хв


def _cleanup_loop():
    while True:
        time.sleep(CLEANUP_INTERVAL_SEC)
        removed = cleanup_idle(SESSION_TTL_SECONDS)
        if removed:
            log.info("Cleanup: removed %d idle sessions (active: %d)", removed, session_count())


threading.Thread(target=_cleanup_loop, daemon=True).start()


# ── Запуск (тільки для development) ──────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  SemanticUA — Document Search")
    print("  http://localhost:5000")
    print("  ⚠️  Development-сервер. Для production: ./gunicorn.sh")
    print("=" * 55)
    app.run(debug=False, port=5000, threaded=True)
