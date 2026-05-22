"""
Flask-маршрути API з per-user ізоляцією.

Кожен браузер отримує signed cookie `sid` (UUID). За цим sid витягується
персональний state з core.state. Документи / FAISS / BM25 — окремі для
кожного користувача. ML-моделі (Bi, CE) — спільні.
"""

import os
import threading
import uuid

from flask import Blueprint, request, jsonify, send_from_directory, session as flask_session

from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS
from core.state import ml, get_state, clear_state, session_count
from core.document import extract_text, split_into_chunks
from core.bm25 import rebuild_bm25
from core.models import build_faiss_index, delete_cache, load_cache
from core.search import semantic_search, bm25_search, highlight, tokenize

api = Blueprint("api", __name__)

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "..", "templates")


# ── Сесія: cookie-based ──────────────────────────────

def _ensure_sid() -> str:
    """Повертає поточний session_id, створюючи якщо нема."""
    if "sid" not in flask_session:
        flask_session["sid"] = uuid.uuid4().hex
        flask_session.permanent = True
    return flask_session["sid"]


def _current_state() -> tuple[str, dict]:
    """sid + state. Якщо in-memory state порожній — пробуємо відновити з диску."""
    sid = _ensure_sid()
    state = get_state(sid)
    if not state["chunks"]:
        # Сервер міг рестартитися — пробуємо відновити з кешу
        if load_cache(sid, state) and ml["status"] == "ready":
            # FAISS відновили, треба ще побудувати BM25
            rebuild_bm25(state)
    return sid, state


# ── Upload ────────────────────────────────────────────

@api.route("/upload", methods=["POST"])
def upload():
    sid, state = _current_state()

    if "file" not in request.files:
        return jsonify({"error": "Файл не передано"}), 400

    f        = request.files["file"]
    filename = f.filename or "document.txt"
    ext      = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": f"Формат {ext} не підтримується. Використовуйте TXT, PDF або DOCX."
        }), 400

    tmp_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}{ext}")
    f.save(tmp_path)

    try:
        text   = extract_text(tmp_path, filename)
        chunks = split_into_chunks(text)

        if not chunks:
            return jsonify({"error": "Документ порожній або текст не вдалося витягти."}), 400

        # Старий кеш цієї сесії — видаляємо
        delete_cache(sid)
        state["filename"] = filename
        state["chunks"]   = chunks
        state["meta"] = {
            "chars":      len(text),
            "words":      len(text.split()),
            "paragraphs": len(chunks),
            "filename":   filename,
        }
        state["index"] = None
        state["embs"]  = None

        # BM25 — синхронно (швидко)
        rebuild_bm25(state)

        # FAISS — у фоні (повільно)
        if ml["status"] == "ready":
            threading.Thread(target=build_faiss_index, args=(sid, state), daemon=True).start()

        return jsonify({
            "ok":       True,
            "filename": filename,
            "meta":     state["meta"],
            "preview":  chunks[0][:200],
        })

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# ── Search ────────────────────────────────────────────

@api.route("/search")
def search():
    sid, state = _current_state()

    query = request.args.get("q", "").strip()
    algo  = request.args.get("algo", "semantic")

    if not state["chunks"]:
        return jsonify({"error": "no_document"})
    if not query:
        return jsonify({"results": [], "total": 0})

    import time
    t0     = time.time()
    tokens = tokenize(query)

    use_semantic = (
        algo == "semantic"
        and ml["status"] == "ready"
        and state["index"] is not None
    )

    if use_semantic:
        try:
            raw, pipeline = semantic_search(state, query)
            used_algo = "semantic"
        except Exception:
            raw, tokens = bm25_search(state, query)
            used_algo = "bm25_fallback"
            pipeline  = {}
        if used_algo == "semantic":
            results = [
                {
                    "id":        r["id"],
                    "text":      r["text"],
                    "snippet":   highlight(r["text"], tokens, r.get("sim_words")),
                    "bi_pct":    r["bi_pct"],
                    "ce_pct":    r["ce_pct"],
                    "bi_rank":   r["bi_rank"],
                    "ce_rank":   rank + 1,
                    "sim_words": r.get("sim_words", []),
                }
                for rank, r in enumerate(raw)
            ]
        else:
            results = [
                {
                    "id":      r["id"],
                    "text":    r["text"],
                    "snippet": highlight(r["text"], tokens),
                    "bm25":    r["bm25"],
                }
                for r in raw
            ]
    else:
        raw, tokens = bm25_search(state, query)
        used_algo = "bm25_fallback" if algo == "semantic" else "bm25"
        pipeline  = {}
        results   = [
            {
                "id":      r["id"],
                "text":    r["text"],
                "snippet": highlight(r["text"], tokens),
                "bm25":    r["bm25"],
            }
            for r in raw
        ]

    return jsonify({
        "results":  results,
        "total":    len(results),
        "query":    query,
        "tokens":   tokens,
        "algo":     used_algo,
        "time_ms":  round((time.time() - t0) * 1000, 1),
        "pipeline": pipeline,
        "doc_meta": state["meta"],
    })


# ── Status ────────────────────────────────────────────

@api.route("/status")
def status():
    sid, state = _current_state()
    return jsonify({
        "status":         ml["status"],
        "error":          ml["error"],
        "bi_ready":       ml["bi"] is not None,
        "ce_ready":       ml["ce"] is not None,
        "faiss_ready":    state["index"] is not None,
        "has_document":   len(state["chunks"]) > 0,
        "doc_meta":       state["meta"],
        "active_sessions": session_count(),
    })


# ── Clear ─────────────────────────────────────────────

@api.route("/clear", methods=["POST"])
def clear():
    sid, _ = _current_state()
    clear_state(sid)
    delete_cache(sid)
    return jsonify({"ok": True})


# ── Frontend ──────────────────────────────────────────

@api.route("/")
def index():
    _ensure_sid()  # одразу видаємо cookie на головній
    return send_from_directory(TEMPLATES_DIR, "index.html")
