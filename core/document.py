"""
Читання документів і розбивка тексту на чанки.

Підтримувані формати: TXT, PDF, DOCX/DOC, MD
"""

import os
import re

from config import CHUNK_MIN_LEN, CHUNK_MAX_LEN


# ── Зчитування файлів ─────────────────────────────────

def read_txt(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def read_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
    except ImportError:
        return "[pypdf не встановлено. Встановіть: pip install pypdf]"


def read_docx(path: str) -> str:
    try:
        import docx
        doc = docx.Document(path)
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        return "[python-docx не встановлено. Встановіть: pip install python-docx]"


def extract_text(path: str, filename: str) -> str:
    """Вибирає потрібний reader за розширенням файлу."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return read_pdf(path)
    if ext in (".docx", ".doc"):
        return read_docx(path)
    return read_txt(path)


# ── Розбивка на чанки ─────────────────────────────────

def _sliding_window(text: str, max_len: int, overlap: int = 80) -> list[str]:
    """
    Ріже занадто довге речення на вікна по словах з невеликим overlap-ом,
    щоб контекст не губився на стиках.
    """
    words = text.split()
    pieces: list[str] = []
    current: list[str] = []
    current_len = 0
    for w in words:
        if current_len + len(w) + 1 > max_len and current:
            pieces.append(" ".join(current))
            # залишаємо хвіст для overlap
            tail, tail_len = [], 0
            for tw in reversed(current):
                if tail_len + len(tw) + 1 > overlap:
                    break
                tail.insert(0, tw)
                tail_len += len(tw) + 1
            current = tail
            current_len = tail_len
        current.append(w)
        current_len += len(w) + 1
    if current:
        pieces.append(" ".join(current))
    return pieces


def split_into_chunks(
    text: str,
    min_len: int = CHUNK_MIN_LEN,
    max_len: int = CHUNK_MAX_LEN,
) -> list[str]:
    """
    Розбиває текст на смислові абзаци (chunks).
    Тришарова стратегія:
      1. Розбивка по \n\n (абзаци)
      2. Якщо абзац > max_len → нарізаємо по реченнях
      3. Якщо одне речення > max_len → sliding window по словах з overlap
    """
    raw = re.split(r"\n{2,}", text.strip())
    chunks: list[str] = []
    buf = ""

    def _flush_long(block: str) -> None:
        """Один блок > max_len → ріжемо по реченнях, надто довгі речення — вікнами."""
        sentences = re.split(r"(?<=[.!?])\s+", block)
        current = ""
        for sent in sentences:
            if len(sent) > max_len:
                # Спершу зберегти накопичене, потім розрізати довге речення
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(_sliding_window(sent, max_len))
                continue
            if len(current) + len(sent) + 1 <= max_len:
                current = (current + " " + sent).strip()
            else:
                if current:
                    chunks.append(current)
                current = sent
        if current:
            chunks.append(current)

    for para in raw:
        para = re.sub(r"\s+", " ", para).strip()
        if not para:
            continue

        buf = (buf + " " + para).strip() if buf else para

        if len(buf) >= min_len:
            if len(buf) > max_len:
                _flush_long(buf)
            else:
                chunks.append(buf)
            buf = ""

    if buf and len(buf) >= min_len:
        chunks.append(buf)

    return chunks
