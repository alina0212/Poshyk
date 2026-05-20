"""
Тести BM25-токенізації та пошуку.
"""
from core.bm25 import BM25, rebuild_bm25, tokenize


# ── tokenize() ────────────────────────────────────────

def test_tokenize_basic_ukrainian():
    assert tokenize("Семантичний пошук") == ["семантичний", "пошук"]


def test_tokenize_filters_stop_words():
    # "та", "в", "для" — стоп-слова, мають відсіятись
    tokens = tokenize("Семантичний пошук та фільтр в документах для користувача")
    assert "та" not in tokens
    assert "в" not in tokens
    assert "для" not in tokens
    assert "семантичний" in tokens
    assert "користувача" in tokens


def test_tokenize_filters_short_tokens():
    # Токени довжиною 1 символ відкидаються
    tokens = tokenize("я зробив це")
    assert "я" not in tokens
    assert "зробив" in tokens


def test_tokenize_mixed_languages():
    tokens = tokenize("Python програма works on macOS")
    assert "python" in tokens
    assert "програма" in tokens
    assert "works" in tokens
    assert "macos" in tokens


def test_tokenize_normalizes_case():
    assert tokenize("Зображення") == tokenize("ЗОБРАЖЕННЯ") == tokenize("зображення")


def test_tokenize_empty_string():
    assert tokenize("") == []
    assert tokenize("   ") == []


# ── BM25 пошук ───────────────────────────────────────

CORPUS = [
    "Семантичний пошук дозволяє знаходити документи за змістом.",
    "BM25 — класичний алгоритм ранжування на основі TF-IDF.",
    "Cross-Encoder оцінює пари запит-документ точніше за Bi-Encoder.",
    "Зображення обробляються нейронною мережею SRGAN для підвищення якості.",
    "Програмне забезпечення працює на macOS, Linux та Windows.",
]


def test_bm25_finds_exact_match():
    bm25 = BM25(CORPUS)
    results = bm25.search("BM25")
    assert len(results) > 0
    # Документ 1 містить "BM25" — має бути на першому місці
    top_id = results[0][0]
    assert top_id == 1


def test_bm25_finds_morphological_variants():
    bm25 = BM25(CORPUS)
    # "зображення" → знаходить документ де воно є
    results = bm25.search("зображення")
    assert any(doc_id == 3 for doc_id, _ in results)


def test_bm25_empty_for_unknown_word():
    bm25 = BM25(CORPUS)
    results = bm25.search("ксилофон")
    # Слова немає в корпусі — нуль результатів зі скором > 0
    assert len(results) == 0


def test_bm25_idf_for_rare_word_high():
    bm25 = BM25(CORPUS)
    # "srgan" зустрічається 1 раз із 5 — IDF високий
    # "семантичний" зустрічається 1 раз із 5 — теж високий
    idf_rare = bm25.idf("srgan")
    idf_unknown = bm25.idf("ксилофон")
    assert idf_rare > 0
    assert idf_unknown == 0  # невідоме слово → 0


def test_bm25_rebuild_via_state():
    state = {"chunks": CORPUS, "bm25": None}
    rebuild_bm25(state)
    assert state["bm25"] is not None
    assert state["bm25"].N == len(CORPUS)


def test_bm25_score_order():
    """Документ з більшою концентрацією токену запиту має вищий скор."""
    bm25 = BM25(CORPUS)
    results = bm25.search("BM25 алгоритм")
    # Документ 1 містить обидва токени — топ
    assert results[0][0] == 1
