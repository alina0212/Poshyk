"""
Тести per-session state та cleanup.
"""
import time

from core.state import (
    _sessions,
    cleanup_idle,
    clear_state,
    get_state,
    session_count,
)


def test_get_state_creates_new():
    state = get_state("user_a")
    assert state["chunks"] == []
    assert state["index"] is None
    assert state["filename"] is None
    assert "last_active" in state


def test_get_state_returns_same_for_same_sid():
    s1 = get_state("user_b")
    s1["chunks"] = ["doc1"]
    s2 = get_state("user_b")
    assert s2["chunks"] == ["doc1"]


def test_get_state_isolated_per_sid():
    """Різні sid → ізольовані state. Це ядро multi-user."""
    a = get_state("alice")
    b = get_state("bob")
    a["chunks"] = ["alice doc"]
    b["chunks"] = ["bob doc"]
    assert get_state("alice")["chunks"] == ["alice doc"]
    assert get_state("bob")["chunks"] == ["bob doc"]


def test_get_state_updates_last_active():
    s1 = get_state("user_c")
    t1 = s1["last_active"]
    time.sleep(0.01)
    s2 = get_state("user_c")
    assert s2["last_active"] > t1


def test_clear_state_removes_session():
    get_state("user_d")
    assert session_count() == 1
    clear_state("user_d")
    assert session_count() == 0


def test_clear_state_idempotent():
    """Подвійний clear не падає."""
    clear_state("nonexistent")
    clear_state("nonexistent")


def test_cleanup_idle_removes_old_sessions():
    """Сесія без активності > ttl має видалитись."""
    state_old = get_state("idle_user")
    state_old["last_active"] = time.time() - 7200  # 2 години тому

    state_active = get_state("active_user")
    # last_active щойно встановлено через get_state

    removed = cleanup_idle(ttl_seconds=3600)  # видалити > 1 год
    assert removed == 1
    assert session_count() == 1
    # Активний користувач лишився
    assert "active_user" in _sessions
    assert "idle_user" not in _sessions


def test_cleanup_idle_keeps_recent():
    get_state("fresh_user")
    removed = cleanup_idle(ttl_seconds=3600)
    assert removed == 0
    assert session_count() == 1


def test_cleanup_idle_multiple():
    """Кілька старих сесій видаляються одночасно."""
    for i in range(5):
        s = get_state(f"old_{i}")
        s["last_active"] = time.time() - 10000

    get_state("new_user")

    removed = cleanup_idle(ttl_seconds=3600)
    assert removed == 5
    assert session_count() == 1
