"""
Спільні фікстури для всіх тестів.
"""
import pytest

from core.state import _sessions


@pytest.fixture(autouse=True)
def clear_sessions_between_tests():
    """Кожен тест отримує чистий state — без артефактів від попередніх."""
    _sessions.clear()
    yield
    _sessions.clear()
