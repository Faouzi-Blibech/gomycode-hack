import pytest

from s2c.reading.service import DEFAULT_CACHE


@pytest.fixture(autouse=True)
def _isolated_reading(tmp_path, monkeypatch):
    """Every test gets its own reading log and an empty shared crop cache."""
    monkeypatch.setenv("READING_LOG", str(tmp_path / "reading.jsonl"))
    DEFAULT_CACHE.clear()
    yield
    DEFAULT_CACHE.clear()
