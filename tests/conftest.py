import pytest

from s2c.reading.service import DEFAULT_CACHE


@pytest.fixture(autouse=True)
def _isolated_reading(tmp_path, monkeypatch):
    """Every test gets its own reading log and an empty shared crop cache."""
    monkeypatch.setenv("READING_LOG", str(tmp_path / "reading.jsonl"))
    DEFAULT_CACHE.clear()
    yield
    DEFAULT_CACHE.clear()


@pytest.fixture(autouse=True)
def _no_trocr_download(monkeypatch):
    """default_pipeline() warms TrOCR in a background thread; a unit test must never hit the network or
    load 1.3 GB of weights. `-m gpu` tests load the model directly instead of through warm(), so they
    still get a real load (see tests/reading/test_trocr.py)."""
    monkeypatch.setattr("s2c.reading.trocr.TrocrReader.warm", lambda self: None)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
