import pytest


@pytest.fixture(autouse=True)
def classical_text_detector(monkeypatch):
    """Synthetic tests must not depend on whether PaddleOCR happens to be installed."""
    monkeypatch.setenv("SKETCH_TEXT_DETECTOR", "classical")
