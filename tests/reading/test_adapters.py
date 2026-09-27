import numpy as np

from s2c.reading import BatchFnReader, Crop, CropFnReader, ReaderResult, as_reader, read_timeout_s

C = Crop(np.zeros((10, 10, 3), np.uint8), (0, 0, 10, 10))


def test_a_batch_function_becomes_a_reader():
    r = BatchFnReader("qwen", lambda images: [("⌀6", 0.9)] * len(images))
    assert r.read([C, C]) == [ReaderResult(text="⌀6", confidence=0.9)] * 2
    assert (r.name, r.calibrated, r.cache_key) == ("qwen", False, None)
    assert BatchFnReader("q", lambda images: None).read([C]) is None


def test_a_crop_function_becomes_a_reader_with_clipped_confidence():
    r = CropFnReader("trocr", lambda image: ("60", 1.3))
    assert r.read([C]) == [ReaderResult(text="60", confidence=1.0)] and r.calibrated is True


def test_as_reader_passes_readers_through_and_wraps_functions():
    reader = CropFnReader("x", lambda image: ("1", 1.0))
    assert as_reader(reader, "ignored", calibrated=False, batch=True) is reader
    assert as_reader(None, "x", calibrated=True, batch=False) is None
    wrapped = as_reader(lambda images: [], "qwen", calibrated=False, batch=True, timeout_s=7.0, cache_key="qwen:m")
    assert isinstance(wrapped, BatchFnReader) and wrapped.timeout_s == 7.0 and wrapped.cache_key == "qwen:m"


def test_read_timeout_comes_from_the_environment(monkeypatch):
    assert read_timeout_s() == 20.0
    monkeypatch.setenv("READ_TIMEOUT_S", "7.5")
    assert read_timeout_s() == 7.5
    monkeypatch.setenv("READ_TIMEOUT_S", "soon")
    assert read_timeout_s(12.0) == 12.0
