"""Wrap the Studio's function readers (one call for every crop, or one call per crop) as Readers."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from s2c.reading.base import Crop, Reader, ReaderResult

BatchFn = Callable[[list[np.ndarray]], list[tuple[str, float]] | None]
CropFn = Callable[[np.ndarray], tuple[str, float]]


def _result(text: str, confidence: float) -> ReaderResult:
    return ReaderResult(text=str(text), confidence=min(max(float(confidence), 0.0), 1.0))


class BatchFnReader:
    def __init__(self, name: str, fn: BatchFn, calibrated: bool = False, timeout_s: float = 20.0,
                 cache_key: str | None = None):
        self.name, self.fn, self.calibrated, self.timeout_s, self.cache_key = name, fn, calibrated, timeout_s, cache_key

    def read(self, crops: list[Crop]) -> list[ReaderResult] | None:
        out = self.fn([c.image for c in crops])
        return None if out is None else [_result(t, c) for t, c in out]


class CropFnReader:
    def __init__(self, name: str, fn: CropFn, calibrated: bool = True, timeout_s: float = 60.0,
                 cache_key: str | None = None):
        self.name, self.fn, self.calibrated, self.timeout_s, self.cache_key = name, fn, calibrated, timeout_s, cache_key

    def read(self, crops: list[Crop]) -> list[ReaderResult] | None:
        return [_result(*self.fn(c.image)) for c in crops]


def as_reader(obj, name: str, *, calibrated: bool, batch: bool, timeout_s: float | None = None,
              cache_key: str | None = None) -> Reader | None:
    """A Reader stays as it is; a function is wrapped; None stays None."""
    if obj is None:
        return None
    if hasattr(obj, "read") and hasattr(obj, "name"):
        return obj
    cls = BatchFnReader if batch else CropFnReader
    kwargs = {"calibrated": calibrated, "cache_key": cache_key}
    if timeout_s is not None:
        kwargs["timeout_s"] = timeout_s
    return cls(name, obj, **kwargs)
