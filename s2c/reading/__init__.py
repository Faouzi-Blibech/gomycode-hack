"""Shared handwriting reading: readers behind one interface, run in parallel with time budgets, a cache,
a warm-up and a call log. Readers only transcribe; each consumer decides what agreement means."""
from s2c.reading.adapters import BatchFnReader, CropFnReader, as_reader
from s2c.reading.base import MIN_CONFIDENCE, Crop, Reader, ReaderResult, read_timeout_s
from s2c.reading.env import readers_from_env
from s2c.reading.service import DEFAULT_CACHE, CropCache, ReaderRun, ReadingService, crop_key

__all__ = ["DEFAULT_CACHE", "MIN_CONFIDENCE", "BatchFnReader", "Crop", "CropCache", "CropFnReader", "Reader",
           "ReaderResult", "ReaderRun", "ReadingService", "as_reader", "crop_key", "read_timeout_s",
           "readers_from_env"]
