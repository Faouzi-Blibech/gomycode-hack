# Shared Reading Service and Studio OCR Trust Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One reading layer (`s2c/reading/`) that runs every handwriting reader in parallel with timeouts, a cache, a warm-up and a call log; the Studio reads through it and stops trusting junk numbers (project review P0-6); then the sketch package moves onto the same layer.

**Architecture:** `s2c/reading/` owns transport concerns only: the `Reader` interface, the `ReadingService` (threads, per-reader time budget, crop cache, JSONL log, background warm-up), adapters for the Studio's function readers, and one shared TrOCR reader. Each consumer keeps its own trust policy: the Studio's `ocr.decide` (numbers must agree between readers, a lone calibrated reader is trusted) and the sketch's `text._decide` (two readers must agree). Untrusted envelope sizes become pre-filled suggestions the user confirms; untrusted hole diameters get the amber `inferred` badge.

**Tech Stack:** Python 3.11, uv, NumPy, OpenCV (headless), Pydantic 2, `concurrent.futures`, pytest, ruff. Optional (`ai` extra): torch, transformers (TrOCR).

**Spec:** No separate spec file. The design was agreed in the session of 2026-09-27 and is recorded in the "Design decisions" section below. Background: `docs/superpowers/reviews/2026-09-24-project-review.md` (P0-6 and the measured facts), `docs/superpowers/specs/2026-09-23-studio-design.md`. Phase C continues `docs/superpowers/plans/2026-09-24-sketch-recognition-plan.md` (on branch `sketch/recognition-design`).

## Design decisions (agreed 2026-09-27)

1. Readers run **in parallel**; a request waits for the slowest reader or its time budget, never the sum.
2. **Trust (Studio):** a read is usable when it parses as a value and, for a calibrated (local) reader, its confidence is at least 0.7. A value is **confirmed** when two or more usable reads give the same number (the sign may differ: TrOCR drops ⌀), or when the only configured reader is calibrated and usable (single-reader mode, today's TrOCR-only behaviour). Otherwise it is **unconfirmed**. A lone VLM read is never confirmed: its confidence is a constant, not evidence.
3. **Unconfirmed envelope size** (x, y, z): does not count as written. The axis is missing, and the abstain pre-fills the read value as the suggestion with the remedy "Check the width: the sketch reads 60 mm. Confirm or correct it." Typing or accepting it makes it `user_edited`. The `MultiViewSpec` contract is unchanged (every envelope size keeps a trusted source).
4. **Unconfirmed hole diameter:** the written value is kept with provenance `inferred` (amber "AI · check"). Snapping may move it by at most 0.4 mm to a standard clearance size; the row stays amber.
5. **Scale trap:** a written size whose mm-per-pixel is more than 3× off the median of at least two other written sizes is unconfirmed, with a warning. A written hole diameter more than 3× off the hole's measured or scaled size is `inferred`.
6. **Parsing and linking:** digits separated by whitespace are never joined ("12 48" is rejected); a ⌀ or R value links to a hole only when the text centre lies within 25 % of the part's longer side plus the hole's radius; at most 16 crops per image, the ones nearest the part.
7. **Cache:** keyed by reader cache key plus a SHA-1 of the crop pixels and shape; only readers that declare a `cache_key` are cached; failed reads are never cached.
8. **Time budgets:** remote reader `READ_TIMEOUT_S` (default 20 s); TrOCR 60 s (covers a cold load). A reader that errors, returns `None` or times out is reported, never raised.
9. **Warm-up:** TrOCR loads once per process under a lock, in a background thread started by `default_pipeline()`.
10. **Log:** one JSON line per reader per call in `READING_LOG` (default `logs/reading.jsonl`): `ts`, `reader`, `crops`, `cached`, `status`, `latency_ms`. No image and no text is logged.
11. **Confidence gate:** one value, `MIN_CONFIDENCE = 0.7`, for both the Studio and the sketch package (the sketch spec's 0.6 is amended in Task 6).

## Global Constraints

- Python `>=3.11,<3.13`; run everything through `uv run --no-sync` (no agent runs `uv sync`: the shared `.venv` must not lose or gain extras).
- `uv run --no-sync ruff check .` and `uv run --no-sync pytest -q` must pass after every task, apart from the known pre-existing failure `tests/test_mv_gradio.py::test_a_bad_number_on_a_later_rebuild_keeps_the_current_faces_and_warnings` (fails on `origin/main` too; report it, do not fix it here).
- Ruff line length 120 (repo `pyproject.toml`).
- Rule 2: a model never produces a millimetre. Readers only transcribe; `parse_value` and `link` decide meaning.
- No model name or provider hard-coded in logic: `VLM_*`, `READ_TIMEOUT_S`, `READING_LOG`, `TROCR_MODEL` (default `microsoft/trocr-base-handwritten`).
- Tests never touch the network and never load a real model unless marked `@pytest.mark.gpu` with `pytest.importorskip("transformers")`.
- Commit messages are plain, in the team's voice. **No `Co-Authored-By`, no "Generated with", no AI attribution** (repo `CLAUDE.md`). **Never push.**
- Phases A and B commit on branch `recognition/reading-service`; Phase C on `sketch/recognition-design`.

## Review Focus

1. **A reader hangs** (DashScope stalls): the request returns after the budget with the other reader's reads, and the hung thread never blocks the caller. Test in Task 1 (`test_a_slow_reader_times_out_without_holding_the_others`).
2. **The same crop read by two different reader setups** (a test double and the real Qwen, or two models behind one name): a cached answer must never cross readers. Test in Task 1 (`test_the_cache_never_crosses_reader_keys`, `test_readers_without_a_cache_key_are_never_cached`).
3. **Batched TrOCR pads short answers after the end token**: padding must not raise the confidence of a short read toward 1. Test in Task 2 (`test_padding_after_the_end_token_does_not_count`).
4. **Qwen answers, TrOCR fails** (not installed, crashed or timed out): the value is amber, never green and never lost. Test in Task 4 (`test_a_batch_read_alone_is_unconfirmed_when_the_local_reader_fails`).
5. **A thin plate drawn with an exaggerated thickness** (100 × 3 mm drawn 600 × 120 px): the scale trap needs two other sizes and flags only the outlier, so the user confirms one value instead of the part abstaining silently. Test in Task 3 (`test_a_size_far_off_the_drawing_scale_is_only_a_suggestion`).

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `s2c/reading/__init__.py` | public names of the reading layer |
| `s2c/reading/base.py` | `Crop`, `ReaderResult`, `Reader` protocol, `MIN_CONFIDENCE`, `read_timeout_s()` |
| `s2c/reading/service.py` | `CropCache`, `DEFAULT_CACHE`, `ReaderRun`, `ReadingService` (parallel, budgets, cache, log, warm-up) |
| `s2c/reading/adapters.py` | `BatchFnReader`, `CropFnReader`, `as_reader` for the Studio's function readers |
| `s2c/reading/trocr.py` | the one TrOCR reader: lazy locked load, batched generate, padding-aware confidence |
| `s2c/multiview/ocr.py` | Studio: parse, crops nearest the part, `decide` (trust), `read_values` on the service, near-hole `link` |
| `s2c/multiview/fuse.py` | Studio: unconfirmed and off-scale values become suggestions (sizes) or `inferred` (holes) |
| `s2c/multiview/pipeline.py` | `MvPipeline.reading()`, service per `observe`, warm-up in `default_pipeline` |
| `scripts/reading_latency.py` | measures each reader alone and all in parallel on one sketch |
| `docs/models.md`, `.env.example` | reading layer, env vars, measured latency |
| `tests/conftest.py` | isolates the reading log and cache per test |
| `tests/reading/` | tests of the reading layer |

---

## Phase A: the reading layer (branch `recognition/reading-service`)

### Task 1: Reader interface and the `ReadingService`

**Files:**
- Create: `s2c/reading/__init__.py`, `s2c/reading/base.py`, `s2c/reading/service.py`, `s2c/reading/adapters.py`
- Create: `tests/reading/__init__.py` (empty), `tests/reading/test_service.py`, `tests/reading/test_adapters.py`
- Modify: `tests/conftest.py` (create it if it does not exist)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces:
  - `s2c.reading.base`: `MIN_CONFIDENCE = 0.7`; dataclass `Crop(image: np.ndarray, box: tuple[int, int, int, int])` (BGR, box `x, y, w, h` on the source image); Pydantic `ReaderResult(text: str, confidence: float in [0, 1])`; `Reader` protocol (`name: str`, `calibrated: bool`, `timeout_s: float`, `read(crops: list[Crop]) -> list[ReaderResult] | None`, optional `cache_key: str | None`, optional `warm() -> None`); `read_timeout_s(default: float = 20.0) -> float` (env `READ_TIMEOUT_S`).
  - `s2c.reading.service`: `crop_key(crop) -> str`; `CropCache(size=4096)` with `get(reader_key, key)`, `put(reader_key, key, result)`, `clear()`, `__len__`; `DEFAULT_CACHE`; dataclass `ReaderRun(name: str, calibrated: bool, results: list[ReaderResult] | None, status: Literal["ok", "error", "timeout", "none"], latency_ms: int, cached: int = 0)`; `ReadingService(readers, cache=DEFAULT_CACHE, log_path=None)` with `.readers`, `read(crops) -> list[ReaderRun]` (one run per reader, in reader order), `warm() -> threading.Thread`.
  - `s2c.reading.adapters`: `BatchFnReader(name, fn, calibrated=False, timeout_s=20.0, cache_key=None)` where `fn(list[np.ndarray]) -> list[tuple[str, float]] | None`; `CropFnReader(name, fn, calibrated=True, timeout_s=60.0, cache_key=None)` where `fn(np.ndarray) -> tuple[str, float]`; `as_reader(obj, name, *, calibrated, batch, timeout_s=None, cache_key=None) -> Reader | None` (returns `obj` unchanged when it already has `read` and `name`; `None` for `None`).
  - `s2c.reading`: re-exports all of the above.

- [ ] **Step 1: Write the test isolation fixture**

If `tests/conftest.py` exists, append the fixture and imports; otherwise create it with exactly this content:

```python
# tests/conftest.py
import pytest

from s2c.reading.service import DEFAULT_CACHE


@pytest.fixture(autouse=True)
def _isolated_reading(tmp_path, monkeypatch):
    """Every test gets its own reading log and an empty shared crop cache."""
    monkeypatch.setenv("READING_LOG", str(tmp_path / "reading.jsonl"))
    DEFAULT_CACHE.clear()
    yield
    DEFAULT_CACHE.clear()
```

- [ ] **Step 2: Write the failing service tests**

```python
# tests/reading/test_service.py
import json
import threading
import time

import numpy as np

from s2c.reading import Crop, CropCache, ReaderResult, ReadingService


def crop(value=0):
    return Crop(np.full((20, 30, 3), value, np.uint8), (0, 0, 30, 20))


class Fake:
    def __init__(self, name, text="60", conf=0.95, calibrated=True, delay=0.0, fail=False, none=False,
                 short=False, timeout_s=5.0, cache_key=None):
        self.name, self.text, self.conf, self.calibrated = name, text, conf, calibrated
        self.delay, self.fail, self.none, self.short, self.timeout_s = delay, fail, none, short, timeout_s
        self.cache_key = cache_key
        self.seen = []  # number of crops per call

    def read(self, crops):
        self.seen.append(len(crops))
        time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("boom")
        if self.none:
            return None
        n = len(crops) - 1 if self.short else len(crops)
        return [ReaderResult(text=self.text, confidence=self.conf) for _ in range(n)]


def test_every_reader_reads_every_crop_in_reader_order():
    runs = ReadingService([Fake("qwen", "⌀6", calibrated=False), Fake("trocr", "6")]).read([crop(), crop(1)])
    assert [r.name for r in runs] == ["qwen", "trocr"]
    assert [[x.text for x in r.results] for r in runs] == [["⌀6", "⌀6"], ["6", "6"]]
    assert [r.status for r in runs] == ["ok", "ok"] and [r.calibrated for r in runs] == [False, True]


def test_readers_run_in_parallel():
    t0 = time.perf_counter()
    ReadingService([Fake("a", delay=0.4), Fake("b", delay=0.4)]).read([crop()])
    assert time.perf_counter() - t0 < 0.7


def test_a_slow_reader_times_out_without_holding_the_others():
    t0 = time.perf_counter()
    runs = ReadingService([Fake("slow", delay=2.0, timeout_s=0.2), Fake("fast")]).read([crop()])
    assert time.perf_counter() - t0 < 1.0
    assert (runs[0].status, runs[0].results) == ("timeout", None)
    assert runs[1].status == "ok" and runs[1].results[0].text == "60"


def test_a_failing_reader_is_reported_not_raised():
    runs = ReadingService([Fake("err", fail=True), Fake("none", none=True), Fake("short", short=True)]).read(
        [crop(), crop(1)])
    assert [(r.status, r.results) for r in runs] == [("error", None), ("none", None), ("none", None)]


def test_no_crops_makes_no_call():
    reader = Fake("a")
    runs = ReadingService([reader]).read([])
    assert reader.seen == [] and runs[0].results == [] and runs[0].status == "ok"


def test_the_cache_reads_each_crop_once_per_reader():
    reader = Fake("trocr", cache_key="trocr:m")
    service = ReadingService([reader])
    service.read([crop(), crop(1)])
    second = service.read([crop(), crop(1), crop(2)])[0]
    assert reader.seen == [2, 1] and second.cached == 2
    assert [r.text for r in second.results] == ["60", "60", "60"]


def test_the_cache_never_crosses_reader_keys():
    ReadingService([Fake("qwen", "60", cache_key="qwen:a")]).read([crop()])
    other = Fake("qwen", "80", cache_key="qwen:b")
    assert ReadingService([other]).read([crop()])[0].results[0].text == "80" and other.seen == [1]


def test_readers_without_a_cache_key_are_never_cached():
    reader = Fake("fn")
    service = ReadingService([reader])
    service.read([crop()])
    service.read([crop()])
    assert reader.seen == [1, 1]


def test_failed_reads_are_not_cached():
    cache = CropCache()
    ReadingService([Fake("a", none=True, cache_key="a")], cache=cache).read([crop()])
    assert len(cache) == 0


def test_the_cache_evicts_the_oldest_entry():
    cache = CropCache(size=2)
    for k in "abc":
        cache.put("r", k, ReaderResult(text=k, confidence=1.0))
    assert cache.get("r", "a") is None and cache.get("r", "c").text == "c" and len(cache) == 2


def test_each_reader_call_is_logged_without_text(tmp_path):
    path = tmp_path / "log.jsonl"
    ReadingService([Fake("a"), Fake("b", fail=True)], log_path=path).read([crop()])
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [(x["reader"], x["status"], x["crops"]) for x in lines] == [("a", "ok", 1), ("b", "error", 1)]
    assert set(lines[0]) == {"ts", "reader", "crops", "cached", "status", "latency_ms"}


def test_warm_loads_readers_in_the_background():
    loaded = threading.Event()

    class Warmable(Fake):
        def warm(self):
            time.sleep(0.2)
            loaded.set()

    t0 = time.perf_counter()
    thread = ReadingService([Warmable("t"), Fake("plain")]).warm()
    assert time.perf_counter() - t0 < 0.1
    thread.join(2)
    assert loaded.is_set()


def test_a_reader_without_budget_or_calibration_attributes_still_runs():
    class Bare:
        name = "bare"

        def read(self, crops):
            return [ReaderResult(text="5", confidence=0.9) for _ in crops]

    run = ReadingService([Bare()]).read([crop()])[0]
    assert run.status == "ok" and run.calibrated is False
```

- [ ] **Step 3: Write the failing adapter tests**

```python
# tests/reading/test_adapters.py
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
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `uv run --no-sync pytest -q tests/reading`
Expected: collection errors, `ModuleNotFoundError: No module named 's2c.reading'`.

- [ ] **Step 5: Write `base.py`**

```python
# s2c/reading/base.py
"""What every handwriting reader shares. Readers only transcribe; they never estimate a size (rule 2)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from pydantic import BaseModel, Field

# A calibrated read below this is not evidence: phantom reads scored 0.31-0.51, real ones 0.99+
# (project review 2026-09-24, measured facts)
MIN_CONFIDENCE = 0.7


@dataclass
class Crop:
    image: np.ndarray                   # BGR
    box: tuple[int, int, int, int]      # x, y, w, h on the source image


class ReaderResult(BaseModel):
    text: str
    confidence: float = Field(ge=0, le=1)


class Reader(Protocol):
    """One result per crop, same order; None means the reader failed.
    `calibrated`: the confidence is a real probability (local models), not a VLM's constant.
    Optional: `cache_key` (str; readers without one are never cached) and `warm()` (load the model)."""
    name: str
    calibrated: bool
    timeout_s: float

    def read(self, crops: list[Crop]) -> list[ReaderResult] | None: ...


def read_timeout_s(default: float = 20.0) -> float:
    """Time budget of a remote reader, from READ_TIMEOUT_S."""
    try:
        return float(os.environ.get("READ_TIMEOUT_S", default))
    except ValueError:
        return default
```

- [ ] **Step 6: Write `service.py`**

```python
# s2c/reading/service.py
"""Run every reader on the same crops in parallel, each within its time budget, with a crop cache,
a background warm-up and one log line per reader call. Trust decisions belong to the caller."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np

from s2c.reading.base import Crop, Reader, ReaderResult

log = logging.getLogger(__name__)
Status = Literal["ok", "error", "timeout", "none"]
DEFAULT_TIMEOUT_S = 30.0


def crop_key(crop: Crop) -> str:
    image = np.ascontiguousarray(crop.image)
    return hashlib.sha1(image.tobytes() + str(image.shape).encode()).hexdigest()


class CropCache:
    """Thread-safe LRU of (reader cache key, crop fingerprint) -> result."""

    def __init__(self, size: int = 4096):
        self.size = size
        self._items: OrderedDict[tuple[str, str], ReaderResult] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, reader_key: str, key: str) -> ReaderResult | None:
        with self._lock:
            value = self._items.get((reader_key, key))
            if value is not None:
                self._items.move_to_end((reader_key, key))
            return value

    def put(self, reader_key: str, key: str, result: ReaderResult) -> None:
        with self._lock:
            self._items[(reader_key, key)] = result
            self._items.move_to_end((reader_key, key))
            while len(self._items) > self.size:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


DEFAULT_CACHE = CropCache()


@dataclass
class ReaderRun:
    name: str
    calibrated: bool
    results: list[ReaderResult] | None   # None: the reader failed, returned nothing usable or ran out of time
    status: Status
    latency_ms: int
    cached: int = 0


def _ms(t0: float) -> int:
    return round((time.perf_counter() - t0) * 1000)


class ReadingService:
    def __init__(self, readers: list[Reader], cache: CropCache | None = DEFAULT_CACHE,
                 log_path: str | Path | None = None):
        self.readers = list(readers)
        self.cache = cache
        self.log_path = Path(log_path or os.environ.get("READING_LOG", "logs/reading.jsonl"))
        self._log_lock = threading.Lock()

    def read(self, crops: list[Crop]) -> list[ReaderRun]:
        """One run per reader, in reader order. The call returns when every reader answered or ran out of time."""
        if not crops:
            return [ReaderRun(r.name, getattr(r, "calibrated", False), [], "ok", 0) for r in self.readers]
        if not self.readers:
            return []
        keys = [crop_key(c) for c in crops]
        pool = ThreadPoolExecutor(max_workers=len(self.readers), thread_name_prefix="reader")
        runs: list[ReaderRun] = []
        t0 = time.perf_counter()
        try:
            futures = [pool.submit(self._run_one, r, crops, keys) for r in self.readers]
            for reader, future in zip(self.readers, futures):
                budget = getattr(reader, "timeout_s", DEFAULT_TIMEOUT_S)
                try:
                    runs.append(future.result(timeout=max(budget - (time.perf_counter() - t0), 0.0)))
                except FutureTimeout:
                    log.warning("reader %s ran out of time (%.1f s)", reader.name, budget)
                    runs.append(ReaderRun(reader.name, getattr(reader, "calibrated", False), None, "timeout", _ms(t0)))
        finally:
            pool.shutdown(wait=False, cancel_futures=True)  # a hung reader must not hold the request
        for run in runs:
            self._log(run, len(crops))
        return runs

    def _run_one(self, reader: Reader, crops: list[Crop], keys: list[str]) -> ReaderRun:
        t0 = time.perf_counter()
        calibrated = getattr(reader, "calibrated", False)
        reader_key = getattr(reader, "cache_key", None)
        hits: dict[int, ReaderResult] = {}
        if self.cache is not None and reader_key:
            for i, k in enumerate(keys):
                cached = self.cache.get(reader_key, k)
                if cached is not None:
                    hits[i] = cached
        todo = [i for i in range(len(crops)) if i not in hits]
        fresh: list[ReaderResult] | None = []
        if todo:
            try:
                fresh = reader.read([crops[i] for i in todo])
            except Exception as e:  # noqa: BLE001 - a reader must never break the caller
                log.warning("reader %s failed: %s", reader.name, e)
                return ReaderRun(reader.name, calibrated, None, "error", _ms(t0), len(hits))
            if fresh is None or len(fresh) != len(todo):
                if fresh is not None:
                    log.warning("reader %s returned %d results for %d crops", reader.name, len(fresh), len(todo))
                return ReaderRun(reader.name, calibrated, None, "none", _ms(t0), len(hits))
            if self.cache is not None and reader_key:
                for i, result in zip(todo, fresh):
                    self.cache.put(reader_key, keys[i], result)
        merged = {**hits, **dict(zip(todo, fresh))}
        return ReaderRun(reader.name, calibrated, [merged[i] for i in range(len(crops))], "ok", _ms(t0), len(hits))

    def warm(self) -> threading.Thread:
        """Load every reader that has a warm() in one background thread, so no request pays for a cold load."""
        def run():
            for reader in self.readers:
                warm = getattr(reader, "warm", None)
                if warm is None:
                    continue
                t0 = time.perf_counter()
                try:
                    warm()
                    log.info("reader %s ready in %.1f s", reader.name, time.perf_counter() - t0)
                except Exception as e:  # noqa: BLE001 - a failed warm-up only means a slower first read
                    log.warning("reader %s warm-up failed: %s", reader.name, e)

        thread = threading.Thread(target=run, name="reader-warmup", daemon=True)
        thread.start()
        return thread

    def _log(self, run: ReaderRun, crops: int) -> None:
        record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "reader": run.name,
                  "crops": crops, "cached": run.cached, "status": run.status, "latency_ms": run.latency_ms}
        try:
            with self._log_lock:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
        except OSError as e:
            log.warning("reading log not written: %s", e)
```

- [ ] **Step 7: Write `adapters.py` and `__init__.py`**

```python
# s2c/reading/adapters.py
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
```

```python
# s2c/reading/__init__.py
"""Shared handwriting reading: readers behind one interface, run in parallel with time budgets, a cache,
a warm-up and a call log. Readers only transcribe; each consumer decides what agreement means."""
from s2c.reading.adapters import BatchFnReader, CropFnReader, as_reader
from s2c.reading.base import MIN_CONFIDENCE, Crop, Reader, ReaderResult, read_timeout_s
from s2c.reading.service import DEFAULT_CACHE, CropCache, ReaderRun, ReadingService, crop_key

__all__ = ["DEFAULT_CACHE", "MIN_CONFIDENCE", "BatchFnReader", "Crop", "CropCache", "CropFnReader", "Reader",
           "ReaderResult", "ReaderRun", "ReadingService", "as_reader", "crop_key", "read_timeout_s"]
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `uv run --no-sync pytest -q tests/reading`
Expected: 17 passed.

- [ ] **Step 9: Run the whole suite and ruff**

Run: `uv run --no-sync ruff check . && uv run --no-sync pytest -q`
Expected: ruff clean; only the known `test_mv_gradio` failure.

- [ ] **Step 10: Commit**

```bash
git add s2c/reading tests/reading tests/conftest.py
git commit -m "Add the shared reading layer: readers run in parallel with time budgets, a crop cache, a warm-up and a call log"
```

---

### Task 2: One shared TrOCR reader

**Files:**
- Create: `s2c/reading/trocr.py`, `tests/reading/test_trocr.py`

**Interfaces:**
- Consumes: `Crop`, `ReaderResult` (Task 1).
- Produces: `token_confidences(logprobs: np.ndarray, mask: np.ndarray) -> list[float]`; `TrocrReader(model_id=None, device=None, timeout_s=60.0, max_new_tokens=12)` with `name = "trocr"`, `calibrated = True`, `cache_key = f"trocr:{model_id}"`, `warm()`, `read(crops)`; module constant `DEFAULT_MODEL = "microsoft/trocr-base-handwritten"`. Construction raises `ImportError` when transformers is missing (callers catch it). Read errors propagate; the `ReadingService` turns them into status `error`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/reading/test_trocr.py
import cv2
import numpy as np
import pytest

from s2c.reading import Crop
from s2c.reading.trocr import token_confidences


def test_confidence_is_the_geometric_mean_of_the_real_tokens():
    logprobs = np.log(np.array([[0.9, 0.8, 1.0]]))
    assert token_confidences(logprobs, np.array([[1, 1, 1]])) == pytest.approx([(0.9 * 0.8) ** (1 / 3)])


def test_padding_after_the_end_token_does_not_count():
    logprobs = np.array([[np.log(0.5), np.log(0.5), 0.0, 0.0, 0.0]])  # two real tokens, then padding at log 1
    assert token_confidences(logprobs, np.array([[1, 1, 0, 0, 0]])) == pytest.approx([0.5])


def test_a_row_without_real_tokens_scores_zero():
    assert token_confidences(np.zeros((1, 3)), np.zeros((1, 3))) == [0.0]


@pytest.mark.gpu
def test_trocr_reads_a_batch_of_printed_digits():
    pytest.importorskip("transformers")
    from s2c.reading.trocr import TrocrReader

    def digits(s):
        img = np.full((80, 200, 3), 255, np.uint8)
        cv2.putText(img, s, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
        return Crop(img, (0, 0, 200, 80))

    reader = TrocrReader()
    reader.warm()
    out = reader.read([digits("60"), digits("125")])
    assert [r.text.replace(" ", "") for r in out] == ["60", "125"]
    assert all(r.confidence > 0.3 for r in out)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --no-sync pytest -q tests/reading/test_trocr.py`
Expected: `ModuleNotFoundError: No module named 's2c.reading.trocr'`.

- [ ] **Step 3: Write `trocr.py`**

```python
# s2c/reading/trocr.py
"""TrOCR handwritten, the one local reader shared by the Studio and the sketch package.
Optional dependency: the ai extra (torch, transformers). The model loads once per process and device,
under a lock, on warm() or on the first read."""
from __future__ import annotations

import os
import threading

import cv2
import numpy as np

from s2c.reading.base import Crop, ReaderResult

DEFAULT_MODEL = "microsoft/trocr-base-handwritten"
_LOADED: dict[tuple[str, str], tuple] = {}
_LOCK = threading.Lock()


def token_confidences(logprobs: np.ndarray, mask: np.ndarray) -> list[float]:
    """exp(mean log-probability) over each row's real tokens. Batched generation pads finished rows with
    tokens at log-probability 0; counting them would push a short read's confidence toward 1."""
    out = []
    for row, keep in zip(np.asarray(logprobs, float), np.asarray(mask).astype(bool)):
        real = row[keep]
        out.append(float(np.clip(np.exp(real.mean()), 0.0, 1.0)) if real.size else 0.0)
    return out


def _load(model_id: str, device: str) -> tuple:
    with _LOCK:
        if (model_id, device) not in _LOADED:
            from huggingface_hub import snapshot_download
            from transformers import RobertaTokenizer, TrOCRProcessor, VisionEncoderDecoderModel, ViTImageProcessor
            # transformers 5 does not fetch vocab.json and merges.txt for this repo by itself; take the small files
            local = snapshot_download(model_id, allow_patterns=["*.json", "*.txt"])
            processor = TrOCRProcessor(image_processor=ViTImageProcessor.from_pretrained(local),
                                       tokenizer=RobertaTokenizer.from_pretrained(local))
            model = VisionEncoderDecoderModel.from_pretrained(model_id).to(device).eval()
            _LOADED[(model_id, device)] = (processor, model)
        return _LOADED[(model_id, device)]


class TrocrReader:
    name = "trocr"
    calibrated = True

    def __init__(self, model_id: str | None = None, device: str | None = None, timeout_s: float = 60.0,
                 max_new_tokens: int = 12):
        import torch
        import transformers  # noqa: F401 - fail early when the ai extra is missing

        self.model_id = model_id or os.environ.get("TROCR_MODEL", DEFAULT_MODEL)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.timeout_s, self.max_new_tokens = timeout_s, max_new_tokens
        self.cache_key = f"trocr:{self.model_id}"

    def warm(self) -> None:
        _load(self.model_id, self.device)

    def read(self, crops: list[Crop]) -> list[ReaderResult] | None:
        if not crops:
            return []
        import torch
        from PIL import Image

        processor, model = _load(self.model_id, self.device)
        images = [Image.fromarray(cv2.cvtColor(c.image, cv2.COLOR_BGR2RGB)) for c in crops]
        pixels = processor(images=images, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            out = model.generate(pixels, max_new_tokens=self.max_new_tokens, num_beams=1, output_scores=True,
                                 return_dict_in_generate=True)
        texts = processor.batch_decode(out.sequences, skip_special_tokens=True)
        scores = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
        generated = out.sequences[:, -scores.shape[1]:]  # the tokens the scores describe
        pad = model.generation_config.pad_token_id
        if pad is None:
            pad = processor.tokenizer.pad_token_id
        mask = (generated != pad).cpu().numpy()
        confidences = token_confidences(scores.cpu().numpy(), mask)
        return [ReaderResult(text=t.strip(), confidence=c) for t, c in zip(texts, confidences)]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --no-sync pytest -q tests/reading/test_trocr.py`
Expected: 3 passed; the `gpu` test is skipped when transformers is not installed (it is not installed in the shared `.venv` today; the controller runs it after the user installs the `ai` extra).

- [ ] **Step 5: Run the whole suite and ruff**

Run: `uv run --no-sync ruff check . && uv run --no-sync pytest -q`
Expected: ruff clean; only the known `test_mv_gradio` failure.

- [ ] **Step 6: Commit**

```bash
git add s2c/reading/trocr.py tests/reading/test_trocr.py
git commit -m "Share one TrOCR reader: load it once under a lock, read every crop in one batch, ignore padding in its confidence"
```

---

## Phase B: the Studio reads through the service (branch `recognition/reading-service`)

### Task 3: Unconfirmed and off-scale values are suggestions, not written values

Runs in parallel with Task 2 (disjoint files).

**Files:**
- Modify: `s2c/multiview/ocr.py` (the `Reading` dataclass only)
- Modify: `s2c/multiview/fuse.py` (`_Candidate`, `_envelope_candidates`, `fuse_envelope`, `features_from`, new `_scale_trap`, constant `SCALE_TRAP`)
- Test: `tests/test_mv_fuse.py`

**Interfaces:**
- Consumes: nothing from Tasks 1-2.
- Produces: `Reading.confirmed: bool = True` (last field of `s2c.multiview.ocr.Reading`, after `text`); `fuse.SCALE_TRAP = 3.0`. `fuse_envelope` and `features_from` keep their signatures and return shapes.

- [ ] **Step 1: Write the failing tests**

In `tests/test_mv_fuse.py`, replace the `written` helper with this version (adds `confirmed`, keeps every existing call working):

```python
def written(value, axis=None, hole=None, kind="linear", conf=0.9, confirmed=True):
    return Linked(Reading(float(value), kind, (0, 0, 10, 10), conf, str(value), confirmed), axis, hole)
```

Append these tests:

```python
def test_an_unconfirmed_size_is_only_a_suggestion():
    front = Observation(face="front", kind="sketch", outline=px_outline(),
                        values=[written(60, "a", confirmed=False), written(40, "b")])
    res = fuse_envelope([front], {"envelope.z_mm": 5})
    assert isinstance(res, MvAbstain) and res.reason == "missing_x"
    assert res.remedy == "Check the width: the sketch reads 60 mm. Confirm or correct it."
    assert res.partial["suggested"]["envelope.x_mm"] == 60
    assert res.partial["known"] == {"envelope.y_mm": 40, "envelope.z_mm": 5}


def test_a_confirmed_size_on_another_view_wins_over_an_unconfirmed_one():
    front = Observation(face="front", kind="sketch", outline=px_outline(),
                        values=[written(60, "a", confirmed=False), written(40, "b")])
    top = Observation(face="top", kind="sketch", outline=px_outline(h=101), values=[written(60, "a")])
    env, prov, _ = fuse_envelope([front, top], {"envelope.z_mm": 5})
    assert env.x_mm == 60 and prov["envelope.x_mm"] == "user_written"


def test_a_size_far_off_the_drawing_scale_is_only_a_suggestion():
    # front 601 x 401 px reads 60 x 40 (0.1 mm/px); top 601 x 101 px reads 60 and 1248 (12.4 mm/px)
    front = front_obs()
    top = Observation(face="top", kind="sketch", outline=px_outline(h=101),
                      values=[written(60, "a"), written(1248, "b")])
    res = fuse_envelope([front, top])
    assert isinstance(res, MvAbstain) and res.reason == "missing_z"
    assert res.partial["suggested"]["envelope.z_mm"] == 1248
    assert res.remedy == "Check the depth: the sketch reads 1248 mm. Confirm or correct it."


def test_the_scale_trap_needs_two_other_sizes():
    front = Observation(face="front", kind="sketch", outline=px_outline(h=121),
                        values=[written(100, "a"), written(3, "b")])  # a thin plate drawn thick: 6x off
    env, prov, _ = fuse_envelope([front], {"envelope.z_mm": 50})
    assert (env.x_mm, env.y_mm) == (100, 3) and prov["envelope.y_mm"] == "user_written"


def test_a_size_within_the_scale_trap_stays_written_and_warns_nothing():
    top = Observation(face="top", kind="sketch", outline=px_outline(h=101), values=[written(60, "a"), written(25, "b")])
    env, prov, warnings = fuse_envelope([front_obs(), top])  # 25 mm on 100 px is 2.5x the 0.1 mm/px of the rest
    assert env.z_mm == 25 and prov["envelope.z_mm"] == "user_written"
    assert not any("scale" in w for w in warnings)


def test_an_unconfirmed_hole_diameter_is_kept_amber():
    env = Envelope(x_mm=60, y_mm=40, z_mm=5)
    front = Observation(face="front", kind="sketch", outline=px_outline(circles=[PixelCircle(500, 600, 58)]),
                        values=[written(6, None, 0, "diameter", confirmed=False)])
    feats, prov = features_from([front], env)
    assert feats[0]["diameter_mm"] == 6 and prov["features[0].diameter_mm"] == "inferred"


def test_a_hole_diameter_far_off_the_drawing_is_kept_amber():
    env = Envelope(x_mm=60, y_mm=40, z_mm=5)
    front = Observation(face="front", kind="sketch", outline=px_outline(circles=[PixelCircle(500, 600, 58)]),
                        values=[written(1, None, 0, "diameter")])  # the drawing says about 5.8 mm
    feats, prov = features_from([front], env)
    assert feats[0]["diameter_mm"] == 1 and prov["features[0].diameter_mm"] == "inferred"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --no-sync pytest -q tests/test_mv_fuse.py`
Expected: every test errors with `TypeError: Reading.__init__() takes ... positional arguments` (the `confirmed` field does not exist yet).

- [ ] **Step 3: Add the field to `Reading`**

In `s2c/multiview/ocr.py`, the dataclass becomes:

```python
@dataclass
class Reading:
    value_mm: float
    kind: Literal["linear", "diameter", "radius"]
    bbox: tuple[int, int, int, int]
    confidence: float
    text: str
    confirmed: bool = True  # False: the readers did not agree, so the user checks the value before it is trusted
```

- [ ] **Step 4: Run the tests again**

Run: `uv run --no-sync pytest -q tests/test_mv_fuse.py`
Expected: the seven new tests FAIL on their assertions (for example `missing_x` expected but an `Envelope` returned); every older test passes.

- [ ] **Step 5: Implement in `fuse.py`**

Add the constant under `DISAGREE = 0.05`:

```python
SCALE_TRAP = 3.0  # sketches are not to scale, but a written size 3x off the rest is a misread until confirmed
```

Replace `_Candidate` and `_envelope_candidates`, and add `_scale_trap`:

```python
@dataclass
class _Candidate:
    value: float
    prov: str          # user_written | unconfirmed | measured; "unconfirmed" never leaves this module
    confidence: float
    face: str
    px: float = 0.0    # the span the value measures, in image pixels


def _envelope_candidates(observations: list[Observation]) -> dict[str, list[_Candidate]]:
    cands: dict[str, list[_Candidate]] = {"x": [], "y": [], "z": []}
    for o in observations:
        a_axis, b_axis, _ = S.FACE_AXES[o.face]
        _, _, w, h = o.outline.bbox
        for which, axis, px in (("a", a_axis, w - 1), ("b", b_axis, h - 1)):
            readings = [lv.reading for lv in o.values
                        if (lv.axis == which and lv.reading.kind == "linear") or lv.axis == "ab"]
            if readings:
                r = max(readings, key=_value)
                prov = "user_written" if r.confirmed else "unconfirmed"
                cands[axis].append(_Candidate(_value(r), prov, r.confidence, o.face, float(px)))
            if o.mm_per_px:
                cands[axis].append(_Candidate(px * o.mm_per_px, "measured", o.confidence, o.face, float(px)))
    return cands


def _scale_trap(cands: dict[str, list[_Candidate]]) -> list[str]:
    """A written size whose mm per pixel is more than SCALE_TRAP times off the median of at least two other
    written sizes is demoted to unconfirmed."""
    written = [c for axis in "xyz" for c in cands[axis] if c.prov in ("user_written", "unconfirmed") and c.px > 0]
    notes = []
    for c in written:
        others = [o.value / o.px for o in written if o is not c]
        if len(others) < 2:
            continue
        median, scale = float(np.median(others)), c.value / c.px
        if c.prov == "user_written" and max(scale / median, median / scale) > SCALE_TRAP:
            c.prov = "unconfirmed"
            notes.append(f"{c.face}: {c.value:g} mm does not fit the drawing's scale; check it")
    return notes
```

Replace `fuse_envelope`:

```python
def fuse_envelope(observations: list[Observation], user_values: dict | None = None):
    user_values = user_values or {}
    cands = _envelope_candidates(observations)
    warnings: list[str] = _scale_trap(cands)
    values: dict[str, float] = {}
    prov: dict[str, str] = {}
    pending: dict[str, float] = {}
    for axis in "xyz":
        key, name = f"envelope.{axis}_mm", S.AXIS_NAMES[axis]
        if key in user_values:
            values[axis], prov[key] = float(user_values[key]), "user_edited"
            continue
        written = [c for c in cands[axis] if c.prov == "user_written"]
        measured = [c for c in cands[axis] if c.prov == "measured"]
        unconfirmed = [c for c in cands[axis] if c.prov == "unconfirmed"]
        if written:
            best = max(written, key=lambda c: c.confidence)
            for c in written:
                if c is not best and _differs(c.value, best.value):
                    warnings.append(f"{name}: {c.face} says {c.value:g} mm, {best.face} says {best.value:g} mm; "
                                    f"using {best.value:g}")
            for c in measured:
                if _differs(c.value, best.value):
                    warnings.append(f"{name}: written {best.value:g} mm, measured {c.value:.1f} mm; "
                                    "using the written value")
            values[axis], prov[key] = best.value, "user_written"
        elif measured:
            best = max(measured, key=lambda c: c.confidence)
            values[axis], prov[key] = round(best.value, 2), "measured"
        elif unconfirmed:
            pending[axis] = max(unconfirmed, key=lambda c: c.confidence).value
    missing = [a for a in "xyz" if a not in values]
    if missing:
        first = missing[0]
        name = S.AXIS_NAMES[first]
        remedy = (f"Check the {name}: the sketch reads {pending[first]:g} mm. Confirm or correct it."
                  if first in pending else f"Enter the {name} in mm.")
        suggested = {**_suggest(observations, values, missing),
                     **{f"envelope.{a}_mm": v for a, v in pending.items()}}
        return S.MvAbstain(
            stage="dimensions", reason=f"missing_{first}", remedy=remedy,
            partial={"known": {f"envelope.{a}_mm": v for a, v in values.items()},
                     "missing": [f"envelope.{a}_mm" for a in missing],
                     "suggested": suggested})
    return S.Envelope(x_mm=values["x"], y_mm=values["y"], z_mm=values["z"]), prov, warnings
```

In `features_from`, replace the `written = {...}` line and the `if i in written:` branch:

```python
        written = {lv.hole_index: (_value(lv.reading), lv.reading.confirmed)
                   for lv in o.values if lv.hole_index is not None}
        axis_len = env.length(S.FACE_AXES[o.face][2])
        for i, c in enumerate(o.outline.circles):
            (a, b), = to_face_mm(np.array([[c.cx, c.cy]]), o.outline.bbox, sa, sb)
            drawn = c.d * o.mm_per_px if o.mm_per_px else c.d * (sa + sb) / 2
            if i in written:
                d, confirmed = written[i]
                off = drawn > 0 and max(d / drawn, drawn / d) > SCALE_TRAP
                d_prov = "user_written" if confirmed and not off else "inferred"
            elif o.mm_per_px:
                d, d_prov = drawn, "measured"
            else:
                d, d_prov = drawn, "scaled"
```

(The rest of the loop body, from `depth, depth_prov = None, None` on, is unchanged.)

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run --no-sync pytest -q tests/test_mv_fuse.py`
Expected: all pass.

- [ ] **Step 7: Run the whole suite and ruff**

Run: `uv run --no-sync ruff check . && uv run --no-sync pytest -q`
Expected: ruff clean; only the known `test_mv_gradio` failure.

- [ ] **Step 8: Commit**

```bash
git add s2c/multiview/ocr.py s2c/multiview/fuse.py tests/test_mv_fuse.py
git commit -m "Keep unconfirmed and off-scale values out of the trusted envelope: suggest sizes for the user to confirm, badge holes amber"
```

---

### Task 4: The Studio reads through the service and decides trust

**Files:**
- Modify: `s2c/multiview/ocr.py` (constants, `parse_value`, new `crops_for`, `_gap`, `decide`, `read_values`, `link`, `trocr_reader`; remove `_trocr` and the `lru_cache` import)
- Modify: `s2c/multiview/pipeline.py` (`MvPipeline.reading`, `observe`, `default_pipeline`)
- Test: `tests/test_mv_ocr.py`, `tests/test_mv_qwen_reader.py`, `tests/test_mv_pipeline.py`

**Interfaces:**
- Consumes: `Crop`, `ReaderResult`, `ReaderRun`, `ReadingService`, `as_reader`, `read_timeout_s`, `MIN_CONFIDENCE` (Task 1); `TrocrReader` (Task 2); `Reading.confirmed` (Task 3).
- Produces: `ocr.MAX_CROPS = 16`, `ocr.NEAR_HOLE = 0.25`; `ocr.crops_for(image_bgr, outline) -> list[Crop]`; `ocr.decide(runs: list[ReaderRun], k: int) -> tuple[str, tuple[float, str], float, bool] | None` (text, parsed value and kind, confidence, confirmed); `ocr.read_values(image_bgr, outline, service: ReadingService | None) -> list[Reading]`; `ocr.trocr_reader(model_name=None) -> Reader` (per-crop function over the shared `TrocrReader`); `MvPipeline.reading() -> ReadingService | None` (readers in trust order: `qwen` batch reader first, then `trocr`).

- [ ] **Step 1: Write the failing OCR tests**

In `tests/test_mv_ocr.py`: change the import line to

```python
from s2c.multiview.ocr import Reading, decide, link, parse_value, read_values, text_regions
from s2c.reading import ReaderResult, ReaderRun, ReadingService, as_reader
```

add this helper under `sketch_with_values`:

```python
def svc(reader=None, batch=None):
    """The Studio's reader setup: the batch (Qwen-VL) reader first, then the per-crop (TrOCR) reader."""
    readers = [r for r in (as_reader(batch, "qwen", calibrated=False, batch=True),
                           as_reader(reader, "trocr", calibrated=True, batch=False)) if r]
    return ReadingService(readers, cache=None)
```

replace `test_read_values_keeps_numbers_and_drops_words` with

```python
def test_read_values_keeps_numbers_and_drops_words():
    img = sketch_with_values()
    o = extract(img)
    assert [r.value_mm for r in read_values(img, o, svc(lambda crop: ("60", 0.9)))] == [60.0, 60.0]
    assert read_values(img, o, svc(lambda crop: ("sixty", 0.9))) == []
    assert read_values(img, o, None) == []
```

and append:

```python
def test_parse_value_never_joins_digits_across_a_space():
    assert parse_value("12 48") is None
    assert parse_value("0 1") is None
    assert parse_value("⌀ 8") == (8.0, "diameter")
    assert parse_value("12 mm") == (12.0, "linear")


def test_a_diameter_far_from_every_hole_is_not_linked():
    far, near = reading(6, "diameter", 1400, 1100), reading(6, "diameter", 1000, 680)
    linked = link([far, near], outline_with_holes())
    assert [(lv.axis, lv.hole_index) for lv in linked] == [(None, None), (None, 1)]


def run(name, calibrated, *reads):
    return ReaderRun(name, calibrated, [ReaderResult(text=t, confidence=c) for t, c in reads], "ok", 1)


def failed(name, calibrated):
    return ReaderRun(name, calibrated, None, "timeout", 1)


def test_decide_confirms_when_two_readers_give_the_same_number():
    text, parsed, _, confirmed = decide([run("qwen", False, ("⌀6", 0.9)), run("trocr", True, ("6", 0.95))], 0)
    assert (text, parsed, confirmed) == ("⌀6", (6.0, "diameter"), True)


def test_decide_leaves_a_disagreement_unconfirmed_with_the_first_readers_text():
    assert decide([run("qwen", False, ("60", 0.9)), run("trocr", True, ("80", 0.95))], 0)[::3] == ("60", False)


def test_a_batch_read_alone_is_unconfirmed_when_the_local_reader_fails():
    assert decide([run("qwen", False, ("60", 0.9)), failed("trocr", True)], 0)[::3] == ("60", False)
    assert decide([run("qwen", False, ("60", 0.9)), run("trocr", True, ("60", 0.5))], 0)[::3] == ("60", False)


def test_a_lone_reader_is_trusted_only_when_it_is_calibrated():
    assert decide([run("trocr", True, ("60", 0.9))], 0)[3] is True
    assert decide([run("trocr", True, ("60", 0.5))], 0) is None
    assert decide([run("qwen", False, ("60", 0.9))], 0)[3] is False


def test_decide_drops_a_crop_nobody_could_read():
    assert decide([run("qwen", False, ("", 0.9)), failed("trocr", True)], 0) is None


def test_read_values_sends_at_most_16_crops_nearest_the_part():
    img = np.full((1200, 1600, 3), 255, np.uint8)
    cv2.rectangle(img, (500, 400), (1100, 800), (0, 0, 0), 4)
    for k in range(10):  # a row of values under the part and a row far above it
        cv2.putText(img, "8", (480 + 70 * k, 900), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
        cv2.putText(img, "8", (480 + 70 * k, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    o = extract(img)
    assert len(text_regions(img, o)) == 20
    seen = []
    readings = read_values(img, o, svc(batch=lambda crops: seen.append(len(crops)) or [("8", 0.9)] * len(crops)))
    assert seen == [16] and sum(r.bbox[1] > 800 for r in readings) == 10


def test_read_values_marks_which_values_the_readers_agree_on():
    img = sketch_with_values()
    o = extract(img)

    def batch(crops):  # crops come nearest the part first: the "60" under it, then the "40" to its left
        return [("60", 0.9), ("40 mm", 0.9)]

    got = {r.value_mm: r.confirmed for r in read_values(img, o, svc(lambda crop: ("40", 0.95), batch))}
    assert got == {60.0: False, 40.0: True}
```

- [ ] **Step 2: Update the Qwen reader test and the pipeline test**

In `tests/test_mv_qwen_reader.py`, change `from tests.test_mv_ocr import sketch_with_values` to `from tests.test_mv_ocr import sketch_with_values, svc` and replace `test_read_values_uses_the_batch_once_and_falls_back_to_the_reader` with:

```python
def test_read_values_calls_the_batch_once_and_keeps_the_local_reads_when_it_fails():
    img = sketch_with_values()
    o = extract(img)
    calls = []

    def batch(crops):
        calls.append(len(crops))
        return [("60", 0.9), ("40 mm", 0.9)]

    def failing(crops):
        return None

    assert sorted(r.value_mm for r in read_values(img, o, svc(batch=batch))) == [40.0, 60.0] and calls == [2]
    kept = read_values(img, o, svc(lambda crop: ("60", 0.8), failing))
    assert [(r.value_mm, r.confirmed) for r in kept] == [(60.0, False), (60.0, False)]
    assert read_values(img, o, svc(batch=failing)) == []
```

In `tests/test_mv_pipeline.py`, replace `test_the_batch_reader_feeds_ocr` with:

```python
def test_the_batch_reader_feeds_ocr(monkeypatch):
    seen = []

    def fake(bgr, outline, service):
        seen.append([r.name for r in service.readers])
        return []

    def batch(crops):
        return []

    monkeypatch.setattr(pipeline, "read_values", fake)
    observed = MvPipeline(batch_reader=batch).observe([ImageInput(sketch(600, 400), "front", "sketch")])
    assert seen == [["qwen"]]
    assert "OCR unavailable: enter the dimensions by hand" not in observed.warnings


def test_the_pipeline_reads_with_qwen_first_then_trocr():
    pipe = MvPipeline(reader=lambda crop: ("", 0.0), batch_reader=lambda crops: [])
    assert [r.name for r in pipe.reading().readers] == ["qwen", "trocr"]
    assert [r.calibrated for r in pipe.reading().readers] == [False, True]
    assert MvPipeline().reading() is None
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run --no-sync pytest -q tests/test_mv_ocr.py tests/test_mv_qwen_reader.py tests/test_mv_pipeline.py`
Expected: import error `cannot import name 'decide' from 's2c.multiview.ocr'`.

- [ ] **Step 4: Implement in `ocr.py`**

Imports: remove `from functools import lru_cache`; add

```python
from s2c.reading import MIN_CONFIDENCE, Crop, ReaderRun, ReadingService
```

Constants, under `STROKE_PX = 25`:

```python
MAX_CROPS = 16    # the text boxes nearest the part; the rest of a sheet is rarely a dimension
NEAR_HOLE = 0.25  # a ⌀ or R links to a hole within this share of the part's longer side, plus the hole's radius
```

`parse_value` gains a first check:

```python
def parse_value(text: str) -> tuple[float, str] | None:
    if re.search(r"\d\s+\d", text):
        return None  # "12 48" is two values or a misread, never 1248
    t = text.strip().replace(" ", "").replace(",", ".")
    # ... the rest is unchanged
```

Replace `read_values` with:

```python
def _gap(box: tuple[int, int, int, int], bbox: tuple[int, int, int, int]) -> float:
    """Distance from a text box's centre to the part's bounding box; 0 inside it."""
    x, y, w, h = box
    bx, by, bw, bh = bbox
    cx, cy = x + w / 2, y + h / 2
    return float(np.hypot(max(bx - cx, 0, cx - bx - bw), max(by - cy, 0, cy - by - bh)))


def crops_for(image_bgr: np.ndarray, outline: PixelOutline) -> list[Crop]:
    """At most MAX_CROPS text crops, nearest the part first, with a 6 px margin."""
    boxes = sorted(text_regions(image_bgr, outline), key=lambda b: _gap(b, outline.bbox))[:MAX_CROPS]
    return [Crop(image_bgr[max(y - 6, 0): y + h + 6, max(x - 6, 0): x + w + 6], (x, y, w, h))
            for x, y, w, h in boxes]


def decide(runs: list[ReaderRun], k: int) -> tuple[str, tuple[float, str], float, bool] | None:
    """Crop k's reads -> (text, (value, kind), confidence, confirmed), or None when no read is usable.
    Usable: parses as a value, and a calibrated reader is at least MIN_CONFIDENCE sure.
    Confirmed: two or more usable reads give the same number, or the only configured reader is calibrated.
    The first usable read in reader order gives the text: the VLM goes first because it keeps ⌀ and R."""
    usable = []
    for run in runs:
        if run.results is None:
            continue
        r = run.results[k]
        parsed = parse_value(r.text)
        if parsed is None or (run.calibrated and r.confidence < MIN_CONFIDENCE):
            continue
        usable.append((r, parsed))
    if not usable:
        return None
    first, parsed = usable[0]
    agree = len(usable) >= 2 and len({p[0] for _, p in usable}) == 1
    alone = len(runs) == 1 and runs[0].calibrated
    return first.text, parsed, first.confidence, agree or alone


def read_values(image_bgr: np.ndarray, outline: PixelOutline, service: ReadingService | None) -> list[Reading]:
    """Every reader reads every crop in parallel; decide() says which values the user must check."""
    if service is None:
        return []
    crops = crops_for(image_bgr, outline)
    runs = service.read(crops)
    out = []
    for k, crop in enumerate(crops):
        decided = decide(runs, k)
        if decided is None:
            log.info("no usable read at %s", crop.box)
            continue
        text, (value, kind), confidence, confirmed = decided
        out.append(Reading(value, kind, crop.box, float(confidence), text, confirmed))
    return out
```

Replace the body of `link` up to and including the hole branch:

```python
def link(readings: list[Reading], outline: PixelOutline) -> list[Linked]:
    """Below or above the outline: axis a. Left or right: axis b. Diameters: the nearest hole, if it is near."""
    bx, by, bw, bh = outline.bbox
    reach = NEAR_HOLE * max(bw, bh)
    out = []
    for r in readings:
        x, y, w, h = r.bbox
        cx, cy = x + w / 2, y + h / 2
        inside = bx <= cx <= bx + bw and by <= cy <= by + bh
        if r.kind != "linear":
            dist = [float(np.hypot(c.cx - cx, c.cy - cy)) - c.d / 2 for c in outline.circles]
            k = min(range(len(dist)), key=dist.__getitem__) if dist else None
            if k is not None and (inside or not outline.circular) and dist[k] <= reach:
                out.append(Linked(r, None, k))
            else:
                out.append(Linked(r, "ab" if outline.circular else None, None))
        elif cy < by or cy > by + bh:
            out.append(Linked(r, "a", None))
        elif cx < bx or cx > bx + bw:
            out.append(Linked(r, "b", None))
        else:
            out.append(Linked(r, None, None))  # written inside the part: kept for the lab view, not linked
    return out
```

Replace `_trocr` and `trocr_reader` with:

```python
def trocr_reader(model_name: str | None = None) -> Reader:
    """One crop at a time over the shared TrOCR reader (s2c.reading.trocr), for callers that need a function.
    The pipeline passes the shared reader itself, so every crop of an image is read in one batch."""
    from s2c.reading.trocr import TrocrReader
    shared = TrocrReader(model_name)

    def read(crop_bgr: np.ndarray) -> tuple[str, float]:
        out = shared.read([Crop(crop_bgr, (0, 0, crop_bgr.shape[1], crop_bgr.shape[0]))])
        return (out[0].text, out[0].confidence) if out else ("", 0.0)

    return read
```

- [ ] **Step 5: Implement in `pipeline.py`**

Imports: add `from s2c.reading import ReadingService, as_reader, read_timeout_s`.

Add this method to `MvPipeline` (after `__init__`):

```python
    def reading(self) -> ReadingService | None:
        """The readers in trust order: the batch (Qwen-VL) reader first, it keeps the ⌀ and R signs; then TrOCR."""
        model = os.environ.get("VLM_MODEL")
        readers = [r for r in (
            as_reader(self.batch_reader, "qwen", calibrated=False, batch=True, timeout_s=read_timeout_s(),
                      cache_key=f"qwen:{model}" if model and self.batch_reader is not None else None),
            as_reader(self.reader, "trocr", calibrated=True, batch=False)) if r is not None]
        return ReadingService(readers) if readers else None
```

(`as_reader` returns a shared `TrocrReader` unchanged, so it keeps its own `cache_key` and 60 s budget. A test double wrapped from a function has no `cache_key` and is never cached. The `qwen` key is set only for a real configured model; when `VLM_MODEL` is unset the batch reader is a test double.)

In `observe`, build the service once before the `for item in images:` loop:

```python
        service = self.reading() if reads else None
```

and replace

```python
                values = link(read_values(bgr, outline, self.reader, self.batch_reader), outline)
```

with

```python
                values = link(read_values(bgr, outline, service), outline)
```

(`reads` is the existing local that decides whether OCR runs; keep its current definition.)

In `default_pipeline`, replace the TrOCR block with the shared reader and start the warm-up:

```python
    reader = provider = None
    try:
        from s2c.reading.trocr import TrocrReader
        reader = TrocrReader()
        ReadingService([reader]).warm()  # loads the model in the background; the first request does not wait
    except Exception as e:  # transformers missing
        log.warning("TrOCR unavailable: %s", e)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run --no-sync pytest -q tests/test_mv_ocr.py tests/test_mv_qwen_reader.py tests/test_mv_pipeline.py`
Expected: all pass (the `gpu` test is skipped without transformers).

- [ ] **Step 7: Run the whole suite and ruff**

Run: `uv run --no-sync ruff check . && uv run --no-sync pytest -q`
Expected: ruff clean; only the known `test_mv_gradio` failure. If `tests/test_studio_pipeline.py` or `tests/test_mv_gradio.py` fail on `read_values` or `reader` usage, update them to the `service` form above, never by weakening an assertion.

- [ ] **Step 8: Commit**

```bash
git add s2c/multiview/ocr.py s2c/multiview/pipeline.py tests/test_mv_ocr.py tests/test_mv_qwen_reader.py tests/test_mv_pipeline.py
git commit -m "Read Studio dimensions through the shared service: both readers in parallel, trusted only when they agree, no joined digits, diameters only on a nearby hole"
```

---

### Task 5: Measure the latency and document the reading layer

**Files:**
- Create: `scripts/reading_latency.py`, `tests/test_reading_latency.py`
- Modify: `docs/models.md` (new section "Reading handwriting"), `.env.example` (three commented lines)

**Interfaces:**
- Consumes: `ReadingService`, `Reader`, `BatchFnReader`, `read_timeout_s` (Task 1); `TrocrReader` (Task 2); `ocr.crops_for` (Task 4); `outline.extract`, `outline.resize_long_side`; `label.env_chat`; `qwen_reader.qwen_batch_reader`.
- Produces: `measure(readers: list[Reader], crops: list[Crop], runs: int = 3) -> list[dict]` with one row per reader alone plus a row `"parallel"` for all readers together; each row has `setup`, `median_ms`, `min_ms`, `max_ms`, `crops`, `status`. `main(argv=None) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reading_latency.py
import time

import numpy as np

from s2c.reading import Crop, ReaderResult
from scripts.reading_latency import measure


class Sleepy:
    calibrated, timeout_s = True, 5.0

    def __init__(self, name, delay):
        self.name, self.delay = name, delay

    def read(self, crops):
        time.sleep(self.delay)
        return [ReaderResult(text="1", confidence=1.0) for _ in crops]


def test_measure_times_each_reader_alone_and_all_in_parallel():
    crops = [Crop(np.zeros((10, 10, 3), np.uint8), (0, 0, 10, 10))]
    rows = measure([Sleepy("a", 0.2), Sleepy("b", 0.3)], crops, runs=2)
    by = {r["setup"]: r for r in rows}
    assert list(by) == ["a", "b", "parallel"]
    assert 180 <= by["a"]["median_ms"] < 280 and 280 <= by["b"]["median_ms"] < 380
    assert by["parallel"]["median_ms"] < 450 and by["parallel"]["crops"] == 1
    assert all(r["status"] == "ok" for r in rows)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --no-sync pytest -q tests/test_reading_latency.py`
Expected: `ModuleNotFoundError: No module named 'scripts.reading_latency'` (if `scripts/` has no `__init__.py`, create an empty one in this task; check first with `ls scripts/__init__.py`).

- [ ] **Step 3: Write the script**

```python
# scripts/reading_latency.py
"""Time the handwriting readers on one sketch: each reader alone, then all of them in parallel.
Needs the ai extra for TrOCR and VLM_BASE_URL / VLM_MODEL / VLM_API_KEY for Qwen-VL; a reader that is not
available is skipped. The cache is off, so every run reads for real.

    uv run python scripts/reading_latency.py examples/sketch_front.png --runs 3
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import cv2

from s2c.reading import BatchFnReader, Crop, Reader, ReadingService, read_timeout_s


def _time(readers: list[Reader], crops: list[Crop], runs: int) -> dict:
    service = ReadingService(readers, cache=None)
    times, status = [], "ok"
    for _ in range(runs):
        t0 = time.perf_counter()
        result = service.read(crops)
        times.append((time.perf_counter() - t0) * 1000)
        bad = [r.status for r in result if r.status != "ok"]
        status = bad[0] if bad else status
    return {"median_ms": round(statistics.median(times)), "min_ms": round(min(times)),
            "max_ms": round(max(times)), "crops": len(crops), "status": status}


def measure(readers: list[Reader], crops: list[Crop], runs: int = 3) -> list[dict]:
    rows = [{"setup": r.name, **_time([r], crops, runs)} for r in readers]
    if len(readers) > 1:
        rows.append({"setup": "parallel", **_time(readers, crops, runs)})
    return rows


def _readers() -> list[Reader]:
    from s2c.multiview.label import env_chat
    from s2c.multiview.qwen_reader import qwen_batch_reader
    readers: list[Reader] = []
    chat = env_chat(stage="mv_read")
    if chat is not None:
        readers.append(BatchFnReader("qwen", qwen_batch_reader(chat), timeout_s=read_timeout_s()))
    try:
        from s2c.reading.trocr import TrocrReader
        trocr = TrocrReader()
        t0 = time.perf_counter()
        trocr.warm()
        print(f"TrOCR loaded in {time.perf_counter() - t0:.1f} s", file=sys.stderr)
        readers.append(trocr)
    except Exception as e:  # noqa: BLE001 - a missing extra only skips the reader
        print(f"TrOCR skipped: {e}", file=sys.stderr)
    return readers


def main(argv: list[str] | None = None) -> int:
    from s2c.multiview.ocr import crops_for
    from s2c.multiview.outline import extract, resize_long_side
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("image")
    args.add_argument("--runs", type=int, default=3)
    ns = args.parse_args(argv)
    image = cv2.imread(ns.image)
    if image is None:
        print(f"cannot read {ns.image}", file=sys.stderr)
        return 2
    image = resize_long_side(image)
    outline = extract(image)
    if not hasattr(outline, "bbox"):
        print(f"no outline: {outline.reason}", file=sys.stderr)
        return 2
    readers = _readers()
    if not readers:
        print("no reader available: set VLM_* or install the ai extra", file=sys.stderr)
        return 2
    print("| setup | median ms | min ms | max ms | crops | status |\n| --- | --- | --- | --- | --- | --- |")
    for row in measure(readers, crops_for(image, outline), ns.runs):
        print(f"| {row['setup']} | {row['median_ms']} | {row['min_ms']} | {row['max_ms']} | {row['crops']} "
              f"| {row['status']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --no-sync pytest -q tests/test_reading_latency.py`
Expected: 1 passed.

- [ ] **Step 5: Document**

Append to `docs/models.md`:

```markdown
## Reading handwriting

The Studio reads written dimensions with two readers at once, through `s2c/reading/`:

- **Qwen-VL** (the `VLM_*` provider): reads every crop of an image in one call and keeps the ⌀ and R signs. Its confidence is a constant, so a Qwen read alone is never trusted.
- **TrOCR** (`microsoft/trocr-base-handwritten`, local, needs `uv sync --extra ai`): reads every crop in one batch and gives a real confidence. Reads below 0.7 are ignored.

A value is trusted (green, `user_written`) when both readers give the same number. When they disagree, or only Qwen answered, the Studio asks the user to confirm the size (pre-filled) or shows the hole diameter amber. With only TrOCR configured, its confident reads are trusted, as before.

Both readers run in parallel, so a request waits for the slower one, not the sum. TrOCR loads in the background when the Studio starts. Settings:

| Variable | Default | Meaning |
| --- | --- | --- |
| `READ_TIMEOUT_S` | 20 | time budget of the Qwen-VL read, in seconds |
| `TROCR_MODEL` | `microsoft/trocr-base-handwritten` | the local handwriting model |
| `READING_LOG` | `logs/reading.jsonl` | one line per reader call: reader, crops, cached, status, latency; no image, no text |

Measured latency: run `uv run python scripts/reading_latency.py <sketch image>` and paste the table here with the date and the machine.
```

Append to `.env.example`:

```bash
# Handwriting reading (s2c/reading): Qwen-VL time budget, local TrOCR model, per-call log
# READ_TIMEOUT_S=20
# TROCR_MODEL=microsoft/trocr-base-handwritten
# READING_LOG=logs/reading.jsonl
```

- [ ] **Step 6: Run the whole suite and ruff**

Run: `uv run --no-sync ruff check . && uv run --no-sync pytest -q`
Expected: ruff clean; only the known `test_mv_gradio` failure.

- [ ] **Step 7: Commit**

```bash
git add scripts/reading_latency.py tests/test_reading_latency.py docs/models.md .env.example
git add scripts/__init__.py 2>/dev/null || true
git commit -m "Add a latency script for the readers and document the reading layer"
```

- [ ] **Step 8 (controller, with the user): measure for real**

Needs the `ai` extra installed and `VLM_*` set. The user runs (or approves) `uv sync --extra ai`, then the controller runs `uv run python scripts/reading_latency.py <the demo sketch>` and pastes the table into `docs/models.md` under "Measured latency" with the date and machine, and commits "Record the measured reading latency".

---

## Phase C: the sketch package on the reading layer (branch `sketch/recognition-design`)

Phase C starts after Tasks 1-5 are committed. The controller merges `recognition/reading-service` into `sketch/recognition-design` (local merge, no push) and then runs Task 6, then the sketch plan's remaining tasks.

### Task 6: Move the sketch readers into `s2c/reading/`

**Files:**
- Create: `s2c/reading/vlm.py` (`SYSTEM`, `tile_grid`, `VlmReader`), `s2c/reading/paddle.py` (`PaddleReader`), `s2c/reading/env.py` (`readers_from_env`)
- Delete: `s2c/sketch/readers.py`
- Move: `tests/sketch/test_readers.py` → `tests/reading/test_sketch_readers.py`
- Modify: `tests/sketch/synth.py` (import), `s2c/reading/__init__.py` (exports), `docs/superpowers/specs/2026-09-24-sketch-recognition-design.md` (0.6 → 0.7), `docs/superpowers/plans/2026-09-24-sketch-recognition-plan.md` (Global Constraints tolerance line and the Task 5 amendments below)

**Interfaces:**
- Consumes: `Crop`, `ReaderResult`, `Reader`, `MIN_CONFIDENCE` (Task 1); `TrocrReader` (Task 2).
- Produces: `s2c.reading.vlm.VlmReader(client, batch=24, timeout_s=None)` with `name = "vlm"`, `calibrated = False`, `timeout_s = read_timeout_s()` when not given, `cache_key = None`; `s2c.reading.paddle.PaddleReader(model_name=None)` with `name = "paddle"`, `calibrated = True`, `timeout_s = 60.0`, `cache_key = f"paddle:{model_name}"`; `s2c.reading.env.readers_from_env(spec=None) -> list[Reader]` (names `vlm`, `paddle`, `trocr`; `trocr` builds the shared `TrocrReader`); all re-exported from `s2c.reading`. `SKETCH_TROCR_MODEL` is replaced by `TROCR_MODEL`.

- [ ] **Step 1: Move the tests first and point them at the new modules**

`git mv tests/sketch/test_readers.py tests/reading/test_sketch_readers.py`. In it, replace every `from s2c.sketch.readers import X` with the new home: `SYSTEM`, `VlmReader`, `tile_grid` from `s2c.reading.vlm`; `PaddleReader` from `s2c.reading.paddle`; `readers_from_env` from `s2c.reading.env`; `Crop` from `s2c.reading`; `TrocrReader` from `s2c.reading.trocr`. Append:

```python
def test_moved_readers_declare_calibration_budget_and_cache():
    from s2c.reading.vlm import VlmReader
    r = VlmReader(client=object())
    assert (r.name, r.calibrated, r.cache_key) == ("vlm", False, None) and r.timeout_s == 20.0
```

In `tests/sketch/synth.py`, change `from s2c.sketch.readers import ReaderResult` to `from s2c.reading import ReaderResult`.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --no-sync pytest -q tests/reading/test_sketch_readers.py`
Expected: `ModuleNotFoundError: No module named 's2c.reading.vlm'`.

- [ ] **Step 3: Move the code**

Create `s2c/reading/vlm.py` with the sketch module's `SYSTEM`, `_Read`, `_Reply`, `_json_block`, `tile_grid` and `VlmReader` exactly as they are in `s2c/sketch/readers.py`, importing `Crop`, `ReaderResult` from `s2c.reading.base`, plus these class attributes and constructor:

```python
class VlmReader:
    name = "vlm"
    calibrated = False
    cache_key = None

    def __init__(self, client: VLMClient, batch: int = 24, timeout_s: float | None = None):
        self.client, self.batch = client, batch
        self.timeout_s = read_timeout_s() if timeout_s is None else timeout_s
```

Create `s2c/reading/paddle.py` with `PaddleReader` as it is, plus `calibrated = True`, `timeout_s = 60.0` and, in `__init__`, `self.cache_key = f"paddle:{name}"` where `name` is the resolved model name.

Create `s2c/reading/env.py`:

```python
"""Pick readers by name from SKETCH_READERS (default "paddle,vlm"); an unavailable reader is skipped."""
from __future__ import annotations

import logging
import os

from s2c.reading.base import Reader

log = logging.getLogger(__name__)


def readers_from_env(spec: str | None = None) -> list[Reader]:
    names = [n.strip() for n in (spec or os.environ.get("SKETCH_READERS", "paddle,vlm")).split(",")]
    out: list[Reader] = []
    for name in filter(None, names):
        try:
            if name == "vlm":
                from s2c.reading.vlm import VlmReader
                from s2c.vision.client import VLMClient
                out.append(VlmReader(VLMClient.from_env()))
            elif name == "paddle":
                from s2c.reading.paddle import PaddleReader
                out.append(PaddleReader())
            elif name == "trocr":
                from s2c.reading.trocr import TrocrReader
                out.append(TrocrReader())
            else:
                log.warning("unknown reader %r skipped", name)
        except Exception as exc:  # noqa: BLE001 - a missing optional dependency must not stop
            log.warning("reader %s unavailable: %s", name, exc)
    return out
```

Delete `s2c/sketch/readers.py`. Add `readers_from_env` to `s2c/reading/__init__.py` (import from `s2c.reading.env`, add to `__all__`); do not re-export `vlm` or `paddle` classes there (they pull optional imports).

- [ ] **Step 4: Amend the spec and the sketch plan**

In `docs/superpowers/specs/2026-09-24-sketch-recognition-design.md`, change "a confidence is below 0.6" to "a calibrated reader's confidence is below 0.7". In `docs/superpowers/plans/2026-09-24-sketch-recognition-plan.md`, change the Global Constraints tolerance "reader confidence below 0.6 is `uncertain`" to "reader confidence below 0.7 (`s2c.reading.MIN_CONFIDENCE`) is `uncertain`", replace every `s2c.sketch.readers` import in the plan's code with the `s2c.reading` home listed in Step 1, and add under Task 5's heading the block "Amendments (2026-09-27)" copied from the section below.

- [ ] **Step 5: Run the tests, the suite and ruff**

Run: `uv run --no-sync pytest -q tests/reading tests/sketch && uv run --no-sync ruff check . && uv run --no-sync pytest -q`
Expected: all pass apart from the known `test_mv_gradio` failure.

- [ ] **Step 6: Commit**

```bash
git add -A s2c/reading s2c/sketch tests/reading tests/sketch docs/superpowers
git commit -m "Move the sketch readers into the shared reading layer and use one confidence gate of 0.7"
```

### Sketch plan Task 5 amendments (2026-09-27)

The sketch plan's Task 5 is implemented as written, with these changes:

1. `s2c/sketch/text.py` imports `from s2c.reading import MIN_CONFIDENCE, Crop, Reader, ReadingService` and does not define its own `MIN_CONFIDENCE`.
2. `read_texts` reads through the service and counts configured readers, so a reader that fails makes values `uncertain`:

```python
def read_texts(sheet_bgr: np.ndarray, boxes, readers: list[Reader]) -> list[TextItem] | SketchAbstain:
    if not boxes:
        return []
    groups = [_crops(sheet_bgr, b) for b in boxes]
    flat = [c for g in groups for c in g]
    runs = ReadingService(readers).read(flat)
    per_reader = [(run.name, run.results) for run in runs if run.results is not None]
    if not per_reader:
        return SketchAbstain(stage="text", reason="readers_unavailable",
                             remedy="Reading service unavailable. Retry in a minute.")
    items: list[TextItem] = []
    start = 0
    for n, (box, group) in enumerate(zip(boxes, groups)):
        readings = []
        for name, res in per_reader:
            text, conf = _best(res[start:start + len(group)])
            readings.append(Reading(reader=name, text=text, confidence=conf))
        start += len(group)
        items.append(_decide(f"t{n}", box, readings, len(readers)))
    return items
```

3. In `_decide`, agreement compares values only (TrOCR drops ⌀, so kinds may differ); the kind comes from the first reader:

```python
        agree = (len(parsed) == n_readers >= 2 and len(values) == 1
                 and all(r.confidence >= MIN_CONFIDENCE for r, _ in parsed))
```

4. The Task 5 test `test_disagreement_is_uncertain_with_both_candidates` and the others stay as written; add:

```python
def test_a_failing_second_reader_makes_values_uncertain():
    sh = Sheet()
    bridge_block(sh)

    class Down:
        name = "down"

        def read(self, crops):
            return None

    items = read_texts(sh.bgr(), find_text_boxes(sh.ink(), 3.0), [TruthReader(sh.texts, "a"), Down()])
    dims = [t for t in items if t.role == "dimension"]
    assert dims and all(t.badge == "uncertain" for t in dims)
```

### Then: the sketch plan, Tasks 5, 6 and 8 to 13

Executed from `docs/superpowers/plans/2026-09-24-sketch-recognition-plan.md` in its order (Task 7 is done), with the amendments above, on `sketch/recognition-design`, with the ledger `.superpowers/sdd/2026-09-24-sketch-recognition-plan/progress.md`.

---

## Execution map

| Wave | Tasks | Agent | Why |
| --- | --- | --- | --- |
| 1 | Task 1 | `worker-high` (opus) | concurrency, time budgets and cache isolation |
| 2 | Task 2 ∥ Task 3 | `worker-medium` (sonnet) each | disjoint files |
| 3 | Task 4 | `worker-medium` (sonnet) | wiring with complete code in the plan |
| 4 | Task 5 | `worker-low` (haiku) | a script and docs with complete code |
| 5 | Task 6 | `worker-medium` (sonnet) | a move with import updates |
| 6+ | sketch Tasks 5, 6, 8-13 | per the sketch plan; Tasks 8 and 10 `worker-high` (opus), the rest `worker-medium` (sonnet) | |

A `worker-medium` (sonnet) reviewer checks each task against this plan before the next wave starts.
