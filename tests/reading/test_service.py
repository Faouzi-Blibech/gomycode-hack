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
    assert time.perf_counter() - t0 < 0.8  # less than the sum of the delays: they ran together, not one after another


def test_a_slow_reader_times_out_without_holding_the_others():
    t0 = time.perf_counter()
    runs = ReadingService([Fake("slow", delay=2.0, timeout_s=0.2), Fake("fast")]).read([crop()])
    assert time.perf_counter() - t0 < 1.5
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


def test_an_empty_read_is_not_cached():
    cache = CropCache()
    reader = Fake("a", text="", cache_key="a")
    ReadingService([reader], cache=cache).read([crop()])
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
