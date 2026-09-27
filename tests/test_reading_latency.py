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
