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
