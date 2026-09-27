import json
import os

import cv2
import numpy as np
import pytest

from s2c.reading import Crop
from s2c.reading.env import readers_from_env
from s2c.reading.vlm import SYSTEM, VlmReader, tile_grid
from s2c.vision.client import VLMClient


def crops(n):
    return [Crop(np.full((30 + i, 60, 3), 255, np.uint8), (i, i, 60, 30 + i)) for i in range(n)]


def client(chat, tmp_path):
    return VLMClient(chat=chat, model="fake", log_path=tmp_path / "vlm.jsonl")


def test_tile_grid_is_a_png_holding_every_crop():
    png = tile_grid(crops(5))
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.shape[0] > 0 and img.shape[1] > 0


def test_one_result_per_crop_in_order(tmp_path):
    def chat(messages):
        return json.dumps({"reads": [{"i": 2, "text": "Ø6", "confidence": 0.8},
                                     {"i": 1, "text": ".50", "confidence": 0.9}]})
    out = VlmReader(client(chat, tmp_path)).read(crops(3))
    assert [r.text for r in out] == [".50", "Ø6", ""]
    assert out[2].confidence == 0.0


def test_prompt_transcribes_and_forbids_estimates():
    s = SYSTEM.lower()
    assert "transcribe" in s and "never estimate" in s and ".50" in SYSTEM


def test_retries_once_then_gives_up(tmp_path):
    calls = []
    def chat(messages):
        calls.append(messages)
        return "not json"
    assert VlmReader(client(chat, tmp_path)).read(crops(2)) is None
    assert len(calls) == 2
    assert "invalid" in calls[1][1]["content"][0]["text"]


def test_fenced_json_is_accepted(tmp_path):
    reply = '```json\n{"reads": [{"i": 1, "text": "4.00", "confidence": 0.9}]}\n```'
    out = VlmReader(client(lambda m: reply, tmp_path)).read(crops(1))
    assert out[0].text == "4.00"


def test_transport_error_returns_none(tmp_path):
    def chat(messages):
        raise TimeoutError("slow")
    assert VlmReader(client(chat, tmp_path)).read(crops(1)) is None


def test_large_sets_are_batched(tmp_path):
    calls = []
    def chat(messages):
        calls.append(1)
        n = int(messages[1]["content"][0]["text"].split(" tiles")[0].split()[-1])
        return json.dumps({"reads": [{"i": i, "text": "1", "confidence": 0.9} for i in range(1, n + 1)]})
    out = VlmReader(client(chat, tmp_path), batch=24).read(crops(50))
    assert len(out) == 50 and len(calls) == 3


def test_empty_input_needs_no_call(tmp_path):
    assert VlmReader(client(lambda m: 1 / 0, tmp_path)).read([]) == []


def test_unknown_reader_names_are_skipped(monkeypatch):
    monkeypatch.setenv("SKETCH_READERS", "nope")
    assert readers_from_env() == []


@pytest.mark.skipif(os.environ.get("SKETCH_MODEL_TESTS") != "1", reason="downloads a model")
def test_paddle_reader_reads_printed_digits():
    from s2c.reading.paddle import PaddleReader
    img = np.full((60, 160, 3), 255, np.uint8)
    cv2.putText(img, "40", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    out = PaddleReader().read([Crop(img, (0, 0, 160, 60))])
    assert out is not None and "40" in out[0].text


@pytest.mark.skipif(os.environ.get("SKETCH_MODEL_TESTS") != "1", reason="downloads a model")
def test_trocr_reader_reads_printed_digits():
    from s2c.reading.trocr import TrocrReader
    img = np.full((60, 160, 3), 255, np.uint8)
    cv2.putText(img, "40", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    out = TrocrReader().read([Crop(img, (0, 0, 160, 60))])
    assert out is not None and "40" in out[0].text


def test_moved_readers_declare_calibration_budget_and_cache():
    from s2c.reading.vlm import VlmReader
    r = VlmReader(client=object())
    assert (r.name, r.calibrated, r.cache_key) == ("vlm", False, None) and r.timeout_s == 20.0


def test_default_readers_include_trocr_so_a_cold_vlm_never_leaves_none(monkeypatch):
    monkeypatch.delenv("SKETCH_READERS", raising=False)
    names = [r.name for r in readers_from_env()]
    assert "trocr" in names
