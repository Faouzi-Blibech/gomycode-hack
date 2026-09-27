import cv2
import numpy as np
import pytest
import torch

from s2c.reading import Crop
from s2c.reading.trocr import ReaderResult, memory_fraction, token_confidences


def test_confidence_is_the_geometric_mean_of_the_real_tokens():
    logprobs = np.log(np.array([[0.9, 0.8, 1.0]]))
    assert token_confidences(logprobs, np.array([[1, 1, 1]])) == pytest.approx([(0.9 * 0.8) ** (1 / 3)])


def test_padding_after_the_end_token_does_not_count():
    logprobs = np.array([[np.log(0.5), np.log(0.5), 0.0, 0.0, 0.0]])  # two real tokens, then padding at log 1
    assert token_confidences(logprobs, np.array([[1, 1, 0, 0, 0]])) == pytest.approx([0.5])


def test_a_row_without_real_tokens_scores_zero():
    assert token_confidences(np.zeros((1, 3)), np.zeros((1, 3))) == [0.0]


def test_memory_fraction_scales_the_budget_against_the_total():
    total = 8 * 1024**3
    assert memory_fraction(1.0, total) == pytest.approx(1 / 8)


def test_memory_fraction_is_clamped_to_at_most_one():
    assert memory_fraction(100.0, 8 * 1024**3) == 1.0


def test_memory_fraction_never_reaches_zero():
    assert 0.0 < memory_fraction(0.0, 8 * 1024**3) <= 1.0
    assert 0.0 < memory_fraction(-1.0, 8 * 1024**3) <= 1.0


def test_an_out_of_memory_error_moves_the_reader_to_cpu_and_retries_once(monkeypatch):
    from s2c.reading import trocr
    from s2c.reading.trocr import TrocrReader

    reader = TrocrReader(device="cuda")
    calls = []

    def fake_load(model_id, device):
        calls.append(("load", device))
        return ("processor", "model")

    def fake_run(processor, model, device, crops, max_new_tokens):
        calls.append(("run", device))
        if device == "cuda":
            raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB")
        return [ReaderResult(text="60", confidence=0.9)]

    monkeypatch.setattr(trocr, "_load", fake_load)
    monkeypatch.setattr(trocr, "_run", fake_run)
    crop = Crop(np.zeros((10, 10, 3), np.uint8), (0, 0, 10, 10))
    out = reader.read([crop])
    assert reader.device == "cpu"
    assert [r.text for r in out] == ["60"]
    assert calls == [("load", "cuda"), ("run", "cuda"), ("load", "cpu"), ("run", "cpu")]


@pytest.mark.gpu
def test_trocr_reads_a_batch_of_printed_digits(monkeypatch):
    pytest.importorskip("transformers")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    from s2c.reading import trocr
    from s2c.reading.trocr import TrocrReader

    def digits(s):
        img = np.full((80, 200, 3), 255, np.uint8)
        cv2.putText(img, s, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
        return Crop(img, (0, 0, 200, 80))

    reader = TrocrReader()
    _, model = trocr._load(reader.model_id, reader.device)  # a real load, bypassing warm() which conftest no-ops
    assert model.dtype == torch.float16  # cuda: half precision keeps GPU memory down alongside Qwen
    out = reader.read([digits("60"), digits("125")])
    # TrOCR was trained on sentences and often ends a read with a period; parsing, not the reader, drops it
    assert [r.text.replace(" ", "").rstrip(".") for r in out] == ["60", "125"]
    assert all(r.confidence > 0.3 for r in out)
