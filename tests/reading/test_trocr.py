import cv2
import numpy as np
import pytest
import torch

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
