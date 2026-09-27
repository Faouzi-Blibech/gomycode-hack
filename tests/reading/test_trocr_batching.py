import numpy as np

from s2c.reading import trocr
from s2c.reading.base import Crop, ReaderResult


def test_a_whole_sheet_is_read_in_small_batches(monkeypatch):
    sizes = []
    monkeypatch.setattr(trocr, "_load", lambda model_id, device: (None, None))

    def fake_run(processor, model, device, crops, max_new_tokens):
        sizes.append(len(crops))
        return [ReaderResult(text=str(c.box[0]), confidence=0.9) for c in crops]

    monkeypatch.setattr(trocr, "_run", fake_run)
    monkeypatch.setenv("TROCR_BATCH", "8")
    crops = [Crop(image=np.zeros((8, 8, 3), np.uint8), box=(i, 0, 8, 8)) for i in range(20)]
    out = trocr.TrocrReader(device="cpu").read(crops)
    assert sizes == [8, 8, 4]
    assert [r.text for r in out] == [str(i) for i in range(20)]
