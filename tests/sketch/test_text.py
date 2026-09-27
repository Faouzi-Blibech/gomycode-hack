import os

import numpy as np
import pytest

from s2c.sketch.models import SketchAbstain
from s2c.sketch.text import erase_mask, find_text_boxes, read_texts
from tests.sketch.synth import Sheet, TruthReader, _overlap, bridge_block


def sheet():
    return bridge_block(Sheet())


def test_boxes_cover_every_written_text():
    sh = sheet()
    boxes = find_text_boxes(sh.ink(), 3.0)
    for s, b in sh.texts:
        assert any(_overlap(box, b) > 0.6 for box in boxes), s
    for box in boxes:  # no box swallows two texts
        assert sum(_overlap(box, b) > 0.6 for _, b in sh.texts) <= 1


def test_two_agreeing_readers_give_written_values_and_labels():
    sh = sheet()
    items = read_texts(sh.bgr(), find_text_boxes(sh.ink(), 3.0),
                       [TruthReader(sh.texts, "a"), TruthReader(sh.texts, "b")])
    dims = [t for t in items if t.role == "dimension"]
    labels = {t.label for t in items if t.role == "label"}
    assert labels == {"top", "front", "right"}
    assert len(dims) == 18
    assert all(t.badge == "written" for t in dims)
    assert sorted(t.parsed.value for t in dims).count(12.5) == 11


def test_disagreement_is_uncertain_with_both_candidates():
    sh = sheet()
    items = read_texts(sh.bgr(), find_text_boxes(sh.ink(), 3.0),
                       [TruthReader(sh.texts, "a"), TruthReader(sh.texts, "b", swap={"100": "700"})])
    t = next(t for t in items if t.role == "dimension" and 100.0 in t.candidates)
    assert t.badge == "uncertain" and t.candidates == [100.0, 700.0]
    assert [r.reader for r in t.readings] == ["a", "b"]


def test_a_single_reader_is_never_trusted_alone():
    sh = sheet()
    items = read_texts(sh.bgr(), find_text_boxes(sh.ink(), 3.0), [TruthReader(sh.texts)])
    assert all(t.badge == "uncertain" for t in items if t.role == "dimension")


def test_a_small_circle_read_as_O_stays_in_the_geometry():
    sh = Sheet()
    sh.circle((600, 500), 14)
    sh.texts.append(("O", (582, 482, 36, 36)))  # what a reader would say about the circle
    sh.text("40", (900, 500))
    items = read_texts(sh.bgr(), find_text_boxes(sh.ink(), 3.0),
                       [TruthReader(sh.texts, "a"), TruthReader(sh.texts, "b")])
    mask = erase_mask(items, sh.img.shape[:2])
    assert mask[500, 586] == 0 and mask[486, 600] == 0  # the circle is not erased
    assert mask[500, 900] == 255                         # the number is


def test_sideways_vertical_text_is_read():
    sh = Sheet()
    sh.line((400, 300), (400, 700), 1)
    sh.text("37.5", (380, 500), rotate=True)
    readers = [TruthReader(sh.texts, "a", upright_only=True),
               TruthReader(sh.texts, "b", upright_only=True)]
    items = read_texts(sh.bgr(), find_text_boxes(sh.ink(), 3.0), readers)
    dims = [t for t in items if t.role == "dimension"]
    assert len(dims) == 1 and dims[0].parsed.value == 37.5 and dims[0].badge == "written"


def test_readers_down_abstains():
    class Down:
        name = "down"

        def read(self, crops):
            return None
    sh = sheet()
    out = read_texts(sh.bgr(), find_text_boxes(sh.ink(), 3.0), [Down(), Down()])
    assert isinstance(out, SketchAbstain) and out.reason == "readers_unavailable"


def test_no_text_needs_no_reader():
    assert read_texts(np.full((100, 100, 3), 255, np.uint8), [], []) == []


def test_classical_detector_is_the_fallback(monkeypatch):
    from s2c.sketch.text import detect_text_boxes
    sh = sheet()
    monkeypatch.setenv("SKETCH_TEXT_DETECTOR", "classical")
    assert detect_text_boxes(sh.bgr(), sh.ink(), 3.0) == find_text_boxes(sh.ink(), 3.0)


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


@pytest.mark.skipif(os.environ.get("SKETCH_MODEL_TESTS") != "1", reason="downloads a model")
def test_paddle_detector_finds_the_written_values(monkeypatch):
    from s2c.sketch.text import detect_text_boxes
    sh = sheet()
    monkeypatch.setenv("SKETCH_TEXT_DETECTOR", "paddle")
    boxes = detect_text_boxes(sh.bgr(), sh.ink(), 3.0)
    found = sum(any(_overlap(box, b) > 0.5 for box in boxes) for _, b in sh.texts)
    assert found >= 0.9 * len(sh.texts)
