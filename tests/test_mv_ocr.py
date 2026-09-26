import cv2
import numpy as np
import pytest

from s2c.multiview.ocr import Reading, link, parse_value, read_values, text_regions
from s2c.multiview.outline import PixelCircle, PixelOutline, extract


def test_parse_value():
    assert parse_value("60") == (60.0, "linear")
    assert parse_value(" 12,5 mm") == (12.5, "linear")
    assert parse_value("Ø6") == (6.0, "diameter")
    assert parse_value("⌀ 8") == (8.0, "diameter")
    assert parse_value("o6") == (6.0, "diameter")
    assert parse_value("R3") == (3.0, "radius")
    assert parse_value("D10") == (10.0, "diameter")
    assert parse_value("1O") == (10.0, "linear")
    assert parse_value("hello") is None
    assert parse_value("0") is None


def outline_with_holes(circular=False):
    return PixelOutline(outer=np.array([[400, 300], [1000, 300], [1000, 700], [400, 700]]),
                        circles=[PixelCircle(500, 400, 60), PixelCircle(900, 600, 60)],
                        bbox=(400, 300, 601, 401), circular=circular, shape=(1200, 1600))


def reading(value, kind, cx, cy):
    return Reading(float(value), kind, (cx - 20, cy - 15, 40, 30), 0.9, str(value))


def test_link_by_position():
    linked = link([reading(60, "linear", 700, 760), reading(40, "linear", 330, 500),
                   reading(6, "diameter", 560, 380), reading(99, "linear", 700, 500)], outline_with_holes())
    assert [(lv.axis, lv.hole_index) for lv in linked] == [("a", None), ("b", None), (None, 0), (None, None)]


def test_diameter_outside_a_round_outline_is_its_envelope():
    assert link([reading(80, "diameter", 700, 760)], outline_with_holes(circular=True))[0].axis == "ab"


def sketch_with_values():
    img = np.full((1200, 1600, 3), 255, np.uint8)
    cv2.rectangle(img, (400, 300), (1000, 700), (0, 0, 0), 4)
    cv2.putText(img, "60", (660, 790), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    cv2.putText(img, "40", (250, 520), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    return img


def test_text_regions_find_values_written_outside_the_outline():
    img = sketch_with_values()
    boxes = text_regions(img, extract(img))
    centres = sorted((x + w // 2, y + h // 2) for x, y, w, h in boxes)
    assert len(boxes) == 2
    assert abs(centres[0][0] - 290) < 40 and abs(centres[1][1] - 770) < 40


def test_read_values_keeps_numbers_and_drops_words():
    img = sketch_with_values()
    o = extract(img)
    assert [r.value_mm for r in read_values(img, o, lambda crop: ("60", 0.9))] == [60.0, 60.0]
    assert read_values(img, o, lambda crop: ("sixty", 0.9)) == []


@pytest.mark.gpu
def test_trocr_reads_printed_digits():
    pytest.importorskip("transformers")
    from s2c.multiview.ocr import trocr_reader
    img = np.full((80, 160, 3), 255, np.uint8)
    cv2.putText(img, "60", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    text, confidence = trocr_reader()(img)
    assert parse_value(text) == (60.0, "linear") and confidence > 0.3
