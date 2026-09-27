import pytest
from pydantic import ValidationError

from s2c.sketch.models import Dimension, Issue, Reading, Size, SketchAbstain, SketchReading, View


def minimal(**over):
    data = {
        "image_size_px": (1200, 1600),
        "views": [View(name="front", label_text="FRONT", bbox_px=(0, 0, 10, 10), size_mm=(40.0, 20.0))],
        "entities": [], "dimensions": [], "features": [],
        "envelope": {"x": Size(value=40.0, badge="written", evidence="reader")},
        "issues": [], "timings_ms": {"total": 1.0},
    }
    data.update(over)
    return SketchReading(**data)


def test_round_trip_json():
    r = minimal()
    again = SketchReading.model_validate_json(r.model_dump_json())
    assert again == r
    assert again.version == "sketch-1" and again.units == "mm"


def test_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        View(name="front", label_text=None, bbox_px=(0, 0, 1, 1), size_mm=(1, 1), width_mm=3)


def test_badge_is_a_closed_set():
    with pytest.raises(ValidationError):
        Size(value=1.0, badge="guessed", evidence="reader")


def test_dimension_keeps_both_readings():
    d = Dimension(id="d1", view="front", kind="linear", value=1.5, text_raw="1.50",
                  readings=[Reading(reader="paddle", text="1.50", confidence=0.9),
                            Reading(reader="vlm", text="1.5O", confidence=0.7)],
                  measures=["r1", "r2"], axis="a", badge="written", evidence="reader",
                  bbox_px=(1, 2, 3, 4))
    assert len(d.readings) == 2 and d.candidates == []


def test_abstain_and_issue():
    a = SketchAbstain(stage="capture", reason="sheet_not_found", remedy="Retake.")
    r = minimal(abstain=a, issues=[Issue(severity="red", kind="unit", message="mm?", targets=[])])
    assert r.abstain.reason == "sheet_not_found"
