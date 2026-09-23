import cv2
import numpy as np
import pytest

from s2c.multiview import pipeline
from s2c.multiview.ocr import Reading
from s2c.multiview.pipeline import ImageInput, MvPipeline
from s2c.multiview.spec import MvAbstain


def sketch(w_px, h_px, circles=()):
    img = np.full((1200, 1600, 3), 255, np.uint8)
    x0, y0 = (1600 - w_px) // 2, (1200 - h_px) // 2
    cv2.rectangle(img, (x0, y0), (x0 + w_px, y0 + h_px), (0, 0, 0), 4)
    for cx, cy, r in circles:
        cv2.circle(img, (x0 + cx, y0 + cy), r, (0, 0, 0), 3)
    return cv2.imencode(".png", img)[1].tobytes()


def fake_reads(per_image):
    calls = iter(per_image)

    def read_values(bgr, outline, reader, batch=None):
        x, y, w, h = outline.bbox
        out = []
        for value, where in next(calls):
            box = (x + w // 2 - 20, y + h + 30, 40, 30) if where == "below" else (x - 80, y + h // 2 - 15, 40, 30)
            out.append(Reading(float(value), "linear", box, 0.9, str(value)))
        return out

    return read_values


def test_front_and_top_sketches_to_a_built_part(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "read_values", fake_reads([[(60, "below"), (40, "left")], [(60, "below")]]))
    pipe = MvPipeline(reader=lambda crop: ("", 0.0))
    observed = pipe.observe([ImageInput(sketch(600, 400, circles=[(100, 300, 30)]), "front", "sketch"),
                             ImageInput(sketch(600, 100), "top", "sketch")])
    first = pipe.fuse(observed)
    assert isinstance(first, MvAbstain) and first.reason == "missing_z"
    assert abs(first.partial["suggested"]["envelope.z_mm"] - 10) <= 0.5
    spec = pipe.fuse(observed, {"envelope.z_mm": 10.0})
    assert spec.views.top.source == "observed" and spec.views.right.source == "assumed"
    assert spec.features[0].diameter_mm == pytest.approx(5.5)  # about 57 px -> 5.7 mm -> M5 clearance
    result = pipe.build(spec, tmp_path, observed.masks)
    assert result.stl.exists() and result.step.exists()
    assert result.iou["front"] > 0.85 and result.iou["top"] > 0.85


def test_an_untagged_image_without_a_model_abstains():
    res = MvPipeline().observe([ImageInput(sketch(600, 400))])
    assert isinstance(res, MvAbstain) and res.reason == "face_unknown"


def test_a_file_that_is_not_an_image_abstains():
    res = MvPipeline().observe([ImageInput(b"not an image", "front")])
    assert isinstance(res, MvAbstain) and res.reason == "bad_image"


def test_build_errors_become_abstentions(tmp_path):
    from s2c.multiview.spec import MultiViewSpec
    pipe = MvPipeline()
    observed = pipe.observe([ImageInput(sketch(600, 400), "front", "sketch")])
    spec = pipe.fuse(observed, {"envelope.x_mm": 60, "envelope.y_mm": 40, "envelope.z_mm": 5})
    data = spec.model_dump()
    data["features"] = [{"type": "hole", "face": "front", "a_mm": 70.0, "b_mm": 10.0, "diameter_mm": 6.0}]
    data["provenance"].update({"features[0].a_mm": "user_edited", "features[0].b_mm": "user_edited",
                               "features[0].diameter_mm": "user_edited"})
    res = pipe.build(MultiViewSpec.model_validate(data), tmp_path)
    assert isinstance(res, MvAbstain) and res.stage == "build" and res.reason == "feature_outside_part"


def test_the_batch_reader_feeds_ocr(monkeypatch):
    seen = []

    def fake(bgr, outline, reader, batch=None):
        seen.append((reader, batch))
        return []

    def batch(crops):
        return []

    monkeypatch.setattr(pipeline, "read_values", fake)
    observed = MvPipeline(batch_reader=batch).observe([ImageInput(sketch(600, 400), "front", "sketch")])
    assert seen == [(None, batch)]
    assert "OCR unavailable: enter the dimensions by hand" not in observed.warnings


def test_default_pipeline_reads_with_qwen_vl_when_configured(monkeypatch):
    for key, value in {"VLM_BASE_URL": "http://localhost:9/v1", "VLM_MODEL": "m", "VLM_API_KEY": "k"}.items():
        monkeypatch.setenv(key, value)
    assert pipeline.default_pipeline().batch_reader is not None
