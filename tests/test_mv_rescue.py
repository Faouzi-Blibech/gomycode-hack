import cv2
import numpy as np
import pytest

from s2c.multiview.outline import extract
from s2c.multiview.pipeline import ImageInput, MvPipeline
from s2c.multiview.qwen_faces import rescue_sketch
from s2c.multiview.qwen_image import ImageGenError
from s2c.multiview.spec import MvAbstain
from tests.test_mv_qwen_faces import fake_gen


def broken_sketch(gap=30):
    """A 600 x 400 rectangle drawn in pen whose left side stops short: a gap at the top-left corner."""
    img = np.full((1200, 1600, 3), 255, np.uint8)
    x0, y0, x1, y1 = 500, 400, 1100, 800
    for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0 + gap))):
        cv2.line(img, a, b, (0, 0, 0), 4)
    return img


def clean(w=600, h=400):
    img = np.full((1200, 1600, 3), 255, np.uint8)
    cv2.rectangle(img, (500, 400), (500 + w, 400 + h), (0, 0, 0), -1)
    return img


def test_a_broken_sketch_has_no_outline():
    res = extract(broken_sketch())
    assert isinstance(res, MvAbstain) and res.reason == "no_outline"


def test_rescue_accepts_a_faithful_redraw():
    outline = rescue_sketch(broken_sketch(), fake_gen(clean()))
    assert outline is not None and abs(outline.bbox[2] - 601) <= 4


def test_rescue_rejects_a_distorted_redraw_or_a_failure():
    assert rescue_sketch(broken_sketch(), fake_gen(clean(600, 250))) is None
    assert rescue_sketch(broken_sketch(), fake_gen(ImageGenError("503"))) is None


def test_the_pipeline_rescues_a_broken_sketch_and_warns():
    data = cv2.imencode(".png", broken_sketch())[1].tobytes()
    observed = MvPipeline(image_gen=fake_gen(clean())).observe([ImageInput(data, "front", "sketch")])
    assert not isinstance(observed, MvAbstain)
    assert "front: sketch cleaned by Qwen-Image, check it" in observed.warnings
    assert observed.observations[0].confidence == pytest.approx(0.9 * 0.8)
    assert isinstance(MvPipeline().observe([ImageInput(data, "front", "sketch")]), MvAbstain)
