import logging

import cv2
import numpy as np

from s2c.fakes import builder, metrology, ocr, views
from s2c.partspec.models import PartSpec
from s2c.pipeline import default_pipeline, fake_pipeline


def sketch_bytes():
    img = np.full((600, 800, 3), 255, np.uint8)
    cv2.rectangle(img, (250, 200), (550, 400), (0, 0, 0), 3)
    _ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def test_fake_pipeline_runs_sketch_end_to_end(tmp_path):
    p = fake_pipeline()
    res = p.analyze(sketch_bytes(), "sketch")
    assert res.abstain is None and isinstance(res.partspec, PartSpec)
    assert res.input_mask.shape == (512, 512)
    built = p.build_and_verify(res.partspec, res.input_mask, tmp_path)
    assert built.stl_path.exists()
    assert 0.0 <= built.iou <= 1.0


def test_default_pipeline_falls_back_to_fakes_when_modules_missing(caplog):
    # The real s2c.ocr, s2c.metrology, s2c.builder and s2c.views modules do not
    # exist on this branch yet, so default_pipeline must resolve every stage to
    # the fake implementation and warn once per missing module.
    with caplog.at_level(logging.WARNING):
        p = default_pipeline(client=None)
    assert p.annotations is ocr.read_annotations
    assert p.measure is metrology.measure
    assert p.build is builder.build
    assert p.export is builder.export
    assert p.silhouettes is views.silhouettes
    for name in ("s2c.ocr", "s2c.metrology", "s2c.builder", "s2c.views"):
        assert any(name in message for message in caplog.messages), f"missing warning for {name}"
