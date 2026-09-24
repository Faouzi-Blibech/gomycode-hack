import logging

import cv2
import numpy as np

from s2c.fakes import builder, metrology, ocr, views
from s2c.partspec.models import Abstain, Annotation, Annotations, PartSpec, Topology
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


def test_build_and_verify_logs_the_exception_and_abstains_without_leaking_it(tmp_path, caplog):
    def boom(spec):
        raise RuntimeError("boom")

    p = fake_pipeline()
    p.build = boom
    res = p.analyze(sketch_bytes(), "sketch")
    assert res.partspec is not None

    with caplog.at_level(logging.ERROR, logger="s2c.pipeline"):
        out = p.build_and_verify(res.partspec, res.input_mask, tmp_path)

    assert isinstance(out, Abstain)
    assert out.stage == "build" and out.reason == "build_failed"
    assert "boom" not in out.reason and "boom" not in out.remedy

    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert error_records, "expected an ERROR log record for the build failure"
    assert any(r.exc_info for r in error_records), "expected the log record to carry exc_info/traceback"


def test_remerge_fills_a_missing_value_from_user_edits():
    topology = Topology.model_validate({"part_type": "plate", "holes": [], "confidence": 0.9})
    annotations = Annotations(items=[
        Annotation(value_mm=60.0, kind="linear", bbox_px=(0.0, 0.0, 1.0, 1.0),
                  linked_to="width", hole_index=None, confidence=0.9),
        Annotation(value_mm=40.0, kind="linear", bbox_px=(0.0, 0.0, 1.0, 1.0),
                  linked_to="height", hole_index=None, confidence=0.9),
    ], confidence=0.9)

    p = fake_pipeline()
    out = p.remerge(topology, source_input="sketch", annotations=annotations)
    assert isinstance(out, Abstain) and out.reason == "missing_thickness"

    spec = p.remerge(topology, source_input="sketch", annotations=annotations,
                     user_values={"thickness": 5.0})
    assert isinstance(spec, PartSpec)
    assert spec.part.thickness_mm == 5.0
    assert spec.provenance["part.thickness_mm"] == "user_edited"
