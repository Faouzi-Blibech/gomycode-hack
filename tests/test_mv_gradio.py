import gradio as gr

import app_mv_gradio as A
from s2c.multiview.pipeline import MvPipeline
from tests.test_mv_pipeline import sketch

ENVELOPE = {"envelope.x_mm": 60, "envelope.y_mm": 40, "envelope.z_mm": 10}


def analyzed(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "OUT_ROOT", tmp_path / "out")
    handlers = A.Handlers(MvPipeline())
    paths = []
    for name, (w, h) in {"front.png": (600, 400), "top.png": (600, 100)}.items():
        path = tmp_path / name
        path.write_bytes(sketch(w, h))
        paths.append(str(path))
    tags = [["front.png", "front", "sketch"], ["top.png", "top", "sketch"]]
    return handlers, dict(zip(A.OUTPUTS, handlers.analyze(paths, tags, "none")))


def test_the_app_builds():
    assert isinstance(A.build_app(MvPipeline()), gr.Blocks)


def test_analyze_asks_for_the_envelope_and_rebuild_makes_the_files(tmp_path, monkeypatch):
    handlers, out = analyzed(tmp_path, monkeypatch)
    assert "missing_x" in out["message"]
    assert [r[0] for r in out["values"]] == list(ENVELOPE) and out["values"][0][2] == A.MISSING
    rows = [[path, ENVELOPE[path], source] for path, _, source in out["values"]]
    done = dict(zip(A.OUTPUTS, handlers.rebuild(rows, [], out["state"])))
    assert done["model"].endswith("part.stl") and len(done["files"]) >= 2
    assert [r[2] for r in done["values"][:3]] == ["user_edited"] * 3
    assert [caption.split(":")[0] for _, caption in done["faces"]] == ["front", "top", "right"]


def test_bad_numbers_are_reported_not_raised(tmp_path, monkeypatch):
    handlers, out = analyzed(tmp_path, monkeypatch)
    rows = [["envelope.x_mm", "abc", A.MISSING], ["envelope.y_mm", "-5", A.MISSING], ["envelope.z_mm", "0", A.MISSING]]
    done = dict(zip(A.OUTPUTS, handlers.rebuild(rows, [], out["state"])))
    assert "not a number" in done["message"] and "more than 0" in done["message"] and done["model"] is None


def test_a_wrong_face_tag_is_explained(tmp_path):
    path = tmp_path / "a.png"
    path.write_bytes(sketch(600, 400))
    out = dict(zip(A.OUTPUTS, A.Handlers(MvPipeline()).analyze([str(path)], [["a.png", "side", "sketch"]], "none")))
    assert "face must be one of" in out["message"]


def test_rebuild_before_analyze_explains():
    out = dict(zip(A.OUTPUTS, A.Handlers(MvPipeline()).rebuild([], [], None)))
    assert out["message"] == "Analyze images first."


def test_outputs_and_uploads_are_deleted_after_an_hour(tmp_path):
    import os
    import time
    old, fresh = tmp_path / "old", tmp_path / "fresh"
    old.mkdir()
    fresh.mkdir()
    (old / "part.stl").write_text("solid")
    two_hours_ago = time.time() - 7200
    os.utime(old, (two_hours_ago, two_hours_ago))
    A.sweep_outputs(tmp_path)
    assert not old.exists() and fresh.exists()
    assert A.build_app(MvPipeline()).delete_cache == (3600, 3600)
