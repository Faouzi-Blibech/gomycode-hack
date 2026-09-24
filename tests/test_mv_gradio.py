import gradio as gr

import app_mv_gradio as A
from s2c.multiview.pipeline import MvPipeline, Observed
from s2c.multiview.spec import MvAbstain
from tests.mv_helpers import make_spec, rect
from tests.test_mv_pipeline import sketch
from tests.test_mv_qwen_faces import fake_gen, silhouette

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


# ---- finding 1: merged badge on the opposite face, reject list scoped to AI-filled faces ----------------------

def test_merged_badge_shows_for_a_photo_merged_on_the_opposite_face():
    spec = make_spec((40, 30, 20)).model_copy(update={"warnings": ["back: merged 2 photos, agreement 0.85"]})
    observed = Observed([], [], {}, [], filled_by={"front": "mirrored", "top": "observed", "right": "observed"})
    front_caption = next(c for _, c in A.face_gallery(observed, spec) if c.startswith("front:"))
    assert "merged 2 photos" in front_caption


def test_the_reject_list_only_offers_ai_drawn_faces(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "OUT_ROOT", tmp_path / "out")
    gen = fake_gen(silhouette(rect(10, 40), 10, 40))
    handlers = A.Handlers(MvPipeline(image_gen=gen))
    paths = []
    for name, (w, h) in {"front.png": (600, 400), "top.png": (600, 100)}.items():
        path = tmp_path / name
        path.write_bytes(sketch(w, h))
        paths.append(str(path))
    tags = [["front.png", "front", "sketch"], ["top.png", "top", "sketch"]]
    out = dict(zip(A.OUTPUTS, handlers.analyze(paths, tags, "none")))
    rows = [[path, ENVELOPE[path], source] for path, _, source in out["values"]]
    done = dict(zip(A.OUTPUTS, handlers.rebuild(rows, [], out["state"])))
    assert done["rejected"]["choices"] == ["right"]


# ---- finding 2: an error must not wipe the other outputs, only a changed row is validated/recorded -----------

def test_a_bad_number_on_a_later_rebuild_keeps_the_current_faces_and_warnings(tmp_path, monkeypatch):
    handlers, out = analyzed(tmp_path, monkeypatch)
    rows = [[path, ENVELOPE[path], source] for path, _, source in out["values"]]
    done = dict(zip(A.OUTPUTS, handlers.rebuild(rows, [], out["state"])))
    assert done["model"] is not None
    bad_rows = [list(r) for r in done["values"]]
    bad_rows[0][1] = "abc"
    redo = dict(zip(A.OUTPUTS, handlers.rebuild(bad_rows, [], done["state"])))
    assert "not a number" in redo["message"] and redo["model"] is None
    assert [c.split(":")[0] for _, c in redo["faces"]] == [c.split(":")[0] for _, c in done["faces"]]
    assert redo["warnings"] == done["warnings"]


def test_unchanged_rows_are_not_recorded_as_edits():
    shown = {"envelope.x_mm": (60.0, "user_written")}
    edits, errors = A.parse_edits([["envelope.x_mm", "60", "user_written"]], shown)
    assert edits == {} and errors == []


def test_a_cleared_box_reverts_the_edit():
    shown = {"envelope.x_mm": (60.0, "user_edited")}
    edits, errors = A.parse_edits([["envelope.x_mm", "", "user_edited"]], shown)
    assert edits == {"envelope.x_mm": None} and errors == []


# ---- finding 3: a suggestion is a placeholder, never a silently confirmed value -------------------------------

def test_a_suggested_envelope_value_is_not_pre_filled():
    abstain = MvAbstain(stage="dimensions", reason="missing_x", remedy="Enter the width in mm.",
                        partial={"known": {}, "missing": ["envelope.x_mm"], "suggested": {"envelope.x_mm": 60.0}})
    rows = A.value_rows(None, abstain, {})
    assert rows[0] == ["envelope.x_mm", "", A.MISSING]


def test_rebuild_without_touching_a_suggestion_does_not_confirm_it(tmp_path, monkeypatch):
    handlers, out = analyzed(tmp_path, monkeypatch)
    done = dict(zip(A.OUTPUTS, handlers.rebuild(out["values"], [], out["state"])))
    assert done["model"] is None and "missing" in done["message"]
