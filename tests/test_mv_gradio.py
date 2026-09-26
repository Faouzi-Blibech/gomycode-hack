import gradio as gr

import app_mv_gradio as A
from s2c.multiview import pipeline
from s2c.multiview.pipeline import MvPipeline, Observed
from s2c.multiview.spec import MvAbstain
from tests.mv_helpers import make_spec, rect
from tests.test_mv_pipeline import fake_reads, sketch
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
    assert "not a number" in done["message"] and "more than 0" in done["message"]
    assert done["model"] == gr.update()  # no build ran: leave whatever the 3D viewer already shows alone
    assert done["values"] == rows  # the user's own typed (bad) rows, so they can see and fix them


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


def test_a_rejected_face_stays_rejected_after_the_next_rebuild(tmp_path, monkeypatch):
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
    first = dict(zip(A.OUTPUTS, handlers.rebuild(rows, ["right"], out["state"])))
    assert first["rejected"]["choices"] == ["right"] and first["rejected"]["value"] == ["right"]
    assert "assumed rectangle" in next(c for _, c in first["faces"] if c.startswith("right:"))
    # rebuild again with exactly the checkbox value the app just returned: right must stay rejected, not come back
    second = dict(zip(A.OUTPUTS, handlers.rebuild(rows, first["rejected"]["value"], first["state"])))
    assert second["rejected"]["choices"] == ["right"] and second["rejected"]["value"] == ["right"]
    assert "assumed rectangle" in next(c for _, c in second["faces"] if c.startswith("right:"))


# ---- finding 2: an error must not wipe the other outputs, only a changed row is validated/recorded -----------

def test_a_bad_number_on_a_later_rebuild_keeps_the_current_faces_and_warnings(tmp_path, monkeypatch):
    handlers, out = analyzed(tmp_path, monkeypatch)
    rows = [[path, ENVELOPE[path], source] for path, _, source in out["values"]]
    done = dict(zip(A.OUTPUTS, handlers.rebuild(rows, [], out["state"])))
    assert done["model"] is not None
    bad_rows = [list(r) for r in done["values"]]
    bad_rows[0][1] = "abc"
    redo = dict(zip(A.OUTPUTS, handlers.rebuild(bad_rows, [], done["state"])))
    assert "not a number" in redo["message"]
    # the previous model, views, files and stats are kept (left alone), not cleared to empty
    assert redo["model"] == gr.update() and redo["views"] == gr.update()
    assert redo["files"] == gr.update() and redo["stats"] == gr.update()
    assert [c for _, c in redo["faces"]] == [c for _, c in done["faces"]]
    assert redo["warnings"] == done["warnings"]
    assert redo["values"] == bad_rows  # the user's own typed rows, so they can fix the bad cell


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
    assert rows[0][0] == "envelope.x_mm" and rows[0][1] == ""
    assert rows[0][2] == f"{A.MISSING} — suggested 60 mm"


def test_rebuild_without_touching_a_suggestion_does_not_confirm_it(tmp_path, monkeypatch):
    handlers, out = analyzed(tmp_path, monkeypatch)
    done = dict(zip(A.OUTPUTS, handlers.rebuild(out["values"], [], out["state"])))
    assert done["model"] is None and "missing" in done["message"]


def test_a_suggested_size_shows_as_a_hint_and_rebuild_does_not_confirm_it(tmp_path, monkeypatch):
    """front gives x and y (written), top gives x again: z is missing but scale-suggested from the top sketch's
    aspect ratio (spec 2026-09-23 section 5) - the exact scenario the source-text hint exists for."""
    monkeypatch.setattr(A, "OUT_ROOT", tmp_path / "out")
    monkeypatch.setattr(pipeline, "read_values", fake_reads([[(60, "below"), (40, "left")], [(60, "below")]]))
    handlers = A.Handlers(MvPipeline(reader=lambda crop: ("", 0.0)))
    paths = []
    for name, (w, h) in {"front.png": (600, 400), "top.png": (600, 100)}.items():
        path = tmp_path / name
        path.write_bytes(sketch(w, h))
        paths.append(str(path))
    tags = [["front.png", "front", "sketch"], ["top.png", "top", "sketch"]]
    out = dict(zip(A.OUTPUTS, handlers.analyze(paths, tags, "none")))
    z_row = next(r for r in out["values"] if r[0] == "envelope.z_mm")
    assert z_row[1] == "" and "suggested" in z_row[2]
    done = dict(zip(A.OUTPUTS, handlers.rebuild(out["values"], [], out["state"])))
    assert done["model"] is None and "envelope.z_mm" not in done["state"]["edits"]
