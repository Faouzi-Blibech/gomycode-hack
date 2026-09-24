import os
import time
from pathlib import Path

import gradio as gr
import numpy as np
import pytest

from s2c.multiview import artifacts
from s2c.multiview.artifacts import ExportResult
from s2c.multiview.pipeline import MvPipeline
from s2c.multiview.settings import AiSettings, ExportSettings, GeometrySettings, MeshSettings, PrintSettings
from s2c.multiview.spec import MvAbstain
from s2c.studio import handlers
from s2c.studio import session as session_mod
from s2c.studio.app import build_app
from s2c.studio.handlers import Studio, parse_size
from s2c.studio.session import SessionStore
from tests.test_mv_pipeline import sketch
from tests.test_mv_qwen_faces import fake_gen


@pytest.fixture
def studio(tmp_path, monkeypatch):
    artifacts.clear_cache()
    monkeypatch.setattr("s2c.multiview.slice.find_slicer", lambda: None)
    return Studio(MvPipeline(), root=tmp_path / "files")


def with_images(studio, tmp_path, faces=(("front", 600, 400), ("top", 600, 100))):
    sid = studio.store.new()
    paths = []
    for face, w, h in faces:
        path = tmp_path / f"{face}.png"
        path.write_bytes(sketch(w, h))
        paths.append(str(path))
    studio.add_images(sid, paths)
    for item, (face, _, _) in zip(studio.store.get(sid).items, faces):
        studio.set_face(sid, item.id, face)
        studio.set_kind(sid, item.id, "sketch")
    return sid


def test_the_app_builds():
    assert isinstance(build_app(MvPipeline()), gr.Blocks)


def test_coverage_shows_which_faces_are_given(studio, tmp_path):
    sid = with_images(studio, tmp_path)
    html = studio.coverage_html(sid)
    assert "front" in html and "top" in html and "AI will draw" in html


def test_analyze_asks_for_sizes_then_build_and_export(studio, tmp_path):
    sid = with_images(studio, tmp_path)
    review = studio.analyze(sid, "none", AiSettings())
    assert not review.ok and "Missing x" in review.message_html
    assert all(review.sizes[a]["required"] for a in "xyz")
    review, model = studio.build(sid, {"x": "60", "y": "40", "z": "10"}, review.rows, [], GeometrySettings())
    assert model.ok and Path(model.preview).exists() and len(model.views) == 6
    exported = studio.export(sid, ExportSettings(formats=["stl", "step", "dxf"]), MeshSettings(), PrintSettings())
    assert {Path(f).name for f in exported.files} == {"part.stl", "part.step", "drawing.dxf"}
    assert Path(exported.zip_path).exists()


def test_sizes_accept_commas_and_units_and_reject_junk():
    assert [parse_size(t) for t in ("42,5", "42 mm", " 42 ")] == [42.5, 42.0, 42.0]
    for junk in ("abc", "0", "-3", "", "nan"):
        assert parse_size(junk) is None


def test_bad_sizes_are_a_message(studio, tmp_path):
    sid = with_images(studio, tmp_path)
    review = studio.analyze(sid, "none", AiSettings())
    review, model = studio.build(sid, {"x": "abc", "y": "40", "z": "-3"}, review.rows, [], GeometrySettings())
    assert not model.ok and "Width" in review.message_html and "Depth" in review.message_html


def fail_big_fillets(monkeypatch):
    real = handlers.build_part

    def fails_on_big_fillets(spec, geometry, root):  # the real failure is pinned in the artifacts tests
        if geometry.finish == "fillet" and geometry.finish_mm > 5:
            return MvAbstain(stage="build", reason="fillet_failed", remedy="Reduce the fillet radius.")
        return real(spec, geometry, root)

    monkeypatch.setattr(handlers, "build_part", fails_on_big_fillets)


def test_a_failed_finish_keeps_the_review(studio, tmp_path, monkeypatch):
    fail_big_fillets(monkeypatch)
    sid = with_images(studio, tmp_path)
    review = studio.analyze(sid, "none", AiSettings())
    sizes = {"x": "60", "y": "40", "z": "10"}
    _, model = studio.build(sid, sizes, review.rows, [], GeometrySettings(finish="fillet", finish_mm=6.0))
    assert not model.ok and "fillet" in model.message_html.lower()
    _, model = studio.rebuild_geometry(sid, GeometrySettings(finish="fillet", finish_mm=1.0))
    assert model.ok


def test_a_failed_finish_opens_the_model_step(studio, tmp_path, monkeypatch):
    fail_big_fillets(monkeypatch)
    sid = with_images(studio, tmp_path)
    review = studio.analyze(sid, "none", AiSettings())
    sizes = {"x": "60", "y": "40", "z": "10"}
    _, model = studio.build(sid, sizes, review.rows, [], GeometrySettings(finish="fillet", finish_mm=6.0))
    assert model.ok is False and model.open_step == 2  # the finish controls live in step 3
    assert studio.rebuild_geometry(sid, GeometrySettings(finish="none"))[1].ok


def test_a_failed_part_without_a_finish_stays_on_the_review(studio, tmp_path, monkeypatch):
    monkeypatch.setattr(handlers, "build_part",
                        lambda spec, geometry, root: MvAbstain(stage="build", reason="bad", remedy="Fix it."))
    sid = with_images(studio, tmp_path)
    review = studio.analyze(sid, "none", AiSettings())
    _, model = studio.build(sid, {"x": "60", "y": "40", "z": "10"}, review.rows, [], GeometrySettings())
    assert model.ok is False and model.open_step is None


def app_fn(app, name):
    return next(f.fn for f in app.fns.values() if f.name == name)


def app_event(app, name):
    return next(f for f in app.fns.values() if f.name == name)


def test_the_build_button_shows_a_failed_finish_in_step_3(studio, tmp_path, monkeypatch):
    fail_big_fillets(monkeypatch)
    sid = with_images(studio, tmp_path)
    review = studio.analyze(sid, "none", AiSettings())
    on_build = app_fn(build_app(studio=studio), "on_build")
    out = on_build(sid, "60", "40", "10", review.rows, [], True, "medium", "fillet", 6.0, "all_vertical")
    step, model_msg = out[0], out[-4]
    assert isinstance(step, gr.Walkthrough) and step.selected == 2 and "Fillet failed" in model_msg


def test_old_builds_are_swept_when_a_part_is_built(studio, tmp_path):
    old = studio.root / "old"
    old.mkdir(parents=True)
    past = time.time() - 7200
    os.utime(old, (past, past))
    sid = with_images(studio, tmp_path)
    review = studio.analyze(sid, "none", AiSettings())
    _, model = studio.build(sid, {"x": "60", "y": "40", "z": "10"}, review.rows, [], GeometrySettings())
    assert model.ok and not old.exists()


def test_suggested_sizes_fill_the_axes_that_have_one(studio, tmp_path):
    sid = with_images(studio, tmp_path)
    assert studio.suggested_sizes(sid) == {}  # nothing analysed yet
    review = studio.analyze(sid, "none", AiSettings())
    review, model = studio.build(sid, {"x": "60", "y": "40", "z": ""}, review.rows, [], GeometrySettings())
    assert not model.ok and "Missing z" in review.message_html
    hint = review.sizes["z"]["placeholder"].removeprefix("suggested ")
    assert parse_size(hint) and studio.suggested_sizes(sid) == {"z": hint}


def test_cancel_stops_analyze():
    app = build_app(MvPipeline())
    analyze = next(i for i, f in app.fns.items() if f.name == "on_analyze")
    cancels = [f for f in app.fns.values() if f.is_cancel_function and analyze in f.cancels]
    assert cancels and app.blocks[cancels[0].targets[0][0]].value == "Cancel"


def test_export_stats_show_filament_in_grams_and_metres(studio, tmp_path, monkeypatch):
    sid = with_images(studio, tmp_path)
    review = studio.analyze(sid, "none", AiSettings())
    studio.build(sid, {"x": "60", "y": "40", "z": "10"}, review.rows, [], GeometrySettings())
    gcode = tmp_path / "part.gcode"
    gcode.write_text("; fake")
    monkeypatch.setattr(handlers, "export_part",
                        lambda *a, **k: ExportResult({"gcode": gcode}, {"gcode": 6}, 2692, 12.4))
    exported = studio.export(sid, ExportSettings(formats=["gcode"]), MeshSettings(), PrintSettings())
    assert "12.4 g · 4.16 m" in exported.stats_html


def test_a_file_that_is_not_an_image_is_explained(studio, tmp_path):
    sid = studio.store.new()
    bad = tmp_path / "notes.png"
    bad.write_bytes(b"not an image at all")
    studio.add_images(sid, [str(bad)])
    studio.set_face(sid, studio.store.get(sid).items[0].id, "front")
    review = studio.analyze(sid, "none", AiSettings())
    assert not review.ok and review.stage == "capture" and "JPEG or PNG" in review.message_html


def test_redraw_moves_to_new_seeds(tmp_path, monkeypatch):
    monkeypatch.setattr("s2c.multiview.slice.find_slicer", lambda: None)
    gen = fake_gen(np.zeros((300, 300, 3), np.uint8))
    studio = Studio(MvPipeline(image_gen=gen), root=tmp_path / "files")
    sid = with_images(studio, tmp_path, faces=(("front", 600, 400),))
    review = studio.analyze(sid, "none", AiSettings(seed=10, attempts=1))
    studio.build(sid, {"x": "60", "y": "40", "z": "10"}, review.rows, [], GeometrySettings())
    seeds_before = {c[2] for c in gen.calls}
    studio.redraw(sid)
    assert seeds_before == {10} and {c[2] for c in gen.calls} == {10, 11}


def test_redraw_draws_a_random_seed_when_asked(tmp_path, monkeypatch):
    monkeypatch.setattr("s2c.multiview.slice.find_slicer", lambda: None)
    seeds = iter([100, 5000])
    monkeypatch.setattr(handlers.random, "randint", lambda a, b: next(seeds))
    gen = fake_gen(np.zeros((300, 300, 3), np.uint8))
    studio = Studio(MvPipeline(image_gen=gen), root=tmp_path / "files")
    sid = with_images(studio, tmp_path, faces=(("front", 600, 400),))
    review = studio.analyze(sid, "none", AiSettings(seed=10, attempts=1, randomize_seed=True))
    studio.build(sid, {"x": "60", "y": "40", "z": "10"}, review.rows, [], GeometrySettings())
    assert {c[2] for c in gen.calls} == {100}
    review = studio.redraw(sid)
    assert review.seed == 5000 and {c[2] for c in gen.calls} == {100, 5000}  # not 100 + attempts


def test_print_time_is_floored_to_hours_and_minutes():
    assert [handlers.duration_text(s) for s in (2692, 3600, 6300)] == ["45 min", "1 h 0 min", "1 h 45 min"]


def test_unchecked_counts_only_the_values_the_table_shows(studio, tmp_path):
    sid = with_images(studio, tmp_path, faces=(("front", 600, 400),))
    top = tmp_path / "top-hole.png"
    top.write_bytes(sketch(600, 100, circles=((200, 50, 20),)))
    studio.add_images(sid, [str(top)])
    item = studio.store.get(sid).items[-1]
    studio.set_face(sid, item.id, "top")
    studio.set_kind(sid, item.id, "sketch")
    review = studio.analyze(sid, "none", AiSettings())
    review, _ = studio.build(sid, {"x": "60", "y": "40", "z": "10"}, review.rows, [], GeometrySettings())
    assert review.rows and review.unchecked == len(review.rows)  # the outlines are shown as faces, not as rows
    review, _ = studio.build(sid, {}, [[r[0], r[1] + 0.5, r[2]] for r in review.rows], [], GeometrySettings())
    assert review.unchecked == 0


def test_the_example_loads_two_tagged_sketches(studio):
    sid = studio.store.new()
    studio.load_examples(sid)
    items = studio.store.get(sid).items
    assert [(i.face, i.kind) for i in items] == [("front", "sketch"), ("top", "sketch")]


def test_the_example_images_are_small():
    pngs = sorted(handlers.EXAMPLES.glob("*.png"))
    assert pngs and all(p.stat().st_size < 300_000 for p in pngs)


def test_the_example_analyzes_and_builds(studio):
    sid = studio.store.new()
    studio.load_examples(sid)
    review = studio.analyze(sid, "none", AiSettings())
    assert review.stage == "review" and len(review.faces) == 2  # both outlines found
    _, model = studio.build(sid, {"x": "50", "y": "30", "z": "20"}, review.rows, [], GeometrySettings())
    assert model.ok
    scores = {c.split(" ")[0]: float(c.rsplit(" ", 1)[1]) for _, c in model.views if "match" in c}  # "front · match 0.97"
    assert set(scores) == {"front", "top"} and min(scores.values()) > 0.9


def test_cleared_size_reverts(studio, tmp_path):
    sid = with_images(studio, tmp_path)
    review = studio.analyze(sid, "none", AiSettings())
    review, model = studio.build(sid, {"x": "60", "y": "40", "z": "10"}, review.rows, [], GeometrySettings())
    assert model.ok
    session = studio.store.get(sid)
    assert session.edits["envelope.x_mm"] == 60
    review, model = studio.build(sid, {"x": "", "y": "40", "z": "10"}, review.rows, [], GeometrySettings())
    assert "envelope.x_mm" not in session.edits
    assert not model.ok  # x is unmeasured in this fixture, so clearing it makes the envelope incomplete again
    assert review.sizes["x"]["value"] == ""


def test_bad_image_names_file(studio, tmp_path):
    sid = studio.store.new()
    bad = tmp_path / "notes.pdf"
    bad.write_bytes(b"%PDF-1.4")
    studio.add_images(sid, [str(bad)])
    studio.set_face(sid, studio.store.get(sid).items[0].id, "front")
    review = studio.analyze(sid, "none", AiSettings())
    assert not review.ok and "notes.pdf" in review.message_html
    assert "bad_image" not in review.message_html


def test_abstain_titles_are_words():
    html = handlers._abstain_card(MvAbstain(stage="outline", reason="no_outline", remedy="Draw a closed outline."))
    assert "no_outline" not in html


def test_clearance_change_refreshes_review(studio, tmp_path):
    sid = with_images(studio, tmp_path, faces=(("front", 600, 400),))
    top = tmp_path / "top-hole.png"
    top.write_bytes(sketch(600, 100, circles=((200, 50, 20),)))
    studio.add_images(sid, [str(top)])
    item = studio.store.get(sid).items[-1]
    studio.set_face(sid, item.id, "top")
    studio.set_kind(sid, item.id, "sketch")
    review = studio.analyze(sid, "none", AiSettings())
    review, model = studio.build(sid, {"x": "60", "y": "40", "z": "10"}, review.rows, [],
                                 GeometrySettings(clearance="fine"))
    assert model.ok and review.rows
    idx = next(i for i, r in enumerate(review.rows) if "diameter" in r[0].lower())
    fine_diam = review.rows[idx][1]

    review, model = studio.rebuild_geometry(sid, GeometrySettings(clearance="coarse"))
    assert model.ok
    coarse_diam = review.rows[idx][1]
    assert coarse_diam != fine_diam

    session = studio.store.get(sid)
    path = session.row_paths[idx]
    assert session.shown[path] == coarse_diam  # so the next Build does not read the table as an edit


def test_get_none_uses_one_id():
    store = SessionStore()
    session = store.get(None)
    assert session.id in store._items and store._items[session.id] is session


def test_idle_sessions_swept_without_new_visitor(monkeypatch):
    monkeypatch.setattr(session_mod, "SWEEP_EVERY_S", 0)
    store = SessionStore(ttl_s=0.05)
    sid1, sid2 = store.new(), store.new()
    time.sleep(0.1)
    store.get(sid1)
    assert sid2 not in store._items


def test_analyze_guard(studio, tmp_path):
    sid = with_images(studio, tmp_path)
    on_analyze = app_fn(build_app(studio=studio), "on_analyze")
    out = on_analyze(sid, "none", True, True, True, True, True, 7, False, 99)  # attempts=99 is out of range
    assert "Check the AI settings" in out[1]


def test_on_geometry_wiring(studio):
    app = build_app(studio=studio)
    event = app_event(app, "on_geometry")
    assert event.trigger_mode == "always_last"
    n_outputs = len(event.outputs)  # [model_msg, model, views, stats, *sizes.values(), values]
    on_geometry = event.fn

    # failure path: nothing analyzed yet, still returns exactly as many values as the wired outputs
    empty_sid = studio.store.new()
    out = on_geometry(empty_sid, True, "medium", "none", 1.0, "all_vertical")
    assert len(out) == n_outputs

    # success path: analyze + build the example, then change the clearance
    sid = studio.store.new()
    studio.load_examples(sid)
    review = studio.analyze(sid, "none", AiSettings())
    review, model = studio.build(sid, {"x": "50", "y": "30", "z": "20"}, review.rows, [], GeometrySettings())
    assert model.ok and review.rows  # the example has a hole, so the values table is not empty
    out = on_geometry(sid, True, "coarse", "none", 1.0, "all_vertical")
    assert len(out) == n_outputs
    refreshed_rows = out[-1]  # the values Dataframe is the last wired output
    assert refreshed_rows and refreshed_rows != review.rows  # carries the new snapped diameter, not the stale one
    assert refreshed_rows[-1][1] == 4.8  # medium (5.0) -> coarse snaps the drawn hole's diameter down
    session = studio.store.get(sid)
    assert session.shown[session.row_paths[-1]] == refreshed_rows[-1][1]
