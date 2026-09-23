from pathlib import Path

import gradio as gr
import numpy as np
import pytest

from s2c.multiview import artifacts
from s2c.multiview.pipeline import MvPipeline
from s2c.multiview.settings import AiSettings, ExportSettings, GeometrySettings, MeshSettings, PrintSettings
from s2c.multiview.spec import MvAbstain
from s2c.studio import handlers
from s2c.studio.app import build_app
from s2c.studio.handlers import Studio, parse_size
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
    assert not review.ok and "missing_x" in review.message_html
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


def test_a_failed_finish_keeps_the_review(studio, tmp_path, monkeypatch):
    real = handlers.build_part

    def fails_on_big_fillets(spec, geometry, root):  # the real failure is pinned in the artifacts tests
        if geometry.finish == "fillet" and geometry.finish_mm > 5:
            return MvAbstain(stage="build", reason="fillet_failed", remedy="Reduce the fillet radius.")
        return real(spec, geometry, root)

    monkeypatch.setattr(handlers, "build_part", fails_on_big_fillets)
    sid = with_images(studio, tmp_path)
    review = studio.analyze(sid, "none", AiSettings())
    sizes = {"x": "60", "y": "40", "z": "10"}
    _, model = studio.build(sid, sizes, review.rows, [], GeometrySettings(finish="fillet", finish_mm=6.0))
    assert not model.ok and "fillet" in model.message_html.lower()
    model = studio.rebuild_geometry(sid, GeometrySettings(finish="fillet", finish_mm=1.0))
    assert model.ok


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
