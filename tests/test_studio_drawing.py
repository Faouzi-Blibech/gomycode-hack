import math
from pathlib import Path

import ezdxf
import pytest
from ezdxf import bbox

from s2c.multiview.build import build
from s2c.multiview.drawing import DRAWING_FILES, write_drawings
from s2c.multiview.settings import EDGE_LABELS
from s2c.multiview.spec import MultiViewSpec
from tests.mv_helpers import make_spec

SPEC = MultiViewSpec.model_validate_json(
    (Path(__file__).parents[1] / "examples" / "mv" / "l_bracket.json").read_text())

# one through hole on each far face (back -> front, left -> right, bottom -> top) plus a fillet; z-positions are
# staggered (10, 25, 20) and diameters distinct (6, 4, 8) so no two holes' axes meet inside the part
FAR_SPEC = make_spec(
    (60.0, 40.0, 30.0),
    features=[
        {"type": "hole", "face": "back", "a_mm": 20.0, "b_mm": 15.0, "diameter_mm": 6.0},
        {"type": "hole", "face": "left", "a_mm": 10.0, "b_mm": 25.0, "diameter_mm": 4.0},
        {"type": "hole", "face": "bottom", "a_mm": 30.0, "b_mm": 20.0, "diameter_mm": 8.0},
    ],
    finishes=[{"type": "fillet", "edges": "all_vertical", "radius_mm": 2.0}],
)


def _hole_center(solid, diameter_mm):
    """Centre of mass of a full-circle boundary edge of that diameter; it lies on the hole's own axis, so its
    in-plane coordinates are exact regardless of which of the two boundary circles is picked."""
    target = math.pi * diameter_mm
    matches = [e for e in solid.val().Edges() if e.geomType() == "CIRCLE" and abs(e.Length() - target) < 1e-3]
    assert matches, f"no hole edge of diameter {diameter_mm}"
    return matches[0].Center()


def _rendered_text(doc, dim):
    """The literal TEXT/MTEXT content ezdxf composed into this dimension's geometry block: what actually renders,
    with ezdxf's own formatting codes (e.g. "%%c" for the diameter sign) still in place. ezdxf's drawing add-on
    (SVG, PDF and the matplotlib preview alike) turns every glyph into filled vector paths, not markup or literal
    characters, so this block -- not the rendered SVG/PDF bytes -- is the only place the composed text is still a
    string; it is exactly what a viewer will draw."""
    block = doc.blocks.get(dim.dxf.geometry)
    for e in block:
        if e.dxftype() == "TEXT":
            return e.dxf.text
        if e.dxftype() == "MTEXT":
            return e.text
    raise AssertionError("dimension geometry block holds no text entity")


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    return write_drawings(build(SPEC), SPEC, tmp_path_factory.mktemp("drawing"), ["dxf", "svg", "pdf"])


def test_the_dxf_is_clean_and_in_millimetres(files):
    doc = ezdxf.readfile(files["dxf"])
    assert not doc.audit().has_errors
    assert doc.header["$INSUNITS"] == 4
    assert files["dxf"].name == DRAWING_FILES["dxf"]


def test_three_views_with_visible_and_hidden_lines(files):
    msp = ezdxf.readfile(files["dxf"]).modelspace()
    layers = {e.dxf.layer for e in msp}
    assert {"VISIBLE", "HIDDEN", "DIMENSIONS"} <= layers
    ext = bbox.extents(msp.query('*[layer=="VISIBLE"]'))
    assert ext.size.x > 50 + 20 and ext.size.y > 30 + 20  # front + right side by side, top above front


def test_overall_sizes_and_hole_diameters_are_dimensioned(files):
    msp = ezdxf.readfile(files["dxf"]).modelspace()
    measured = {round(d.get_measurement(), 1) for d in msp.query("DIMENSION")}
    assert {50.0, 30.0, 20.0, 5.5} <= measured


def test_svg_and_pdf_are_rendered(files):
    assert "<svg" in files["svg"].read_text(encoding="utf-8")[:500]
    assert files["pdf"].read_bytes()[:5] == b"%PDF-"


def test_only_the_requested_formats_are_written(tmp_path):
    assert set(write_drawings(build(SPEC), SPEC, tmp_path, ["svg"])) == {"svg"}
    assert write_drawings(build(SPEC), SPEC, tmp_path, ["stl"]) == {}


def test_far_side_hole_is_dimensioned(tmp_path):
    solid = build(FAR_SPEC)
    doc = ezdxf.readfile(write_drawings(solid, FAR_SPEC, tmp_path, ["dxf"])["dxf"])
    dias = [d for d in doc.modelspace().query("DIMENSION") if d.dimtype & 7 == 3]
    env = FAR_SPEC.envelope
    w, h, depth = env.x_mm, env.y_mm, env.z_mm
    gap = max(15.0, 0.3 * max(w, h, depth))

    def far_side(face, diameter_mm):
        dim = next(dm for dm in dias if f"({face})" in dm.dxf.text)
        text = _rendered_text(doc, dim)
        assert text.startswith("%%c") and text.count("%%c") == 1  # exactly one diameter sign, right before the value
        center = (dim.dxf.defpoint + dim.dxf.defpoint4) / 2
        return center, _hole_center(solid, diameter_mm)

    center, axis = far_side("back", 6.0)  # front view: (a, b) of the back face's own frame land at (x, y)
    assert center.x == pytest.approx(axis.x, abs=0.01)
    assert center.y == pytest.approx(axis.y, abs=0.01)

    center, axis = far_side("left", 4.0)  # right view: mirrored along the width, unchanged in height
    assert center.x == pytest.approx((w + gap) + (depth - axis.z), abs=0.01)
    assert center.y == pytest.approx(axis.y, abs=0.01)

    center, axis = far_side("bottom", 8.0)  # top view: unchanged in width, mirrored along the depth
    assert center.x == pytest.approx(axis.x, abs=0.01)
    assert center.y == pytest.approx((h + gap) + (depth - axis.z), abs=0.01)


def test_finish_callout(tmp_path):
    doc = ezdxf.readfile(write_drawings(build(FAR_SPEC), FAR_SPEC, tmp_path, ["dxf"])["dxf"])
    texts = [t.dxf.text for t in doc.modelspace().query('TEXT[layer=="TEXT"]')]
    assert f"Fillet R2 on {EDGE_LABELS['all_vertical']}" in texts


def test_chamfer_callout_uses_c_not_r(tmp_path):
    spec = make_spec((60.0, 40.0, 30.0), finishes=[{"type": "chamfer", "edges": "all_vertical", "radius_mm": 1.5}])
    doc = ezdxf.readfile(write_drawings(build(spec), spec, tmp_path, ["dxf"])["dxf"])
    texts = [t.dxf.text for t in doc.modelspace().query('TEXT[layer=="TEXT"]')]
    assert f"Chamfer C1.5 on {EDGE_LABELS['all_vertical']}" in texts
    assert not any("Chamfer R" in t for t in texts)  # R means radius; a chamfer size is C, not R


def _view_boxes(env):
    """(x0, x1, y0, y1) of the front, right and top view rectangles, from the placement drawing_document uses."""
    w, h, d = env.x_mm, env.y_mm, env.z_mm
    gap = max(15.0, 0.3 * max(w, h, d))
    return [(0.0, w, 0.0, h), (w + gap, w + gap + d, 0.0, h), (0.0, w, h + gap, h + gap + d)]


def test_text_block_never_overlaps_a_view(tmp_path):
    thin = make_spec((60.0, 40.0, 3.0), finishes=[{"type": "chamfer", "edges": "all_vertical", "radius_mm": 1.0}])
    for name, spec in [("l_bracket", SPEC), ("thin_plate", thin)]:
        doc = ezdxf.readfile(write_drawings(build(spec), spec, tmp_path / name, ["dxf"])["dxf"])
        boxes = _view_boxes(spec.envelope)
        for e in doc.modelspace().query('TEXT[layer=="TEXT"]'):
            tb = bbox.extents([e])
            for x0, x1, y0, y1 in boxes:
                overlaps = tb.extmax.x > x0 and tb.extmin.x < x1 and tb.extmax.y > y0 and tb.extmin.y < y1
                assert not overlaps, f"{name}: {e.dxf.text!r} overlaps view box {(x0, x1, y0, y1)}"


def test_drawing_write_is_atomic(tmp_path, monkeypatch):
    from s2c.multiview import drawing

    def fake_pdf(doc, path):
        Path(path).write_bytes(b"XXXXX")
        raise RuntimeError("boom")

    monkeypatch.setattr(drawing, "_pdf", fake_pdf)
    with pytest.raises(RuntimeError):
        write_drawings(build(SPEC), SPEC, tmp_path, ["dxf", "pdf"])
    assert not (tmp_path / "drawing.pdf").exists()
    assert not any(p.suffix == ".tmp" for p in tmp_path.iterdir())
