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

# one through hole on a far face per canonical view (back -> front, left -> right) plus a fillet
FAR_SPEC = make_spec(
    (60.0, 40.0, 30.0),
    features=[
        {"type": "hole", "face": "back", "a_mm": 20.0, "b_mm": 15.0, "diameter_mm": 6.0},
        {"type": "hole", "face": "left", "a_mm": 10.0, "b_mm": 25.0, "diameter_mm": 4.0},
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

    back = next(dm for dm in dias if "⌀" in dm.dxf.text and "(back)" in dm.dxf.text)
    center = (back.dxf.defpoint + back.dxf.defpoint4) / 2
    back_axis = _hole_center(solid, 6.0)  # front view: (a, b) of the back face's own frame land at (x, y)
    assert center.x == pytest.approx(back_axis.x, abs=0.01)
    assert center.y == pytest.approx(back_axis.y, abs=0.01)

    left = next(dm for dm in dias if "⌀" in dm.dxf.text and "(left)" in dm.dxf.text)
    center = (left.dxf.defpoint + left.dxf.defpoint4) / 2
    left_axis = _hole_center(solid, 4.0)
    assert center.x == pytest.approx((w + gap) + (depth - left_axis.z), abs=0.01)
    assert center.y == pytest.approx(left_axis.y, abs=0.01)


def test_finish_callout(tmp_path):
    doc = ezdxf.readfile(write_drawings(build(FAR_SPEC), FAR_SPEC, tmp_path, ["dxf"])["dxf"])
    texts = [t.dxf.text for t in doc.modelspace().query('TEXT[layer=="TEXT"]')]
    assert f"Fillet R2 on {EDGE_LABELS['all_vertical']}" in texts


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
