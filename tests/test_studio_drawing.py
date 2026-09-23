from pathlib import Path

import ezdxf
import pytest
from ezdxf import bbox

from s2c.multiview.build import build
from s2c.multiview.drawing import DRAWING_FILES, write_drawings
from s2c.multiview.spec import MultiViewSpec

SPEC = MultiViewSpec.model_validate_json(
    (Path(__file__).parents[1] / "examples" / "mv" / "l_bracket.json").read_text())


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
