import math

import pytest

from s2c.multiview.build import build
from s2c.multiview.slice import (choose_down, find_slicer, load_profile, orient_for_print, parse_duration,
                                 parse_gcode_stats, slice_solid)
from s2c.multiview.spec import MvAbstain
from tests.mv_helpers import make_spec, outline

L_SHAPE = [(0, 0), (50, 0), (50, 30), (47, 30), (47, 3), (0, 3)]


def test_default_profile():
    p = load_profile()
    assert (p.bed_x, p.bed_y, p.max_height) == (220, 220, 250) and p.center == (110, 110)


def test_a_standing_plate_is_laid_flat():
    solid = build(make_spec((5.0, 40.0, 60.0)))
    assert choose_down(solid) in ("-x", "+x")
    bb = orient_for_print(solid).val().BoundingBox()
    assert math.isclose(bb.zlen, 5, abs_tol=1e-6) and math.isclose(bb.zmin, 0, abs_tol=1e-6)


def test_an_l_bracket_prints_on_its_long_leg():
    solid = build(make_spec((50.0, 30.0, 20.0), front=outline(L_SHAPE)))
    assert choose_down(solid) == "-y"
    assert math.isclose(orient_for_print(solid).val().BoundingBox().zlen, 30, abs_tol=1e-6)


def test_too_big_for_the_bed(tmp_path):
    res = slice_solid(build(make_spec((300.0, 20.0, 10.0))), tmp_path)
    assert isinstance(res, MvAbstain) and res.reason == "too_big_for_bed"


def test_without_a_slicer_the_print_stl_is_still_written(tmp_path, monkeypatch):
    monkeypatch.setattr("s2c.multiview.slice.find_slicer", lambda: None)
    res = slice_solid(build(make_spec((60.0, 40.0, 5.0))), tmp_path)
    assert res.gcode is None and res.print_stl.exists()
    assert res.warnings == ["G-code unavailable: slicer not installed"]


def test_gcode_stats():
    text = "G1 X1\n; filament used [g] = 12.34\n; estimated printing time (normal mode) = 1h 2m 3s\n"
    assert parse_gcode_stats(text) == (3723.0, 12.34)
    assert parse_duration("1d 0h 0m 5s") == 86405.0


@pytest.mark.slicer
@pytest.mark.skipif(find_slicer() is None, reason="PrusaSlicer not installed")
def test_real_slice_produces_gcode(tmp_path):
    res = slice_solid(build(make_spec((60.0, 40.0, 5.0))), tmp_path)
    text = res.gcode.read_text()
    assert "G1" in text
    assert abs(text.count(";LAYER_CHANGE") - 25) <= 1
    assert res.print_time_s > 0 and res.filament_g > 0
