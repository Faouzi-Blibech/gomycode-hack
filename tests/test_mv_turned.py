import math
from itertools import pairwise

import cv2
import numpy as np

from s2c.multiview.build import build, volume
from s2c.multiview.pipeline import ImageInput, MvPipeline
from s2c.multiview.turned import SNAP, WARNING, profile, turned_axis
from tests.mv_helpers import circle, make_spec, outline, rect

STEP = [(0, 0), (20, 0), (20, 5), (15, 5), (15, 15), (5, 15), (5, 5), (0, 5)]


def test_a_stepped_round_part_is_built_round():
    s = make_spec((20.0, 15.0, 20.0), front=outline(STEP), top=outline(circle(10, 10, 10, 180)), right=outline(STEP))
    assert turned_axis(s) == "y"
    assert math.isclose(volume(build(s)), math.pi * (10 ** 2 * 5 + 5 ** 2 * 10), rel_tol=0.015)


def test_a_square_plate_is_not_turned():
    s = make_spec((20.0, 5.0, 20.0))
    assert turned_axis(s) is None
    assert math.isclose(volume(build(s)), 2000, rel_tol=0.005)


def test_different_side_views_are_not_turned():
    s = make_spec((20.0, 15.0, 20.0), front=outline(STEP), top=outline(circle(10, 10, 10, 180)),
                  right=outline(rect(20, 15)))
    assert turned_axis(s) is None


def test_a_round_disk_with_a_through_hole_keeps_its_hole():
    hole = {"type": "hole", "face": "front", "a_mm": 15.0, "b_mm": 15.0, "diameter_mm": 8.0}
    s = make_spec((30.0, 30.0, 6.0), front=outline(circle(15, 15, 15, 180)), features=[hole])
    assert turned_axis(s) == "z"
    assert math.isclose(volume(build(s)), math.pi * (15 ** 2 - 4 ** 2) * 6, rel_tol=0.015)


def _area(pts):
    return abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]))) / 2


def test_teeth_that_reach_past_the_side_views_are_kept():
    """Teeth between the side views reach past R; a revolve sized from the side views alone would cut them."""
    rho = [5.0 if k % 4 == 0 else 5.3 for k in range(16)]
    teeth = [(5 + p * math.cos(k * math.pi / 8), 5 + p * math.sin(k * math.pi / 8)) for k, p in enumerate(rho)]
    s = make_spec((10.0, 10.0, 5.0), front=outline(teeth))
    assert turned_axis(s) == "z"
    assert math.isclose(volume(build(s)), _area(teeth) * 5, rel_tol=0.005)


def test_side_outlines_that_stop_short_of_the_ends_leave_no_needle():
    """Pixel outlines stop a hair short of the envelope ends; a needle on the axis made OCC return nothing."""
    skew = [(0.0, 19.88), (0.06, 0.06), (10.0, 0.12), (9.94, 19.94)]
    s = make_spec((10.0, 20.0, 10.0), front=outline(skew), top=outline(circle(5, 5, 5, 180)), right=outline(skew))
    assert turned_axis(s) == "y"
    assert all(r > 0.5 for r, _ in profile(s, "y")[1:-1])
    assert math.isclose(volume(build(s)), math.pi * 5 ** 2 * 20, rel_tol=0.015)


def test_leaning_outline_edges_still_make_cylinders():
    """A cone that is almost a cylinder makes OCC booleans return nothing, so near-equal radii are made equal."""
    lean = [(0, 0), (20, 0), (20, 5), (15, 5), (15.0001, 15), (4.9999, 15), (5, 5), (0, 5)]
    s = make_spec((20.0, 15.0, 20.0), front=outline(lean), top=outline(circle(10, 10, 10, 180)), right=outline(lean))
    pts = profile(s, "y")
    snap = SNAP * 20
    assert all(a[0] == b[0] or a[1] == b[1] or (abs(a[0] - b[0]) > snap and abs(a[1] - b[1]) > snap)
               for a, b in pairwise(pts))
    assert math.isclose(volume(build(s)), math.pi * (10 ** 2 * 5 + 5 ** 2 * 10), rel_tol=0.015)


def _circle_sketch(r_px):
    img = np.full((1200, 1600, 3), 255, np.uint8)
    cv2.circle(img, (800, 600), r_px, (0, 0, 0), 4)
    return cv2.imencode(".png", img)[1].tobytes()


def test_fuse_warns_when_the_part_is_turned():
    pipe = MvPipeline()
    observed = pipe.observe([ImageInput(_circle_sketch(300), "top", "sketch")])
    spec = pipe.fuse(observed, {"envelope.x_mm": 40, "envelope.y_mm": 10, "envelope.z_mm": 40})
    assert spec.views.front.source == "assumed" and spec.views.right.source == "assumed"
    assert WARNING.format(axis="y") in spec.warnings
