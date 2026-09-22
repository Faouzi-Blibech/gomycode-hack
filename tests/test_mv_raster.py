import cadquery as cq
import numpy as np
import pytest

from s2c.multiview.raster import face_mask, iou, mask_to_mm, normalize_mask, outline_mask, solid_mesh
from s2c.multiview.spec import FACES, Envelope, face_size
from tests.mv_helpers import box_mesh, circle, rect

ENV = Envelope(x_mm=60.0, y_mm=40.0, z_mm=5.0)


def plate_with_hole():
    return cq.Workplane("XY").box(60, 40, 5, centered=False).cut(
        cq.Workplane("XY").center(20, 20).circle(5).extrude(5))


def test_box_front_is_filled_although_front_and_back_triangles_overlap():
    mask, s = face_mask(box_mesh(60, 40, 5), "front", ENV, px=512)
    expected = (60 * s) * (40 * s) / 512 ** 2
    assert abs((mask == 255).mean() - expected) < 0.01


@pytest.mark.parametrize("face", FACES)
def test_every_face_of_a_box_is_its_envelope_rectangle(face):
    mask, _ = face_mask(box_mesh(60, 40, 5), face, ENV, px=512)
    a, b = face_size(face, ENV)
    assert iou(mask, outline_mask(rect(a, b), [], a, b, px=512)) > 0.97


def test_solid_with_a_hole_matches_the_analytic_front():
    mask, _ = face_mask(solid_mesh(plate_with_hole()), "front", ENV, px=512)
    assert iou(mask, outline_mask(rect(60, 40), [circle(20, 20, 5)], 60, 40, px=512)) > 0.98


def test_mask_to_mm_recovers_outline_and_opening():
    mask, s = face_mask(solid_mesh(plate_with_hole()), "front", ENV, px=512)
    outer, inner = mask_to_mm(mask, s, 512)
    xs, ys = zip(*outer)
    assert min(xs) == pytest.approx(0, abs=0.3) and max(xs) == pytest.approx(60, abs=0.3)
    assert min(ys) == pytest.approx(0, abs=0.3) and max(ys) == pytest.approx(40, abs=0.3)
    assert len(inner) == 1
    ixs, _ = zip(*inner[0])
    assert max(ixs) - min(ixs) == pytest.approx(10, abs=0.5)


def test_iou_and_normalize():
    a = np.zeros((100, 100), np.uint8)
    a[10:30, 10:50] = 255
    b = np.zeros((100, 100), np.uint8)
    b[50:90, 20:100] = 255  # the same shape at twice the size
    assert iou(a, b) == 0.0
    assert iou(normalize_mask(a, 128), normalize_mask(b, 128)) > 0.95
