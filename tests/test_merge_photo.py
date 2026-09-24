import numpy as np

from s2c.fakes.metrology import measure
from s2c.merge import merge
from s2c.partspec.models import Abstain, Bbox, Circle, Coin, Measurements, PartSpec, Topology


def topo(**kw):
    base = {"part_type": "plate", "holes": [], "confidence": 0.9}
    base.update(kw)
    return Topology.model_validate(base)


def test_photo_plate_needs_thickness_then_builds():
    m = measure(np.zeros((1, 1, 3), np.uint8))
    out = merge(topo(), source_input="photo", measurements=m)
    assert isinstance(out, Abstain) and out.reason == "missing_thickness"
    spec = merge(topo(), source_input="photo", measurements=m, user_values={"thickness": 5})
    assert isinstance(spec, PartSpec)
    assert spec.provenance["part.width_mm"] == "measured"
    assert len(spec.features) == 2 and spec.provenance["features[0].x_mm"] == "measured"


def test_photo_flange_derives_bolt_circle():
    m = Measurements(
        mm_per_px=0.1, coin=Coin(name="1 TND", pixel_diameter=250, eccentricity=0.02, confidence=0.9),
        outer_contour_mm=[(0, 0), (80, 0), (80, 80), (0, 80)], bbox_mm=Bbox(width=80, height=80),
        circles_mm=[Circle(x=40, y=40, diameter=30), Circle(x=70, y=40, diameter=6), Circle(x=10, y=40, diameter=6),
                    Circle(x=40, y=70, diameter=6), Circle(x=40, y=10, diameter=6)],
        confidence=0.85,
    )
    spec = merge(topo(part_type="flange"), source_input="photo", measurements=m, user_values={"thickness": 8})
    assert isinstance(spec, PartSpec)
    assert spec.part.bolt_count == 4
    assert abs(spec.part.bolt_circle_diameter_mm - 60) < 0.01
    assert spec.part.bore_diameter_mm == 30


def test_photo_l_bracket_maps_bbox_to_leg_a_and_width_then_asks_for_leg_b():
    # a top-down photo of an L-bracket shows leg_a by width; leg_b stands up out of the photo
    m = Measurements(
        mm_per_px=0.1, coin=Coin(name="1 TND", pixel_diameter=250, eccentricity=0.02, confidence=0.9),
        outer_contour_mm=[(0, 0), (50, 0), (50, 30), (0, 30)], bbox_mm=Bbox(width=50, height=30),
        confidence=0.85,
    )
    out = merge(topo(part_type="l_bracket"), source_input="photo", measurements=m)
    assert isinstance(out, Abstain) and out.reason == "missing_leg_b"
    assert out.partial == {"leg_a": 50.0, "width": 30.0}
    assert "leg b" in out.remedy
    spec = merge(topo(part_type="l_bracket"), source_input="photo", measurements=m,
                 user_values={"leg_b": 20.0, "thickness": 3.0})
    assert isinstance(spec, PartSpec)
    assert (spec.part.leg_a_mm, spec.part.leg_b_mm, spec.part.width_mm) == (50, 20, 30)
    assert spec.provenance["part.leg_a_mm"] == "measured" and spec.provenance["part.width_mm"] == "measured"
    assert spec.provenance["part.leg_b_mm"] == "user_edited"


def test_photo_profile_extrusion_simplifies_contour():
    m = Measurements(
        mm_per_px=0.1, coin=Coin(name="1 TND", pixel_diameter=250, eccentricity=0.02, confidence=0.9),
        outer_contour_mm=[(0, 0), (30, 0), (60, 0), (60, 20), (30, 40), (0, 40)], bbox_mm=Bbox(width=60, height=40),
        confidence=0.85,
    )
    spec = merge(topo(part_type="profile_extrusion"), source_input="photo", measurements=m, user_values={"depth": 10})
    assert isinstance(spec, PartSpec)
    assert (30.0, 0.0) not in spec.part.points_mm  # collinear point removed
    assert len(spec.part.points_mm) == 5
