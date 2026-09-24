"""Every value merge reads either lands in the spec or comes back as a warning. None vanish,
and none get relabelled as a different measurement."""
import pytest

from s2c.merge import merge
from s2c.partspec.models import (
    Abstain,
    Annotation,
    Annotations,
    Bbox,
    Circle,
    Coin,
    Measurements,
    PartSpec,
    Topology,
)


def ann(value, kind="linear", linked_to="unknown", hole_index=None, conf=0.9):
    return Annotation(value_mm=value, kind=kind, bbox_px=(0.0, 0.0, 1.0, 1.0),
                      linked_to=linked_to, hole_index=hole_index, confidence=conf)


def topo(**kw):
    base = {"part_type": "plate", "holes": [], "confidence": 0.9}
    base.update(kw)
    return Topology.model_validate(base)


def meas(w, h, circles=()):
    return Measurements(
        mm_per_px=0.1, coin=Coin(name="1 TND", pixel_diameter=250, eccentricity=0.02, confidence=0.9),
        outer_contour_mm=[(0, 0), (w, 0), (w, h), (0, h)], bbox_mm=Bbox(width=w, height=h),
        circles_mm=list(circles), confidence=0.85,
    )


def warned(warnings, *needles):
    return [w for w in warnings if all(n in w for n in needles)]


WIDTH, HEIGHT, THICK = ann(60.0, linked_to="width"), ann(40.0, linked_to="height"), ann(5.0, linked_to="thickness")
ONE_HOLE = [{"u": 0.2, "v": 0.25}]


def test_slot_length_never_becomes_thickness():  # C3
    plain = [ann(60.0, linked_to="width"), ann(40.0, linked_to="height")]
    without_slot = merge(topo(), source_input="sketch", annotations=Annotations(items=plain, confidence=0.9))
    a = Annotations(items=[*plain, ann(25.0, linked_to="slot_length")], confidence=0.9)
    out = merge(topo(), source_input="sketch", annotations=a)

    # exactly the abstention there is with no slot annotation at all: the slot length is not a thickness
    assert isinstance(out, Abstain), f"slot length was used as a part dimension: {out}"
    assert without_slot.reason == "missing_thickness"
    assert (out.reason, out.remedy, out.partial) == (without_slot.reason, without_slot.remedy, without_slot.partial)

    # the slot value is not lost: answering the abstention brings its warning back on the spec
    spec = merge(topo(), source_input="sketch", annotations=a, user_values={"thickness": 4.0})
    assert isinstance(spec, PartSpec)
    assert spec.part.thickness_mm == 4.0 and spec.provenance["part.thickness_mm"] == "user_edited"
    assert warned(spec.warnings, "slot_length", "25 mm", "not used")


def test_slot_width_is_never_assigned_positionally():  # C3
    a = Annotations(items=[ann(60.0), ann(40.0), ann(5.0), ann(8.0, linked_to="slot_width")], confidence=0.9)
    spec = merge(topo(), source_input="sketch", annotations=a)
    assert isinstance(spec, PartSpec)
    assert (spec.part.width_mm, spec.part.height_mm, spec.part.thickness_mm) == (60, 40, 5)
    assert warned(spec.warnings, "slot_width", "8 mm", "not used")


def test_corner_radius_is_never_assigned_to_another_dimension():  # same defect class as C3
    a = Annotations(items=[ann(20.0, "diameter", "outer_diameter"), ann(2.0, linked_to="corner_radius")],
                    confidence=0.9)
    out = merge(topo(part_type="spacer"), source_input="sketch", annotations=a)
    assert isinstance(out, Abstain) and out.reason == "missing_length", f"corner radius became a length: {out}"
    spec = merge(topo(part_type="spacer"), source_input="sketch", annotations=a, user_values={"length": 30.0})
    assert spec.part.length_mm == 30.0
    assert warned(spec.warnings, "corner_radius", "2 mm", "not used")


# Abstain.partial is a value map (the UI and the golden harness read it as numbers), so it never
# carries warnings. Nothing is lost: answering the abstention re-runs merge and the spec warns.
ABSTAINS_WITH_UNPLACED_VALUES = {
    "slot length, thickness missing": (
        {}, "sketch", [WIDTH, HEIGHT, ann(25.0, linked_to="slot_length")], None,
        {"thickness": 4.0}, ("slot_length", "25 mm")),
    "extra measured circle, length missing": (
        {"part_type": "spacer"}, "photo", [],
        meas(50.0, 50.0, [Circle(x=25, y=25, diameter=20), Circle(x=10, y=10, diameter=4)]),
        {"length": 30.0}, ("measured circle", "4 mm")),
    "hole past the detected holes, height missing": (
        {"holes": ONE_HOLE}, "sketch", [WIDTH, THICK, ann(8.0, "diameter", "hole_diameter", hole_index=2)], None,
        {"height": 40.0}, ("hole 3", "8 mm")),
}


@pytest.mark.parametrize("case", list(ABSTAINS_WITH_UNPLACED_VALUES.values()), ids=list(ABSTAINS_WITH_UNPLACED_VALUES))
def test_abstain_partial_is_numbers_only_and_the_warning_returns_on_the_spec(case):
    topo_kw, source, items, m, answer, needles = case
    kw = {"source_input": source, "annotations": Annotations(items=items, confidence=0.9), "measurements": m}
    out = merge(topo(**topo_kw), **kw)
    assert isinstance(out, Abstain) and out.reason.startswith("missing_")
    assert all(isinstance(v, float) for v in out.partial.values()), out.partial

    spec = merge(topo(**topo_kw), **kw, user_values=answer)
    assert isinstance(spec, PartSpec)
    assert warned(spec.warnings, *needles, "not used")


def test_inconsistent_dimensions_partial_is_numbers_only():
    a = Annotations(items=[ann(20.0, "diameter", "outer_diameter"), ann(30.0, "diameter", "inner_diameter"),
                           ann(30.0, linked_to="length"), ann(3.0, "radius")], confidence=0.9)
    out = merge(topo(part_type="spacer"), source_input="sketch", annotations=a)
    assert isinstance(out, Abstain) and out.reason == "inconsistent_dimensions"
    assert all(isinstance(v, float) for v in out.partial.values()), out.partial


def test_radius_on_a_part_without_a_radius_is_warned_not_dropped():  # C1
    a = Annotations(items=[ann(20.0, "diameter", "outer_diameter"), ann(30.0, linked_to="length"),
                           ann(3.0, "radius")], confidence=0.9)
    spec = merge(topo(part_type="spacer"), source_input="sketch", annotations=a)
    assert isinstance(spec, PartSpec)
    assert warned(spec.warnings, "radius", "3 mm", "not used")


def test_hole_diameter_on_a_flange_is_warned_not_dropped():  # C2
    a = Annotations(items=[ann(100.0, "diameter", "outer_diameter"), ann(30.0, "diameter", "inner_diameter"),
                           ann(8.0, linked_to="thickness"), ann(70.0, "diameter", "bolt_circle_diameter"),
                           ann(9.0, "diameter", "bolt_hole_diameter"), ann(6.0, "diameter", "hole_diameter")],
                    confidence=0.9)
    spec = merge(topo(part_type="flange", bolt_count=4), source_input="sketch", annotations=a)
    assert isinstance(spec, PartSpec)
    assert spec.part.bolt_hole_diameter_mm == 9.0  # the plan maps no flange field to hole_diameter
    assert spec.features == []
    assert warned(spec.warnings, "hole_diameter", "6 mm", "not used")


def test_hole_annotation_past_the_detected_holes_is_warned():  # C4
    a = Annotations(items=[WIDTH, HEIGHT, THICK, ann(6.0, "diameter", "hole_diameter", hole_index=0),
                           ann(8.0, "diameter", "hole_diameter", hole_index=2)], confidence=0.9)
    spec = merge(topo(holes=ONE_HOLE), source_input="sketch", annotations=a)
    assert isinstance(spec, PartSpec)
    assert len(spec.features) == 1 and spec.features[0].diameter_mm == 6.0  # no invented third hole
    assert warned(spec.warnings, "hole 3", "hole_diameter", "8 mm", "not used")


def test_measured_circle_overrides_written_hole_diameter_with_a_warning():  # I2
    a = Annotations(items=[ann(6.0, "diameter", "hole_diameter", hole_index=0)], confidence=0.9)
    spec = merge(topo(holes=ONE_HOLE), source_input="photo", annotations=a,
                 measurements=meas(60.0, 40.0, [Circle(x=10, y=10, diameter=6.2)]), user_values={"thickness": 5.0})
    assert isinstance(spec, PartSpec)
    assert spec.features[0].diameter_mm == 6.2  # measured still wins
    assert spec.provenance["features[0].diameter_mm"] == "measured"
    assert warned(spec.warnings, "hole 1", "6 mm", "not used", "measured")


@pytest.mark.parametrize("part_type, user_values",
                         [("spacer", {"length": 30.0}), ("profile_extrusion", {"depth": 10.0})])
def test_measured_circle_that_becomes_no_feature_is_warned(part_type, user_values):  # R
    circles = [Circle(x=25, y=25, diameter=20), Circle(x=10, y=10, diameter=4)]
    spec = merge(topo(part_type=part_type), source_input="photo", measurements=meas(50.0, 50.0, circles),
                 user_values=user_values)
    assert isinstance(spec, PartSpec)
    assert spec.features == []
    assert warned(spec.warnings, "measured circle", "4 mm", "not used")
    if part_type == "spacer":
        assert spec.part.inner_diameter_mm == 20.0  # the largest circle is still the bore
    else:
        assert warned(spec.warnings, "measured circle", "20 mm", "not used")


def test_written_dimension_under_a_measured_one_is_warned():
    a = Annotations(items=[ann(62.0, linked_to="width")], confidence=0.9)
    spec = merge(topo(), source_input="photo", annotations=a, measurements=meas(60.0, 40.0),
                 user_values={"thickness": 5.0})
    assert spec.part.width_mm == 60.0 and spec.provenance["part.width_mm"] == "measured"
    assert warned(spec.warnings, "width", "62 mm", "not used", "measured")

    agreeing = Annotations(items=[ann(60.0, linked_to="width")], confidence=0.9)
    spec = merge(topo(), source_input="photo", annotations=agreeing, measurements=meas(60.0, 40.0),
                 user_values={"thickness": 5.0})
    assert not warned(spec.warnings, "width")  # nothing was lost, nothing to say


OTHER_DROP_PATHS = {
    "lower-confidence duplicate": (
        {}, [WIDTH, HEIGHT, THICK, ann(58.0, linked_to="width", conf=0.5)], {}, ("width", "58 mm")),
    "unlinked length beyond the open fields": (
        {}, [ann(60.0), ann(40.0), ann(5.0), ann(3.0)], {}, ("unlinked", "3 mm")),
    "second unlinked diameter on a plate": (
        {"holes": ONE_HOLE}, [WIDTH, HEIGHT, THICK, ann(6.0, "diameter"), ann(4.0, "diameter")], {},
        ("unlinked", "4 mm")),
    "unlinked diameter beyond the spacer fields": (
        {"part_type": "spacer"},
        [ann(20.0, "diameter"), ann(8.0, "diameter"), ann(3.0, "diameter"), ann(30.0, linked_to="length")], {},
        ("unlinked", "3 mm")),
    "hole_x with no hole_y": (
        {"holes": ONE_HOLE}, [WIDTH, HEIGHT, THICK, ann(12.0, linked_to="hole_x", hole_index=0)], {},
        ("hole 1", "hole_x", "12 mm")),
    "hole position with no hole number": (
        {"holes": ONE_HOLE}, [WIDTH, HEIGHT, THICK, ann(12.0, linked_to="hole_x"), ann(9.0, linked_to="hole_y")], {},
        ("hole_x", "12 mm")),
    "hole diameter with no holes": (
        {}, [WIDTH, HEIGHT, THICK, ann(6.0, "diameter", "hole_diameter")], {}, ("hole_diameter", "6 mm")),
    "hole_x on a spacer": (
        {"part_type": "spacer"},
        [ann(20.0, "diameter", "outer_diameter"), ann(30.0, linked_to="length"),
         ann(7.0, linked_to="hole_x", hole_index=0)], {}, ("hole_x", "7 mm")),
    "entered value the part has no field for": (
        {}, [WIDTH, HEIGHT, THICK], {"slot_length": 25.0}, ("slot_length", "25 mm")),
}


@pytest.mark.parametrize("case", list(OTHER_DROP_PATHS.values()), ids=list(OTHER_DROP_PATHS))
def test_other_drop_paths_warn(case):
    topo_kw, items, user_values, needles = case
    spec = merge(topo(**topo_kw), source_input="sketch", annotations=Annotations(items=items, confidence=0.9),
                 user_values=user_values)
    assert isinstance(spec, PartSpec)
    assert warned(spec.warnings, *needles, "not used"), spec.warnings
