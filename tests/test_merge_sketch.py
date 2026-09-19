from s2c.merge import merge
from s2c.partspec.models import Abstain, Annotation, Annotations, PartSpec, Topology


def ann(value, kind="linear", linked_to="unknown", hole_index=None, conf=0.9):
    return Annotation(value_mm=value, kind=kind, bbox_px=(0.0, 0.0, 1.0, 1.0),
                      linked_to=linked_to, hole_index=hole_index, confidence=conf)


def topo(**kw):
    base = {"part_type": "plate", "holes": [{"u": 0.2, "v": 0.25}], "confidence": 0.9}
    base.update(kw)
    return Topology.model_validate(base)


def test_linked_annotations_fill_the_plate():
    a = Annotations(items=[ann(60.0, linked_to="width"), ann(40.0, linked_to="height"),
                           ann(5.0, linked_to="thickness"), ann(6.0, "diameter", "hole_diameter")], confidence=0.9)
    spec = merge(topo(), source_input="sketch", annotations=a)
    assert isinstance(spec, PartSpec)
    assert spec.part.width_mm == 60 and spec.part.height_mm == 40 and spec.part.thickness_mm == 5
    assert spec.provenance["part.width_mm"] == "user_written"
    hole = spec.features[0]
    assert (hole.x_mm, hole.y_mm, hole.diameter_mm) == (12.0, 10.0, 6.0)
    assert spec.provenance["features[0].x_mm"] == "default"
    assert any("hole position" in w for w in spec.warnings)


def test_unlinked_values_are_assigned_largest_first():
    a = Annotations(items=[ann(5.0), ann(60.0), ann(40.0)], confidence=0.8)
    spec = merge(topo(holes=[]), source_input="sketch", annotations=a)
    assert (spec.part.width_mm, spec.part.height_mm, spec.part.thickness_mm) == (60, 40, 5)


def test_missing_thickness_abstains_with_partial():
    a = Annotations(items=[ann(60.0, linked_to="width"), ann(40.0, linked_to="height")], confidence=0.8)
    out = merge(topo(holes=[]), source_input="sketch", annotations=a)
    assert isinstance(out, Abstain)
    assert out.stage == "merge" and out.reason == "missing_thickness"
    assert out.partial == {"width": 60.0, "height": 40.0}
    assert "thickness" in out.remedy


def test_user_values_fill_the_gap_as_user_edited():
    a = Annotations(items=[ann(60.0, linked_to="width"), ann(40.0, linked_to="height")], confidence=0.8)
    spec = merge(topo(holes=[]), source_input="sketch", annotations=a, user_values={"thickness": 4.0})
    assert spec.part.thickness_mm == 4.0
    assert spec.provenance["part.thickness_mm"] == "user_edited"


def test_unsupported_topology_abstains():
    out = merge(topo(part_type="unsupported", unsupported_reason="curved surface", holes=[]),
                source_input="sketch", annotations=Annotations(confidence=0.5))
    assert isinstance(out, Abstain) and out.reason == "unsupported_geometry"
    assert "curved surface" in out.remedy


def test_profile_extrusion_from_sketch_abstains():
    out = merge(topo(part_type="profile_extrusion", holes=[]), source_input="sketch",
                annotations=Annotations(confidence=0.5))
    assert isinstance(out, Abstain) and out.reason == "profile_needs_photo"


def test_spacer_uses_diameter_annotations():
    a = Annotations(items=[ann(20.0, "diameter"), ann(8.0, "diameter"), ann(30.0, linked_to="length")], confidence=0.9)
    spec = merge(topo(part_type="spacer", holes=[]), source_input="sketch", annotations=a)
    assert (spec.part.outer_diameter_mm, spec.part.inner_diameter_mm, spec.part.length_mm) == (20, 8, 30)


def test_confidence_is_the_minimum_of_stages():
    a = Annotations(items=[ann(60.0, linked_to="width"), ann(40.0, linked_to="height"),
                           ann(5.0, linked_to="thickness")], confidence=0.6)
    spec = merge(topo(holes=[]), source_input="sketch", annotations=a)
    assert spec.confidence == 0.6
