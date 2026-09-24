"""Every value merge reads either lands in the spec or comes back as a warning. None vanish,
and none get relabelled as a different measurement."""
from s2c.merge import merge
from s2c.partspec.models import Abstain, Annotation, Annotations, PartSpec, Topology


def ann(value, kind="linear", linked_to="unknown", hole_index=None, conf=0.9):
    return Annotation(value_mm=value, kind=kind, bbox_px=(0.0, 0.0, 1.0, 1.0),
                      linked_to=linked_to, hole_index=hole_index, confidence=conf)


def topo(**kw):
    base = {"part_type": "plate", "holes": [], "confidence": 0.9}
    base.update(kw)
    return Topology.model_validate(base)


def warned(warnings, *needles):
    return [w for w in warnings if all(n in w for n in needles)]


def test_slot_length_never_becomes_thickness():
    plain = [ann(60.0, linked_to="width"), ann(40.0, linked_to="height")]
    without_slot = merge(topo(), source_input="sketch", annotations=Annotations(items=plain, confidence=0.9))
    a = Annotations(items=[*plain, ann(25.0, linked_to="slot_length")], confidence=0.9)
    out = merge(topo(), source_input="sketch", annotations=a)

    # same abstention as with no slot annotation at all: the slot length is not a thickness
    assert isinstance(out, Abstain), f"slot length was used as a part dimension: {out}"
    assert without_slot.reason == "missing_thickness"
    assert (out.reason, out.remedy) == (without_slot.reason, without_slot.remedy)
    assert {k: v for k, v in out.partial.items() if k != "warnings"} == without_slot.partial
    assert "thickness" not in out.partial
    # and the slot value is carried, not silently lost
    assert warned(out.partial["warnings"], "slot_length", "25 mm", "not used")

    spec = merge(topo(), source_input="sketch", annotations=a, user_values={"thickness": 4.0})
    assert isinstance(spec, PartSpec)
    assert spec.part.thickness_mm == 4.0 and spec.provenance["part.thickness_mm"] == "user_edited"
    assert warned(spec.warnings, "slot_length", "25 mm", "not used")


def test_slot_width_is_never_assigned_positionally():
    a = Annotations(items=[ann(60.0), ann(40.0), ann(5.0), ann(8.0, linked_to="slot_width")], confidence=0.9)
    spec = merge(topo(), source_input="sketch", annotations=a)
    assert isinstance(spec, PartSpec)
    assert (spec.part.width_mm, spec.part.height_mm, spec.part.thickness_mm) == (60, 40, 5)
    assert warned(spec.warnings, "slot_width", "8 mm", "not used")
