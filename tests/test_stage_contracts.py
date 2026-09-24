import json

import pytest
from pydantic import ValidationError

from s2c.partspec.models import Abstain, Annotation, Annotations, Measurements, Topology
from s2c.partspec.schema import topology_json_schema


def test_topology_accepts_normalised_positions():
    t = Topology.model_validate({
        "part_type": "plate", "holes": [{"u": 0.2, "v": 0.5, "kind": "through"}],
        "confidence": 0.8,
    })
    assert t.holes[0].u == 0.2
    assert t.view == "unknown"


def test_topology_rejects_any_dimension_field():
    with pytest.raises(ValidationError):
        Topology.model_validate({"part_type": "plate", "width_mm": 60, "confidence": 0.8})


def test_topology_rejects_position_outside_unit_square():
    with pytest.raises(ValidationError):
        Topology.model_validate({"part_type": "plate", "holes": [{"u": 1.5, "v": 0}], "confidence": 0.8})


def test_annotations_and_measurements_validate():
    a = Annotations.model_validate({"items": [
        {"value_mm": 60, "kind": "linear", "bbox_px": [1, 2, 30, 12], "linked_to": "width", "confidence": 0.9}
    ], "confidence": 0.9})
    assert a.items[0].hole_index is None
    m = Measurements.model_validate({
        "mm_per_px": 0.1, "coin": {"name": "1 TND", "pixel_diameter": 250, "eccentricity": 0.05, "confidence": 0.95},
        "outer_contour_mm": [[0, 0], [60, 0], [60, 40], [0, 40]], "bbox_mm": {"width": 60, "height": 40},
        "circles_mm": [{"x": 10, "y": 10, "diameter": 6}], "confidence": 0.9,
    })
    assert m.bbox_mm.width == 60


def test_abstain_carries_reason_and_remedy():
    a = Abstain(stage="metrology", reason="coin_tilted", remedy="Lay the coin flat, shoot top-down, retake.")
    assert a.partial is None


def test_annotation_hole_index_rejects_negative_value():
    """A negative hole_index never raises on its own in Python (negative
    indexing), so a later stage would silently select the wrong hole, or
    silently drop a user-written measurement. Reject it here, loudly."""
    with pytest.raises(ValidationError) as exc_info:
        Annotation.model_validate({
            "value_mm": 60, "kind": "linear", "bbox_px": [1, 2, 30, 12],
            "hole_index": -1, "confidence": 0.9,
        })
    assert "hole_index" in str(exc_info.value)


def test_topology_bolt_count_rejects_negative_value():
    with pytest.raises(ValidationError) as exc_info:
        Topology.model_validate({"part_type": "plate", "bolt_count": -1, "confidence": 0.8})
    assert "bolt_count" in str(exc_info.value)


def test_topology_annotation_count_rejects_negative_value():
    with pytest.raises(ValidationError) as exc_info:
        Topology.model_validate({"part_type": "plate", "annotation_count": -1, "confidence": 0.8})
    assert "annotation_count" in str(exc_info.value)


_MEASUREMENT_WORDS = {
    "width", "height", "thickness", "diameter", "radius", "depth",
    "length", "distance", "size", "dimension", "spacing", "offset",
    "pitch", "angle",
}


def _collect_property_names(node):
    """Walk every 'properties' block in a JSON schema, including nested
    $defs (e.g. TopoHole, TopoSlot), and return every field name found."""
    names = []
    if isinstance(node, dict):
        props = node.get("properties")
        if isinstance(props, dict):
            names.extend(props.keys())
        for value in node.values():
            names.extend(_collect_property_names(value))
    elif isinstance(node, list):
        for item in node:
            names.extend(_collect_property_names(item))
    return names


def test_topology_schema_exposes_no_measurement_fields():
    """The schema text is pasted into the prompt sent to the vision model,
    so a measurement-shaped field name (a "_mm" suffix, or a bare word like
    "width") would invite the model to return a dimension -- which this
    project forbids. Walk the actual field names, including nested
    definitions, rather than substring-searching the whole schema: a word
    like "symmetric" contains "mm" by coincidence of spelling and is not a
    measurement."""
    schema = json.loads(topology_json_schema())
    assert schema["properties"]["part_type"]["enum"][0] == "plate"

    for name in _collect_property_names(schema):
        lowered = name.lower()
        assert "_mm" not in lowered, f"measurement-shaped field name found: {name!r}"
        offenders = [word for word in _MEASUREMENT_WORDS if word in lowered]
        assert not offenders, f"measurement-shaped field name found: {name!r} (matches {offenders})"
