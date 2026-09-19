import pytest
from pydantic import ValidationError

from s2c.partspec.models import Hole, PartSpec, Plate, numeric_field_paths


def plate_spec(**overrides):
    data = {
        "source_input": "sketch",
        "part": {"type": "plate", "width_mm": 60, "height_mm": 40, "thickness_mm": 5},
        "features": [{"type": "hole", "x_mm": 10, "y_mm": 10, "diameter_mm": 6}],
        "provenance": {
            "part.width_mm": "user_written",
            "part.height_mm": "user_written",
            "part.thickness_mm": "user_written",
            "part.corner_radius_mm": "default",
            "features[0].x_mm": "default",
            "features[0].y_mm": "default",
            "features[0].diameter_mm": "user_written",
        },
        "confidence": 0.9,
    }
    data.update(overrides)
    return data


def test_valid_plate_spec_round_trips():
    spec = PartSpec.model_validate(plate_spec())
    assert isinstance(spec.part, Plate)
    assert isinstance(spec.features[0], Hole)
    assert PartSpec.model_validate_json(spec.model_dump_json()) == spec


def test_numeric_field_paths_lists_every_number():
    spec = PartSpec.model_validate(plate_spec())
    assert numeric_field_paths(spec) == [
        "part.width_mm", "part.height_mm", "part.thickness_mm", "part.corner_radius_mm",
        "features[0].x_mm", "features[0].y_mm", "features[0].diameter_mm",
    ]


def test_missing_provenance_is_rejected():
    data = plate_spec()
    del data["provenance"]["part.thickness_mm"]
    with pytest.raises(ValidationError, match="part.thickness_mm"):
        PartSpec.model_validate(data)


def test_negative_dimension_is_rejected():
    data = plate_spec()
    data["part"]["width_mm"] = -1
    with pytest.raises(ValidationError):
        PartSpec.model_validate(data)


def test_unit_strings_are_rejected():
    data = plate_spec()
    data["part"]["width_mm"] = "60mm"
    with pytest.raises(ValidationError):
        PartSpec.model_validate(data)


def test_unknown_part_type_is_rejected():
    data = plate_spec()
    data["part"] = {"type": "gear", "teeth": 12}
    with pytest.raises(ValidationError):
        PartSpec.model_validate(data)


def test_flange_bore_must_fit_inside_bolt_circle():
    data = plate_spec(part={
        "type": "flange", "outer_diameter_mm": 80, "bore_diameter_mm": 70, "thickness_mm": 6,
        "bolt_circle_diameter_mm": 60, "bolt_count": 4, "bolt_hole_diameter_mm": 6,
    }, features=[], provenance={
        "part.outer_diameter_mm": "measured", "part.bore_diameter_mm": "measured",
        "part.thickness_mm": "user_edited", "part.bolt_circle_diameter_mm": "measured",
        "part.bolt_count": "measured", "part.bolt_hole_diameter_mm": "measured",
    })
    with pytest.raises(ValidationError, match="bore"):
        PartSpec.model_validate(data)


def test_model_invented_extra_field_is_rejected():
    # extra="forbid" is the mechanism that enforces "the model never supplies a dimension".
    # Prove it rejects an unexpected field on the part itself...
    data = plate_spec()
    data["part"]["estimated_thickness_mm"] = 5.0
    with pytest.raises(ValidationError):
        PartSpec.model_validate(data)

    # ...and on a nested feature, not just the top-level part.
    data = plate_spec()
    data["features"][0]["width"] = 3.0
    with pytest.raises(ValidationError):
        PartSpec.model_validate(data)


def test_spacer_inner_diameter_must_be_smaller_than_outer():
    spacer_provenance = {
        "part.outer_diameter_mm": "measured",
        "part.inner_diameter_mm": "measured",
        "part.length_mm": "measured",
    }

    equal_data = plate_spec(
        part={"type": "spacer", "outer_diameter_mm": 20, "inner_diameter_mm": 20, "length_mm": 10},
        features=[],
        provenance=spacer_provenance,
    )
    with pytest.raises(ValidationError, match="inner diameter must be smaller than outer diameter"):
        PartSpec.model_validate(equal_data)

    larger_data = plate_spec(
        part={"type": "spacer", "outer_diameter_mm": 20, "inner_diameter_mm": 25, "length_mm": 10},
        features=[],
        provenance=spacer_provenance,
    )
    with pytest.raises(ValidationError, match="inner diameter must be smaller than outer diameter"):
        PartSpec.model_validate(larger_data)


def test_numeric_field_paths_covers_slot_and_finishes():
    data = plate_spec(
        features=[{
            "type": "slot", "x_mm": 15, "y_mm": 5, "width_mm": 4, "length_mm": 12,
            "angle_deg": 45.0, "depth_mm": 2.0,
        }],
        finishes=[
            {"type": "fillet", "radius_mm": 2.0},
            {"type": "chamfer", "radius_mm": 1.5},
        ],
        provenance={
            "part.width_mm": "user_written",
            "part.height_mm": "user_written",
            "part.thickness_mm": "user_written",
            "part.corner_radius_mm": "default",
            "features[0].x_mm": "default",
            "features[0].y_mm": "default",
            "features[0].width_mm": "user_written",
            "features[0].length_mm": "user_written",
            "features[0].angle_deg": "default",
            "features[0].depth_mm": "default",
            "finishes[0].radius_mm": "user_written",
            "finishes[1].radius_mm": "user_written",
        },
    )
    spec = PartSpec.model_validate(data)
    assert numeric_field_paths(spec) == [
        "part.width_mm", "part.height_mm", "part.thickness_mm", "part.corner_radius_mm",
        "features[0].x_mm", "features[0].y_mm", "features[0].width_mm", "features[0].length_mm",
        "features[0].angle_deg", "features[0].depth_mm",
        "finishes[0].radius_mm", "finishes[1].radius_mm",
    ]


def test_profile_extrusion_requires_at_least_three_points():
    data = plate_spec(
        part={"type": "profile_extrusion", "points_mm": [[0, 0], [10, 0]], "depth_mm": 5},
        features=[],
        provenance={"part.points_mm": "measured", "part.depth_mm": "user_written"},
    )
    with pytest.raises(ValidationError, match="points_mm"):
        PartSpec.model_validate(data)
