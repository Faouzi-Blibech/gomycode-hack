"""Frozen contracts. Changing anything here needs a PR approved by all three owners."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Mm = Annotated[float, Field(gt=0, description="millimetres")]
NonNegMm = Annotated[float, Field(ge=0, description="millimetres")]
Provenance = Literal["measured", "user_written", "user_edited", "default"]
SourceInput = Literal["sketch", "photo", "drawing"]
EdgeSelector = Literal["all", "all_vertical", "top", "bottom"]


class _Strict(BaseModel):
    """extra="forbid" is what rejects a model-invented width_mm. Not strict mode: lists must
    still validate as tuples when specs arrive as Python dicts from json.loads."""
    model_config = ConfigDict(extra="forbid")


# ---- features -------------------------------------------------------------

class Hole(_Strict):
    type: Literal["hole"] = "hole"
    x_mm: float
    y_mm: float
    diameter_mm: Mm
    depth_mm: Mm | None = None  # None means through
    leg: Literal["a", "b"] | None = None  # l_bracket only


class Slot(_Strict):
    type: Literal["slot"] = "slot"
    x_mm: float
    y_mm: float
    width_mm: Mm
    length_mm: Mm
    angle_deg: float = 0.0
    depth_mm: Mm | None = None
    leg: Literal["a", "b"] | None = None


class Fillet(_Strict):
    type: Literal["fillet"] = "fillet"
    edges: EdgeSelector = "all_vertical"
    radius_mm: Mm


class Chamfer(_Strict):
    type: Literal["chamfer"] = "chamfer"
    edges: EdgeSelector = "all_vertical"
    radius_mm: Mm


# ---- part types -----------------------------------------------------------

class Plate(_Strict):
    type: Literal["plate"] = "plate"
    width_mm: Mm
    height_mm: Mm
    thickness_mm: Mm
    corner_radius_mm: NonNegMm = 0.0


class LBracket(_Strict):
    type: Literal["l_bracket"] = "l_bracket"
    leg_a_mm: Mm
    leg_b_mm: Mm
    width_mm: Mm
    thickness_mm: Mm
    angle_deg: float = 90.0


class Flange(_Strict):
    type: Literal["flange"] = "flange"
    outer_diameter_mm: Mm
    bore_diameter_mm: Mm
    thickness_mm: Mm
    bolt_circle_diameter_mm: Mm
    bolt_count: int = Field(ge=0)
    bolt_hole_diameter_mm: Mm

    @model_validator(mode="after")
    def _nested_diameters(self) -> Flange:
        if not self.bore_diameter_mm < self.bolt_circle_diameter_mm < self.outer_diameter_mm:
            raise ValueError("bore must be inside the bolt circle, bolt circle inside the outer diameter")
        return self


class Spacer(_Strict):
    type: Literal["spacer"] = "spacer"
    outer_diameter_mm: Mm
    inner_diameter_mm: NonNegMm = 0.0  # 0 means solid
    length_mm: Mm

    @model_validator(mode="after")
    def _inner_inside_outer(self) -> Spacer:
        if self.inner_diameter_mm >= self.outer_diameter_mm:
            raise ValueError("inner diameter must be smaller than outer diameter")
        return self


class ProfileExtrusion(_Strict):
    type: Literal["profile_extrusion"] = "profile_extrusion"
    points_mm: list[tuple[float, float]] = Field(min_length=3)
    depth_mm: Mm


Part = Annotated[Plate | LBracket | Flange | Spacer | ProfileExtrusion, Field(discriminator="type")]
Feature = Annotated[Hole | Slot, Field(discriminator="type")]
Finish = Annotated[Fillet | Chamfer, Field(discriminator="type")]
PartType = Literal["plate", "l_bracket", "flange", "spacer", "profile_extrusion"]


# ---- PartSpec -------------------------------------------------------------

class PartSpec(_Strict):
    version: Literal["1"] = "1"
    source_input: SourceInput
    part: Part
    features: list[Feature] = []
    finishes: list[Finish] = []
    provenance: dict[str, Provenance]
    confidence: float = Field(ge=0, le=1)
    warnings: list[str] = []

    @model_validator(mode="after")
    def _every_number_has_provenance(self) -> PartSpec:
        missing = [p for p in numeric_field_paths(self) if p not in self.provenance]
        if missing:
            raise ValueError(f"missing provenance for {missing}")
        return self


def _numeric_names(model: BaseModel) -> list[str]:
    names = []
    for name, value in model.model_dump().items():
        if name == "type":
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) or name == "points_mm":
            names.append(name)
    return names


def numeric_field_paths(spec: PartSpec) -> list[str]:
    """Every field that must carry provenance, in a stable order."""
    paths = [f"part.{n}" for n in _numeric_names(spec.part)]
    for i, feature in enumerate(spec.features):
        paths += [f"features[{i}].{n}" for n in _numeric_names(feature)]
    for i, finish in enumerate(spec.finishes):
        paths += [f"finishes[{i}].{n}" for n in _numeric_names(finish)]
    return paths
