"""The SketchReading contract (spec section 10.1). The team builds against this; change it only by
agreement."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Badge = Literal["written", "edited", "derived", "uncertain", "predicted", "conflict"]
Evidence = Literal["reader", "user", "arithmetic", "cross_view", "symmetry", "pattern",
                   "standard", "proportion", "geometry"]
ViewName = Literal["front", "top", "right", "left", "bottom", "back"]
Unit = float
Box = tuple[float, float, float, float]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class View(_Strict):
    name: ViewName
    label_text: str | None
    bbox_px: Box
    size_mm: tuple[float, float]


class Entity(_Strict):
    id: str
    view: ViewName
    type: Literal["line", "arc", "circle", "curve"]
    line_type: Literal["visible", "hidden", "centre"]
    mm: dict
    px: dict
    confidence: float = Field(ge=0, le=1)


class Reading(_Strict):
    reader: str
    text: str
    confidence: float = Field(ge=0, le=1)


class Dimension(_Strict):
    id: str
    view: ViewName | None
    kind: Literal["linear", "diameter", "radius", "angle", "chamfer", "thread"]
    value: float | None
    text_raw: str
    readings: list[Reading]
    candidates: list[float] = []
    implied: float | None = None
    tolerance: str | None = None
    measures: list[str]
    axis: Literal["a", "b"] | None = None
    badge: Badge
    evidence: Evidence
    bbox_px: Box


class Feature(_Strict):
    id: str
    type: Literal["hole", "slot", "other"]
    axis: Literal["x", "y", "z"]
    position_mm: tuple[float, float, float]
    diameter: float | None = None
    width: float | None = None
    length: float | None = None
    through: bool | None = None
    depth: float | None = None
    evidence: list[str]
    badge: Badge


class Size(_Strict):
    value: float
    badge: Badge
    evidence: Evidence


class Issue(_Strict):
    severity: Literal["red", "amber"]
    kind: Literal["conflict", "uncertain", "predicted", "unplaced", "decimal", "unit", "label",
                  "meaning_check", "not_in_3d", "gap_closed"]
    message: str
    targets: list[str]


class SketchAbstain(_Strict):
    stage: Literal["capture", "text", "views", "vectorize", "link", "solve"]
    reason: str
    remedy: str


class SketchReading(_Strict):
    version: Literal["sketch-1"] = "sketch-1"
    units: Literal["mm"] = "mm"
    image_size_px: tuple[int, int]
    views: list[View]
    entities: list[Entity]
    dimensions: list[Dimension]
    features: list[Feature]
    envelope: dict[Literal["x", "y", "z"], Size]
    issues: list[Issue]
    abstain: SketchAbstain | None = None
    timings_ms: dict[str, float]
