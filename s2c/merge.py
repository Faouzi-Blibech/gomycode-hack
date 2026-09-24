"""Fuse Topology + Annotations + Measurements into a PartSpec. The only place numbers meet topology."""
from __future__ import annotations

from pydantic import ValidationError

from s2c.partspec.models import (
    Abstain,
    Annotation,
    Annotations,
    Measurements,
    PartSpec,
    Provenance,
    SourceInput,
    Topology,
)

# annotation key -> PartSpec part field, per type
FIELD_MAP: dict[str, dict[str, str]] = {
    "plate": {"width": "width_mm", "height": "height_mm", "thickness": "thickness_mm",
              "corner_radius": "corner_radius_mm"},
    "l_bracket": {"leg_a": "leg_a_mm", "leg_b": "leg_b_mm", "width": "width_mm", "thickness": "thickness_mm"},
    "spacer": {"outer_diameter": "outer_diameter_mm", "inner_diameter": "inner_diameter_mm",
               "length": "length_mm"},
    "flange": {"outer_diameter": "outer_diameter_mm", "inner_diameter": "bore_diameter_mm",
               "thickness": "thickness_mm", "bolt_circle_diameter": "bolt_circle_diameter_mm",
               "bolt_hole_diameter": "bolt_hole_diameter_mm"},
    "profile_extrusion": {"depth": "depth_mm"},
}
REQUIRED: dict[str, list[str]] = {
    "plate": ["width", "height", "thickness"],
    "l_bracket": ["leg_a", "leg_b", "width", "thickness"],
    "spacer": ["outer_diameter", "length"],
    "flange": ["outer_diameter", "inner_diameter", "thickness", "bolt_circle_diameter", "bolt_hole_diameter"],
    "profile_extrusion": ["depth"],
}
# where unlinked values go, largest first
LINEAR_ORDER: dict[str, list[str]] = {
    "plate": ["width", "height", "thickness"],
    "l_bracket": ["leg_a", "leg_b", "width", "thickness"],
    "spacer": ["length"],
    "flange": ["thickness"],
    "profile_extrusion": ["depth"],
}
DIAMETER_ORDER: dict[str, list[str]] = {
    "plate": [], "l_bracket": [], "profile_extrusion": [],
    "spacer": ["outer_diameter", "inner_diameter"],
    "flange": ["outer_diameter", "bolt_circle_diameter", "inner_diameter", "bolt_hole_diameter"],
}

# known link targets merge cannot build yet: they go to the collector, never to positional assignment
SLOT_KEYS = ("slot_length", "slot_width")

Values = dict[str, tuple[float, Provenance]]


class _Unplaced:
    """The one collector for values merge read but cannot use. Each becomes one warning, so a
    written or measured number never disappears silently or turns into a different measurement."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def written(self, a: Annotation, why: str) -> None:
        link = "unlinked" if a.linked_to == "unknown" else a.linked_to
        self._add(f"written {link} ({a.kind})", a.value_mm, why)

    def _add(self, what: str, value_mm: float, why: str) -> None:
        self.warnings.append(f"{what} {value_mm:g} mm not used: {why}")


def merge(
    topology: Topology,
    *,
    source_input: SourceInput,
    annotations: Annotations | None = None,
    measurements: Measurements | None = None,
    user_values: dict[str, float] | None = None,
) -> PartSpec | Abstain:
    if topology.part_type == "unsupported":
        return Abstain(stage="merge", reason="unsupported_geometry",
                       remedy=f"Unsupported geometry: {topology.unsupported_reason or 'not a flat profile'}.")
    if topology.part_type == "profile_extrusion" and measurements is None:
        return Abstain(stage="merge", reason="profile_needs_photo",
                       remedy="Free-form outlines need a top-down photo with a coin for scale.")

    kind = topology.part_type
    unplaced = _Unplaced()
    values: Values = {}
    hole_values: dict[int | None, dict[str, float]] = {}
    confidences = [topology.confidence]

    if measurements is not None:
        values.update(_values_from_measurements(measurements, kind))
        confidences.append(measurements.confidence)
    if annotations is not None:
        ann_values, hole_values = _values_from_annotations(annotations, kind, unplaced)
        for k, v in ann_values.items():
            values.setdefault(k, v)
        confidences.append(annotations.confidence)
    for k, v in (user_values or {}).items():
        values[k] = (float(v), "user_edited")

    missing = [k for k in REQUIRED[kind] if k not in values]
    if missing:
        field = missing[0]
        return Abstain(
            stage="merge", reason=f"missing_{field}",
            remedy=f"We could not read the {field.replace('_', ' ')}. Enter it below.",
            partial=_partial(values, unplaced),
        )

    part, provenance = _assemble_part(kind, values, measurements, topology)
    features, feat_prov, feat_warn = _assemble_holes(kind, topology, values, hole_values, measurements)
    provenance.update(feat_prov)

    try:
        return PartSpec(
            source_input=source_input, part=part, features=features, finishes=[],
            provenance=provenance, confidence=min(confidences), warnings=unplaced.warnings + feat_warn,
        )
    except ValidationError as e:
        return Abstain(stage="merge", reason="inconsistent_dimensions",
                       remedy=f"The dimensions do not fit together: {e.errors()[0]['msg']}.",
                       partial=_partial(values, unplaced))


def _partial(values: Values, unplaced: _Unplaced) -> dict:
    """What was recovered, for the UI. Abstain has no warnings field, so the unplaced-value
    warnings ride in `partial` under "warnings", only when there are any."""
    partial: dict = {k: v for k, (v, _) in values.items()}
    if unplaced.warnings:
        partial["warnings"] = list(unplaced.warnings)
    return partial


def _values_from_annotations(ann: Annotations, kind: str, unplaced: _Unplaced) -> tuple[Values, dict]:
    values: Values = {}
    hole_values: dict[int | None, dict[str, float]] = {}
    unknown_linear: list[float] = []
    unknown_diam: list[float] = []
    for a in sorted(ann.items, key=lambda a: -a.confidence):
        if a.linked_to in ("hole_diameter", "hole_x", "hole_y"):
            hole_values.setdefault(a.hole_index, {}).setdefault(a.linked_to, a.value_mm)
        elif a.linked_to in SLOT_KEYS:  # a slot dimension is never a part dimension
            unplaced.written(a, "merge does not build slots yet")
        elif a.linked_to != "unknown" and a.linked_to in FIELD_MAP[kind]:
            values.setdefault(a.linked_to, (a.value_mm, "user_written"))
        else:  # unknown, or linked to a field this part type does not have: never drop a value
            if a.kind == "diameter":
                unknown_diam.append(a.value_mm)
            elif a.kind == "radius":
                if kind == "plate":
                    values.setdefault("corner_radius", (a.value_mm, "user_written"))
            else:
                unknown_linear.append(a.value_mm)
    for key, val in zip([k for k in LINEAR_ORDER[kind] if k not in values], sorted(unknown_linear, reverse=True)):
        values[key] = (val, "user_written")
    for key, val in zip([k for k in DIAMETER_ORDER[kind] if k not in values], sorted(unknown_diam, reverse=True)):
        values[key] = (val, "user_written")
    if kind in ("plate", "l_bracket") and unknown_diam and None not in hole_values:
        hole_values[None] = {"hole_diameter": max(unknown_diam)}
    return values, hole_values


def _values_from_measurements(m: Measurements, kind: str) -> Values:
    w, h = m.bbox_mm.width, m.bbox_mm.height
    circles = sorted(m.circles_mm, key=lambda c: -c.diameter)
    v: Values = {}
    if kind == "plate":
        v["width"], v["height"] = (w, "measured"), (h, "measured")
    elif kind == "l_bracket":
        v["leg_a"], v["leg_b"] = (w, "measured"), (h, "measured")
    elif kind == "spacer":
        v["outer_diameter"] = (max(w, h), "measured")
        v["inner_diameter"] = ((circles[0].diameter if circles else 0.0), "measured")
    elif kind == "flange":
        v["outer_diameter"] = (max(w, h), "measured")
        if circles:
            v["inner_diameter"] = (circles[0].diameter, "measured")
        bolts = circles[1:]
        if bolts:
            cx, cy = w / 2, h / 2
            radius = sum(((c.x - cx) ** 2 + (c.y - cy) ** 2) ** 0.5 for c in bolts) / len(bolts)
            v["bolt_circle_diameter"] = (2 * radius, "measured")
            v["bolt_hole_diameter"] = (sum(c.diameter for c in bolts) / len(bolts), "measured")
    return v


def _assemble_part(kind: str, values: Values, m: Measurements | None, topology: Topology):
    fields: dict = {"type": kind}
    prov: dict[str, Provenance] = {}
    for key, field in FIELD_MAP[kind].items():
        if key in values:
            value, source = values[key]
            fields[field], prov[f"part.{field}"] = float(value), source
    if kind == "plate" and "corner_radius_mm" not in fields:
        fields["corner_radius_mm"], prov["part.corner_radius_mm"] = 0.0, "default"
    if kind == "l_bracket":
        fields["angle_deg"], prov["part.angle_deg"] = 90.0, "default"
    if kind == "spacer" and "inner_diameter_mm" not in fields:
        fields["inner_diameter_mm"], prov["part.inner_diameter_mm"] = 0.0, "default"
    if kind == "flange":
        if m is not None and m.circles_mm:
            count, source = max(len(m.circles_mm) - 1, 0), "measured"
        elif topology.bolt_count:
            count, source = topology.bolt_count, "default"  # a count is topology, not a dimension
        else:
            count, source = 0, "default"
        fields["bolt_count"], prov["part.bolt_count"] = count, source
    if kind == "profile_extrusion":
        fields["points_mm"] = _simplify(m.outer_contour_mm) if m is not None else []
        prov["part.points_mm"] = "measured"
    return fields, prov


def _assemble_holes(kind, topology: Topology, values: Values, hole_values: dict, m: Measurements | None):
    if kind not in ("plate", "l_bracket"):
        return [], {}, []
    if kind == "plate":
        w, h = values["width"][0], values["height"][0]
    else:
        w, h = values["leg_a"][0], values["width"][0]
    features, prov, warnings = [], {}, []
    shared_d = hole_values.get(None, {}).get("hole_diameter")

    if m is not None and m.circles_mm:
        for i, c in enumerate(m.circles_mm):
            features.append({"type": "hole", "x_mm": c.x, "y_mm": c.y, "diameter_mm": c.diameter})
            prov.update({f"features[{i}].x_mm": "measured", f"features[{i}].y_mm": "measured",
                         f"features[{i}].diameter_mm": "measured"})
        return features, prov, warnings

    for i, th in enumerate(topology.holes):
        own = hole_values.get(i, {})
        d = own.get("hole_diameter", shared_d)
        if d is None:
            warnings.append(f"hole {i + 1}: no diameter written, defaulting to 5 mm")
            d, d_prov = 5.0, "default"
        else:
            d_prov = "user_written"
        if "hole_x" in own and "hole_y" in own:
            x, y, p_prov = own["hole_x"], own["hole_y"], "user_written"
        else:
            x, y, p_prov = round(th.u * w, 1), round(th.v * h, 1), "default"
            warnings.append(f"hole {i + 1}: hole position estimated from the sketch, confirm it")
        features.append({"type": "hole", "x_mm": x, "y_mm": y, "diameter_mm": d, "leg": "a" if kind == "l_bracket" else None})
        prov.update({f"features[{i}].x_mm": p_prov, f"features[{i}].y_mm": p_prov, f"features[{i}].diameter_mm": d_prov})
    return features, prov, warnings


def _simplify(points: list[tuple[float, float]], epsilon_mm: float = 0.5) -> list[tuple[float, float]]:
    import cv2
    import numpy as np
    arr = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(arr, epsilon_mm, True).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in approx]
