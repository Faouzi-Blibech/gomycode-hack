"""Fuse Topology + Annotations + Measurements into a PartSpec. The only place numbers meet topology."""
from __future__ import annotations

from pydantic import ValidationError

from s2c.partspec.models import (
    Abstain,
    Annotation,
    Annotations,
    Circle,
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

HOLE_KEYS = ("hole_diameter", "hole_x", "hole_y")
# known link targets merge cannot build yet: they go to the collector, never to positional assignment
SLOT_KEYS = ("slot_length", "slot_width")
HOLE_TYPES = ("plate", "l_bracket")  # the only part types merge places holes on
NAMES = {"plate": "a plate", "l_bracket": "an L-bracket", "flange": "a flange", "spacer": "a spacer",
         "profile_extrusion": "a profile extrusion"}
# link targets whose field holds a diameter or a radius (from the *_diameter_mm / *_radius_mm model
# fields they fill); a written R on a diameter field is doubled, a written diameter on a radius halved
DIAMETER_KEYS = ("outer_diameter", "inner_diameter", "bolt_circle_diameter", "bolt_hole_diameter", "hole_diameter")
RADIUS_KEYS = ("corner_radius",)

Values = dict[str, tuple[float, Provenance]]
Written = dict[str, Annotation]  # annotation key -> the annotation that fills it
WrittenHoles = dict[int | None, dict[str, Annotation]]  # hole index (None: every hole) -> key -> annotation


class _Unplaced:
    """The one collector for values merge read but cannot use. Each becomes one warning, so a
    written or measured number never disappears silently or turns into a different measurement."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def written(self, a: Annotation, why: str) -> None:
        hole = f"hole {a.hole_index + 1}: " if a.linked_to in HOLE_KEYS and a.hole_index is not None else ""
        link = "unlinked" if a.linked_to == "unknown" else a.linked_to
        self._add(f"{hole}written {link} ({a.kind})", a.value_mm, why)

    def measured_circle(self, c: Circle, why: str) -> None:
        self._add(f"measured circle at ({c.x:g}, {c.y:g}), diameter", c.diameter, why)

    def entered(self, key: str, value_mm: float, why: str) -> None:
        self._add(f"entered {key}", value_mm, why)

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
    written: Written = {}
    written_holes: WrittenHoles = {}
    confidences = [topology.confidence]

    if measurements is not None:
        values.update(_values_from_measurements(measurements, kind, unplaced))
        confidences.append(measurements.confidence)
    if annotations is not None:
        written, written_holes = _values_from_annotations(annotations, kind, unplaced)
        for key, a in written.items():
            if key not in values:
                values[key] = (_mm(a, key), "user_written")
            elif _mm(a, key) != values[key][0]:  # measured wins over written, but never silently
                unplaced.written(a, f"the measured {key} of {values[key][0]:g} mm takes precedence")
        confidences.append(annotations.confidence)
    usable_holes = _usable_hole_values(kind, topology, written_holes, measurements, unplaced)
    for key, v in (user_values or {}).items():
        if key in FIELD_MAP[kind]:
            values[key] = (float(v), "user_edited")
        else:
            unplaced.entered(key, float(v), f"{NAMES[kind]} has no {key}")

    missing = [k for k in REQUIRED[kind] if k not in values]
    if missing:
        field = missing[0]
        return Abstain(
            stage="merge", reason=f"missing_{field}",
            remedy=f"We could not read the {field.replace('_', ' ')}. Enter it below.",
            partial=_partial(values),
        )

    part, provenance = _assemble_part(kind, values, measurements, topology)
    hole_values = {i: {k: _mm(a, k) for k, a in keyed.items()} for i, keyed in usable_holes.items()}
    features, feat_prov, feat_warn = _assemble_holes(kind, topology, values, hole_values, measurements)
    provenance.update(feat_prov)
    conversions = _conversion_notes(written, values, usable_holes)

    try:
        return PartSpec(
            source_input=source_input, part=part, features=features, finishes=[], provenance=provenance,
            confidence=min(confidences), warnings=unplaced.warnings + conversions + feat_warn,
        )
    except ValidationError as e:
        return Abstain(stage="merge", reason="inconsistent_dimensions",
                       remedy=f"The dimensions do not fit together: {e.errors()[0]['msg']}.",
                       partial=_partial(values))


def _partial(values: Values) -> dict[str, float]:
    """What was recovered, as a value map of numbers only. Unplaced-value warnings are not carried
    on an Abstain: answering it re-runs merge with the same inputs and the PartSpec warns."""
    return {k: v for k, (v, _) in values.items()}


def _mm(a: Annotation, key: str) -> float:
    """The written value in the unit its field expects: R10 on a diameter is 20, Ø10 on a radius is 5.
    Plain arithmetic on a user-written number, so it stays user_written."""
    if a.kind == "radius" and key in DIAMETER_KEYS:
        return a.value_mm * 2
    if a.kind == "diameter" and key in RADIUS_KEYS:
        return a.value_mm / 2
    return a.value_mm


def _conversion_notes(written: Written, values: Values, holes: WrittenHoles) -> list[str]:
    """One warning per converted value that reached the spec, so the user confirms it."""
    used = [(k, a) for k, a in written.items() if values[k][1] == "user_written"]  # not measured or edited
    used += [(k, a) for keyed in holes.values() for k, a in keyed.items()]
    notes = []
    for key, a in used:
        if _mm(a, key) != a.value_mm:
            hole = f"hole {a.hole_index + 1}: " if key in HOLE_KEYS and a.hole_index is not None else ""
            symbol = "R" if a.kind == "radius" else "Ø"
            notes.append(f"{hole}{key.replace('_', ' ')} {_mm(a, key):g} mm taken from written "
                         f"{symbol}{a.value_mm:g}, confirm it")
    return notes


def _values_from_annotations(ann: Annotations, kind: str,
                             unplaced: _Unplaced) -> tuple[Written, WrittenHoles]:
    written: Written = {}
    holes: WrittenHoles = {}
    unknown_linear: list[Annotation] = []
    unknown_diam: list[Annotation] = []
    for a in sorted(ann.items, key=lambda a: -a.confidence):
        if a.linked_to in HOLE_KEYS:
            _first_wins(holes.setdefault(a.hole_index, {}), a.linked_to, a, unplaced)
        elif a.linked_to in FIELD_MAP[kind]:
            _first_wins(written, a.linked_to, a, unplaced)
        elif a.linked_to in SLOT_KEYS:  # a slot dimension is never a part dimension
            unplaced.written(a, "merge does not build slots yet")
        elif a.kind == "radius" or a.linked_to == "corner_radius":  # a radius is never a length or diameter
            if kind == "plate":
                _first_wins(written, "corner_radius", a, unplaced)
            else:
                unplaced.written(a, f"{NAMES[kind]} has no radius dimension")
        elif a.kind == "diameter":  # unknown, or linked to a field this part type does not have
            unknown_diam.append(a)
        else:
            unknown_linear.append(a)
    left_over = _fill_largest_first(LINEAR_ORDER[kind], unknown_linear, written)
    spare_diameters = _fill_largest_first(DIAMETER_ORDER[kind], unknown_diam, written)
    if kind in HOLE_TYPES and spare_diameters and "hole_diameter" not in holes.get(None, {}):
        holes.setdefault(None, {})["hole_diameter"] = spare_diameters.pop(0)  # the largest sizes the holes
    for a in left_over + spare_diameters:
        unplaced.written(a, f"no open dimension is left for it on {NAMES[kind]}")
    return written, holes


def _first_wins(filled: dict[str, Annotation], key: str, a: Annotation, unplaced: _Unplaced) -> None:
    """Annotations arrive most confident first; a later, different value for the same key is reported."""
    if key not in filled:
        filled[key] = a
    elif _mm(filled[key], key) != _mm(a, key):
        unplaced.written(a, f"a {key} of {_mm(filled[key], key):g} mm is used instead")


def _fill_largest_first(keys: list[str], items: list[Annotation], written: Written) -> list[Annotation]:
    """Unlinked values fill the open keys, largest value first. Returns the ones left over."""
    items = sorted(items, key=lambda a: -a.value_mm)
    open_keys = [k for k in keys if k not in written]
    written.update(zip(open_keys, items))
    return items[len(open_keys):]


def _usable_hole_values(kind: str, topology: Topology, holes: WrittenHoles, m: Measurements | None,
                        unplaced: _Unplaced) -> WrittenHoles:
    """The written hole values _assemble_holes will place. Every other one goes to the collector."""
    usable: WrittenHoles = {}
    for index, keyed in holes.items():
        for key, a in keyed.items():
            why = _why_hole_value_unused(kind, topology, holes, m, index, key)
            if why:
                unplaced.written(a, why)
            else:
                usable.setdefault(index, {})[key] = a
    return usable


def _why_hole_value_unused(kind: str, topology: Topology, holes: WrittenHoles, m: Measurements | None,
                           index: int | None, key: str) -> str | None:
    n = len(topology.holes)
    if kind not in HOLE_TYPES:
        return "merge places holes only on plates and L-brackets"
    if m is not None and m.circles_mm:
        return "the holes measured in the photo take precedence"
    if index is None and key != "hole_diameter":
        return "no hole number was read for it"
    if n == 0:
        return "no holes were found"
    if index is not None and index >= n:
        return f"only {n} hole{'s' if n > 1 else ''} found"
    if index is None and all("hole_diameter" in holes.get(i, {}) for i in range(n)):
        return "every hole has its own written diameter"
    partner = {"hole_x": "hole_y", "hole_y": "hole_x"}.get(key)
    if partner and partner not in holes[index]:
        return f"no {partner} was read for it, so the position is estimated"
    return None


def _values_from_measurements(m: Measurements, kind: str, unplaced: _Unplaced) -> Values:
    w, h = m.bbox_mm.width, m.bbox_mm.height
    circles = sorted(m.circles_mm, key=lambda c: -c.diameter)
    v: Values = {}
    if kind == "plate":
        v["width"], v["height"] = (w, "measured"), (h, "measured")
    elif kind == "l_bracket":
        # a top-down photo shows leg_a by width; leg_b stands up out of it, so the user is asked for it
        v["leg_a"], v["width"] = (w, "measured"), (h, "measured")
    elif kind == "spacer":
        v["outer_diameter"] = (max(w, h), "measured")
        v["inner_diameter"] = ((circles[0].diameter if circles else 0.0), "measured")
        for c in circles[1:]:
            unplaced.measured_circle(c, "only the largest circle is used, as the bore")
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
    elif kind == "profile_extrusion":
        for c in circles:
            unplaced.measured_circle(c, "merge places holes only on plates and L-brackets")
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
    """Places holes from measured circles, or from topology plus the usable written hole values."""
    if kind not in HOLE_TYPES:
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
