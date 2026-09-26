"""Fuse per-image observations into a MultiViewSpec. Spec sections 4 and 5.
Envelope trust: typed > written > measured. Nothing else passes the gate."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from s2c.multiview import spec as S
from s2c.multiview.ocr import Linked, Reading
from s2c.multiview.outline import PixelOutline, to_face_mm
from s2c.multiview.raster import iou, outline_mask

CLEARANCE_CLASSES = {  # ISO 273 clearance holes for M2, M2.5, M3, M4, M5, M6, M8, M10
    "fine": (2.2, 2.7, 3.2, 4.3, 5.3, 6.4, 8.4, 10.5),
    "medium": (2.4, 2.9, 3.4, 4.5, 5.5, 6.6, 9.0, 11.0),
    "coarse": (2.6, 3.1, 3.6, 4.8, 5.8, 7.0, 10.0, 12.0),
}
THICKNESS_MM = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
SNAPPABLE = frozenset({"scaled", "inferred", "estimated"})
DISAGREE = 0.05
_FEATURE_PATH = re.compile(r"features\[(\d+)\]\.(\w+)")


@dataclass
class Observation:
    face: str
    kind: str                                   # sketch | photo | drawing
    outline: PixelOutline
    values: list[Linked] = field(default_factory=list)
    mm_per_px: float | None = None              # reference-object scale, photos only
    blind: dict[int, bool] = field(default_factory=dict)             # circle index -> blind, Solaria only (rule 2)
    depth_estimates: dict[int, float] = field(default_factory=dict)  # unused; kept for Solaria's cleanup pop
    depth_ratio: dict[int, float] = field(default_factory=dict)      # circle index -> blind depth / axis length, Solaria
    depth_from_image: set[int] = field(default_factory=set)          # circles whose blind flag came from Solaria
    confidence: float = 0.9


def attach_label(obs: Observation, label) -> None:
    """No-op: the vision model's hole flags and depth guesses never reach geometry (rule 2). Only Solaria
    (depth.py apply_depth) may mark a hole blind or give it a depth."""


def _value(r: Reading) -> float:
    return r.value_mm * 2 if r.kind == "radius" else r.value_mm


def _differs(a: float, b: float) -> bool:
    return abs(a - b) / max(a, b) > DISAGREE


# ---- envelope ---------------------------------------------------------------

@dataclass
class _Candidate:
    value: float
    prov: str
    confidence: float
    face: str


def _envelope_candidates(observations: list[Observation]) -> dict[str, list[_Candidate]]:
    cands: dict[str, list[_Candidate]] = {"x": [], "y": [], "z": []}
    for o in observations:
        a_axis, b_axis, _ = S.FACE_AXES[o.face]
        _, _, w, h = o.outline.bbox
        for which, axis, px in (("a", a_axis, w - 1), ("b", b_axis, h - 1)):
            readings = [lv.reading for lv in o.values
                        if (lv.axis == which and lv.reading.kind == "linear") or lv.axis == "ab"]
            if readings:
                r = max(readings, key=_value)
                cands[axis].append(_Candidate(_value(r), "user_written", r.confidence, o.face))
            if o.mm_per_px:
                cands[axis].append(_Candidate(px * o.mm_per_px, "measured", o.confidence, o.face))
    return cands


def fuse_envelope(observations: list[Observation], user_values: dict | None = None):
    user_values = user_values or {}
    cands = _envelope_candidates(observations)
    values: dict[str, float] = {}
    prov: dict[str, str] = {}
    warnings: list[str] = []
    for axis in "xyz":
        key, name = f"envelope.{axis}_mm", S.AXIS_NAMES[axis]
        if key in user_values:
            values[axis], prov[key] = float(user_values[key]), "user_edited"
            continue
        written = [c for c in cands[axis] if c.prov == "user_written"]
        measured = [c for c in cands[axis] if c.prov == "measured"]
        if written:
            best = max(written, key=lambda c: c.confidence)
            for c in written:
                if c is not best and _differs(c.value, best.value):
                    warnings.append(f"{name}: {c.face} says {c.value:g} mm, {best.face} says {best.value:g} mm; "
                                    f"using {best.value:g}")
            for c in measured:
                if _differs(c.value, best.value):
                    warnings.append(f"{name}: written {best.value:g} mm, measured {c.value:.1f} mm; "
                                    "using the written value")
            values[axis], prov[key] = best.value, "user_written"
        elif measured:
            best = max(measured, key=lambda c: c.confidence)
            values[axis], prov[key] = round(best.value, 2), "measured"
    missing = [a for a in "xyz" if a not in values]
    if missing:
        first = missing[0]
        return S.MvAbstain(
            stage="dimensions", reason=f"missing_{first}", remedy=f"Enter the {S.AXIS_NAMES[first]} in mm.",
            partial={"known": {f"envelope.{a}_mm": v for a, v in values.items()},
                     "missing": [f"envelope.{a}_mm" for a in missing],
                     "suggested": _suggest(observations, values, missing)})
    return S.Envelope(x_mm=values["x"], y_mm=values["y"], z_mm=values["z"]), prov, warnings


def _suggest(observations, values, missing) -> dict[str, float]:
    """A scaled pre-fill for each missing axis, from a view that shows it next to a known axis."""
    out = {}
    for axis in missing:
        for o in observations:
            a_axis, b_axis, _ = S.FACE_AXES[o.face]
            _, _, w, h = o.outline.bbox
            if a_axis == axis and b_axis in values:
                out[f"envelope.{axis}_mm"] = round(values[b_axis] * (w - 1) / (h - 1) * 2) / 2
                break
            if b_axis == axis and a_axis in values:
                out[f"envelope.{axis}_mm"] = round(values[a_axis] * (h - 1) / (w - 1) * 2) / 2
                break
    return out


# ---- outlines -----------------------------------------------------------------

def _scales(o: Observation, env: S.Envelope) -> tuple[float, float]:
    a_len, b_len = S.face_size(o.face, env)
    _, _, w, h = o.outline.bbox
    return a_len / max(w - 1, 1), b_len / max(h - 1, 1)


def _clamp(pts, a_len, b_len):
    return [(min(max(a, 0.0), a_len), min(max(b, 0.0), b_len)) for a, b in pts]


def _outline_prov(o: Observation) -> str:
    return "measured" if o.mm_per_px else "scaled"


def observed_outline(o: Observation, env: S.Envelope) -> S.Outline:
    sa, sb = _scales(o, env)
    a_len, b_len = S.face_size(o.face, env)
    bbox = o.outline.bbox

    def mm(points):
        return S.to_canonical(o.face, _clamp(to_face_mm(points, bbox, sa, sb), a_len, b_len), env)

    source = "observed" if o.face in S.CANONICAL_FACES else "mirrored"
    return S.Outline(outer=mm(o.outline.outer), inner=[mm(loop) for loop in o.outline.inner], source=source,
                     confidence=o.confidence)


def canonical_outlines(observations: list[Observation], env: S.Envelope):
    """Best observed or mirrored outline per canonical face, with its provenance, and warnings."""
    out, warnings = {}, []
    for o in observations:
        if o.kind == "photo":
            a_len, b_len = S.face_size(o.face, env)
            _, _, w, h = o.outline.bbox
            if _differs((w - 1) / (h - 1), a_len / b_len):
                warnings.append(f"{o.face}: photo is not square-on, retake it facing the part")
    for face in S.CANONICAL_FACES:
        group = [(o, observed_outline(o, env)) for o in observations if S.CANONICAL_OF[o.face] == face]
        if not group:
            continue
        best_o, best = max(group, key=lambda t: (t[0].confidence, t[1].source == "observed"))
        a_len, b_len = S.face_size(face, env)
        ref = outline_mask(best.outer, best.inner, a_len, b_len)
        for o, ol in group:
            if o is not best_o and iou(outline_mask(ol.outer, ol.inner, a_len, b_len), ref) < 0.9:
                warnings.append(f"{o.face} and {best_o.face} outlines disagree; using {best_o.face}")
        out[face] = (best, _outline_prov(best_o))
    return out, warnings


# ---- features -------------------------------------------------------------------

def _duplicate_through(feats: list[dict], face: str, a: float, b: float, env: S.Envelope) -> bool:
    here = S.to_global(face, a, b, env)
    for f in feats:
        if f["depth_mm"] is None and S.CANONICAL_OF[f["face"]] == S.CANONICAL_OF[face]:
            there = S.to_global(f["face"], f["a_mm"], f["b_mm"], env)
            if all(abs(here[k] - there[k]) <= 1.0 for k in here):
                return True
    return False


def features_from(observations: list[Observation], env: S.Envelope):
    """Every circle becomes a hole on its own face; a through hole seen from both sides is kept once."""
    feats: list[dict] = []
    prov: dict[str, str] = {}
    for o in observations:
        sa, sb = _scales(o, env)
        written = {lv.hole_index: _value(lv.reading) for lv in o.values if lv.hole_index is not None}
        axis_len = env.length(S.FACE_AXES[o.face][2])
        for i, c in enumerate(o.outline.circles):
            (a, b), = to_face_mm(np.array([[c.cx, c.cy]]), o.outline.bbox, sa, sb)
            if i in written:
                d, d_prov = written[i], "user_written"
            elif o.mm_per_px:
                d, d_prov = c.d * o.mm_per_px, "measured"
            else:
                d, d_prov = c.d * (sa + sb) / 2, "scaled"
            depth, depth_prov = None, None
            if o.blind.get(i):
                if i in o.depth_ratio:
                    depth, depth_prov = o.depth_ratio[i] * axis_len, "estimated"
                else:
                    depth, depth_prov = axis_len / 2, "default"
            elif _duplicate_through(feats, o.face, a, b, env):
                continue
            k = len(feats)
            feats.append({"type": "hole", "face": o.face, "a_mm": a, "b_mm": b, "diameter_mm": float(d),
                          "depth_mm": depth})
            pos = _outline_prov(o)
            prov.update({f"features[{k}].a_mm": pos, f"features[{k}].b_mm": pos, f"features[{k}].diameter_mm": d_prov})
            if depth is not None:
                prov[f"features[{k}].depth_mm"] = depth_prov
    return feats, prov


# ---- snapping -----------------------------------------------------------------

def _grid(v: float) -> float:
    return round(v * 2) / 2


def snap_diameter(d: float, clearance: str = "medium") -> float:
    best = min(CLEARANCE_CLASSES[clearance], key=lambda c: abs(c - d))
    return best if abs(best - d) <= 0.4 else _grid(d)


def snap_coord(v: float, length: float) -> float:
    """Envelope edges exactly, thin walls to standard thicknesses, everything else to 0.5 mm."""
    if v <= 0.5:
        return 0.0
    if length - v <= 0.5:
        return float(length)
    wall = min(v, length - v)
    if wall < 12:
        t = min(THICKNESS_MM, key=lambda t: abs(t - wall))
        if abs(t - wall) <= 0.3:
            return t if v < length / 2 else float(length - t)
    return min(max(_grid(v), 0.0), float(length))


EDGE_SLOPE = 0.035  # tan(2 deg): how far off-axis a "straight" outline edge may drift
EDGE_MIN_MM = 1.0    # shorter edges are curve segments, not sketched straight lines
MOVE_LIMIT = 0.02    # a level does not move more than this fraction of its axis


def _level_targets(levels: set, axis_len: float, all_coords) -> dict:
    """snap_coord per level, kept only if it moves, the move is small, and it lands neither on another
    level's snapped or original value nor on any other vertex's coordinate on this axis (spec 4.5)."""
    candidates = {v: snap_coord(v, axis_len) for v in levels}
    targets = {}
    for v, new in candidates.items():
        if abs(new - v) <= 1e-9 or abs(new - v) > MOVE_LIMIT * axis_len:
            targets[v] = v
            continue
        collides = any(w != v and (abs(new - w) <= 1e-6 or abs(new - candidates[w]) <= 1e-6) for w in levels)
        if not collides:
            collides = any(abs(c - v) > 1e-6 and abs(new - c) <= 1e-6 for c in all_coords)
        targets[v] = v if collides else new
    return targets


def _snap_outline(points: list, a_len: float, b_len: float) -> list:
    """Snap only straight axis-parallel edges (spec 4.5); curves and sloped or thin edges pass through."""
    n = len(points)
    if n < 3:
        return list(points)
    on_horiz, on_vert = [False] * n, [False] * n
    for i in range(n):
        a1, b1 = points[i]
        a2, b2 = points[(i + 1) % n]
        da, db = a2 - a1, b2 - b1
        length = (da * da + db * db) ** 0.5
        if length < EDGE_MIN_MM:
            continue
        if abs(db) <= EDGE_SLOPE * abs(da):
            on_horiz[i] = on_horiz[(i + 1) % n] = True
        elif abs(da) <= EDGE_SLOPE * abs(db):
            on_vert[i] = on_vert[(i + 1) % n] = True
    all_a = [p[0] for p in points]
    all_b = [p[1] for p in points]
    a_targets = _level_targets({round(points[i][0], 6) for i in range(n) if on_vert[i]}, a_len, all_a)
    b_targets = _level_targets({round(points[i][1], 6) for i in range(n) if on_horiz[i]}, b_len, all_b)
    out = []
    for i, (a, b) in enumerate(points):
        na = a_targets.get(round(a, 6), a) if on_vert[i] else a
        nb = b_targets.get(round(b, 6), b) if on_horiz[i] else b
        out.append((na, nb))
    return out


def snap(data: dict, clearance: str = "medium") -> None:
    """Snap scaled, inferred and estimated values of a spec dict in place (spec 4.5)."""
    prov, snapped = data["provenance"], data.setdefault("snapped", [])
    for k, f in enumerate(data["features"]):
        for name in ("a_mm", "b_mm", "diameter_mm", "depth_mm", "width_mm", "length_mm"):
            path = f"features[{k}].{name}"
            if f.get(name) is None or prov.get(path) not in SNAPPABLE:
                continue
            new = snap_diameter(f[name], clearance) if name == "diameter_mm" else _grid(f[name])
            if new > 0 and abs(new - f[name]) > 1e-9:
                f[name] = new
                snapped.append(path)
    lengths = {"x": data["envelope"]["x_mm"], "y": data["envelope"]["y_mm"], "z": data["envelope"]["z_mm"]}
    for face in S.CANONICAL_FACES:
        path = f"views.{face}.outer"
        if prov.get(path) not in SNAPPABLE:
            continue
        a_axis, b_axis, _ = S.FACE_AXES[face]
        outline = data["views"][face]
        orig = [tuple(p) for p in outline["outer"]]
        new = _snap_outline(outline["outer"], lengths[a_axis], lengths[b_axis])
        if len(set(new)) >= len(set(orig)) and new != orig:
            outline["outer"] = new
            snapped.append(path)


def assemble(env: S.Envelope, env_prov: dict, outlines: dict, feats: list[dict], feat_prov: dict,
             warnings: list[str], user_values: dict | None = None, accepted=(), snap_values: bool = True,
             clearance: str = "medium") -> S.MultiViewSpec:
    """outlines: canonical face -> (Outline, provenance). Applies the user's edits, then snapping."""
    data = {
        "envelope": env.model_dump(),
        "views": {face: ol.model_dump() for face, (ol, _) in outlines.items()},
        "features": [dict(f) for f in feats],
        "finishes": [],
        "provenance": {**env_prov, **{f"views.{face}.outer": p for face, (_, p) in outlines.items()}, **feat_prov},
        "warnings": list(dict.fromkeys(warnings)),
        "confidence": round(min(ol.confidence for ol, _ in outlines.values()), 3),
    }
    for face in accepted:
        if f"views.{face}.outer" in data["provenance"]:
            data["provenance"][f"views.{face}.outer"] = "user_edited"
    for path, value in (user_values or {}).items():
        m = _FEATURE_PATH.fullmatch(path)
        if m and int(m.group(1)) < len(data["features"]):
            data["features"][int(m.group(1))][m.group(2)] = float(value)
            data["provenance"][path] = "user_edited"
    if snap_values:
        snap(data, clearance)
    return S.MultiViewSpec.model_validate(data)
