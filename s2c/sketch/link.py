"""Stage 6: reference positions per view axis, and which positions or circles each value measures."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.optimize import linear_sum_assignment

from s2c.sketch.classify import Classified, DimLine
from s2c.sketch.models import Issue, ViewName
from s2c.sketch.text import TextItem

TOL_FRAC, TOL_MIN = 0.015, 4.0
INVALID = 1e6


@dataclass
class Ref:
    id: str
    view: ViewName
    axis: Literal["a", "b"]
    px: float
    sources: list[str]


@dataclass
class Link:
    text: TextItem
    view: ViewName
    kind: str
    axis: Literal["a", "b"] | None
    refs: tuple[str, str] | None
    circles: list[str] = field(default_factory=list)
    measures: list[str] = field(default_factory=list)
    how: Literal["dimension_line", "leader", "across", "position", "unplaced"] = "unplaced"
    confidence: float = 0.0


def _pos(p, axis: str) -> float:
    return float(p[0]) if axis == "a" else -float(p[1])


def ref_tolerance(refs: list[Ref]) -> float:
    extent = refs[-1].px - refs[0].px if refs else 0.0
    return max(TOL_MIN, TOL_FRAC * extent)


def build_refs(view: ViewName, c: Classified) -> dict[str, list[Ref]]:
    raw: dict[str, list[tuple[float, str]]] = {"a": [], "b": []}
    for p in c.visible + c.hidden + c.centre:
        if p.kind == "line":
            for e in (p.p0, p.p1):
                raw["a"].append((_pos(e, "a"), p.id))
                raw["b"].append((_pos(e, "b"), p.id))
        elif p.kind in ("circle", "arc"):
            cx, cy, r = p.center[0], p.center[1], p.radius
            raw["a"] += [(cx - r, p.id), (cx, p.id), (cx + r, p.id)]
            raw["b"] += [(-cy - r, p.id), (-cy, p.id), (-cy + r, p.id)]
        else:
            xs, ys = p.pts[:, 0], p.pts[:, 1]
            raw["a"] += [(float(xs.min()), p.id), (float(xs.max()), p.id)]
            raw["b"] += [(-float(ys.max()), p.id), (-float(ys.min()), p.id)]
    refs: dict[str, list[Ref]] = {}
    for axis, vals in raw.items():
        vals.sort()
        if not vals:
            refs[axis] = []
            continue
        tol = max(TOL_MIN, TOL_FRAC * (vals[-1][0] - vals[0][0]))
        clusters = [[vals[0]]]
        for v in vals[1:]:
            if v[0] - clusters[-1][0][0] <= tol:
                clusters[-1].append(v)
            else:
                clusters.append([v])
        refs[axis] = [Ref(f"{view}.{axis}{k}", view, axis, float(np.mean([v[0] for v in cl])),
                          sorted({v[1] for v in cl})) for k, cl in enumerate(clusters)]
    return refs


def _nearest(refs: list[Ref], px: float, tol: float) -> Ref | None:
    best = min(refs, key=lambda r: abs(r.px - px), default=None)
    return best if best is not None and abs(best.px - px) <= tol else None


def _centre(t: TextItem) -> np.ndarray:
    x, y, w, h = t.box
    return np.array([x + w / 2, y + h / 2], float)


def _cost(t: TextItem, d: DimLine) -> float:
    c = _centre(t)
    axis = d.p1 - d.p0
    length = float(np.linalg.norm(axis))
    u = axis / (length + 1e-9)
    s = float(np.dot(c - d.p0, u))
    perp = float(abs((c - d.p0)[0] * u[1] - (c - d.p0)[1] * u[0]))
    size = max(t.box[2], t.box[3])
    margin = 0.25 * length + size
    if s < -margin or s > length + margin or perp > 2 * size + 10:
        return INVALID
    return perp + 0.1 * abs(s - length / 2)


def _assign(texts: list[TextItem], dimlines: list[DimLine]) -> dict[str, DimLine]:
    if not texts or not dimlines:
        return {}
    cost = np.array([[_cost(t, d) for d in dimlines] for t in texts])
    rows, cols = linear_sum_assignment(cost)
    return {texts[r].id: dimlines[c] for r, c in zip(rows, cols) if cost[r, c] < INVALID}


def _linear(view, t, d: DimLine, c: Classified, refs, tol) -> Link | None:
    axis = "a" if d.horizontal else "b"
    found = []
    for k, end in enumerate((d.p0, d.p1)):
        ext = c.extensions.get(f"{d.id}:{k}")
        if ext is not None:
            px = (_pos(ext.p0, axis) + _pos(ext.p1, axis)) / 2
        else:
            tip = min((a.tip for a in d.arrows), key=lambda q: np.linalg.norm(q - end), default=end)
            px = _pos(tip if np.linalg.norm(tip - end) < 20 else end, axis)
        found.append(_nearest(refs[axis], px, 2 * tol[axis]))
    if None in found or found[0].id == found[1].id:
        return None
    a, b = sorted(found, key=lambda r: r.px)
    return Link(t, view, t.parsed.kind, axis, (a.id, b.id), [], [a.id, b.id], "dimension_line",
                t.confidence)


def _by_position(view, t, c: Classified, refs, tol) -> Link | None:
    """No dimension line: the nearest parallel visible edge on the side where the text sits."""
    if not refs["a"] or not refs["b"]:
        return None
    ca, cb = _pos(_centre(t), "a"), _pos(_centre(t), "b")
    a0, a1, b0, b1 = refs["a"][0].px, refs["a"][-1].px, refs["b"][0].px, refs["b"][-1].px
    reach = 0.25 * max(a1 - a0, b1 - b0)
    lines = [p for p in c.visible if p.kind == "line"]
    if a0 <= ca <= a1 and (b1 < cb <= b1 + reach or b0 - reach <= cb < b0):
        axis, along, across = "a", ca, cb
        cands = [p for p in lines if abs(p.p1[1] - p.p0[1]) < 0.2 * abs(p.p1[0] - p.p0[0])
                 and min(p.p0[0], p.p1[0]) <= along <= max(p.p0[0], p.p1[0])]
        key = (lambda p: abs(-p.p0[1] - across))
    elif b0 <= cb <= b1 and (a1 < ca <= a1 + reach or a0 - reach <= ca < a0):
        axis, along, across = "b", cb, ca
        cands = [p for p in lines if abs(p.p1[0] - p.p0[0]) < 0.2 * abs(p.p1[1] - p.p0[1])
                 and min(-p.p0[1], -p.p1[1]) <= along <= max(-p.p0[1], -p.p1[1])]
        key = (lambda p: abs(p.p0[0] - across))
    else:
        return None
    if not cands:
        return None
    edge = min(cands, key=key)
    r0 = _nearest(refs[axis], _pos(edge.p0, axis), 2 * tol[axis])
    r1 = _nearest(refs[axis], _pos(edge.p1, axis), 2 * tol[axis])
    if r0 is None or r1 is None or r0.id == r1.id:
        return None
    a, b = sorted((r0, r1), key=lambda r: r.px)
    return Link(t, view, t.parsed.kind, axis, (a.id, b.id), [], [a.id, b.id], "position",
                0.7 * t.confidence)


def _round(view, t, c: Classified, used: set[str], stroke) -> Link | None:
    circles = [p for p in c.visible + c.hidden if p.kind in ("circle", "arc")]
    if not circles:
        return None
    centre = _centre(t)
    reach = 1.5 * max(t.box[2], t.box[3]) + 20
    lead = min(c.leaders, key=lambda L: np.linalg.norm(L.tail - centre), default=None)
    how, target = None, None
    if lead is not None and np.linalg.norm(lead.tail - centre) <= reach:
        target = min(circles, key=lambda p: abs(np.linalg.norm(lead.tip - np.array(p.center)) - p.radius))
        if abs(np.linalg.norm(lead.tip - np.array(target.center)) - target.radius) > 2 * stroke + 3 + 0.1 * target.radius:
            target = None
        how = "leader"
    if target is None:
        for d in c.dimlines:
            if d.id in used:
                continue
            mid, length = (d.p0 + d.p1) / 2, float(np.linalg.norm(d.p1 - d.p0))
            for p in circles:
                if np.linalg.norm(mid - np.array(p.center)) <= 0.3 * p.radius \
                        and abs(length - 2 * p.radius) <= 0.15 * 2 * p.radius and _cost(t, d) < INVALID:
                    target, how = p, "across"
                    used.add(d.id)
                    break
            if target is not None:
                break
    if target is None:
        return None
    group = [target.id]
    if t.parsed.count > 1:
        group = [p.id for p in circles if p.kind == "circle"
                 and abs(p.radius - target.radius) <= 0.08 * target.radius]
    return Link(t, view, t.parsed.kind, None, None, group, list(group), how, t.confidence)


def link_view(view: ViewName, c: Classified, texts: list[TextItem], stroke_px: float):
    refs = build_refs(view, c)
    tol = {axis: ref_tolerance(rs) for axis, rs in refs.items()}
    dims = [t for t in texts if t.role == "dimension" and t.parsed is not None]
    linear = [t for t in dims if t.parsed.kind == "linear"]
    assigned = _assign(linear, c.dimlines)
    used = {d.id for d in assigned.values()}
    links: list[Link] = []
    issues: list[Issue] = []
    for t in dims:
        link = None
        if t.parsed.kind == "linear":
            d = assigned.get(t.id)
            link = _linear(view, t, d, c, refs, tol) if d else None
            if link is None:
                link = _by_position(view, t, c, refs, tol)
                if link is not None:
                    issues.append(Issue(severity="amber", kind="unplaced", targets=[t.id],
                                        message=f"{t.readings[0].text} linked by its position only; check it."))
        elif t.parsed.kind in ("diameter", "radius", "thread"):
            link = _round(view, t, c, used, stroke_px)
        if link is None:
            link = Link(t, view, t.parsed.kind, None, None, how="unplaced", confidence=t.confidence)
            issues.append(Issue(severity="amber", kind="unplaced", targets=[t.id],
                                message=f"Could not tell what {t.readings[0].text} measures."))
        links.append(link)
    return refs, links, issues
