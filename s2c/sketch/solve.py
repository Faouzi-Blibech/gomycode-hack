"""Stages 7 and 8 (ranks 1, 2 and 5): one least-squares system over every view's reference positions.
Written values are exact, shared views agree, the drawing's proportions fill the rest."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np

from s2c.sketch.link import Link, Ref
from s2c.sketch.models import Badge, Evidence, Issue, Size, ViewName
from s2c.sketch.vectorize import Prim

GROUPS: dict[str, list[tuple[str, str, int]]] = {
    "x": [("front", "a", 1), ("top", "a", 1), ("bottom", "a", 1), ("back", "a", -1)],
    "y": [("front", "b", 1), ("right", "b", 1), ("left", "b", 1), ("back", "b", 1)],
    "z": [("right", "a", 1), ("top", "b", 1), ("left", "a", -1), ("bottom", "b", -1)],
}
HARD, SOFT = 1e3, 1.0
MATCH_TOL = 0.03            # of the view extent, for cross-view matching
DECIMAL_RATIO = 3.0
FALLBACK_SCALE = 0.25       # mm per px when the sheet has no value at all


@dataclass
class DimResult:
    value: float | None
    badge: Badge
    evidence: Evidence
    implied: float | None = None
    candidates: list[float] = field(default_factory=list)


@dataclass
class PredGap:
    view: ViewName
    axis: str
    refs: tuple[str, str]
    value: float


@dataclass
class Solved:
    mm: dict[str, float]
    scale: dict[tuple[str, str], float]
    dims: dict[str, DimResult]
    circle_d: dict[str, tuple[float, Badge]]
    envelope: dict[str, Size]
    gaps: list[PredGap]
    issues: list[Issue]


@dataclass
class _Row:
    coef: dict[str, float]
    rhs: float
    kind: str          # anchor | match | dim | circle | soft
    tag: str = ""      # text id for dim and circle rows


class _System:
    def __init__(self, ids: list[str]):
        self.ix = {r: k for k, r in enumerate(ids)}
        self.rows: list[_Row] = []

    def add(self, coef, rhs, kind, tag=""):
        self.rows.append(_Row(coef, rhs, kind, tag))

    def matrix(self, rows: list[_Row]) -> tuple[np.ndarray, np.ndarray]:
        A = np.zeros((len(rows), len(self.ix)))
        b = np.zeros(len(rows))
        for i, row in enumerate(rows):
            for rid, c in row.coef.items():
                A[i, self.ix[rid]] += c
            b[i] = row.rhs
        return A, b

    def vector(self, coef: dict[str, float]) -> np.ndarray:
        v = np.zeros(len(self.ix))
        for rid, c in coef.items():
            v[self.ix[rid]] += c
        return v


def _lstsq(A, b):
    if A.size == 0:
        return np.zeros(A.shape[1])
    return np.linalg.lstsq(A, b, rcond=None)[0]


def _in_rowspace(A: np.ndarray, v: np.ndarray) -> bool:
    if A.size == 0:
        return False
    x = np.linalg.lstsq(A.T, v, rcond=None)[0]
    return float(np.linalg.norm(A.T @ x - v)) <= 1e-6 * max(1.0, float(np.linalg.norm(v)))


def _norm(rs: list[Ref], orient: int) -> list[float]:
    lo, hi = rs[0].px, rs[-1].px
    span = max(hi - lo, 1e-9)
    return [(r.px - lo) / span if orient > 0 else (hi - r.px) / span for r in rs]


def _match(rs1, o1, rs2, o2) -> list[tuple[Ref, Ref]]:
    n1 = sorted(zip(_norm(rs1, o1), rs1), key=lambda t: t[0])
    n2 = sorted(zip(_norm(rs2, o2), rs2), key=lambda t: t[0])
    i = j = 0
    out = []
    while i < len(n1) and j < len(n2):
        if abs(n1[i][0] - n2[j][0]) <= MATCH_TOL:
            out.append((n1[i][1], n2[j][1]))
            i, j = i + 1, j + 1
        elif n1[i][0] < n2[j][0]:
            i += 1
        else:
            j += 1
    return out


def _distance(rs: list[Ref], r: Ref, orient: int, sign: float) -> dict[str, float]:
    """Coefficients of the distance from the axis origin face: u, or u_max - u when mirrored."""
    if orient > 0:
        return {r.id: sign}
    coef = {rs[-1].id: sign}
    coef[r.id] = coef.get(r.id, 0.0) - sign
    return coef


def _merge(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0.0) + v
    return out


def _circle_refs(rs: list[Ref], circle_id: str, lo_px: float, hi_px: float):
    mine = [r for r in rs if circle_id in r.sources]
    if len(mine) < 2:
        return None
    lo = min(mine, key=lambda r: abs(r.px - lo_px))
    hi = min(mine, key=lambda r: abs(r.px - hi_px))
    return (lo, hi) if lo.id != hi.id else None


def _circle_rows(link: Link, value: float, views, circles) -> list[tuple[dict[str, float], float]]:
    rows = []
    for cid in link.circles:
        c = circles.get(cid)
        if c is None or link.view not in views:
            continue
        cx, cy, r = c.center[0], c.center[1], c.radius
        span = value if link.kind != "radius" else 2 * value
        for axis, lo, hi in (("a", cx - r, cx + r), ("b", -cy - r, -cy + r)):
            pair = _circle_refs(views[link.view].get(axis, []), cid, lo, hi)
            if pair:
                rows.append(({pair[1].id: 1.0, pair[0].id: -1.0}, span))
    return rows


def _scales(views, links, issues) -> dict[tuple[str, str], float]:
    ref = {r.id: r for axes in views.values() for rs in axes.values() for r in rs}
    ratios: dict[tuple[str, str], list[tuple[float, Link]]] = {}
    for L in links:
        if L.refs and L.kind == "linear" and L.text.badge == "written":
            dpx = abs(ref[L.refs[1]].px - ref[L.refs[0]].px)
            if dpx > 0:
                ratios.setdefault((L.view, L.axis), []).append((L.text.parsed.value / dpx, L))
    scale = {k: float(np.median([r for r, _ in v])) for k, v in ratios.items()}
    for vals in ratios.values():  # decimal trap (spec 7.5)
        for k, (r, L) in enumerate(vals):
            others = [o for m, (o, _) in enumerate(vals) if m != k]
            if not others:
                continue
            med = float(np.median(others))
            if r / med > DECIMAL_RATIO or med / r > DECIMAL_RATIO:
                dpx = L.text.parsed.value / r
                alts = [L.text.parsed.value * f for f in (10, 0.1, 0.01, 100)]
                best = min(alts, key=lambda a: abs(a - med * dpx))
                issues.append(Issue(severity="amber", kind="decimal", targets=[L.text.id],
                                    message=f"{L.text.readings[0].text} looks off scale; did you mean {best:g}?"))
    everything = [r for v in ratios.values() for r, _ in v]
    overall = float(np.median(everything)) if everything else FALLBACK_SCALE
    if not everything:
        issues.append(Issue(severity="red", kind="predicted", targets=[],
                            message="No values found. Write at least the overall width, height and depth."))
    for v, axes in views.items():
        for ax in axes:
            if (v, ax) in scale:
                continue
            partner = [scale[(pv, pa)] for g in GROUPS.values() if (v, ax) in [(m[0], m[1]) for m in g]
                       for pv, pa, _ in g if (pv, pa) in scale]
            scale[(v, ax)] = partner[0] if partner else overall
    return scale


def solve(views: dict[ViewName, dict[str, list[Ref]]], links: list[Link],
          circles: dict[str, Prim]) -> Solved:
    issues: list[Issue] = []
    ids = [r.id for axes in views.values() for rs in axes.values() for r in rs]
    system = _System(ids)
    scale = _scales(views, links, issues)

    for v, axes in views.items():
        for ax, rs in axes.items():
            if rs:
                system.add({rs[0].id: 1.0}, 0.0, "anchor")
            for r0, r1 in pairwise(rs):
                system.add({r1.id: 1.0, r0.id: -1.0}, scale[(v, ax)] * (r1.px - r0.px), "soft")
    for members in GROUPS.values():
        present = [(v, ax, o) for v, ax, o in members if views.get(v, {}).get(ax)]
        for v2, ax2, o2 in present[1:]:
            v1, ax1, o1 = present[0]
            rs1, rs2 = views[v1][ax1], views[v2][ax2]
            for r1, r2 in _match(rs1, o1, rs2, o2):
                system.add(_merge(_distance(rs1, r1, o1, 1.0), _distance(rs2, r2, o2, -1.0)), 0.0, "match")

    dim_rows: dict[str, list[tuple[dict[str, float], float]]] = {}
    for L in links:
        if L.text.parsed is None:
            continue
        value = L.text.parsed.value
        if L.refs and L.kind == "linear":
            dim_rows[L.text.id] = [({L.refs[1]: 1.0, L.refs[0]: -1.0}, value)]
        elif L.circles:
            dim_rows[L.text.id] = _circle_rows(L, value, views, circles)
    by_id = {L.text.id: L for L in links}
    written = [tid for tid in dim_rows if by_id[tid].text.badge == "written" and dim_rows[tid]]
    uncertain = [tid for tid in dim_rows if by_id[tid].text.badge == "uncertain" and dim_rows[tid]]

    def rows_for(tids, extra=None):
        rows = [r for r in system.rows if r.kind in ("anchor", "match")]
        for tid in tids:
            rows += [_Row(c, (extra or {}).get(tid, rhs), "dim", tid) for c, rhs in dim_rows[tid]]
        return rows

    # conflicts on the hard rows alone (leave-one-out, ties to the value most off the drawing)
    def worst(tids) -> float:
        A, b = system.matrix(rows_for(tids))
        x = _lstsq(A, b)
        res = np.abs(A @ x - b)
        tol = np.array([max(0.05, 0.005 * abs(rhs)) for rhs in b])
        return float(np.max(res - tol)) if len(b) else -1.0

    consistent = list(written)
    conflicts: list[str] = []
    ref = {r.id: r for axes in views.values() for rs in axes.values() for r in rs}

    def off_drawing(tid) -> float:
        L = by_id[tid]
        if not L.refs:
            return 0.0
        dpx = abs(ref[L.refs[1]].px - ref[L.refs[0]].px)
        drawn = scale[(L.view, L.axis)] * dpx
        return abs(L.text.parsed.value - drawn) / max(L.text.parsed.value, 1e-9)

    while consistent and worst(consistent) > 0:
        # every removal that leaves a consistent set scores 0, so the tie-break decides among them
        trials = [(max(0.0, round(worst([t for t in consistent if t != tid]), 6)), -off_drawing(tid), tid)
                  for tid in consistent]
        _, _, drop = min(trials)
        consistent.remove(drop)
        conflicts.append(drop)

    # implied sizes for uncertain values (spec 8.2), then the final solve
    def full(tids, extra=None):
        hard = rows_for(tids, extra)
        soft = [r for r in system.rows if r.kind == "soft"]
        A_h, b_h = system.matrix(hard)
        A_s, b_s = system.matrix(soft)
        return _lstsq(np.vstack([HARD * A_h, SOFT * A_s]), np.concatenate([HARD * b_h, SOFT * b_s]))

    first = full(consistent)
    proposals: dict[str, float] = {}
    implied: dict[str, float] = {}
    for tid in uncertain:
        coef, _ = dim_rows[tid][0]
        imp = float(system.vector(coef) @ first)
        implied[tid] = imp
        cands = list(by_id[tid].text.candidates) or [by_id[tid].text.parsed.value]
        cands += [c * f for c in list(cands) for f in (10, 0.1, 0.01)]
        proposals[tid] = min(cands, key=lambda c: abs(c - imp))
    sol = full(consistent + list(proposals), proposals)
    mm = {rid: float(sol[k]) for rid, k in system.ix.items()}

    A_all, _ = system.matrix(rows_for(consistent))
    A_own, _ = system.matrix([r for r in rows_for(consistent) if r.kind != "match"])

    def badge_of(coef) -> tuple[Badge, Evidence]:
        vec = system.vector(coef)
        if _in_rowspace(A_own, vec):
            return "derived", "arithmetic"
        if _in_rowspace(A_all, vec):
            return "derived", "cross_view"
        return "predicted", "proportion"

    dims: dict[str, DimResult] = {}
    for L in links:
        tid = L.text.id
        value = L.text.parsed.value if L.text.parsed else None
        if tid in conflicts:
            dims[tid] = DimResult(value, "conflict", "reader", candidates=list(L.text.candidates))
            issues.append(Issue(severity="red", kind="conflict", targets=[tid],
                                message=f"{L.text.readings[0].text} contradicts other written values."))
        elif tid in proposals:
            dims[tid] = DimResult(proposals[tid], "uncertain", "reader", implied[tid],
                                  list(L.text.candidates))
            issues.append(Issue(severity="amber", kind="uncertain", targets=[tid],
                                message=f"Readers disagree on {L.text.readings[0].text}; "
                                        f"the drawing suggests {proposals[tid]:g}."))
        else:
            dims[tid] = DimResult(value, L.text.badge or "uncertain", "reader",
                                  candidates=list(L.text.candidates))

    circle_d: dict[str, tuple[float, Badge]] = {}
    for L in links:
        for cid in L.circles:
            if L.text.id in consistent:
                d = L.text.parsed.value * (2 if L.kind == "radius" else 1)
                circle_d[cid] = (d, "written")
    for cid, c in circles.items():
        if cid in circle_d:
            continue
        view = next((v for v in views if cid.startswith(f"{v}-")), None)
        pair = _circle_refs(views.get(view, {}).get("a", []), cid, c.center[0] - c.radius,
                            c.center[0] + c.radius) if view else None
        if pair:
            b, _ = badge_of({pair[1].id: 1.0, pair[0].id: -1.0})
            circle_d[cid] = (mm[pair[1].id] - mm[pair[0].id], b)

    gaps: list[PredGap] = []
    for v, axes in views.items():
        for ax, rs in axes.items():
            for r0, r1 in pairwise(rs):
                b, _ = badge_of({r1.id: 1.0, r0.id: -1.0})
                if b == "predicted":
                    gaps.append(PredGap(v, ax, (r0.id, r1.id), mm[r1.id] - mm[r0.id]))

    envelope: dict[str, Size] = {}
    direct = {tuple(sorted(by_id[t].refs)) for t in consistent if by_id[t].refs}
    for g, members in GROUPS.items():
        present = [(v, ax) for v, ax, _ in members if views.get(v, {}).get(ax)]
        if not present:
            issues.append(Issue(severity="red", kind="predicted", targets=[],
                                message=f"No view shows the {g.upper()} size; draw it or type it."))
            continue
        v, ax = present[0]
        lo, hi = views[v][ax][0], views[v][ax][-1]
        value = mm[hi.id] - mm[lo.id]
        if tuple(sorted((lo.id, hi.id))) in direct:
            envelope[g] = Size(value=value, badge="written", evidence="reader")
        else:
            b, e = badge_of({hi.id: 1.0, lo.id: -1.0})
            envelope[g] = Size(value=value, badge=b, evidence=e)
            if b == "predicted":
                issues.append(Issue(severity="amber", kind="predicted", targets=[g],
                                    message=f"The overall {g.upper()} size is predicted from the drawing; confirm it."))
    return Solved(mm, scale, dims, circle_d, envelope, gaps, issues)


def px_to_mm(solved: Solved, refs: list[Ref], view: str, axis: str, px: float) -> float:
    if not refs:
        return 0.0
    xs = np.array([r.px for r in refs])
    ys = np.array([solved.mm[r.id] for r in refs])
    s = solved.scale.get((view, axis), FALLBACK_SCALE)
    if px < xs[0]:
        return float(ys[0] - s * (xs[0] - px))
    if px > xs[-1]:
        return float(ys[-1] + s * (px - xs[-1]))
    return float(np.interp(px, xs, ys))
