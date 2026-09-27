"""Stage 9: holes from a circle in one view and a pair of lines along its axis in another."""
from __future__ import annotations

import numpy as np

from s2c.sketch.classify import Classified
from s2c.sketch.link import Ref
from s2c.sketch.models import Feature, Issue, ViewName
from s2c.sketch.solve import GROUPS, Solved, px_to_mm

LOS = {"front": "z", "back": "z", "top": "y", "bottom": "y", "right": "x", "left": "x"}
AXES = {v: {ax: g for g, members in GROUPS.items() for mv, ax, _ in members if mv == v} for v in LOS}
NEAR_VIEWER = {"front": min, "back": max, "top": max, "bottom": min, "right": max, "left": min}
ORDER = ["front", "top", "right", "left", "bottom", "back"]


def _orient(view: str, axis: str) -> int:
    return next((o for members in GROUPS.values() for v, ax, o in members if v == view and ax == axis), 1)


def _axis_px(point, axis: str) -> float:
    return float(point[0]) if axis == "a" else -float(point[1])


def to_distance(solved: Solved, refs: dict[ViewName, dict[str, list[Ref]]], view: str, axis: str, px: float) -> float:
    rs = refs[view][axis]
    u = px_to_mm(solved, rs, view, axis, px)
    return u if _orient(view, axis) > 0 else solved.mm[rs[-1].id] - u


def _team_frame(d: dict[str, float], solved: Solved) -> tuple[float, float, float]:
    depth = solved.envelope["z"].value if "z" in solved.envelope else 0.0
    return d.get("x", 0.0), d.get("y", 0.0), depth - d.get("z", 0.0)


def _lines_along(c: Classified, view, axis_along, refs, solved, axis_pos, kinds=("hidden", "visible")):
    out = []
    for p in [q for k in kinds for q in getattr(c, k)]:
        if p.kind != "line":
            continue
        d = p.p1 - p.p0
        if np.linalg.norm(d) < 1 or (abs(d[0]) > abs(d[1])) != (axis_along == "a"):
            continue
        pos = to_distance(solved, refs, view, axis_pos, (_axis_px(p.p0, axis_pos) + _axis_px(p.p1, axis_pos)) / 2)
        span = sorted(to_distance(solved, refs, view, axis_along, _axis_px(e, axis_along)) for e in (p.p0, p.p1))
        out.append((pos, span, p))
    return out


def _faces(c: Classified, view, ax_pos, ax_along, refs, solved, at: float, tol: float) -> list[float]:
    """Positions along the hole axis of visible lines that cross the hole's position `at`."""
    out = []
    for p in c.visible:
        if p.kind != "line":
            continue
        d = p.p1 - p.p0
        if (abs(d[0]) > abs(d[1])) != (ax_pos == "a"):
            continue
        span = sorted(to_distance(solved, refs, view, ax_pos, _axis_px(e, ax_pos)) for e in (p.p0, p.p1))
        if span[0] - tol <= at <= span[1] + tol:
            out.append(to_distance(solved, refs, view, ax_along,
                                   (_axis_px(p.p0, ax_along) + _axis_px(p.p1, ax_along)) / 2))
    return out


def find_holes(classified: dict[ViewName, Classified], refs: dict[ViewName, dict[str, list[Ref]]],
               solved: Solved) -> tuple[list[Feature], list[Issue]]:
    feats: list[Feature] = []
    issues: list[Issue] = []
    for va in ORDER:
        if va not in classified:
            continue
        hole_axis = LOS[va]
        for circ in [p for p in classified[va].visible + classified[va].hidden if p.kind == "circle"]:
            centre = {AXES[va][ax]: to_distance(solved, refs, va, ax,
                                                circ.center[0] if ax == "a" else -circ.center[1])
                      for ax in ("a", "b")}
            dia, badge = solved.circle_d.get(circ.id, (2 * circ.radius * solved.scale[(va, "a")], "predicted"))
            r, tol = dia / 2, max(0.5, 0.1 * dia / 2)
            pairs = []
            for vb in ORDER:
                if vb == va or vb not in classified or hole_axis not in AXES[vb].values():
                    continue
                shared = [g for g in centre if g in AXES[vb].values()]
                if not shared:
                    continue
                g = shared[0]
                ax_pos = next(a for a, gg in AXES[vb].items() if gg == g)
                ax_along = next(a for a, gg in AXES[vb].items() if gg == hole_axis)
                cands = _lines_along(classified[vb], vb, ax_along, refs, solved, ax_pos)
                lows = [c for c in cands if abs(c[0] - (centre[g] - r)) <= tol]
                highs = [c for c in cands if abs(c[0] - (centre[g] + r)) <= tol]
                faces = _faces(classified[vb], vb, ax_pos, ax_along, refs, solved, centre[g], tol)
                for lo in lows:
                    for hi in highs:
                        s0, s1 = max(lo[1][0], hi[1][0]), min(lo[1][1], hi[1][1])
                        if s1 - s0 > tol:
                            through = any(abs(f - s0) <= tol for f in faces) and any(abs(f - s1) <= tol for f in faces)
                            pairs.append((s0, s1, through, [lo[2].id, hi[2].id]))
                if pairs:
                    break
            if not pairs:
                ends = {"x": solved.envelope.get("x"), "y": solved.envelope.get("y"), "z": solved.envelope.get("z")}
                full = ends[hole_axis].value if ends[hole_axis] else 0.0
                along = NEAR_VIEWER[va](0.0, full)
                pos = _team_frame({**centre, hole_axis: along}, solved)
                feats.append(Feature(id=f"hole-{len(feats)}", type="hole", axis=hole_axis, position_mm=pos,
                                     diameter=dia, through=True, depth=None, evidence=[circ.id],
                                     badge="predicted"))
                issues.append(Issue(severity="amber", kind="predicted", targets=[f"hole-{len(feats) - 1}"],
                                    message=f"Ø{dia:g} hole has no hidden lines in another view; assumed through."))
                continue
            for s0, s1, through, ids in pairs:
                along = NEAR_VIEWER[va](s0, s1)
                pos = _team_frame({**centre, hole_axis: along}, solved)
                feats.append(Feature(id=f"hole-{len(feats)}", type="hole", axis=hole_axis, position_mm=pos,
                                     diameter=dia, through=through, depth=None if through else s1 - s0,
                                     evidence=[circ.id, *ids], badge=badge))
    return feats, issues
