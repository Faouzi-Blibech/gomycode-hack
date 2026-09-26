"""Turned parts: the hull of three extrusions has a square section, so a round hub or boss comes out square.
When the view along one axis is round and both side views match, the hull is cut down to a solid of revolution.
Deterministic, from the spec alone."""
from __future__ import annotations

import math
from itertools import combinations

import cadquery as cq
import numpy as np

from s2c.multiview.build import _clean
from s2c.multiview.spec import MultiViewSpec, to_global

WARNING = "Built as a turned part around the {axis} axis: its view along that axis is round and both side views match"

# axis -> (view along the axis, ((side view, its radial axis), ...)), in the order the axes are tried
SIDES = {
    "y": ("top", (("front", "x"), ("right", "z"))),
    "z": ("front", (("top", "x"), ("right", "y"))),
    "x": ("right", (("front", "y"), ("top", "z"))),
}
# revolve plane per axis: local x is radial, local y (= normal x xDir) runs along the axis
_PLANES = {"y": ((1, 0, 0), (0, 0, 1)), "z": ((1, 0, 0), (0, -1, 0)), "x": ((0, 1, 0), (0, 0, -1))}
SAMPLES = 256
LENGTH_TOL = 0.02
ROUND_TOL, ROUND_SLACK_MM = 1.03, 0.2
PROFILE_TOL = 0.03
NECK_TOL = 0.02  # a radius near 0 inside the part would make a non-manifold revolve
EPS_MM = 1e-4
# revolve clean-up, as fractions of the part's largest envelope length (the axis length for END_BAND)
MARGIN, SIMPLIFY, SNAP, END_BAND = 1e-3, 1e-3, 2e-3, 1e-2


def _extents(poly: np.ndarray, hs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Min and max radial coordinate of the closed outline's edge crossings at each height; +inf, -inf on a miss.
    poly rows are (height, radial)."""
    h1, t1 = poly[:, 0], poly[:, 1]
    h2, t2 = np.roll(h1, -1), np.roll(t1, -1)
    at = hs[:, None]
    hit = (np.minimum(h1, h2) <= at) & (at <= np.maximum(h1, h2))
    flat = h1 == h2
    t = t1 + (t2 - t1) * (at - h1) / np.where(flat, 1.0, h2 - h1)
    lo = np.where(hit, np.where(flat, np.minimum(t1, t2), t), np.inf).min(axis=1)
    hi = np.where(hit, np.where(flat, np.maximum(t1, t2), t), -np.inf).max(axis=1)
    return lo, hi


def _halves(spec: MultiViewSpec, axis: str, hs: np.ndarray) -> np.ndarray:
    """The four half-profiles: extent from the centre line on each side of both side views, one row each."""
    env = spec.envelope
    rows = []
    for face, rad in SIDES[axis][1]:
        g = [to_global(face, a, b, env) for a, b in getattr(spec.views, face).outer]
        lo, hi = _extents(np.array([(p[axis], p[rad]) for p in g], float), hs)
        c = env.length(rad) / 2
        rows += [hi - c, c - lo]
    return np.clip(np.array(rows), 0.0, None)


def _reach(spec: MultiViewSpec, axis: str) -> tuple[float, float]:
    """How far the view along the axis reaches from the centre line, and R: half the larger length across it."""
    env = spec.envelope
    view, sides = SIDES[axis]
    u, v = (rad for _, rad in sides)
    lu, lv = env.length(u), env.length(v)
    g = [to_global(view, a, b, env) for a, b in getattr(spec.views, view).outer]
    return max(math.hypot(p[u] - lu / 2, p[v] - lv / 2) for p in g), max(lu, lv) / 2


def turned_axis(spec: MultiViewSpec) -> str | None:
    """The axis the part is turned about, or None. Cheap: numpy only, it runs on every fuse."""
    env = spec.envelope
    for axis, (_, sides) in SIDES.items():
        lu, lv = (env.length(rad) for _, rad in sides)
        if abs(lu - lv) > LENGTH_TOL * max(lu, lv):
            continue
        reach, r_max = _reach(spec, axis)
        if reach > ROUND_TOL * r_max + ROUND_SLACK_MM:
            continue
        hs = np.linspace(0.0, env.length(axis), SAMPLES + 2)[1:-1]
        halves = _halves(spec, axis, hs)
        if halves.max(axis=0).min() < NECK_TOL * r_max:
            continue
        if max(np.abs(a - b).mean() for a, b in combinations(halves, 2)) / r_max <= PROFILE_TOL:
            return axis
    return None


def _simplify(pts: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker on an open polyline: fewer, larger faces make the boolean faster and sturdier."""
    keep = np.zeros(len(pts), bool)
    keep[[0, -1]] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        chord, rel = pts[j] - pts[i], pts[i + 1:j] - pts[i]
        dist = np.abs(chord[0] * rel[:, 1] - chord[1] * rel[:, 0]) / max(float(np.hypot(*chord)), 1e-12)
        k = int(np.argmax(dist))
        if dist[k] > tol:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return pts[keep]


def _chains(values: np.ndarray, tol: float) -> list[np.ndarray]:
    """Indices of the values, grouped where sorted neighbours are at most tol apart."""
    order = np.argsort(values, kind="stable")
    return np.split(order, np.flatnonzero(np.diff(values[order]) > tol) + 1)


def _square_up(pts: np.ndarray, tol: float) -> list[tuple[float, float]]:
    """Radii and heights within tol made equal, so every face is a cylinder, a flat ring or a clearly sloped cone:
    OCC booleans silently return nothing on a cone that is almost a cylinder. Radii round up, keeping the part inside."""
    r, h = pts[:, 0].copy(), pts[:, 1].copy()
    inner, last = np.arange(1, len(pts) - 1), len(pts) - 1  # the two points on the axis keep r = 0
    for g in _chains(r[inner], tol):
        r[inner[g]] = r[inner[g]].max()
    for g in _chains(h, tol):
        h[g] = h[0] if 0 in g else h[last] if last in g else h[g].mean()
    out: list[tuple[float, float]] = []
    for p in zip(r.tolist(), h.tolist()):
        if out and p == out[-1]:
            continue
        if len(out) >= 2 and (out[-2][0] == out[-1][0] == p[0] or out[-2][1] == out[-1][1] == p[1]):
            out[-1] = p
        else:
            out.append(p)
    return out


def profile(spec: MultiViewSpec, axis: str) -> list[tuple[float, float]]:
    """Half-section (r, h), closed on the axis. r is the largest half-extent, taken just either side of every
    outline vertex height so steps stay sharp, and scaled up when the view along the axis reaches further out
    (gear teeth between the side views). The true part stays inside, and the revolve never just touches the hull
    (OCC booleans fail on tangent contact): r grows by a margin and the ends run past the envelope. Within END_BAND
    of an end r takes the band's largest value: an outline that stops just short of the envelope end would otherwise
    leave a needle on the axis."""
    env = spec.envelope
    length, size = env.length(axis), max(env.x_mm, env.y_mm, env.z_mm)
    heights = np.array(sorted({0.0, length} | {to_global(face, a, b, env)[axis] for face, _ in SIDES[axis][1]
                                               for a, b in getattr(spec.views, face).outer}))
    hs = np.sort(np.concatenate([heights - EPS_MM, heights + EPS_MM]))
    hs = hs[(hs > 0) & (hs < length)]
    reach, r_max = _reach(spec, axis)
    r = _halves(spec, axis, hs).max(axis=0) * max(1.0, reach / r_max)
    for end in (hs < END_BAND * length, hs > (1 - END_BAND) * length):
        if end.any():
            r[end] = r[end].max()
    r += MARGIN * size
    lo, hi = -MARGIN * size, length + MARGIN * size
    pts = np.array([(0.0, lo), (r[0], lo), *zip(r, hs), (r[-1], hi), (0.0, hi)])
    return _square_up(_simplify(pts, SIMPLIFY * size), SNAP * size)


def revolve(spec: MultiViewSpec, axis: str) -> cq.Workplane:
    """Solid of revolution about the envelope's centre line along `axis`."""
    env = spec.envelope
    origin = tuple(0.0 if k == axis else env.length(k) / 2 for k in "xyz")
    x_dir, normal = _PLANES[axis]
    plane = cq.Plane(origin=origin, xDir=x_dir, normal=normal)
    return cq.Workplane(plane).polyline(_clean(profile(spec, axis))).close().revolve(360, (0, 0, 0), (0, 1, 0))
