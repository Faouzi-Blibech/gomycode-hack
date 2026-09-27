"""Stage 4: skeleton -> paths -> lines, circles, arcs and free curves. Deterministic geometry."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import cv2
import numpy as np
from skan import Skeleton as SkanSkeleton
from skimage.morphology import skeletonize

SMOOTH_TURN_DEG = 35.0


@dataclass
class Prim:
    id: str
    kind: Literal["line", "circle", "arc", "curve"]
    pts: np.ndarray
    width: float
    center: tuple[float, float] | None = None
    radius: float | None = None

    @property
    def p0(self) -> np.ndarray:
        return self.pts[0]

    @property
    def p1(self) -> np.ndarray:
        return self.pts[-1]

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.pts, axis=0), axis=1).sum())

    def direction(self) -> np.ndarray:
        d = self.p1 - self.p0
        return d / (np.linalg.norm(d) + 1e-9)


def trace(mask: np.ndarray) -> list[np.ndarray]:
    """Skeleton paths between endpoints and junctions, plus closed loops, as (x, y) polylines.
    The graph comes from skan (a maintained skeleton-analysis library), not our own tracer."""
    sk = skeletonize(mask > 0)
    if not sk.any():
        return []
    graph = SkanSkeleton(sk)
    out = []
    for i in range(graph.n_paths):
        rc = graph.path_coordinates(i)  # (N, 2) as (row, col)
        if len(rc) >= 2:
            out.append(rc[:, ::-1].astype(float))
    return out


def circle_fit(pts: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Algebraic (Kasa) circle fit: centre, radius, rms distance to the circle."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.stack([x, y, np.ones_like(x)], 1)
    b = x * x + y * y
    (a, bb, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = a / 2, bb / 2
    r = float(np.sqrt(max(c + cx * cx + cy * cy, 0.0)))
    rms = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2)))
    return np.array([cx, cy]), r, rms


def _turns_deg(poly: np.ndarray) -> np.ndarray:
    d = np.diff(poly, axis=0)
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    return np.abs((np.diff(ang) + 180) % 360 - 180)


def _span_deg(pts: np.ndarray, c: np.ndarray) -> float:
    ang = np.unwrap(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
    return float(np.degrees(ang.max() - ang.min()))


def vectorize(ink: np.ndarray, stroke_px: float, view: str) -> list[Prim]:
    dist = cv2.distanceTransform((ink > 0).astype(np.uint8), cv2.DIST_L2, 3)
    prims: list[Prim] = []

    def add(kind, pts, width, center=None, radius=None):
        prims.append(Prim(f"{view}-p{len(prims)}", kind, np.asarray(pts, float), width,
                          center, radius))

    for path in trace(ink):
        length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        if length < max(3.0, stroke_px):
            continue
        ij = np.clip(path.round().astype(int), 0, [ink.shape[1] - 1, ink.shape[0] - 1])
        width = float(2 * dist[ij[:, 1], ij[:, 0]].mean())
        closed = np.linalg.norm(path[0] - path[-1]) <= 2 * stroke_px and length > 6 * stroke_px
        if closed:
            c, r, rms = circle_fit(path)
            if r > 1.5 * stroke_px and rms / r < 0.08:
                add("circle", path, width, (float(c[0]), float(c[1])), r)
                continue
            # start the loop at a corner (the point farthest from the centroid), so RDP does not
            # split one edge in two where the loop happened to start. Drop the duplicate closing
            # point and simplify with closed=True (approxPolyDP with closed=False and a path whose
            # first and last point coincide collapses the segment back to the start), then re-close
            # the simplified polygon so the final line pass emits the last edge too.
            k = int(np.argmax(np.linalg.norm(path - path.mean(0), axis=1)))
            path = np.vstack([path[k:-1], path[:k]])
        eps = max(2.0, 0.8 * stroke_px)
        simp = cv2.approxPolyDP(path.astype(np.float32).reshape(-1, 1, 2), eps, closed).reshape(-1, 2)
        if closed and len(simp) >= 3:
            simp = np.vstack([simp, simp[:1]])
        if len(simp) >= 4 and length > 8 * stroke_px and not closed:
            c, r, rms = circle_fit(path)
            if rms / max(r, 1e-9) < 0.05 and r < 5 * length and _span_deg(path, c) > 30:
                add("arc", path, width, (float(c[0]), float(c[1])), r)
                continue
        if len(simp) >= 5 and _turns_deg(simp).max() < SMOOTH_TURN_DEG:
            add("curve", simp, width)
            continue
        for a, b in pairwise(simp):
            if np.linalg.norm(b - a) >= max(2.0, 0.5 * stroke_px):
                add("line", [a, b], width)
    return prims
