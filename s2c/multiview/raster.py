"""Triangles -> per-face binary masks; mask normalisation and IoU. Spec sections 6.3 and 6.6."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from s2c.multiview.spec import Envelope, face_size

MARGIN = 8


@dataclass
class Mesh:
    vertices: np.ndarray  # (n, 3) float, millimetres
    faces: np.ndarray     # (m, 3) int


def face_coords(face: str, pts: np.ndarray, env: Envelope) -> np.ndarray:
    """Global (n, 3) points -> (n, 2) points in the face frame of spec section 3."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    big_x, big_z = env.x_mm, env.z_mm
    a, b = {
        "front": (x, y), "back": (big_x - x, y),
        "top": (x, big_z - z), "bottom": (x, z),
        "right": (big_z - z, y), "left": (z, y),
    }[face]
    return np.stack([a, b], axis=1)


def _scale(a_len: float, b_len: float, px: int) -> float:
    return (px - 2 * MARGIN) / max(a_len, b_len)


def mm_to_px(points, s: float, px: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return np.stack([MARGIN + pts[:, 0] * s, px - MARGIN - pts[:, 1] * s], axis=1)


def face_mask(mesh: Mesh, face: str, env: Envelope, px: int = 512) -> tuple[np.ndarray, float]:
    """Silhouette of `mesh` seen from `face`, drawn in that face's envelope rectangle.
    Returns the mask and s in pixels per millimetre: pixel (u, v) is a = (u - MARGIN) / s, b = (px - MARGIN - v) / s."""
    s = _scale(*face_size(face, env), px)
    uv = mm_to_px(face_coords(face, mesh.vertices, env), s, px)
    tris = np.round(uv[mesh.faces]).astype(np.int32)
    img = np.zeros((px, px), np.uint8)
    for tri in tris:  # one call per triangle: one call for all of them fills even-odd and erases overlaps
        cv2.fillPoly(img, [tri], 255)
    return img, s


def polygon_mask(outer, holes=(), shape=(512, 512)) -> np.ndarray:
    """Filled outer polygon minus holes; points are pixel (x, y)."""
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(outer, np.float64)).astype(np.int32).reshape(-1, 2)], 255)
    for h in holes:
        cv2.fillPoly(m, [np.round(np.asarray(h, np.float64)).astype(np.int32).reshape(-1, 2)], 0)
    return m


def outline_mask(outer_mm, inner_mm, a_len: float, b_len: float, px: int = 256) -> np.ndarray:
    """Mask of a face-frame outline, drawn the way face_mask draws a mesh."""
    s = _scale(a_len, b_len, px)
    return polygon_mask(mm_to_px(outer_mm, s, px), [mm_to_px(loop, s, px) for loop in inner_mm], (px, px))


def mask_to_mm(mask: np.ndarray, s: float, px: int, min_opening_mm: float = 3.0,
               eps_mm: float = 0.5) -> tuple[list, list]:
    """Largest outline of a face_mask and its openings, back in face-frame millimetres."""
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("empty mask")
    tops = [i for i in range(len(contours)) if hierarchy[0][i][3] == -1]
    outer_i = max(tops, key=lambda i: cv2.contourArea(contours[i]))

    def to_mm(c):
        pts = cv2.approxPolyDP(c, eps_mm * s, True).reshape(-1, 2).astype(np.float64)
        return [(float((u - MARGIN) / s), float((px - MARGIN - v) / s)) for u, v in pts]

    inner = []
    for i, c in enumerate(contours):
        if hierarchy[0][i][3] == outer_i:
            _, _, w, h = cv2.boundingRect(c)
            if min(w, h) / s >= min_opening_mm:
                inner.append(to_mm(c))
    return to_mm(contours[outer_i]), inner


def normalize_mask(mask: np.ndarray, px: int = 512) -> np.ndarray:
    """Crop to the content, pad to a square, resize to px. Same rule as the integrator's normalize_mask."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((px, px), np.uint8)
    crop = mask[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    h, w = crop.shape
    side = max(h, w)
    square = np.zeros((side, side), np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    square[y0: y0 + h, x0: x0 + w] = crop
    out = cv2.resize(square, (px, px), interpolation=cv2.INTER_NEAREST)
    return np.where(out > 127, 255, 0).astype(np.uint8)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a_on, b_on = a > 127, b > 127
    union = np.logical_or(a_on, b_on).sum()
    return 0.0 if union == 0 else float(np.logical_and(a_on, b_on).sum() / union)


def solid_mesh(solid, tolerance: float = 0.05, angular: float = 0.2) -> Mesh:
    """Tessellate a CadQuery Workplane or Shape."""
    shape = solid.val() if hasattr(solid, "val") else solid
    verts, tris = shape.tessellate(tolerance, angular)
    return Mesh(np.array([[v.x, v.y, v.z] for v in verts], np.float64), np.array(tris, np.int64).reshape(-1, 3))
