"""Fill the canonical faces nobody photographed: predicted from a TripoSR mesh, else assumed rectangular.
The mirror rule already ran in fuse. Spec section 6.3. The mesh is only rendered, never exported."""
from __future__ import annotations

import itertools
import logging
from collections.abc import Callable

import numpy as np

from s2c.multiview.raster import Mesh, face_mask, iou, mask_to_mm, normalize_mask
from s2c.multiview.spec import CANONICAL_FACES, Envelope, Outline, face_size

log = logging.getLogger(__name__)
MeshProvider = Callable[[np.ndarray], Mesh]
MIN_ORIENTATION_IOU = 0.6
SEARCH_PX = 128


def rotations() -> list[np.ndarray]:
    """The 24 axis-aligned rotations: signed permutation matrices with determinant +1."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            m = np.zeros((3, 3))
            for row, (col, sign) in enumerate(zip(perm, signs)):
                m[row, col] = sign
            if round(np.linalg.det(m)) == 1:
                out.append(m)
    return out


def _at_origin(v: np.ndarray) -> np.ndarray:
    return v - v.min(axis=0)


def _own_envelope(v: np.ndarray) -> Envelope:
    span = np.maximum(v.max(axis=0) - v.min(axis=0), 1e-6)
    return Envelope(x_mm=float(span[0]), y_mm=float(span[1]), z_mm=float(span[2]))


def orient(mesh: Mesh, face: str, target_mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Rotation that makes the mesh, seen from `face` at its own proportions, look most like the target."""
    target = normalize_mask(target_mask, SEARCH_PX)
    best_r, best = np.eye(3), -1.0
    for r in rotations():
        v = _at_origin(mesh.vertices @ r.T)
        mask, _ = face_mask(Mesh(v, mesh.faces), face, _own_envelope(v), SEARCH_PX)
        score = iou(normalize_mask(mask, SEARCH_PX), target)
        if score > best:
            best_r, best = r, score
    return best_r, best


def fit_to_envelope(mesh: Mesh, r: np.ndarray, env: Envelope) -> Mesh:
    """Rotate, then scale each axis so the bounding box equals the trusted envelope."""
    v = _at_origin(mesh.vertices @ r.T)
    span = np.maximum(v.max(axis=0), 1e-9)
    return Mesh(v / span * np.array([env.x_mm, env.y_mm, env.z_mm]), mesh.faces)


def _clamp(pts, a_len, b_len):
    return [(min(max(a, 0.0), a_len), min(max(b, 0.0), b_len)) for a, b in pts]


def predicted_outline(mesh: Mesh, face: str, env: Envelope, confidence: float) -> Outline:
    mask, s = face_mask(mesh, face, env, 512)
    outer, inner = mask_to_mm(mask, s, 512)
    a_len, b_len = face_size(face, env)
    return Outline(outer=_clamp(outer, a_len, b_len), inner=[_clamp(loop, a_len, b_len) for loop in inner],
                   source="inferred", confidence=round(float(confidence), 3))


def assumed_outline(face: str, env: Envelope) -> Outline:
    a, b = face_size(face, env)
    return Outline(outer=[(0.0, 0.0), (a, 0.0), (a, b), (0.0, b)], source="assumed", confidence=0.3)


def complete(outlines: dict[str, Outline], env: Envelope, target_face: str, target_mask: np.ndarray,
             image: np.ndarray | None, provider: MeshProvider | None, mesh: Mesh | None = None,
             rejected=()) -> tuple[dict[str, Outline], list[str], Mesh | None]:
    """All three canonical outlines, the warnings, and the mesh so the caller can cache it."""
    result, warnings = dict(outlines), []
    missing = [f for f in CANONICAL_FACES if f not in outlines]
    wanted = [f for f in missing if f not in rejected]
    if wanted and mesh is None and provider is not None and image is not None:
        try:
            mesh = provider(image)
        except Exception as e:
            log.warning("3D predictor failed: %s", e)
            warnings.append("3D predictor unavailable")
    fitted, score = None, 0.0
    if wanted and mesh is not None:
        r, score = orient(mesh, target_face, target_mask)
        if score >= MIN_ORIENTATION_IOU:
            fitted = fit_to_envelope(mesh, r, env)
        else:
            warnings.append("predicted view unreliable")
    for face in missing:
        if fitted is not None and face in wanted:
            try:
                result[face] = predicted_outline(fitted, face, env, score)
                continue
            except ValueError:
                log.warning("predicted %s view was empty", face)
        result[face] = assumed_outline(face, env)
        warnings.append(f"assumed rectangular {face}, check it")
    return result, warnings, mesh
