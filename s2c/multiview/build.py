"""MultiViewSpec -> CadQuery solid: intersection of three extruded outlines, then face features and finishes.
Spec section 6.4. Deterministic; no model output is ever executed here."""
from __future__ import annotations

from pathlib import Path

import cadquery as cq

from s2c.multiview import exporters
from s2c.multiview.spec import FACE_AXES, Envelope, FaceHole, Fillet, MultiViewSpec, Outline, face_size


class BuildError(Exception):
    def __init__(self, reason: str, remedy: str):
        super().__init__(f"{reason}: {remedy}")
        self.reason, self.remedy = reason, remedy


INVALID = ("invalid_solid", "Simplify the outline or retake the photo.")
EMPTY = ("intersection_empty", "The views do not describe one part. Check which face each photo shows.")


def volume(solid: cq.Workplane) -> float:
    return float(sum(s.Volume() for s in solid.solids().vals()))


# ---- the three prisms -------------------------------------------------------

def _canonical_plane(face: str, env: Envelope) -> tuple[cq.Plane, float]:
    """Build plane of a canonical face and the extrusion length that fills the envelope.
    Local (u, v): front (X, Y), top (X, Z), right (Z, Y). CadQuery sets yDir = normal x xDir."""
    if face == "front":
        return cq.Plane(origin=(0, 0, 0), xDir=(1, 0, 0), normal=(0, 0, 1)), env.z_mm
    if face == "top":
        return cq.Plane(origin=(0, env.y_mm, 0), xDir=(1, 0, 0), normal=(0, -1, 0)), env.y_mm
    return cq.Plane(origin=(env.x_mm, 0, 0), xDir=(0, 0, 1), normal=(-1, 0, 0)), env.x_mm


def _plane_uv(face: str, pts, env: Envelope) -> list[tuple[float, float]]:
    """Canonical face-frame points -> local (u, v) of the build plane (spec section 3)."""
    if face == "front":
        return [(a, b) for a, b in pts]
    if face == "top":
        return [(a, env.z_mm - b) for a, b in pts]
    return [(env.z_mm - a, b) for a, b in pts]


def _clean(pts) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a, b in pts:
        p = (round(float(a), 6), round(float(b), 6))
        if not out or p != out[-1]:
            out.append(p)
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out


def _prism(face: str, outline: Outline, env: Envelope) -> cq.Workplane:
    plane, length = _canonical_plane(face, env)
    outer = _clean(_plane_uv(face, outline.outer, env))
    if len(outer) < 3:
        raise BuildError(*INVALID)
    solid = cq.Workplane(plane).polyline(outer).close().extrude(length)
    for loop in outline.inner:
        pts = _clean(_plane_uv(face, loop, env))
        if len(pts) >= 3:
            solid = solid.cut(cq.Workplane(plane).polyline(pts).close().extrude(length))
    return solid


def _check(solid: cq.Workplane) -> None:
    solids = solid.solids().vals()
    if not solids or volume(solid) < 1e-6:
        raise BuildError(*EMPTY)
    if len(solids) > 1 or not solids[0].isValid():
        raise BuildError(*INVALID)


# ---- features ----------------------------------------------------------------

def _face_plane(face: str, env: Envelope, offset: float = 0.0) -> cq.Plane:
    """Plane on an envelope face: origin at (a, b) = (0, 0), xDir along +a, normal pointing out of the part.
    For every face, normal x (+a) = +b, so local (u, v) = (a, b)."""
    x, y, z = env.x_mm, env.y_mm, env.z_mm
    origin, x_dir, normal = {
        "front": ((0, 0, z), (1, 0, 0), (0, 0, 1)),
        "back": ((x, 0, 0), (-1, 0, 0), (0, 0, -1)),
        "top": ((0, y, z), (1, 0, 0), (0, 1, 0)),
        "bottom": ((0, 0, 0), (1, 0, 0), (0, -1, 0)),
        "right": ((x, 0, z), (0, 0, -1), (1, 0, 0)),
        "left": ((0, 0, 0), (0, 0, 1), (-1, 0, 0)),
    }[face]
    moved = tuple(o + n * offset for o, n in zip(origin, normal))
    return cq.Plane(origin=moved, xDir=x_dir, normal=normal)


def _cut_feature(solid: cq.Workplane, f, env: Envelope) -> cq.Workplane:
    a_len, b_len = face_size(f.face, env)
    if not (0 <= f.a_mm <= a_len and 0 <= f.b_mm <= b_len):
        raise BuildError("feature_outside_part", "A hole or slot lies outside the part. Check its position.")
    if f.depth_mm is None:  # through: start 1 mm outside, end 1 mm past the far side
        wp, dist = cq.Workplane(_face_plane(f.face, env, 1.0)), env.length(FACE_AXES[f.face][2]) + 2.0
    else:  # blind: depth measured from the envelope face inward
        wp, dist = cq.Workplane(_face_plane(f.face, env)), f.depth_mm
    wp = wp.center(f.a_mm, f.b_mm)
    shape = wp.circle(f.diameter_mm / 2) if isinstance(f, FaceHole) else wp.slot2D(f.length_mm, f.width_mm, f.angle_deg)
    return solid.cut(shape.extrude(-dist))


_EDGE_SELECTORS = {"all": None, "all_vertical": "|Z", "top": ">Z", "bottom": "<Z"}


def _apply_finish(solid: cq.Workplane, finish) -> cq.Workplane:
    kind = "fillet" if isinstance(finish, Fillet) else "chamfer"
    failed = BuildError(f"{kind}_failed", "Reduce the fillet radius." if kind == "fillet" else "Reduce the chamfer size.")
    selector = _EDGE_SELECTORS[finish.edges]
    edges = solid.edges() if selector is None else solid.edges(selector)
    try:
        out = edges.fillet(finish.radius_mm) if kind == "fillet" else edges.chamfer(finish.radius_mm)
    except Exception as e:  # OCC raises StdFail_NotDone and friends
        raise failed from e
    if not out.solids().vals() or not out.val().isValid():
        raise failed
    return out


def build(spec: MultiViewSpec) -> cq.Workplane:
    env = spec.envelope
    try:
        solid = _prism("front", spec.views.front, env)
        for face in ("top", "right"):
            solid = solid.intersect(_prism(face, getattr(spec.views, face), env))
            if not solid.solids().vals():  # an empty result would make CadQuery fall back to an earlier solid
                raise BuildError(*EMPTY)
    except BuildError:
        raise
    except Exception as e:
        raise BuildError(*INVALID) from e
    _check(solid)
    for f in spec.features:
        solid = _cut_feature(solid, f, env)
    for finish in spec.finishes:
        solid = _apply_finish(solid, finish)
    _check(solid)
    return solid


def export(solid: cq.Workplane, out_dir: Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    step, stl = out_dir / "part.step", out_dir / "part.stl"
    with exporters._OCCT_LOCK:
        cq.exporters.export(solid, str(step))
        cq.exporters.export(solid, str(stl), tolerance=0.01, angularTolerance=0.1)
    return step, stl
