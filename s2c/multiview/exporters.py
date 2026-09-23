"""Mesh and CAD file writers with fixed quality presets. Spec 2026-09-23-studio section 6.
Meshes use absolute tolerances on a copy of the shape: OCCT keeps a triangulation on the shape and would
otherwise reuse a finer one, and CadQuery's default tolerance is relative to the part size."""
from __future__ import annotations

import tempfile
import threading
from pathlib import Path

import cadquery as cq
import trimesh

from s2c.multiview.settings import MESH_TOLERANCES

MESH_FORMATS = ("stl", "step", "3mf", "obj", "glb", "ply", "brep")
FILE_NAMES = {"stl": "part.stl", "step": "part.step", "3mf": "part.3mf", "obj": "part.obj", "glb": "part.glb",
              "ply": "part.ply", "brep": "part.brep"}
PART_COLOUR = (79, 70, 229, 255)  # indigo 600, the Studio's primary colour
_STEP_LOCK = threading.Lock()  # OCCT's STEP writer settings are process-wide


def _workplane(solid) -> cq.Workplane:
    return solid if isinstance(solid, cq.Workplane) else cq.Workplane("XY").add(solid)


def _copy(solid) -> cq.Workplane:
    return cq.Workplane("XY").add(_workplane(solid).val().copy())


def write_stl(solid, path: Path, quality: str = "normal") -> Path:
    tol, ang = MESH_TOLERANCES[quality]
    _copy(solid).val().exportStl(str(path), tolerance=tol, angularTolerance=ang, ascii=False, relative=False)
    return Path(path)


def mesh_of(solid, quality: str = "normal") -> trimesh.Trimesh:
    """The same triangles as write_stl, as a trimesh."""
    with tempfile.TemporaryDirectory() as d:
        stl = write_stl(solid, Path(d) / "mesh.stl", quality)
        return trimesh.load(str(stl), force="mesh")


def write_step(solid, path: Path, quality: str = "normal") -> Path:
    with _STEP_LOCK:
        cq.exporters.export(_workplane(solid), str(path), exportType="STEP")
    return Path(path)


def write_3mf(solid, path: Path, quality: str = "normal") -> Path:
    tol, ang = MESH_TOLERANCES[quality]
    cq.exporters.export(_copy(solid), str(path), exportType="3MF", tolerance=tol, angularTolerance=ang)
    return Path(path)


def write_brep(solid, path: Path, quality: str = "normal") -> Path:
    cq.exporters.export(_workplane(solid), str(path), exportType="BREP")
    return Path(path)


def _mesh_writer(file_type: str, scale: float = 1.0, colour: bool = False):
    def write(solid, path: Path, quality: str = "normal") -> Path:
        mesh = mesh_of(solid, quality)
        if scale != 1.0:
            mesh.apply_scale(scale)
        if colour:
            mesh.visual.face_colors = PART_COLOUR
        mesh.export(str(path), file_type=file_type)
        return Path(path)
    return write


write_obj = _mesh_writer("obj")
write_ply = _mesh_writer("ply")
write_glb = _mesh_writer("glb", scale=0.001, colour=True)  # glTF is in metres, Y-up like the design frame
WRITERS = {"stl": write_stl, "step": write_step, "3mf": write_3mf, "obj": write_obj, "glb": write_glb,
           "ply": write_ply, "brep": write_brep}


def export_mesh_formats(solid, out_dir: Path, formats, quality: str = "normal") -> dict[str, Path]:
    unknown = [f for f in formats if f not in WRITERS]
    if unknown:
        raise ValueError(f"unknown mesh formats {unknown}; choose from {list(WRITERS)}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {f: WRITERS[f](solid, out_dir / FILE_NAMES[f], quality) for f in formats}
