"""Blender output. A real .blend is written by a separate Blender or bpy process and never inside the app: bpy is
about 300 MB, Python 3.11 only, and may clash on numpy. Without one, a "Blender kit" zip holds the OBJ and the
import script. Spec 2026-09-23-studio sections 2 and 6."""
from __future__ import annotations

import os
import sys
import tempfile
import zipfile
from pathlib import Path

from s2c.multiview.proc import run, tail

BLEND_TIMEOUT_S = 90
KIT_NAME = "part_blender_kit.zip"
KIT_WARNING = "Blender not configured: the download holds part.obj and open_in_blender.py instead of a .blend"
VENDOR_PYTHON = Path(__file__).resolve().parents[2] / "vendor" / "bpy-env" / (
    "Scripts/python.exe" if os.name == "nt" else "bin/python")
SCRIPT = '''"""Import part.obj into an empty Blender scene in millimetres and save it as a .blend.
Run: blender -b --factory-startup --python open_in_blender.py -- part.obj part.blend"""
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
src, dst = argv[0], argv[1]
bpy.ops.wm.read_factory_settings(use_empty=True)
units = bpy.context.scene.unit_settings
units.system, units.scale_length, units.length_unit = "METRIC", 0.001, "MILLIMETERS"
bpy.ops.wm.obj_import(filepath=src, forward_axis="NEGATIVE_Z", up_axis="Y")  # the OBJ is Y-up, Blender is Z-up
part = bpy.context.selected_objects[0]
part.name = "part"
material = bpy.data.materials.new("Part")
material.diffuse_color = (0.31, 0.27, 0.90, 1.0)
part.data.materials.append(material)
bpy.ops.wm.save_as_mainfile(filepath=dst, compress=True)
'''


def blender_runner() -> list[str] | None:
    """The command prefix that runs SCRIPT: a Blender executable, or a Python that has bpy."""
    exe = os.environ.get("BLENDER_PATH")
    if exe and Path(exe).is_file():
        return [exe, "-b", "--factory-startup", "--python"]
    for py in (os.environ.get("BLENDER_PYTHON"), str(VENDOR_PYTHON)):
        if py and Path(py).is_file():
            return [py]
    return None


def _valid_blend(path: Path) -> bool:
    """Non-empty and starting with a real .blend magic: plain, gzip- or zstd-compressed."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    head = path.read_bytes()[:8]
    return head.startswith((b"BLENDER", b"\x28\xb5\x2f\xfd", b"\x1f\x8b"))


def _kit(obj_path: Path, out_dir: Path, warnings: list[str]) -> tuple[Path, list[str]]:
    kit = out_dir / KIT_NAME
    if not kit.exists():
        fd, name = tempfile.mkstemp(suffix=".part", dir=out_dir)  # unique: concurrent callers never share it
        os.close(fd)
        tmp = Path(name)
        try:
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
                z.write(obj_path, arcname="part.obj")
                z.writestr("open_in_blender.py", SCRIPT)
            os.replace(tmp, kit)
        finally:
            tmp.unlink(missing_ok=True)
    return kit, [*warnings, KIT_WARNING] if warnings else [KIT_WARNING]


def write_blend(obj_path: Path, out_dir: Path) -> tuple[Path, list[str]]:
    obj_path, out_dir = Path(obj_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "part.blend"
    if target.exists():
        if _valid_blend(target):
            return target, []
        target.unlink()
    runner = blender_runner()
    if runner is None:
        return _kit(obj_path, out_dir, [])
    script = out_dir / "open_in_blender.py"
    script.write_text(SCRIPT, encoding="utf-8")
    log = out_dir / "blender.log"
    code = run([*runner, str(script), "--", str(obj_path), str(target)], BLEND_TIMEOUT_S, log)
    if code == 0 and _valid_blend(target):
        (out_dir / KIT_NAME).unlink(missing_ok=True)  # a stale kit from an earlier failed run
        return target, []
    reason = "timed out" if code is None else f"exit code {code}"
    print(tail(log), file=sys.stderr)
    return _kit(obj_path, out_dir, [f"Blender failed ({reason}); see blender.log"])
