"""Solid -> print-oriented STL -> G-code with PrusaSlicer CLI and one fixed FDM profile. Spec section 6.5."""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cadquery as cq

from s2c.multiview.spec import MvAbstain

log = logging.getLogger(__name__)

DEFAULT_PROFILE = Path(__file__).resolve().parents[2] / "profiles" / "fdm_default.ini"
WINDOWS_SLICER = Path("C:/Program Files/Prusa3D/PrusaSlicer/prusa-slicer-console.exe")
VENDOR_DIR = Path(__file__).resolve().parents[2] / "vendor"  # portable PrusaSlicer zip unpacked here, no admin needed
SLICE_TIMEOUT_S = 120
# down direction -> (unit vector, rotation (axis, degrees) that turns it into -Z)
DOWN_DIRECTIONS = {
    "-z": ((0, 0, -1), None),
    "+z": ((0, 0, 1), ((1, 0, 0), 180.0)),
    "-y": ((0, -1, 0), ((1, 0, 0), 90.0)),
    "+y": ((0, 1, 0), ((1, 0, 0), -90.0)),
    "-x": ((-1, 0, 0), ((0, 1, 0), -90.0)),
    "+x": ((1, 0, 0), ((0, 1, 0), 90.0)),
}


@dataclass
class Profile:
    path: Path
    bed_x: float
    bed_y: float
    max_height: float

    @property
    def center(self) -> tuple[float, float]:
        return self.bed_x / 2, self.bed_y / 2


@dataclass
class SliceResult:
    print_stl: Path
    gcode: Path | None
    print_time_s: float | None
    filament_g: float | None
    warnings: list[str] = field(default_factory=list)


def load_profile(path: Path | str | None = None) -> Profile:
    path = Path(path or os.environ.get("SLICER_PROFILE") or DEFAULT_PROFILE)
    values = {}
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    corners = [tuple(float(n) for n in c.split("x")) for c in values["bed_shape"].split(",")]
    xs, ys = zip(*corners)
    return Profile(path, max(xs) - min(xs), max(ys) - min(ys), float(values.get("max_print_height", 200)))


def _down_area(solid: cq.Workplane, direction) -> float:
    """Area of the flat faces that would rest on the bed if `direction` pointed down."""
    d = cq.Vector(*direction)
    bb = solid.val().BoundingBox()
    extreme = max(d.dot(cq.Vector(x, y, z)) for x in (bb.xmin, bb.xmax) for y in (bb.ymin, bb.ymax)
                  for z in (bb.zmin, bb.zmax))
    area = 0.0
    for face in solid.faces().vals():
        if face.geomType() == "PLANE" and face.normalAt().dot(d) > 0.999 and abs(face.Center().dot(d) - extreme) < 1e-3:
            area += face.Area()
    return area


def choose_down(solid: cq.Workplane) -> str:
    """Largest flat face on the bed; on a tie within 1 percent, the lowest part."""
    bb = solid.val().BoundingBox()
    height = {"x": bb.xlen, "y": bb.ylen, "z": bb.zlen}
    areas = {name: _down_area(solid, vec) for name, (vec, _) in DOWN_DIRECTIONS.items()}
    best = max(areas.values())
    tied = [n for n in DOWN_DIRECTIONS if areas[n] >= 0.99 * best]
    return min(tied, key=lambda n: height[n[1]])


def orient_for_print(solid: cq.Workplane) -> cq.Workplane:
    rotation = DOWN_DIRECTIONS[choose_down(solid)][1]
    if rotation is not None:
        axis, angle = rotation
        solid = solid.rotate((0, 0, 0), axis, angle)
    bb = solid.val().BoundingBox()
    return solid.translate((-bb.xmin, -bb.ymin, -bb.zmin))


def fits_bed(solid: cq.Workplane, profile: Profile) -> bool:
    bb = solid.val().BoundingBox()
    flat = ((bb.xlen <= profile.bed_x and bb.ylen <= profile.bed_y)
            or (bb.ylen <= profile.bed_x and bb.xlen <= profile.bed_y))
    return flat and bb.zlen <= profile.max_height


def find_slicer() -> Path | None:
    vendored = sorted(VENDOR_DIR.glob("PrusaSlicer*/prusa-slicer-console.exe"))
    for cand in (os.environ.get("SLICER_PATH"), shutil.which("prusa-slicer-console"), shutil.which("prusa-slicer"),
                 str(WINDOWS_SLICER), *map(str, vendored)):
        if cand and Path(cand).is_file():
            return Path(cand)
    return None


_TIME = re.compile(r"; estimated printing time \(normal mode\) = (.+)")
_GRAMS = re.compile(r"; filament used \[g\] = ([\d.]+)")


def parse_duration(text: str) -> float:
    units = {"d": 86400, "h": 3600, "m": 60, "s": 1}
    return float(sum(int(n) * units[u] for n, u in re.findall(r"(\d+)\s*([dhms])", text)))


def parse_gcode_stats(text: str) -> tuple[float | None, float | None]:
    t, g = _TIME.search(text), _GRAMS.search(text)
    return (parse_duration(t.group(1)) if t else None), (float(g.group(1)) if g else None)


def slice_solid(solid: cq.Workplane, out_dir: Path, profile_path: Path | None = None,
                slicer: Path | None = None) -> SliceResult | MvAbstain:
    profile = load_profile(profile_path)
    printable = orient_for_print(solid)
    if not fits_bed(printable, profile):
        return MvAbstain(stage="slice", reason="too_big_for_bed",
                         remedy="The part is larger than the printer bed. Scale it down or split it.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stl = out_dir / "part_print.stl"
    cq.exporters.export(printable, str(stl), tolerance=0.01, angularTolerance=0.1)
    slicer = slicer or find_slicer()
    if slicer is None:
        return SliceResult(stl, None, None, None, ["G-code unavailable: slicer not installed"])
    gcode = out_dir / "part.gcode"
    cx, cy = profile.center
    cmd = [str(slicer), "--export-gcode", "--load", str(profile.path), "--center", f"{cx:g},{cy:g}",
           "--output", str(gcode), str(stl)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=SLICE_TIMEOUT_S, check=False)
        ok = proc.returncode == 0 and gcode.exists()
        if not ok:
            log.warning("slicer failed (%s): %s", proc.returncode, "\n".join(proc.stderr.splitlines()[-20:]))
    except subprocess.TimeoutExpired:
        ok = False
        log.warning("slicer timed out after %s s", SLICE_TIMEOUT_S)
    if not ok:
        return MvAbstain(stage="slice", reason="slicer_failed",
                         remedy="The slicer could not process this part. Check the model in the viewer.")
    seconds, grams = parse_gcode_stats(gcode.read_text(errors="ignore"))
    return SliceResult(stl, gcode, seconds, grams)
