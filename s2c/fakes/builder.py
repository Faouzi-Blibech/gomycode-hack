"""Stand-in for the geometry owner's CadQuery builder (s2c/builder.py's `build`
and `export`). Never runs CadQuery: `build` just wraps the spec, `export` writes
a fixed triangle mesh. Delete once the real builder module lands.
"""
from dataclasses import dataclass
from pathlib import Path

from s2c.partspec.models import PartSpec

_CUBE_STL = """solid fake
facet normal 0 0 1
 outer loop
  vertex 0 0 1
  vertex 1 0 1
  vertex 1 1 1
 endloop
endfacet
facet normal 0 0 1
 outer loop
  vertex 0 0 1
  vertex 1 1 1
  vertex 0 1 1
 endloop
endfacet
endsolid fake
"""


@dataclass
class FakeSolid:
    spec: PartSpec


def build(spec: PartSpec) -> FakeSolid:
    return FakeSolid(spec)


def export(solid: FakeSolid, out_dir: Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    step, stl = out_dir / "part.step", out_dir / "part.stl"
    step.write_text("ISO-10303-21; FAKE STEP\n")
    stl.write_text(_CUBE_STL)
    return step, stl
