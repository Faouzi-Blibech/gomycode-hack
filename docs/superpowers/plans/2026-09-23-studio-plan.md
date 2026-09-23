# Sketch-to-CAD Studio: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A professional Gradio Studio for the multi-view path: guided Capture → Review → Model & Export flow, Hugging Face–style parameters, and downloads in STL, STEP, 3MF, OBJ, GLB, PLY, BREP, Blender, DXF/SVG/PDF drawings, G-code and a zip.

**Architecture:** Shared backend modules (`settings`, `exporters`, `drawing`, `blend`, `print_settings`, `finish`, `artifacts`) sit beside the existing pipeline; a content-addressed cache means a format or print change never rebuilds geometry. The UI lives in `s2c/studio/` and calls plain handler methods that tests can drive without a browser.

**Tech Stack:** Python 3.11, uv, pytest, ruff; gradio 6.28, CadQuery 2.8 (OCP), trimesh 5.1, ezdxf 1.4.4, matplotlib, PrusaSlicer 2.9.6 CLI, optional Blender/bpy 4.2 in a separate process.

**Spec:** `docs/superpowers/specs/2026-09-23-studio-design.md` (extends `2026-09-22-multiview-gcode-design.md` and `2026-09-23-qwen-solaria-design.md`).

## Global Constraints

- Work in `gomycode-hack/`. Run with `uv run`. Line length 120. No lambda assignments (ruff E731).
- Do not modify `s2c/multiview/spec.py`, `s2c/multiview/build.py`, `s2c/partspec/`, or `fuse_envelope`. (spec §1)
- The model never writes code; no model output is executed. Numbers keep their provenance; nothing here may make an untrusted value pass the envelope gate. (multi-view spec §2)
- Units are millimetres everywhere, except GLB, which is in metres. (spec §6)
- Mesh presets are absolute tolerances on a shape copy: draft 0.1/0.5, normal 0.02/0.2, fine 0.005/0.1 (mm/rad). (spec §2)
- Subprocesses: argument lists, `encoding="utf-8", errors="replace"`, output to a log file, and a timeout that kills the process tree (slicer 120 s, Blender 90 s). (spec §10)
- Files live under `tmp/mv_gradio/<key>/` and are deleted one hour after last use. (spec §5)
- Unit tests never touch the network, the real slicer or real Blender. Mark real slicing `slicer` and real Blender `blender`. (spec §11)
- Implementer agents **do not commit and do not push**. The controller reviews, runs the whole suite and commits. Commit messages are plain, with no AI attribution. (CLAUDE.md)
- Each agent touches only the files its task lists, and runs only its own test files until the controller's wave check.

## Review Focus

1. A fillet or chamfer too big for a thin wall: the user sees a stop card with its remedy, and changing the size fixes it without starting over (Task 8 `test_a_finish_that_cannot_be_built_is_a_card`; Task 9 `test_a_failed_finish_keeps_the_review`).
2. Changing only print settings or formats must not rebuild geometry or rewrite existing files (Task 8 `test_new_formats_reuse_the_part_and_its_files`).
3. No slicer, or a part too big for the bed: every other format still downloads, and G-code is skipped with a warning (Task 8 `test_gcode_problems_never_block_the_other_files`).
4. Sizes typed as "42,5", "42 mm" or " 42 ": accepted; "abc", "0" or "-3" give a message, never an exception (Task 9 `test_sizes_accept_commas_and_units_and_reject_junk`).
5. Uploading a file that is not an image: a clear card in Capture, no crash (Task 9 `test_a_file_that_is_not_an_image_is_explained`).

## Waves and file ownership

| Wave | Task | Owns |
| --- | --- | --- |
| 1 | 1 Settings and process helper | `settings.py`, `proc.py`, `pyproject.toml`, `uv.lock` |
| 2 (parallel) | 2 Exporters | `exporters.py` |
| 2 | 3 Drawings | `drawing.py` |
| 2 | 4 Blender | `blend.py`, `scripts/setup_blender.ps1` |
| 2 | 5 Print settings and slicer | `print_settings.py`, `slice.py` |
| 2 | 6 Geometry finish | `finish.py` |
| 2 | 7 Seed, AI switches, snapping | `pipeline.py`, `complete.py`, `qwen_faces.py`, `fuse.py` |
| 3 | 8 Artifacts | `artifacts.py` |
| 4 | 9 Studio UI and examples | `s2c/studio/*`, `app_mv_studio.py`, `scripts/make_examples.py`, `examples/mv/sketches/*` |
| 5 | 10 API, CLI, docs, end-to-end | `routes.py`, `scripts/mv.py`, `README.md`, `docs/*` |

Each task's tests live in its own new test file (`tests/test_studio_<task>.py`) unless the task says otherwise.

---

### Task 1: Settings and process helper

**Files:**
- Create: `s2c/multiview/settings.py`, `s2c/multiview/proc.py`, `tests/test_studio_settings.py`
- Modify: `pyproject.toml` (dependencies, markers), `uv.lock`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `settings.MESH_TOLERANCES: dict[str, tuple[float, float]]`, `settings.MATERIALS: dict[str, tuple[int, int, int, int]]` (nozzle, first-layer nozzle, bed, first-layer bed), `settings.NOZZLES`, `settings.EDGE_LABELS`, `settings.FORMATS: dict[str, str]` (key → label, registry order)
  - `AiSettings`, `GeometrySettings`, `MeshSettings` (property `tolerances`), `PrintSettings`, `ExportSettings` (field `formats: list[str]`), `StudioSettings` (fields `ai`, `geometry`, `mesh`, `printing`, `export`)
  - `settings_hash(model) -> str` (16 hex chars)
  - `proc.run(cmd: list[str], timeout_s: float, log_path: Path, cwd: Path | None = None) -> int | None` (None = timed out, tree killed); `proc.tail(log_path, n=20) -> str`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_studio_settings.py
import sys
import time

import pytest
from pydantic import ValidationError

from s2c.multiview.proc import run, tail
from s2c.multiview.settings import (
    FORMATS,
    MATERIALS,
    ExportSettings,
    GeometrySettings,
    MeshSettings,
    PrintSettings,
    StudioSettings,
    settings_hash,
)


def test_defaults_are_valid_and_match_the_spec():
    s = StudioSettings()
    assert (s.ai.seed, s.ai.attempts, s.geometry.clearance, s.mesh.quality) == (7, 2, "medium", "normal")
    assert (s.printing.material, s.printing.layer_mm, s.printing.infill_pct, s.printing.supports) == \
        ("PLA", 0.2, 20, "buildplate")
    assert s.export.formats == ["stl", "step", "3mf", "gcode"]
    assert MeshSettings(quality="fine").tolerances == (0.005, 0.1)
    assert MATERIALS["PETG"][0] == 240


def test_ranges_are_enforced():
    for bad in ({"layer_mm": 0.4}, {"nozzle_mm": 0.5}, {"infill_pct": 101}, {"scale_pct": 20}, {"material": "PVC"}):
        with pytest.raises(ValidationError):
            PrintSettings(**bad)
    with pytest.raises(ValidationError):
        GeometrySettings(finish_mm=50)
    with pytest.raises(ValidationError):
        ExportSettings(formats=["stl", "xyz"])


def test_a_layer_must_fit_the_nozzle():
    assert PrintSettings(nozzle_mm=0.6, layer_mm=0.32).layer_mm == 0.32
    with pytest.raises(ValidationError):
        PrintSettings(nozzle_mm=0.2, layer_mm=0.2)


def test_formats_are_kept_in_registry_order_without_duplicates():
    assert ExportSettings(formats=["gcode", "stl", "stl", "pdf"]).formats == ["stl", "pdf", "gcode"]
    assert list(FORMATS)[:3] == ["stl", "step", "3mf"]


def test_the_hash_is_stable_and_changes_with_values():
    assert settings_hash(PrintSettings()) == settings_hash(PrintSettings())
    assert settings_hash(PrintSettings()) != settings_hash(PrintSettings(infill_pct=40))
    assert len(settings_hash(GeometrySettings())) == 16


def test_run_writes_output_to_the_log(tmp_path):
    code = run([sys.executable, "-c", "print('hello from a child')"], 30, tmp_path / "log.txt")
    assert code == 0 and "hello from a child" in tail(tmp_path / "log.txt")


def test_run_kills_a_program_that_hangs(tmp_path):
    t0 = time.monotonic()
    code = run([sys.executable, "-c", "import time; time.sleep(60)"], 1, tmp_path / "log.txt")
    assert code is None and time.monotonic() - t0 < 15
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_studio_settings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 's2c.multiview.proc'` (or `settings`).

- [ ] **Step 3: Write `settings.py`**

```python
"""Studio settings shared by the UI, the API and the command line. Spec 2026-09-23-studio section 4.
Every model has defaults and pydantic ranges, so StudioSettings() is always valid."""
from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Quality = Literal["draft", "normal", "fine"]
Clearance = Literal["fine", "medium", "coarse"]
Material = Literal["PLA", "PETG", "ABS", "ASA", "TPU"]
Supports = Literal["off", "buildplate", "everywhere"]
InfillPattern = Literal["grid", "gyroid", "rectilinear", "honeycomb", "cubic", "lightning"]
FinishKind = Literal["none", "fillet", "chamfer"]
EdgeChoice = Literal["all_vertical", "top", "bottom", "all"]

MESH_TOLERANCES: dict[str, tuple[float, float]] = {"draft": (0.1, 0.5), "normal": (0.02, 0.2), "fine": (0.005, 0.1)}
# material -> (nozzle, first-layer nozzle, bed, first-layer bed), degrees C
MATERIALS: dict[str, tuple[int, int, int, int]] = {
    "PLA": (210, 215, 60, 60), "PETG": (240, 240, 80, 80), "ABS": (250, 255, 100, 100),
    "ASA": (255, 260, 100, 100), "TPU": (225, 225, 50, 50),
}
NOZZLES = (0.2, 0.4, 0.6, 0.8)
EDGE_LABELS = {"all_vertical": "Outline corners", "top": "Front-face edges", "bottom": "Back-face edges",
               "all": "All edges"}
FORMATS: dict[str, str] = {
    "stl": "STL mesh", "step": "STEP (CAD)", "3mf": "3MF (slicers)", "obj": "OBJ mesh", "glb": "GLB (web, AR)",
    "ply": "PLY mesh", "brep": "BREP (OpenCascade)", "blend": "Blender", "dxf": "DXF drawing",
    "svg": "SVG drawing", "pdf": "PDF drawing", "gcode": "G-code",
}


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AiSettings(_Model):
    use_reader: bool = True
    use_qwen_image: bool = True
    use_rescue: bool = True
    use_triposr: bool = True
    use_solaria: bool = True
    seed: int = Field(7, ge=0, le=2**31 - 1)
    randomize_seed: bool = False
    attempts: int = Field(2, ge=1, le=4)


class GeometrySettings(_Model):
    snap: bool = True
    clearance: Clearance = "medium"
    finish: FinishKind = "none"
    finish_mm: float = Field(1.0, ge=0.2, le=10.0)
    finish_edges: EdgeChoice = "all_vertical"


class MeshSettings(_Model):
    quality: Quality = "normal"

    @property
    def tolerances(self) -> tuple[float, float]:
        return MESH_TOLERANCES[self.quality]


class PrintSettings(_Model):
    material: Material = "PLA"
    nozzle_mm: float = 0.4
    layer_mm: float = Field(0.2, ge=0.05, le=0.32)
    infill_pct: int = Field(20, ge=0, le=100)
    infill_pattern: InfillPattern = "grid"
    perimeters: int = Field(3, ge=1, le=8)
    supports: Supports = "buildplate"
    brim_mm: float = Field(0.0, ge=0.0, le=10.0)
    scale_pct: float = Field(100.0, ge=50.0, le=200.0)

    @field_validator("nozzle_mm")
    @classmethod
    def _known_nozzle(cls, v: float) -> float:
        if not any(abs(v - n) < 1e-9 for n in NOZZLES):
            raise ValueError(f"nozzle must be one of {NOZZLES} mm")
        return v

    @model_validator(mode="after")
    def _layer_fits_nozzle(self) -> PrintSettings:
        if self.layer_mm > 0.75 * self.nozzle_mm + 1e-9:
            raise ValueError(f"layer height {self.layer_mm:g} mm is more than 0.75 x the {self.nozzle_mm:g} mm nozzle")
        return self


class ExportSettings(_Model):
    formats: list[str] = Field(default_factory=lambda: ["stl", "step", "3mf", "gcode"])

    @field_validator("formats")
    @classmethod
    def _known_formats(cls, v: list[str]) -> list[str]:
        unknown = sorted(set(v) - set(FORMATS))
        if unknown:
            raise ValueError(f"unknown formats {unknown}; choose from {list(FORMATS)}")
        return [f for f in FORMATS if f in v]


class StudioSettings(_Model):
    ai: AiSettings = Field(default_factory=AiSettings)
    geometry: GeometrySettings = Field(default_factory=GeometrySettings)
    mesh: MeshSettings = Field(default_factory=MeshSettings)
    printing: PrintSettings = Field(default_factory=PrintSettings)
    export: ExportSettings = Field(default_factory=ExportSettings)


def settings_hash(model: BaseModel) -> str:
    body = json.dumps(model.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()[:16]
```

- [ ] **Step 4: Write `proc.py`**

```python
"""Run an outside program safely: argument list, output to a log file, and a timeout that kills the whole
process tree (a Windows grandchild can otherwise keep a pipe open forever). Spec 2026-09-23-studio section 10."""
from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path


def _kill_tree(proc: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, check=False)
    else:
        os.killpg(proc.pid, signal.SIGKILL)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def run(cmd: list[str], timeout_s: float, log_path: Path, cwd: Path | None = None) -> int | None:
    """The exit code, or None when the program ran past `timeout_s` and was killed."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    with log_path.open("w", encoding="utf-8", errors="replace") as out:
        proc = subprocess.Popen([str(c) for c in cmd], stdout=out, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, cwd=cwd, creationflags=flags,
                                start_new_session=os.name != "nt")
        try:
            return proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            return None


def tail(log_path: Path, n: int = 20) -> str:
    try:
        lines = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_studio_settings.py -v`
Expected: PASS (7 tests).

- [ ] **Step 6: Dependencies and marker**

In `pyproject.toml`, add to `[project] dependencies` (all already installed in the venv): `"trimesh>=4.4",`, `"ezdxf>=1.1",`, `"matplotlib>=3.8",`. Keep `trimesh` in the `ai` extra too. Add the marker line `"blender: needs Blender or bpy configured (BLENDER_PATH or BLENDER_PYTHON)",` to `markers`.

Run: `uv lock` then `uv sync --extra ai`
Expected: no package removed.

- [ ] **Step 7: Controller commit** (implementer stops here and reports)

```bash
git add s2c/multiview/settings.py s2c/multiview/proc.py tests/test_studio_settings.py pyproject.toml uv.lock
git commit -m "Add shared Studio settings and a safe subprocess runner"
```

---

### Task 2: Exporters

**Files:**
- Create: `s2c/multiview/exporters.py`, `tests/test_studio_exporters.py`

**Interfaces:**
- Consumes: `settings.MESH_TOLERANCES`.
- Produces: `MESH_FORMATS: tuple[str, ...]` (`stl step 3mf obj glb ply brep`), `FILE_NAMES: dict[str, str]`, `PART_COLOUR`, `write_stl/write_step/write_3mf/write_obj/write_glb/write_ply/write_brep(solid, path, quality="normal") -> Path`, `mesh_of(solid, quality) -> trimesh.Trimesh`, `export_mesh_formats(solid, out_dir, formats, quality="normal") -> dict[str, Path]` (raises `ValueError` on an unknown format). `solid` is a CadQuery `Workplane` or `Shape`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_studio_exporters.py
import zipfile
from pathlib import Path

import cadquery as cq
import pytest
import trimesh

from s2c.multiview.build import build, volume
from s2c.multiview.exporters import FILE_NAMES, MESH_FORMATS, export_mesh_formats, mesh_of
from s2c.multiview.spec import MultiViewSpec

SPEC = MultiViewSpec.model_validate_json(
    (Path(__file__).parents[1] / "examples" / "mv" / "l_bracket.json").read_text())


@pytest.fixture(scope="module")
def part():
    return build(SPEC)


@pytest.fixture(scope="module")
def files(part, tmp_path_factory):
    return export_mesh_formats(part, tmp_path_factory.mktemp("exports"), MESH_FORMATS, "normal")


def test_every_format_is_written_with_its_name(files):
    assert set(files) == set(MESH_FORMATS)
    for fmt, path in files.items():
        assert path.name == FILE_NAMES[fmt] and path.stat().st_size > 0


def test_stl_is_watertight_with_the_right_volume(part, files):
    mesh = trimesh.load(str(files["stl"]), force="mesh")
    assert mesh.is_watertight and mesh.volume == pytest.approx(volume(part), rel=0.005)


def test_step_and_brep_reload_exactly(part, files):
    assert cq.importers.importStep(str(files["step"])).val().Volume() == pytest.approx(volume(part), rel=0.001)
    assert cq.Shape.importBrep(str(files["brep"])).Volume() == pytest.approx(volume(part), rel=0.001)


def test_3mf_holds_a_model(files):
    with zipfile.ZipFile(files["3mf"]) as z:
        assert "3D/3dmodel.model" in z.namelist()


def test_obj_and_ply_reload_with_the_right_volume(part, files):
    for fmt in ("obj", "ply"):
        mesh = trimesh.load(str(files[fmt]), force="mesh")
        assert mesh.volume == pytest.approx(volume(part), rel=0.005), fmt


def test_glb_is_in_metres(files):
    extents = sorted(trimesh.load(str(files["glb"])).extents)
    assert extents == pytest.approx(sorted([0.050, 0.030, 0.020]), rel=0.01)


def test_quality_is_honoured_even_after_a_finer_mesh(part):
    fine = len(mesh_of(part, "fine").faces)
    draft = len(mesh_of(part, "draft").faces)
    assert draft < fine


def test_an_unknown_format_is_refused(part, tmp_path):
    with pytest.raises(ValueError):
        export_mesh_formats(part, tmp_path, ["xyz"])
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_studio_exporters.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 's2c.multiview.exporters'`.

- [ ] **Step 3: Write `exporters.py`**

```python
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
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_studio_exporters.py -v`
Expected: PASS (8 tests). If a CadQuery keyword differs in 2.8 (for example `exportStl`'s `relative`), read the installed signature with `inspect.signature` and keep absolute tolerances; record the ruling.

- [ ] **Step 5: Controller commit**

```bash
git add s2c/multiview/exporters.py tests/test_studio_exporters.py
git commit -m "Export STL, STEP, 3MF, OBJ, GLB, PLY and BREP with quality presets"
```

---

### Task 3: Three-view drawing (DXF, SVG, PDF)

**Files:**
- Create: `s2c/multiview/drawing.py`, `tests/test_studio_drawing.py`

**Interfaces:**
- Consumes: a CadQuery solid and its `MultiViewSpec`.
- Produces: `DRAWING_FORMATS = ("dxf", "svg", "pdf")`, `DRAWING_FILES: dict[str, str]`, `drawing_document(solid, spec) -> ezdxf.document.Drawing`, `write_drawings(solid, spec, out_dir, formats) -> dict[str, Path]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_studio_drawing.py
from pathlib import Path

import ezdxf
import pytest
from ezdxf import bbox

from s2c.multiview.build import build
from s2c.multiview.drawing import DRAWING_FILES, write_drawings
from s2c.multiview.spec import MultiViewSpec

SPEC = MultiViewSpec.model_validate_json(
    (Path(__file__).parents[1] / "examples" / "mv" / "l_bracket.json").read_text())


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    return write_drawings(build(SPEC), SPEC, tmp_path_factory.mktemp("drawing"), ["dxf", "svg", "pdf"])


def test_the_dxf_is_clean_and_in_millimetres(files):
    doc = ezdxf.readfile(files["dxf"])
    assert not doc.audit().has_errors
    assert doc.header["$INSUNITS"] == 4
    assert files["dxf"].name == DRAWING_FILES["dxf"]


def test_three_views_with_visible_and_hidden_lines(files):
    msp = ezdxf.readfile(files["dxf"]).modelspace()
    layers = {e.dxf.layer for e in msp}
    assert {"VISIBLE", "HIDDEN", "DIMENSIONS"} <= layers
    ext = bbox.extents(msp.query('*[layer=="VISIBLE"]'))
    assert ext.size.x > 50 + 20 and ext.size.y > 30 + 20  # front + right side by side, top above front


def test_overall_sizes_and_hole_diameters_are_dimensioned(files):
    msp = ezdxf.readfile(files["dxf"]).modelspace()
    measured = {round(d.get_measurement(), 1) for d in msp.query("DIMENSION")}
    assert {50.0, 30.0, 20.0, 5.5} <= measured


def test_svg_and_pdf_are_rendered(files):
    assert "<svg" in files["svg"].read_text(encoding="utf-8")[:500]
    assert files["pdf"].read_bytes()[:5] == b"%PDF-"


def test_only_the_requested_formats_are_written(tmp_path):
    assert set(write_drawings(build(SPEC), SPEC, tmp_path, ["svg"])) == {"svg"}
    assert write_drawings(build(SPEC), SPEC, tmp_path, ["stl"]) == {}
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_studio_drawing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 's2c.multiview.drawing'`.

- [ ] **Step 3: Write `drawing.py`**

```python
"""Three-view engineering drawing of the final solid: hidden-line projection (OCCT HLR) of the front, top and right
views in a third-angle layout, overall dimensions, hole diameters and a title block. DXF in millimetres; SVG and
PDF are rendered from the DXF. Spec 2026-09-23-studio section 6."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import cadquery as cq
import ezdxf
from cadquery.occ_impl.exporters.dxf import DxfDocument
from ezdxf.addons.drawing import Frontend, RenderContext, layout, svg
from ezdxf.addons.drawing.config import BackgroundPolicy, ColorPolicy, Configuration
from matplotlib.figure import Figure
from OCP.BRepLib import BRepLib
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCP.HLRAlgo import HLRAlgo_Projector
from OCP.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape

from s2c.multiview.spec import FaceHole, MultiViewSpec

DRAWING_FORMATS = ("dxf", "svg", "pdf")
DRAWING_FILES = {"dxf": "drawing.dxf", "svg": "drawing.svg", "pdf": "drawing.pdf"}
# face -> (view direction, drawing x direction). Face frames of the multi-view spec section 3: after shifting by the
# envelope, a point (a, b) of that face lands at (a, b) in its view.
VIEWS = {"front": ((0, 0, 1), (1, 0, 0)), "top": ((0, 1, 0), (1, 0, 0)), "right": ((1, 0, 0), (0, 0, -1))}
DIM_STYLE = {"dimtxt": 3.5, "dimlfac": 1, "dimasz": 2.5, "dimexo": 1.5}  # the EZDXF style scales by 100 otherwise
RENDER = Configuration(background_policy=BackgroundPolicy.WHITE, color_policy=ColorPolicy.BLACK)


def _collect(*compounds) -> cq.Shape | None:
    shapes = [c for c in compounds if c is not None and not c.IsNull()]
    if not shapes:
        return None
    for c in shapes:
        BRepLib.BuildCurves3d_s(c, 1e-6)  # HLR edges carry only 2D curves
    return cq.Compound.makeCompound([cq.Shape.cast(c) for c in shapes])


def _project(shape: cq.Shape, direction, xdir) -> tuple[cq.Shape | None, cq.Shape | None]:
    algo = HLRBRep_Algo()
    algo.Add(shape.wrapped)
    algo.Projector(HLRAlgo_Projector(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(*direction), gp_Dir(*xdir))))
    algo.Update()
    algo.Hide()
    hlr = HLRBRep_HLRToShape(algo)
    return _collect(hlr.VCompound(), hlr.OutLineVCompound()), _collect(hlr.HCompound(), hlr.OutLineHCompound())


def _linear(msp, p1, p2, base, angle=0.0) -> None:
    msp.add_linear_dim(base=base, p1=p1, p2=p2, angle=angle, dimstyle="EZDXF", override=DIM_STYLE,
                       dxfattribs={"layer": "DIMENSIONS"}).render()


def drawing_document(solid, spec: MultiViewSpec) -> ezdxf.document.Drawing:
    shape = solid.val() if isinstance(solid, cq.Workplane) else solid
    env = spec.envelope
    w, h, d = env.x_mm, env.y_mm, env.z_mm
    gap = max(15.0, 0.3 * max(w, h, d))
    shift = {"front": (0.0, 0.0), "top": (0.0, d), "right": (d, 0.0)}      # envelope corner to (0, 0)
    place = {"front": (0.0, 0.0), "top": (0.0, h + gap), "right": (w + gap, 0.0)}  # third-angle layout
    dxf = DxfDocument(setup=True)  # linetypes and the EZDXF dimension style
    dxf.add_layer("VISIBLE", color=7)
    dxf.add_layer("HIDDEN", color=8, linetype="DASHED")
    dxf.add_layer("DIMENSIONS", color=1)
    dxf.add_layer("TEXT", color=7)
    for face, (direction, xdir) in VIEWS.items():
        dx, dy = shift[face][0] + place[face][0], shift[face][1] + place[face][1]
        visible, hidden = _project(shape, direction, xdir)
        for part, layer in ((visible, "VISIBLE"), (hidden, "HIDDEN")):
            if part is not None:
                dxf.add_shape(cq.Workplane("XY").add(part.translate(cq.Vector(dx, dy, 0))), layer)
    doc = dxf.document
    doc.header["$INSUNITS"] = 4
    msp = doc.modelspace()
    _linear(msp, (0, 0), (w, 0), (0, -10))                               # width under the front view
    _linear(msp, (0, 0), (0, h), (-10, 0), angle=90)                     # height left of the front view
    _linear(msp, (w + gap, 0), (w + gap + d, 0), (w + gap, -10))         # depth under the right view
    notes = []
    for k, f in enumerate(spec.features):
        if isinstance(f, FaceHole) and f.face in place:
            cx, cy = place[f.face][0] + f.a_mm, place[f.face][1] + f.b_mm
            msp.add_diameter_dim(center=(cx, cy), radius=f.diameter_mm / 2, angle=45, dimstyle="EZDXF",
                                 override=DIM_STYLE, dxfattribs={"layer": "DIMENSIONS"}).render()
        else:
            notes.append(f"Feature {k + 1} on the {f.face} face")
    lines = ["Sketch-to-CAD", f"Envelope {w:g} x {h:g} x {d:g} mm", "Third-angle projection, units mm",
             f"Date {date.today().isoformat()}", *notes]
    x0, y0 = w + gap, h + gap + d
    for i, text in enumerate(lines):
        msp.add_text(text, height=3.5 if i else 5.0, dxfattribs={"layer": "TEXT"}).set_placement((x0, y0 - 7 * i))
    return doc


def _svg(doc) -> str:
    backend = svg.SVGBackend()
    Frontend(RenderContext(doc), backend, config=RENDER).draw_layout(doc.modelspace())
    return backend.get_string(layout.Page(297, 210, layout.Units.mm, margins=layout.Margins.all(10)))


def _pdf(doc, path: Path) -> None:
    from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
    fig = Figure(figsize=(11.69, 8.27))  # A4 landscape; no pyplot, so no GUI backend is involved
    ax = fig.add_axes((0.03, 0.03, 0.94, 0.94))
    ax.set_axis_off()
    Frontend(RenderContext(doc), MatplotlibBackend(ax), config=RENDER).draw_layout(doc.modelspace(), finalize=True)
    fig.savefig(str(path), format="pdf")


def write_drawings(solid, spec: MultiViewSpec, out_dir: Path, formats) -> dict[str, Path]:
    wanted = [f for f in DRAWING_FORMATS if f in formats]
    if not wanted:
        return {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = drawing_document(solid, spec)
    files = {f: out_dir / DRAWING_FILES[f] for f in wanted}
    if "dxf" in files:
        doc.saveas(files["dxf"])
    if "svg" in files:
        files["svg"].write_text(_svg(doc), encoding="utf-8")
    if "pdf" in files:
        _pdf(doc, files["pdf"])
    return files
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_studio_drawing.py -v`
Expected: PASS (5 tests). OCP and ezdxf signatures were verified by the CAD advisor (HLR snippet, `DxfDocument.add_shape` on a Workplane, `dimlfac=1`); if one differs, fix it against the installed library and record the ruling. Keep the layer names and file names.

- [ ] **Step 5: Controller commit**

```bash
git add s2c/multiview/drawing.py tests/test_studio_drawing.py
git commit -m "Draw three hidden-line views with dimensions as DXF, SVG and PDF"
```

---

### Task 4: Blender export

**Files:**
- Create: `s2c/multiview/blend.py`, `scripts/setup_blender.ps1`, `tests/test_studio_blend.py`

**Interfaces:**
- Consumes: `proc.run`, `proc.tail` (Task 1); an OBJ written by Task 2's `write_obj`.
- Produces: `SCRIPT: str` (the bpy 4.2 script), `KIT_NAME = "part_blender_kit.zip"`, `KIT_WARNING`, `blender_runner() -> list[str] | None`, `write_blend(obj_path, out_dir) -> tuple[Path, list[str]]` (a `.blend`, or the kit zip with `KIT_WARNING`; returns an existing file without rerunning).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_studio_blend.py
import os
import sys
import zipfile

import pytest

from s2c.multiview import blend
from s2c.multiview.blend import KIT_NAME, KIT_WARNING, blender_runner, write_blend


@pytest.fixture
def obj(tmp_path):
    path = tmp_path / "part.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    return path


def test_without_blender_a_kit_is_written(obj, tmp_path, monkeypatch):
    monkeypatch.setattr(blend, "blender_runner", lambda: None)
    path, warnings = write_blend(obj, tmp_path / "out")
    assert path.name == KIT_NAME and warnings == [KIT_WARNING]
    with zipfile.ZipFile(path) as z:
        assert sorted(z.namelist()) == ["open_in_blender.py", "part.obj"]
        assert "obj_import" in z.read("open_in_blender.py").decode()


def test_with_a_blender_python_the_script_runs_with_both_paths(obj, tmp_path, monkeypatch):
    fake = ("import sys\nargv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]\n"
            "open(argv[1], 'wb').write(b'BLENDER-v402' + open(argv[0], 'rb').read()[:1])\n")
    monkeypatch.setattr(blend, "SCRIPT", fake)
    monkeypatch.setattr(blend, "blender_runner", lambda: [sys.executable])
    path, warnings = write_blend(obj, tmp_path / "out")
    assert path.name == "part.blend" and warnings == []
    assert path.read_bytes().startswith(b"BLENDER")


def test_a_failing_blender_falls_back_to_the_kit(obj, tmp_path, monkeypatch):
    monkeypatch.setattr(blend, "SCRIPT", "raise SystemExit(3)\n")
    monkeypatch.setattr(blend, "blender_runner", lambda: [sys.executable])
    path, warnings = write_blend(obj, tmp_path / "out")
    assert path.name == KIT_NAME and any("Blender failed" in w for w in warnings)


def test_the_runner_comes_from_the_environment(tmp_path, monkeypatch):
    for key in ("BLENDER_PATH", "BLENDER_PYTHON"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(blend, "VENDOR_PYTHON", tmp_path / "missing.exe")
    assert blender_runner() is None
    exe = tmp_path / "blender.exe"
    exe.write_text("")
    monkeypatch.setenv("BLENDER_PATH", str(exe))
    assert blender_runner() == [str(exe), "-b", "--factory-startup", "--python"]


@pytest.mark.blender
@pytest.mark.skipif(blender_runner() is None, reason="Blender or bpy not configured")
def test_a_real_blend_file(obj, tmp_path):
    path, warnings = write_blend(obj, tmp_path / "out")
    assert path.suffix == ".blend" and path.read_bytes()[:7] in (b"BLENDER", b"\x28\xb5\x2f\xfd"[:4] + b"\x00" * 3)
    assert os.path.getsize(path) > 1000
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_studio_blend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 's2c.multiview.blend'`.

- [ ] **Step 3: Write `blend.py`**

```python
"""Blender output. A real .blend is written by a separate Blender or bpy process and never inside the app: bpy is
about 300 MB, Python 3.11 only, and may clash on numpy. Without one, a "Blender kit" zip holds the OBJ and the
import script. Spec 2026-09-23-studio sections 2 and 6."""
from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

from s2c.multiview.proc import run, tail

BLEND_TIMEOUT_S = 90
KIT_NAME = "part_blender_kit.zip"
KIT_WARNING = "Blender not configured: the download holds part.obj and open_in_blender.py instead of a .blend"
VENDOR_PYTHON = Path(__file__).resolve().parents[2] / "vendor" / "bpy-env" / (
    "Scripts/python.exe" if os.name == "nt" else "bin/python")
SCRIPT = '''"""Import part.obj into an empty Blender scene in millimetres and save it as a .blend.
Run: blender -b --python open_in_blender.py -- part.obj part.blend"""
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


def _kit(obj_path: Path, out_dir: Path, warnings: list[str]) -> tuple[Path, list[str]]:
    kit = out_dir / KIT_NAME
    if not kit.exists():
        tmp = kit.with_suffix(".part")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(obj_path, arcname="part.obj")
            z.writestr("open_in_blender.py", SCRIPT)
        os.replace(tmp, kit)
    return kit, [*warnings, KIT_WARNING] if warnings else [KIT_WARNING]


def write_blend(obj_path: Path, out_dir: Path) -> tuple[Path, list[str]]:
    obj_path, out_dir = Path(obj_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "part.blend"
    if target.exists():
        return target, []
    runner = blender_runner()
    if runner is None:
        return _kit(obj_path, out_dir, [])
    script = out_dir / "open_in_blender.py"
    script.write_text(SCRIPT, encoding="utf-8")
    log = out_dir / "blender.log"
    code = run([*runner, str(script), "--", str(obj_path), str(target)], BLEND_TIMEOUT_S, log)
    if code == 0 and target.exists():
        return target, []
    reason = "timed out" if code is None else f"exit code {code}"
    print(tail(log), file=sys.stderr)
    return _kit(obj_path, out_dir, [f"Blender failed ({reason}); see blender.log"])
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_studio_blend.py -v`
Expected: 4 PASS, 1 SKIPPED (no Blender configured).

- [ ] **Step 5: Optional real Blender, isolated**

```powershell
# scripts/setup_blender.ps1 — installs bpy 4.2 into vendor/bpy-env (about 300 MB), separate from the app's venv
uv venv vendor/bpy-env --python 3.11
uv pip install --python vendor/bpy-env/Scripts/python.exe "bpy==4.2.23"
```

Try it once: `powershell -File scripts/setup_blender.ps1`, then `uv run pytest tests/test_studio_blend.py -v -m blender`. Expected: the real test PASSES. If the download or install fails, record it in the report and continue: the kit path is the supported fallback.

- [ ] **Step 6: Controller commit**

```bash
git add s2c/multiview/blend.py scripts/setup_blender.ps1 tests/test_studio_blend.py
git commit -m "Write a Blender file through a separate process, or a Blender kit"
```

---

### Task 5: Print settings and slicer overrides

**Files:**
- Create: `s2c/multiview/print_settings.py`, `tests/test_studio_print.py`
- Modify: `s2c/multiview/slice.py` (`slice_solid` and imports only)

**Interfaces:**
- Consumes: `PrintSettings`, `MATERIALS` (Task 1); `proc.run`, `proc.tail` (Task 1).
- Produces: `print_settings.slicer_flags(p: PrintSettings) -> list[str]`; `slice.slice_solid(solid, out_dir, profile_path=None, slicer=None, overrides: list[str] | None = None, scale: float = 1.0) -> SliceResult | MvAbstain` (existing callers unchanged); `slice.DATADIR`.

- [ ] **Step 1: Check the flags against the bundled slicer**

Run: `vendor/PrusaSlicer-2.9.6/prusa-slicer-console.exe --help-fff | findstr /R "layer-height fill-density fill-pattern perimeters support-material brim-width nozzle-diameter filament-type bed-temperature"` and `... --help | findstr /R "threads datadir"`
Expected: every flag used below is listed. If one is not, record a ruling and drop it.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_studio_print.py
import pytest
import trimesh

from s2c.multiview import slice as slicing
from s2c.multiview.build import build
from s2c.multiview.print_settings import slicer_flags
from s2c.multiview.settings import PrintSettings
from s2c.multiview.slice import find_slicer, slice_solid
from s2c.multiview.spec import MvAbstain
from tests.mv_helpers import make_spec


def value(flags, name):
    return flags[flags.index(name) + 1]


def test_defaults_become_pla_flags():
    f = slicer_flags(PrintSettings())
    assert value(f, "--fill-density") == "20%" and value(f, "--layer-height") == "0.2"
    assert value(f, "--temperature") == "210" and value(f, "--bed-temperature") == "60"
    assert "--support-material" in f and "--support-material-buildplate-only" in f


def test_every_choice_maps_to_a_flag():
    f = slicer_flags(PrintSettings(material="PETG", nozzle_mm=0.6, layer_mm=0.3, infill_pct=45,
                                   infill_pattern="gyroid", perimeters=5, supports="off", brim_mm=4))
    assert value(f, "--filament-type") == "PETG" and value(f, "--temperature") == "240"
    assert value(f, "--nozzle-diameter") == "0.6" and value(f, "--first-layer-height") == "0.3"
    assert value(f, "--fill-pattern") == "gyroid" and value(f, "--perimeters") == "5"
    assert value(f, "--brim-width") == "4" and "--no-support-material" in f and "--support-material" not in f
    everywhere = slicer_flags(PrintSettings(supports="everywhere"))
    assert "--no-support-material-buildplate-only" in everywhere


def test_the_first_layer_fits_a_small_nozzle():
    f = slicer_flags(PrintSettings(nozzle_mm=0.2, layer_mm=0.1))
    assert float(value(f, "--first-layer-height")) <= 0.15 + 1e-9


def fake_slicer(calls, write=True):
    def run(cmd, timeout_s, log_path, cwd=None):
        calls.append(cmd)
        if write:
            out = cmd[cmd.index("--output") + 1]
            with open(out, "w", encoding="utf-8") as g:
                g.write("G1 X1\n; filament used [g] = 3.21\n; estimated printing time (normal mode) = 12m 5s\n")
        return 0
    return run


def test_overrides_threads_and_datadir_reach_the_command(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(slicing, "run", fake_slicer(calls))
    res = slice_solid(build(make_spec((60.0, 40.0, 5.0))), tmp_path, slicer=tmp_path / "slicer.exe",
                      overrides=slicer_flags(PrintSettings(infill_pct=35)))
    cmd = calls[0]
    assert value(cmd, "--fill-density") == "35%" and value(cmd, "--threads") == "4" and "--datadir" in cmd
    assert cmd.index("--load") < cmd.index("--fill-density")
    assert (res.print_time_s, res.filament_g) == (725.0, 3.21)


def test_scale_applies_before_the_bed_check(tmp_path, monkeypatch):
    monkeypatch.setattr(slicing, "run", fake_slicer([]))
    res = slice_solid(build(make_spec((60.0, 40.0, 5.0))), tmp_path, slicer=tmp_path / "s.exe", scale=2.0)
    mesh = trimesh.load(str(res.print_stl), force="mesh")
    assert sorted(mesh.extents) == pytest.approx([10.0, 80.0, 120.0], abs=0.05)
    too_big = slice_solid(build(make_spec((150.0, 40.0, 5.0))), tmp_path / "b", slicer=tmp_path / "s.exe", scale=2.0)
    assert isinstance(too_big, MvAbstain) and too_big.reason == "too_big_for_bed"


def test_a_hung_slicer_is_a_clean_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(slicing, "run", lambda cmd, timeout_s, log_path, cwd=None: None)
    res = slice_solid(build(make_spec((60.0, 40.0, 5.0))), tmp_path, slicer=tmp_path / "s.exe")
    assert isinstance(res, MvAbstain) and res.reason == "slicer_failed"


@pytest.mark.slicer
@pytest.mark.skipif(find_slicer() is None, reason="PrusaSlicer not installed")
def test_thicker_layers_mean_fewer_layers(tmp_path):
    solid = build(make_spec((60.0, 40.0, 6.0)))
    thin = slice_solid(solid, tmp_path / "a", overrides=slicer_flags(PrintSettings(layer_mm=0.2)))
    thick = slice_solid(solid, tmp_path / "b", overrides=slicer_flags(PrintSettings(layer_mm=0.3)))
    count = [r.gcode.read_text(encoding="utf-8").count(";LAYER_CHANGE") for r in (thin, thick)]
    assert count[1] < count[0]
```

- [ ] **Step 3: Run them to see them fail**

Run: `uv run pytest tests/test_studio_print.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 's2c.multiview.print_settings'`.

- [ ] **Step 4: Write `print_settings.py`**

```python
"""PrintSettings -> PrusaSlicer 2.9 command-line overrides, from a whitelist; values are validated by pydantic before
they get here, because a bad enum makes the slicer print its help and write no G-code. Spec 2026-09-23-studio §4."""
from __future__ import annotations

from s2c.multiview.settings import MATERIALS, PrintSettings


def slicer_flags(p: PrintSettings) -> list[str]:
    nozzle, first_nozzle, bed, first_bed = MATERIALS[p.material]
    first_layer = min(max(p.layer_mm, 0.2), 0.75 * p.nozzle_mm)
    flags = [
        "--filament-type", p.material,
        "--temperature", str(nozzle), "--first-layer-temperature", str(first_nozzle),
        "--bed-temperature", str(bed), "--first-layer-bed-temperature", str(first_bed),
        "--nozzle-diameter", f"{p.nozzle_mm:g}",
        "--layer-height", f"{p.layer_mm:g}", "--first-layer-height", f"{first_layer:g}",
        "--fill-density", f"{p.infill_pct}%", "--fill-pattern", p.infill_pattern,
        "--perimeters", str(p.perimeters), "--brim-width", f"{p.brim_mm:g}",
    ]
    if p.supports == "off":
        flags.append("--no-support-material")
    else:
        flags += ["--support-material", "--support-material-auto"]
        flags.append("--support-material-buildplate-only" if p.supports == "buildplate"
                     else "--no-support-material-buildplate-only")
    return flags
```

- [ ] **Step 5: Change `slice.py`**

Replace `import subprocess` with `import tempfile` and add `from s2c.multiview.proc import run, tail`. After `SLICE_TIMEOUT_S = 120` add:

```python
DATADIR = Path(tempfile.gettempdir()) / "s2c-prusaslicer"  # never read the user's own PrusaSlicer presets
SLICE_THREADS = 4
```

Replace `slice_solid` with:

```python
def slice_solid(solid: cq.Workplane, out_dir: Path, profile_path: Path | None = None, slicer: Path | None = None,
                overrides: list[str] | None = None, scale: float = 1.0) -> SliceResult | MvAbstain:
    """`overrides` are PrusaSlicer flags applied on top of the profile; `scale` resizes the print only."""
    profile = load_profile(profile_path)
    if scale != 1.0:
        solid = cq.Workplane("XY").add(solid.val().scale(scale))
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
    cmd = [str(slicer), "--export-gcode", "--load", str(profile.path), *(overrides or []),
           "--center", f"{cx:g},{cy:g}", "--threads", str(SLICE_THREADS), "--datadir", str(DATADIR),
           "--output", str(gcode), str(stl)]
    log_path = out_dir / "slicer.log"
    code = run(cmd, SLICE_TIMEOUT_S, log_path)
    if code != 0 or not gcode.exists():
        why = "timed out" if code is None else f"exit code {code}"
        log.warning("slicer failed (%s): %s", why, tail(log_path))
        return MvAbstain(stage="slice", reason="slicer_failed",
                         remedy="The slicer could not process this part. Check the model in the viewer.")
    seconds, grams = parse_gcode_stats(gcode.read_text(encoding="utf-8", errors="replace"))
    return SliceResult(stl, gcode, seconds, grams)
```

Also change `load_profile`'s `path.read_text()` to `path.read_text(encoding="utf-8")`.

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/test_studio_print.py tests/test_mv_slice.py -v`
Expected: all PASS (the real-slicer tests run too, because PrusaSlicer is vendored).

- [ ] **Step 7: Controller commit**

```bash
git add s2c/multiview/print_settings.py s2c/multiview/slice.py tests/test_studio_print.py
git commit -m "Pass print settings and scale to PrusaSlicer, and never let it hang"
```

---

### Task 6: Geometry finish

**Files:**
- Create: `s2c/multiview/finish.py`, `tests/test_studio_finish.py`

**Interfaces:**
- Consumes: `GeometrySettings` (Task 1); `MultiViewSpec` (spec.py, unchanged).
- Produces: `FINISH_LIMIT = 0.45`, `max_finish_mm(spec) -> float`, `apply_geometry(spec, geometry) -> tuple[MultiViewSpec, list[str]]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_studio_finish.py
from pathlib import Path

import pytest

from s2c.multiview.build import BuildError, build, volume
from s2c.multiview.finish import apply_geometry, max_finish_mm
from s2c.multiview.settings import GeometrySettings
from s2c.multiview.spec import MultiViewSpec

SPEC = MultiViewSpec.model_validate_json(
    (Path(__file__).parents[1] / "examples" / "mv" / "l_bracket.json").read_text())


def test_no_finish_returns_the_spec_unchanged():
    spec, warnings = apply_geometry(SPEC, GeometrySettings())
    assert spec is SPEC and warnings == []


def test_a_fillet_is_added_as_a_user_value_and_removes_material():
    spec, warnings = apply_geometry(SPEC, GeometrySettings(finish="fillet", finish_mm=1.0))
    assert warnings == [] and spec.finishes[-1].type == "fillet" and spec.finishes[-1].radius_mm == 1.0
    assert spec.finishes[-1].edges == "all_vertical"
    assert spec.provenance[f"finishes[{len(spec.finishes) - 1}].radius_mm"] == "user_edited"
    assert volume(build(spec)) < volume(build(SPEC))


def test_a_chamfer_on_all_edges():
    spec, _ = apply_geometry(SPEC, GeometrySettings(finish="chamfer", finish_mm=0.5, finish_edges="all"))
    assert (spec.finishes[-1].type, spec.finishes[-1].edges) == ("chamfer", "all")


def test_the_size_is_clamped_to_the_part():
    assert max_finish_mm(SPEC) == pytest.approx(0.45 * 20)
    spec, warnings = apply_geometry(SPEC, GeometrySettings(finish="fillet", finish_mm=10.0))
    assert spec.finishes[-1].radius_mm == pytest.approx(9.0)
    assert warnings == ["Fillet reduced to 9 mm to fit the part"]


def test_a_finish_that_cannot_be_built_still_raises_a_build_error():
    spec, _ = apply_geometry(SPEC, GeometrySettings(finish="fillet", finish_mm=8.0))
    with pytest.raises(BuildError) as e:
        build(spec)
    assert e.value.reason == "fillet_failed"
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_studio_finish.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 's2c.multiview.finish'`.

- [ ] **Step 3: Write `finish.py`**

```python
"""Studio geometry settings applied to a MultiViewSpec. Spec 2026-09-23-studio section 4.
spec.py is unchanged: the finish uses the existing Fillet and Chamfer models, and the size the user chose is
user_edited. "all_vertical" means edges along the depth axis, as the multi-view spec section 5 defines it."""
from __future__ import annotations

from s2c.multiview.settings import GeometrySettings
from s2c.multiview.spec import MultiViewSpec

FINISH_LIMIT = 0.45  # of the smallest envelope side


def max_finish_mm(spec: MultiViewSpec) -> float:
    env = spec.envelope
    return round(FINISH_LIMIT * min(env.x_mm, env.y_mm, env.z_mm), 2)


def apply_geometry(spec: MultiViewSpec, geometry: GeometrySettings) -> tuple[MultiViewSpec, list[str]]:
    if geometry.finish == "none":
        return spec, []
    size = min(geometry.finish_mm, max_finish_mm(spec))
    warnings = [] if size == geometry.finish_mm else [f"{geometry.finish.capitalize()} reduced to {size:g} mm to fit the part"]
    data = spec.model_dump()
    k = len(data["finishes"])
    data["finishes"].append({"type": geometry.finish, "edges": geometry.finish_edges, "radius_mm": size})
    data["provenance"][f"finishes[{k}].radius_mm"] = "user_edited"
    data["warnings"] = [*data["warnings"], *warnings]
    return MultiViewSpec.model_validate(data), warnings
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_studio_finish.py -v`
Expected: PASS (5 tests). If `test_a_finish_that_cannot_be_built...` builds instead of failing, pick the smallest radius above 3 mm that fails on the 3 mm leg, keep the assertion on `fillet_failed`, and record it.

- [ ] **Step 5: Controller commit**

```bash
git add s2c/multiview/finish.py tests/test_studio_finish.py
git commit -m "Apply the Studio's fillet or chamfer to a spec as a user value"
```

---

### Task 7: Seed, AI switches and snapping settings

**Files:**
- Modify: `s2c/multiview/qwen_faces.py` (`qwen_face`), `s2c/multiview/complete.py` (`complete` signature and its `qwen_face` call), `s2c/multiview/fuse.py` (clearance table, `snap_diameter`, `snap`, `assemble`), `s2c/multiview/pipeline.py` (`MvPipeline.__init__`, new `configured`, `_outline`, `fuse`)
- Create: `tests/test_studio_pipeline.py`

**Interfaces:**
- Consumes: `AiSettings`, `GeometrySettings` (Task 1).
- Produces:
  - `qwen_face(outlines, env, face, observed, refs, gen, cache, seed=SEED, attempts=TRIES)`
  - `complete(..., seed: int = SEED, attempts: int = TRIES)`
  - `fuse.CLEARANCE_CLASSES`, `snap_diameter(d, clearance="medium")`, `snap(data, clearance="medium")`, `assemble(..., accepted=(), snap_values=True, clearance="medium")`
  - `MvPipeline.configured(ai: AiSettings) -> MvPipeline`; attributes `seed`, `attempts`, `draw_faces`, `rescue_enabled`
  - `MvPipeline.fuse(observed, user_values=None, accepted=(), rejected=(), geometry: GeometrySettings | None = None)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_studio_pipeline.py
import numpy as np
import pytest

from s2c.multiview import pipeline
from s2c.multiview.fuse import snap_diameter
from s2c.multiview.pipeline import ImageInput, MvPipeline
from s2c.multiview.settings import AiSettings, GeometrySettings
from tests.test_mv_pipeline import fake_reads, sketch
from tests.test_mv_qwen_faces import fake_gen


def test_configured_switches_are_per_request():
    def batch(crops):
        return []

    def provider(img):
        raise RuntimeError

    def depth(img):
        raise RuntimeError

    gen = fake_gen(np.zeros((10, 10, 3), np.uint8))
    base = MvPipeline(batch_reader=batch, mesh_provider=provider, image_gen=gen, depth=depth)
    pipe = base.configured(AiSettings(use_reader=False, use_qwen_image=False, use_rescue=False, use_triposr=False,
                                      use_solaria=False, seed=42, attempts=1))
    assert (pipe.batch_reader, pipe.mesh_provider, pipe.depth) == (None, None, None)
    assert (pipe.draw_faces, pipe.rescue_enabled, pipe.seed, pipe.attempts) == (False, False, 42, 1)
    assert base.batch_reader is batch and base.draw_faces and base.rescue_enabled and base.seed == 7


def test_the_seed_and_attempts_reach_qwen_image():
    gen = fake_gen(np.zeros((300, 300, 3), np.uint8))  # an empty drawing: rejected, so every attempt is used
    pipe = MvPipeline(image_gen=gen).configured(AiSettings(seed=42, attempts=2))
    observed = pipe.observe([ImageInput(sketch(600, 400), "front", "sketch")])
    pipe.fuse(observed, {"envelope.x_mm": 60, "envelope.y_mm": 40, "envelope.z_mm": 10})
    assert [c[2] for c in gen.calls] == [42, 43, 42, 43]


def test_switching_off_qwen_image_leaves_faces_to_the_fallback():
    gen = fake_gen(np.zeros((300, 300, 3), np.uint8))
    pipe = MvPipeline(image_gen=gen).configured(AiSettings(use_qwen_image=False))
    observed = pipe.observe([ImageInput(sketch(600, 400), "front", "sketch")])
    pipe.fuse(observed, {"envelope.x_mm": 60, "envelope.y_mm": 40, "envelope.z_mm": 10})
    assert gen.calls == [] and observed.filled_by["top"] == "assumed"


def test_the_clearance_class_picks_the_table():
    assert (snap_diameter(5.37), snap_diameter(5.37, "fine"), snap_diameter(5.7, "coarse")) == (5.5, 5.3, 5.8)
    assert snap_diameter(7.2) == 7.0  # no clearance hole within 0.4 mm: the 0.5 mm grid


def test_snapping_follows_the_geometry_settings(monkeypatch):
    monkeypatch.setattr(pipeline, "read_values", fake_reads([[(60, "below"), (40, "left")], [(60, "below")]] * 3))
    pipe = MvPipeline(reader=lambda crop: ("", 0.0))
    observed = pipe.observe([ImageInput(sketch(600, 400, circles=[(100, 300, 30)]), "front", "sketch"),
                             ImageInput(sketch(600, 100), "top", "sketch")])
    values = {"envelope.z_mm": 10.0}
    raw = pipe.fuse(observed, values, geometry=GeometrySettings(snap=False)).features[0].diameter_mm
    coarse = pipe.fuse(observed, values, geometry=GeometrySettings(clearance="coarse")).features[0].diameter_mm
    assert raw == pytest.approx(5.7, abs=0.25) and raw not in (5.5, 5.8) and coarse == 5.8
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_studio_pipeline.py -v`
Expected: FAIL (`AttributeError: 'MvPipeline' object has no attribute 'configured'`, `snap_diameter() takes 1 positional argument`).

- [ ] **Step 3: `qwen_faces.py`**

Replace `qwen_face` with:

```python
def qwen_face(outlines: dict[str, Outline], env: Envelope, face: str, observed: list[str], refs,
              gen: ImageGen | None, cache: dict, seed: int = SEED, attempts: int = TRIES) -> Outline | None:
    """The first drawing of `face` that passes the voxel check, trying seeds seed .. seed + attempts - 1."""
    for attempt in range(attempts):
        img = _drawn(gen, refs, face, cache, seed + attempt)
        if img is None and cache.get(FAILED):
            return None  # the call failed: another seed would only wait again
        candidate = None if img is None else outline_from_image(img, face, env)
        if candidate is not None and consistent({**outlines, face: candidate}, env, observed):
            return candidate
    return None
```

- [ ] **Step 4: `complete.py`**

Change the import to `from s2c.multiview.qwen_faces import SEED, TRIES, qwen_face`. Add `seed: int = SEED, attempts: int = TRIES` after `filled_by: dict | None = None` in `complete`'s signature, and change the call to `qwen_face(result, env, face, observed, list(refs), gen, qwen_cache, seed=seed, attempts=attempts)`.

- [ ] **Step 5: `fuse.py`**

Replace `CLEARANCE_MM = (2.7, 3.4, 4.5, 5.5, 6.6, 9.0)` with:

```python
CLEARANCE_CLASSES = {  # ISO 273 clearance holes for M2, M2.5, M3, M4, M5, M6, M8, M10
    "fine": (2.2, 2.7, 3.2, 4.3, 5.3, 6.4, 8.4, 10.5),
    "medium": (2.4, 2.9, 3.4, 4.5, 5.5, 6.6, 9.0, 11.0),
    "coarse": (2.6, 3.1, 3.6, 4.8, 5.8, 7.0, 10.0, 12.0),
}
CLEARANCE_MM = CLEARANCE_CLASSES["medium"]
```

Replace `snap_diameter` with:

```python
def snap_diameter(d: float, clearance: str = "medium") -> float:
    best = min(CLEARANCE_CLASSES[clearance], key=lambda c: abs(c - d))
    return best if abs(best - d) <= 0.4 else _grid(d)
```

In `snap`, change the signature to `def snap(data: dict, clearance: str = "medium") -> None:` and the call inside to `snap_diameter(f[name], clearance)`. In `assemble`, change the signature's end to `accepted=(), snap_values: bool = True, clearance: str = "medium") -> S.MultiViewSpec:` and replace `snap(data)` with:

```python
    if snap_values:
        snap(data, clearance)
```

- [ ] **Step 6: `pipeline.py`**

Add `import copy` to the standard imports, `from s2c.multiview.qwen_faces import RESCUE_PENALTY, SEED, TRIES, rescue_sketch` (replacing the existing qwen_faces import) and `from s2c.multiview.settings import AiSettings, GeometrySettings`.

At the end of `MvPipeline.__init__` add:

```python
        self.seed, self.attempts = SEED, TRIES
        self.draw_faces = self.rescue_enabled = True
```

Add the method after `__init__`:

```python
    def configured(self, ai: AiSettings) -> MvPipeline:
        """A copy for one request with the user's AI switches, seed and attempts; the shared pipeline never changes."""
        pipe = copy.copy(self)
        if not ai.use_reader:
            pipe.batch_reader = None
        if not ai.use_triposr:
            pipe.mesh_provider = None
        if not ai.use_solaria:
            pipe.depth = None
        pipe.draw_faces, pipe.rescue_enabled = ai.use_qwen_image, ai.use_rescue
        pipe.seed, pipe.attempts = ai.seed, ai.attempts
        return pipe
```

In `_outline`, change `and self.image_gen is not None):` to `and self.rescue_enabled and self.image_gen is not None):`, and pass the seed: `fixed = rescue_sketch(bgr, self.image_gen, self.seed)`.

Change `fuse`'s signature to `def fuse(self, observed: Observed, user_values: dict | None = None, accepted=(), rejected=(), geometry: GeometrySettings | None = None) -> S.MultiViewSpec | S.MvAbstain:` and add `geometry = geometry or GeometrySettings()` as its first line. In the `complete(...)` call, replace `gen=self.image_gen` with `gen=self.image_gen if self.draw_faces else None` and add `seed=self.seed, attempts=self.attempts`. Replace the `assemble(...)` call's arguments ending `user_values, accepted)` with `user_values, accepted, snap_values=geometry.snap, clearance=geometry.clearance)`.

- [ ] **Step 7: Run the tests**

Run: `uv run pytest tests/test_studio_pipeline.py tests/test_mv_pipeline.py tests/test_mv_fuse.py tests/test_mv_complete.py tests/test_mv_qwen_faces.py tests/test_mv_rescue.py -v`
Expected: all PASS.

- [ ] **Step 8: Controller commit**

```bash
git add s2c/multiview/qwen_faces.py s2c/multiview/complete.py s2c/multiview/fuse.py s2c/multiview/pipeline.py tests/test_studio_pipeline.py
git commit -m "Expose the seed, attempts, AI switches and snapping settings per request"
```

---

### Task 8: Artifacts (parts, files, zip, cleanup)

**Files:**
- Create: `s2c/multiview/artifacts.py`, `tests/test_studio_artifacts.py`

**Interfaces:**
- Consumes: Tasks 1–6: `settings.*`, `exporters.MESH_FORMATS/FILE_NAMES/export_mesh_formats/write_glb`, `drawing.DRAWING_FORMATS/DRAWING_FILES/write_drawings`, `blend.write_blend`, `print_settings.slicer_flags`, `slice.slice_solid/parse_gcode_stats`, `finish.apply_geometry`; `build.build/volume/BuildError`; `raster.solid_mesh/face_mask/normalize_mask`.
- Produces:
  - `ROOT = Path("tmp/mv_gradio")`, `TTL_S = 3600`
  - `@dataclass Part(key, spec, solid, volume_mm3, bbox_mm, views, preview, folder, warnings)`
  - `@dataclass ExportResult(files: dict[str, Path], sizes: dict[str, int], print_time_s, filament_g, warnings)`
  - `geometry_key(spec, geometry=None) -> str` (20 hex chars)
  - `build_part(spec, geometry=None, root=ROOT) -> Part | MvAbstain`
  - `export_part(part, formats, mesh=None, printing=None, slicer=None, profile=None) -> ExportResult`
  - `bundle(part, result, settings: dict | None = None) -> Path`
  - `sweep(root=ROOT, ttl_s=TTL_S) -> None`, `clear_cache() -> None`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_studio_artifacts.py
import json
import os
import time
import zipfile
from pathlib import Path

import pytest

from s2c.multiview import artifacts
from s2c.multiview.artifacts import build_part, bundle, export_part, geometry_key, sweep
from s2c.multiview.settings import GeometrySettings, MeshSettings, PrintSettings
from s2c.multiview.spec import MultiViewSpec, MvAbstain

SPEC = MultiViewSpec.model_validate_json(
    (Path(__file__).parents[1] / "examples" / "mv" / "l_bracket.json").read_text())


@pytest.fixture(autouse=True)
def no_real_slicer(monkeypatch):
    artifacts.clear_cache()
    monkeypatch.setattr("s2c.multiview.slice.find_slicer", lambda: None)


def test_the_key_ignores_warnings_and_provenance_but_not_geometry():
    other = SPEC.model_copy(update={"warnings": ["something"], "confidence": 0.5})
    assert geometry_key(other) == geometry_key(SPEC)
    assert geometry_key(SPEC, GeometrySettings(finish="fillet")) != geometry_key(SPEC)


def test_a_part_is_built_once(tmp_path, monkeypatch):
    calls = []
    real = artifacts.build
    monkeypatch.setattr(artifacts, "build", lambda spec: calls.append(1) or real(spec))
    first = build_part(SPEC, root=tmp_path)
    second = build_part(SPEC.model_copy(update={"warnings": ["x"]}), root=tmp_path)
    assert first is second and calls == [1]
    assert first.preview.exists() and first.bbox_mm == pytest.approx((50, 30, 20))
    assert set(first.views) == {"front", "back", "left", "right", "top", "bottom"}


def test_a_finish_that_cannot_be_built_is_a_card(tmp_path):
    res = build_part(SPEC, GeometrySettings(finish="fillet", finish_mm=8.0), root=tmp_path)
    assert isinstance(res, MvAbstain) and res.stage == "build" and res.reason == "fillet_failed"


def test_new_formats_reuse_the_part_and_its_files(tmp_path):
    part = build_part(SPEC, root=tmp_path)
    first = export_part(part, ["stl", "step"], MeshSettings())
    stamp = first.files["stl"].stat().st_mtime_ns
    second = export_part(part, ["stl", "step", "obj"], MeshSettings(), PrintSettings(infill_pct=50))
    assert second.files["stl"].stat().st_mtime_ns == stamp and second.files["obj"].exists()
    fine = export_part(part, ["stl"], MeshSettings(quality="fine"))
    assert fine.files["stl"] != first.files["stl"] and fine.files["stl"].stat().st_size > first.sizes["stl"]


def test_every_format_can_be_exported(tmp_path, monkeypatch):
    monkeypatch.setattr("s2c.multiview.blend.blender_runner", lambda: None)
    part = build_part(SPEC, root=tmp_path)
    res = export_part(part, ["stl", "step", "3mf", "obj", "glb", "ply", "brep", "blend", "dxf", "svg", "pdf"])
    assert set(res.files) == {"stl", "step", "3mf", "obj", "glb", "ply", "brep", "blend", "dxf", "svg", "pdf"}
    assert res.files["blend"].name == "part_blender_kit.zip" and all(s > 0 for s in res.sizes.values())


def test_gcode_problems_never_block_the_other_files(tmp_path, monkeypatch):
    part = build_part(SPEC, root=tmp_path)
    res = export_part(part, ["stl", "gcode"])
    assert set(res.files) == {"stl"} and "G-code unavailable: slicer not installed" in res.warnings
    huge = SPEC.model_copy(deep=True)
    res = export_part(part, ["step", "gcode"], printing=PrintSettings(scale_pct=200))
    assert "step" in res.files and "gcode" not in res.files
    assert huge.envelope == SPEC.envelope


def test_the_zip_holds_the_files_and_a_manifest(tmp_path):
    part = build_part(SPEC, root=tmp_path)
    res = export_part(part, ["stl", "step"])
    path = bundle(part, res, {"mesh": {"quality": "normal"}})
    with zipfile.ZipFile(path) as z:
        assert sorted(z.namelist()) == ["manifest.json", "part.step", "part.stl"]
        manifest = json.loads(z.read("manifest.json"))
    assert manifest["spec"]["provenance"]["envelope.x_mm"] == "user_written"
    assert manifest["settings"] == {"mesh": {"quality": "normal"}} and manifest["part"]["key"] == part.key


def test_the_sweep_deletes_old_folders_only(tmp_path):
    old, fresh = tmp_path / "old", tmp_path / "fresh"
    old.mkdir()
    fresh.mkdir()
    past = time.time() - 7200
    os.utime(old, (past, past))
    sweep(tmp_path)
    assert not old.exists() and fresh.exists()
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_studio_artifacts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 's2c.multiview.artifacts'`.

- [ ] **Step 3: Write `artifacts.py`**

```python
"""Built parts and their files, cached by content. Spec 2026-09-23-studio section 5.
A part is keyed by its geometry (never by warnings or provenance); files by mesh quality or print settings. Files live
under ROOT/<key>/ and are deleted an hour after their last use. Nothing the user did not change is recomputed."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
import zipfile
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cadquery as cq
import numpy as np

from s2c.multiview import spec as S
from s2c.multiview.blend import write_blend
from s2c.multiview.build import BuildError, build, volume
from s2c.multiview.drawing import DRAWING_FILES, DRAWING_FORMATS, write_drawings
from s2c.multiview.exporters import FILE_NAMES, MESH_FORMATS, export_mesh_formats, write_glb
from s2c.multiview.finish import apply_geometry
from s2c.multiview.print_settings import slicer_flags
from s2c.multiview.raster import face_mask, normalize_mask, solid_mesh
from s2c.multiview.settings import FORMATS, GeometrySettings, MeshSettings, PrintSettings, settings_hash
from s2c.multiview.slice import parse_gcode_stats, slice_solid

ROOT = Path("tmp/mv_gradio")
TTL_S = 3600
LRU_SIZE = 16
_KEY_FIELDS = {"envelope", "views", "features", "finishes"}


@dataclass
class Part:
    key: str
    spec: S.MultiViewSpec  # after the geometry settings
    solid: cq.Workplane
    volume_mm3: float
    bbox_mm: tuple[float, float, float]
    views: dict[str, np.ndarray]
    preview: Path
    folder: Path
    warnings: list[str] = field(default_factory=list)


@dataclass
class ExportResult:
    files: dict[str, Path]
    sizes: dict[str, int]
    print_time_s: float | None = None
    filament_g: float | None = None
    warnings: list[str] = field(default_factory=list)


class _Lru:
    def __init__(self, size: int):
        self.size, self._items, self._lock = size, OrderedDict(), threading.Lock()

    def get(self, key):
        with self._lock:
            value = self._items.get(key)
            if value is not None:
                self._items.move_to_end(key)
            return value

    def put(self, key, value) -> None:
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.size:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


_parts = _Lru(LRU_SIZE)


def clear_cache() -> None:
    _parts.clear()


def geometry_key(spec: S.MultiViewSpec, geometry: GeometrySettings | None = None) -> str:
    body = json.dumps(spec.model_dump(mode="json", include=_KEY_FIELDS), sort_keys=True)
    return hashlib.sha256(f"{body}|{settings_hash(geometry or GeometrySettings())}".encode()).hexdigest()[:20]


def _touch(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    now = time.time()
    os.utime(folder, (now, now))


def build_part(spec: S.MultiViewSpec, geometry: GeometrySettings | None = None, root: Path = ROOT) -> Part | S.MvAbstain:
    geometry = geometry or GeometrySettings()
    key = geometry_key(spec, geometry)
    folder = Path(root) / key
    cached = _parts.get((str(root), key))
    if cached is not None and cached.preview.exists():  # the sweep may have removed the files
        _touch(folder)
        return cached
    final, warnings = apply_geometry(spec, geometry)
    try:
        solid = build(final)
    except BuildError as e:
        return S.MvAbstain(stage="build", reason=e.reason, remedy=e.remedy)
    bb = solid.val().BoundingBox()
    mesh = solid_mesh(solid)
    views = {f: normalize_mask(face_mask(mesh, f, final.envelope)[0]) for f in S.FACES}
    _touch(folder)
    preview = write_glb(solid, folder / "preview.glb", "normal")
    part = Part(key, final, solid, volume(solid), (bb.xlen, bb.ylen, bb.zlen), views, preview, folder, warnings)
    _parts.put((str(root), key), part)
    return part


def _mesh_files(part: Part, wanted: list[str], quality: str) -> dict[str, Path]:
    folder = part.folder / f"mesh-{quality}"
    missing = [f for f in wanted if not (folder / FILE_NAMES[f]).exists()]
    if missing:
        export_mesh_formats(part.solid, folder, missing, quality)
    return {f: folder / FILE_NAMES[f] for f in wanted}


def _gcode(part: Part, printing: PrintSettings, slicer, profile) -> tuple[Path | None, float | None, float | None,
                                                                         list[str]]:
    folder = part.folder / f"print-{settings_hash(printing)}"
    gcode = folder / "part.gcode"
    if gcode.exists():
        seconds, grams = parse_gcode_stats(gcode.read_text(encoding="utf-8", errors="replace"))
        return gcode, seconds, grams, []
    res = slice_solid(part.solid, folder, profile, slicer, overrides=slicer_flags(printing),
                      scale=printing.scale_pct / 100)
    if isinstance(res, S.MvAbstain):
        return None, None, None, [f"G-code skipped: {res.remedy}"]
    return res.gcode, res.print_time_s, res.filament_g, list(res.warnings)


def export_part(part: Part, formats, mesh: MeshSettings | None = None, printing: PrintSettings | None = None,
                slicer: Path | None = None, profile: Path | None = None) -> ExportResult:
    mesh, printing = mesh or MeshSettings(), printing or PrintSettings()
    unknown = sorted(set(formats) - set(FORMATS))
    if unknown:
        raise ValueError(f"unknown formats {unknown}")
    wanted = [f for f in FORMATS if f in formats]
    need = [f for f in MESH_FORMATS if f in wanted or (f == "obj" and "blend" in wanted)]
    made = _mesh_files(part, need, mesh.quality)
    files = {f: made[f] for f in MESH_FORMATS if f in wanted}
    warnings: list[str] = []
    seconds = grams = None
    if "blend" in wanted:
        files["blend"], more = write_blend(made["obj"], part.folder / f"mesh-{mesh.quality}")
        warnings += more
    drawings = [f for f in DRAWING_FORMATS if f in wanted]
    if drawings:
        folder = part.folder / "drawing"
        missing = [f for f in drawings if not (folder / DRAWING_FILES[f]).exists()]
        if missing:
            write_drawings(part.solid, part.spec, folder, missing)
        files.update({f: folder / DRAWING_FILES[f] for f in drawings})
    if "gcode" in wanted:
        gcode, seconds, grams, more = _gcode(part, printing, slicer, profile)
        warnings += more
        if gcode is not None:
            files["gcode"] = gcode
    _touch(part.folder)
    ordered = {f: files[f] for f in FORMATS if f in files}
    return ExportResult(ordered, {f: p.stat().st_size for f, p in ordered.items()}, seconds, grams, warnings)


def bundle(part: Part, result: ExportResult, settings: dict | None = None) -> Path:
    target = part.folder / f"sketch-to-cad-{part.key[:8]}.zip"
    tmp = target.with_suffix(".part")
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "part": {"key": part.key, "volume_mm3": round(part.volume_mm3, 3),
                 "bbox_mm": [round(v, 3) for v in part.bbox_mm]},
        "files": {f: p.name for f, p in result.files.items()},
        "print": {"time_s": result.print_time_s, "filament_g": result.filament_g},
        "settings": settings or {},
        "warnings": [*part.spec.warnings, *result.warnings],
        "spec": part.spec.model_dump(mode="json"),
    }
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for path in result.files.values():
            z.write(path, arcname=path.name)
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
    os.replace(tmp, target)
    return target


def sweep(root: Path = ROOT, ttl_s: float = TTL_S) -> None:
    root = Path(root)
    if not root.exists():
        return
    now = time.time()
    for d in root.iterdir():
        if d.is_dir() and now - d.stat().st_mtime > ttl_s:
            shutil.rmtree(d, ignore_errors=True)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_studio_artifacts.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Controller commit**

```bash
git add s2c/multiview/artifacts.py tests/test_studio_artifacts.py
git commit -m "Cache built parts and their files by content, with a zip and a manifest"
```

---

### Task 9: The Studio UI and demo examples

**Files:**
- Create: `s2c/studio/__init__.py`, `s2c/studio/theme.py`, `s2c/studio/session.py`, `s2c/studio/status.py`, `s2c/studio/handlers.py`, `s2c/studio/app.py`, `app_mv_studio.py`, `scripts/make_examples.py`, `examples/mv/sketches/front.png`, `examples/mv/sketches/top.png`, `examples/mv/sketches/examples.json`, `tests/test_studio_ui.py`

**Interfaces:**
- Consumes: `MvPipeline` (`configured`, `observe`, `fuse(geometry=)`, attributes), `ImageInput`, `Observed`, `default_pipeline` (pipeline); `artifacts.build_part/export_part/bundle/sweep/ROOT`; `settings.*`; `slice.find_slicer`; `blend.blender_runner`; `raster.outline_mask/iou`; spec constants.
- Produces: `Studio(pipe, store=None, root=ROOT)` with `add_images`, `set_face`, `set_kind`, `remove`, `load_examples`, `coverage_html`, `analyze`, `redraw`, `build`, `rebuild_geometry`, `export`; dataclasses `Review`, `Model`, `Exported`; `build_app(pipe=None, studio=None) -> gr.Blocks`; `launch()`.

- [ ] **Step 1: Generate the demo sketches**

```python
# scripts/make_examples.py
"""Draw the demo sketches for the Studio's "Try an example": the L-bracket of examples/mv/l_bracket.json, front and
top views, in pen, with the dimensions written the way a user would. Run: uv run python scripts/make_examples.py"""
import json
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "examples" / "mv" / "sketches"
PX = 12  # pixels per millimetre
INK = (40, 40, 40)


def canvas():
    img = np.full((1000, 1400, 3), 250, np.uint8)
    noise = np.random.default_rng(3).normal(0, 2, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def to_px(points, x0, y0, height_mm):
    return np.array([(x0 + a * PX, y0 + (height_mm - b) * PX) for a, b in points], np.int32)


def write(img, text, at):
    cv2.putText(img, text, at, cv2.FONT_HERSHEY_SIMPLEX, 1.6, INK, 3, cv2.LINE_AA)


def front():
    img = canvas()
    x0, y0 = 400, 250
    pts = to_px([(0, 0), (50, 0), (50, 30), (47, 30), (47, 3), (0, 3)], x0, y0, 30)
    cv2.polylines(img, [pts], True, INK, 4, cv2.LINE_AA)
    write(img, "50", (x0 + 25 * PX - 30, y0 + 30 * PX + 90))
    write(img, "30", (x0 + 50 * PX + 60, y0 + 15 * PX + 15))
    return img


def top():
    img = canvas()
    x0, y0 = 400, 300
    cv2.polylines(img, [to_px([(0, 0), (50, 0), (50, 20), (0, 20)], x0, y0, 20)], True, INK, 4, cv2.LINE_AA)
    cv2.circle(img, (x0 + 20 * PX, y0 + (20 - 10) * PX), round(2.75 * PX), INK, 3, cv2.LINE_AA)
    write(img, "20", (x0 - 110, y0 + 10 * PX + 15))
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "front.png"), front())
    cv2.imwrite(str(OUT / "top.png"), top())
    (OUT / "examples.json").write_text(json.dumps([
        {"file": "front.png", "face": "front", "kind": "sketch"},
        {"file": "top.png", "face": "top", "kind": "sketch"}], indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
```

Run: `uv run python scripts/make_examples.py`
Expected: prints the folder; the two PNGs and `examples.json` exist. Open both PNGs (Read tool) and check they look like pen sketches.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_studio_ui.py
from pathlib import Path

import gradio as gr
import numpy as np
import pytest

from s2c.multiview import artifacts
from s2c.multiview.pipeline import MvPipeline
from s2c.multiview.settings import AiSettings, ExportSettings, GeometrySettings, MeshSettings, PrintSettings
from s2c.studio.app import build_app
from s2c.studio.handlers import Studio, parse_size
from tests.test_mv_pipeline import sketch
from tests.test_mv_qwen_faces import fake_gen


@pytest.fixture
def studio(tmp_path, monkeypatch):
    artifacts.clear_cache()
    monkeypatch.setattr("s2c.multiview.slice.find_slicer", lambda: None)
    return Studio(MvPipeline(), root=tmp_path / "files")


def with_images(studio, tmp_path, faces=(("front", 600, 400), ("top", 600, 100))):
    sid = studio.store.new()
    paths = []
    for face, w, h in faces:
        path = tmp_path / f"{face}.png"
        path.write_bytes(sketch(w, h))
        paths.append(str(path))
    studio.add_images(sid, paths)
    for item, (face, _, _) in zip(studio.store.get(sid).items, faces):
        studio.set_face(sid, item.id, face)
        studio.set_kind(sid, item.id, "sketch")
    return sid


def test_the_app_builds():
    assert isinstance(build_app(MvPipeline()), gr.Blocks)


def test_coverage_shows_which_faces_are_given(studio, tmp_path):
    sid = with_images(studio, tmp_path)
    html = studio.coverage_html(sid)
    assert "front" in html and "top" in html and "AI will draw" in html


def test_analyze_asks_for_sizes_then_build_and_export(studio, tmp_path):
    sid = with_images(studio, tmp_path)
    review = studio.analyze(sid, "none", AiSettings())
    assert not review.ok and "missing_x" in review.message_html
    assert all(review.sizes[a]["required"] for a in "xyz")
    review, model = studio.build(sid, {"x": "60", "y": "40", "z": "10"}, review.rows, [], GeometrySettings())
    assert model.ok and Path(model.preview).exists() and len(model.views) == 6
    exported = studio.export(sid, ExportSettings(formats=["stl", "step", "dxf"]), MeshSettings(), PrintSettings())
    assert {Path(f).name for f in exported.files} == {"part.stl", "part.step", "drawing.dxf"}
    assert Path(exported.zip_path).exists()


def test_sizes_accept_commas_and_units_and_reject_junk():
    assert [parse_size(t) for t in ("42,5", "42 mm", " 42 ")] == [42.5, 42.0, 42.0]
    for junk in ("abc", "0", "-3", "", "nan"):
        assert parse_size(junk) is None


def test_bad_sizes_are_a_message(studio, tmp_path):
    sid = with_images(studio, tmp_path)
    review = studio.analyze(sid, "none", AiSettings())
    review, model = studio.build(sid, {"x": "abc", "y": "40", "z": "-3"}, review.rows, [], GeometrySettings())
    assert not model.ok and "Width" in review.message_html and "Depth" in review.message_html


def test_a_failed_finish_keeps_the_review(studio, tmp_path):
    sid = with_images(studio, tmp_path)
    review = studio.analyze(sid, "none", AiSettings())
    sizes = {"x": "60", "y": "40", "z": "3"}
    _, model = studio.build(sid, sizes, review.rows, [], GeometrySettings(finish="fillet", finish_mm=1.3))
    assert not model.ok and "fillet" in model.message_html.lower()
    model = studio.rebuild_geometry(sid, GeometrySettings(finish="fillet", finish_mm=0.3))
    assert model.ok


def test_a_file_that_is_not_an_image_is_explained(studio, tmp_path):
    sid = studio.store.new()
    bad = tmp_path / "notes.png"
    bad.write_bytes(b"not an image at all")
    studio.add_images(sid, [str(bad)])
    studio.set_face(sid, studio.store.get(sid).items[0].id, "front")
    review = studio.analyze(sid, "none", AiSettings())
    assert not review.ok and review.stage == "capture" and "JPEG or PNG" in review.message_html


def test_redraw_moves_to_new_seeds(tmp_path, monkeypatch):
    monkeypatch.setattr("s2c.multiview.slice.find_slicer", lambda: None)
    gen = fake_gen(np.zeros((300, 300, 3), np.uint8))
    studio = Studio(MvPipeline(image_gen=gen), root=tmp_path / "files")
    sid = with_images(studio, tmp_path, faces=(("front", 600, 400),))
    review = studio.analyze(sid, "none", AiSettings(seed=10, attempts=1))
    studio.build(sid, {"x": "60", "y": "40", "z": "10"}, review.rows, [], GeometrySettings())
    seeds_before = {c[2] for c in gen.calls}
    studio.redraw(sid)
    assert seeds_before == {10} and {c[2] for c in gen.calls} == {10, 11}


def test_the_example_loads_two_tagged_sketches(studio):
    sid = studio.store.new()
    studio.load_examples(sid)
    items = studio.store.get(sid).items
    assert [(i.face, i.kind) for i in items] == [("front", "sketch"), ("top", "sketch")]
```

- [ ] **Step 3: Run them to see them fail**

Run: `uv run pytest tests/test_studio_ui.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 's2c.studio'`.

- [ ] **Step 4: `s2c/studio/__init__.py` and `theme.py`**

```python
# s2c/studio/__init__.py
"""Sketch-to-CAD Studio: the guided Gradio app for the multi-view path. Spec 2026-09-23-studio."""
```

```python
# s2c/studio/theme.py
"""Look and feel of the Studio: theme, CSS and small HTML pieces. Spec 2026-09-23-studio section 8.
Chip text always says what it means, so colour is never the only signal; every pair meets 4.5:1 contrast."""
from __future__ import annotations

import html

import gradio as gr

THEME = gr.themes.Base(primary_hue="indigo", secondary_hue="amber", neutral_hue="slate",
                       radius_size=gr.themes.sizes.radius_md, spacing_size=gr.themes.sizes.spacing_md).set(
    body_background_fill="*neutral_50", body_background_fill_dark="*neutral_950",
    block_border_width="1px", block_shadow="none", block_radius="*radius_lg",
    button_primary_background_fill="*primary_600", button_primary_background_fill_hover="*primary_700",
)
TONES = {"ok": ("#065F46", "#D1FAE5"), "check": ("#92400E", "#FEF3C7"), "ai": ("#5B21B6", "#EDE9FE"),
         "stop": ("#991B1B", "#FEE2E2"), "info": ("#334155", "#E2E8F0")}
TRUSTED = frozenset({"user_written", "measured", "user_edited"})
SOURCE_TEXT = {"user_written": "written", "measured": "measured", "user_edited": "you set it",
               "scaled": "scaled · check", "inferred": "AI · check", "estimated": "estimated · check",
               "default": "default · check"}
FACE_BADGES = {"observed": ("observed", "ok"), "mirrored": ("mirrored", "ok"),
               "qwen-image": ("drawn by Qwen-Image", "ai"), "triposr": ("predicted by TripoSR", "ai"),
               "assumed": ("assumed rectangle", "check")}
CSS = """
.studio-header {display:flex; flex-wrap:wrap; align-items:center; gap:12px; justify-content:space-between}
.studio-header h1 {margin:0; font-size:1.6rem}
.studio-header p {margin:2px 0 0; opacity:.8}
.chip {display:inline-block; padding:2px 10px; margin:2px 4px 2px 0; border-radius:999px; font-size:.82rem;
       font-weight:600; white-space:nowrap}
.card {border-radius:12px; padding:12px 16px; border:1px solid; margin:4px 0}
.card h4 {margin:0 0 4px}
.stats {display:grid; grid-template-columns:repeat(auto-fit, minmax(120px, 1fr)); gap:8px}
.stat {border:1px solid var(--border-color-primary); border-radius:12px; padding:8px 12px}
.stat b {display:block; font-size:1.15rem}
.size.required textarea, .size.required input {border-color:#DC2626 !important}
.size.required label span {color:#991B1B}
"""


def chip(text: str, tone: str = "info") -> str:
    fg, bg = TONES[tone]
    return f'<span class="chip" style="color:{fg};background:{bg}">{html.escape(text)}</span>'


def source_chip(provenance: str) -> str:
    tone = "ok" if provenance in TRUSTED else "ai" if provenance == "inferred" else "check"
    return chip(SOURCE_TEXT.get(provenance, provenance), tone)


def card(title: str, body: str, tone: str = "info") -> str:
    fg, bg = TONES[tone]
    return (f'<div class="card" style="border-color:{fg};background:{bg};color:{fg}">'
            f"<h4>{html.escape(title)}</h4><div>{html.escape(body)}</div></div>")


def stats_html(items: list[tuple[str, str]]) -> str:
    cells = "".join(f'<div class="stat">{html.escape(k)}<b>{html.escape(v)}</b></div>' for k, v in items)
    return f'<div class="stats">{cells}</div>'


def bullet_html(title: str, lines: list[str], tone: str) -> str:
    if not lines:
        return ""
    items = "".join(f"<li>{html.escape(line)}</li>" for line in lines)
    fg, bg = TONES[tone]
    return f'<div class="card" style="border-color:{fg};background:{bg};color:{fg}"><h4>{title}</h4><ul>{items}</ul></div>'
```

- [ ] **Step 5: `session.py` and `status.py`**

```python
# s2c/studio/session.py
"""Per-browser-session state, kept on the server and keyed by an id: gr.State only holds the id, because the
pipeline's Observed (images, masks, meshes) is too big to deep-copy on every event. Spec 2026-09-23-studio §8."""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

from s2c.multiview.artifacts import ExportResult, Part
from s2c.multiview.pipeline import Observed
from s2c.multiview.settings import AiSettings, GeometrySettings
from s2c.multiview.spec import MultiViewSpec


@dataclass
class Item:
    id: str
    path: str
    name: str
    face: str = "auto"
    kind: str = "auto"


@dataclass
class Session:
    id: str
    items: list[Item] = field(default_factory=list)
    reference: str = "none"
    ai: AiSettings = field(default_factory=AiSettings)
    geometry: GeometrySettings = field(default_factory=GeometrySettings)
    observed: Observed | None = None
    spec: MultiViewSpec | None = None
    edits: dict[str, float] = field(default_factory=dict)
    rejected: tuple[str, ...] = ()
    row_paths: list[str] = field(default_factory=list)
    shown: dict[str, float | None] = field(default_factory=dict)
    part: Part | None = None
    exported: ExportResult | None = None
    touched: float = field(default_factory=time.time)


class SessionStore:
    def __init__(self, ttl_s: float = 3600):
        self.ttl_s, self._items, self._lock = ttl_s, {}, threading.Lock()

    def new(self) -> str:
        sid = uuid.uuid4().hex
        with self._lock:
            self._sweep()
            self._items[sid] = Session(sid)
        return sid

    def get(self, sid: str | None) -> Session:
        """The session, or a fresh one under that id (after a server restart the browser still holds its id)."""
        with self._lock:
            session = self._items.get(sid or "")
            if session is None:
                session = self._items[sid or uuid.uuid4().hex] = Session(sid or uuid.uuid4().hex)
            session.touched = time.time()
            return session

    def drop(self, sid: str | None) -> None:
        with self._lock:
            self._items.pop(sid or "", None)

    def _sweep(self) -> None:
        now = time.time()
        for sid in [s for s, v in self._items.items() if now - v.touched > self.ttl_s]:
            del self._items[sid]
```

```python
# s2c/studio/status.py
"""Which providers this Studio can use, shown as chips in the header. Computed once at start."""
from __future__ import annotations

from s2c.multiview.blend import blender_runner
from s2c.multiview.pipeline import MvPipeline
from s2c.multiview.slice import find_slicer
from s2c.studio.theme import chip


def provider_status(pipe: MvPipeline) -> dict[str, bool]:
    return {
        "Qwen-VL": pipe.batch_reader is not None or pipe.chat is not None,
        "Qwen-Image": pipe.image_gen is not None,
        "TripoSR": pipe.mesh_provider is not None,
        "Solaria": pipe.depth is not None,
        "PrusaSlicer": find_slicer() is not None,
        "Blender": blender_runner() is not None,
    }


def header_html(status: dict[str, bool]) -> str:
    chips = "".join(chip(f"{'●' if ok else '○'} {name}", "ok" if ok else "info") for name, ok in status.items())
    return ('<div class="studio-header"><div><h1>Sketch-to-CAD Studio</h1>'
            "<p>Sketches or photos of a part, one or more per face → an editable CAD part, drawings and G-code.</p>"
            f"</div><div>{chips}{chip('images deleted after 1 h', 'info')}</div></div>")
```

- [ ] **Step 6: `handlers.py`**

```python
# s2c/studio/handlers.py
"""What the Studio's buttons do, as plain methods that tests call without a browser. Each returns a small view model
(Review, Model, Exported) that app.py maps onto components. Spec 2026-09-23-studio section 8."""
from __future__ import annotations

import json
import math
import random
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from s2c.multiview import spec as S
from s2c.multiview.artifacts import ROOT, build_part, bundle, export_part, sweep
from s2c.multiview.pipeline import ImageInput, MvPipeline
from s2c.multiview.raster import iou, outline_mask
from s2c.multiview.settings import AiSettings, ExportSettings, GeometrySettings, MeshSettings, PrintSettings
from s2c.studio.session import Item, SessionStore
from s2c.studio.theme import FACE_BADGES, TRUSTED, bullet_html, card, chip, source_chip, stats_html

FACE_CHOICES = ["auto", *S.FACES]
KIND_CHOICES = ["auto", "sketch", "photo", "drawing"]
REFERENCES = ["none", "1 TND", "1 EUR", "2 EUR", "card", "a4"]
AXES = ("x", "y", "z")
AXIS_LABEL = {"x": "Width (X)", "y": "Height (Y)", "z": "Depth (Z)"}
EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "mv" / "sketches"
_FEATURE = re.compile(r"(features|finishes)\[(\d+)\]\.(\w+)")
_FIELD_WORDS = {"a_mm": "position a", "b_mm": "position b", "diameter_mm": "diameter", "depth_mm": "depth",
                "width_mm": "width", "length_mm": "length", "angle_deg": "angle", "radius_mm": "size"}


@dataclass
class Review:
    ok: bool
    stage: str  # "capture" when the images themselves failed, else "review"
    message_html: str
    sizes: dict[str, dict] = field(default_factory=dict)  # axis -> value, placeholder, info, required
    rows: list[list] = field(default_factory=list)       # field, value, source chip
    faces: list[tuple[np.ndarray, str]] = field(default_factory=list)
    ai_faces: list[str] = field(default_factory=list)
    warnings_html: str = ""
    reads: list[list] = field(default_factory=list)
    seed: int = 7
    unchecked: int = 0


@dataclass
class Model:
    ok: bool
    message_html: str
    preview: str | None = None
    views: list[tuple[np.ndarray, str]] = field(default_factory=list)
    stats_html: str = ""


@dataclass
class Exported:
    message_html: str
    files: list[str] = field(default_factory=list)
    zip_path: str | None = None
    stats_html: str = ""


def parse_size(text) -> float | None:
    """'42,5', '42 mm' and ' 42 ' are 42.5, 42 and 42; anything that is not a positive number is None."""
    t = str(text or "").strip().lower().removesuffix("mm").strip().replace(",", ".")
    try:
        value = float(t)
    except ValueError:
        return None
    return value if math.isfinite(value) and value > 0 else None


def field_label(spec: S.MultiViewSpec, path: str) -> str:
    m = _FEATURE.fullmatch(path)
    if not m:
        return path
    group, k, name = m.group(1), int(m.group(2)), m.group(3)
    if group == "finishes":
        return f"{spec.finishes[k].type.capitalize()} {_FIELD_WORDS.get(name, name)}"
    f = spec.features[k]
    return f"{f.type.capitalize()} {k + 1} ({f.face}) · {_FIELD_WORDS.get(name, name)}"


def _value(spec: S.MultiViewSpec, path: str) -> float:
    data = spec.model_dump()
    m = _FEATURE.fullmatch(path)
    if m:
        return data[m.group(1)][int(m.group(2))][m.group(3)]
    head, name = path.split(".", 1)
    return data[head][name]


def _abstain_card(a: S.MvAbstain) -> str:
    return card(f"Stopped at {a.stage}: {a.reason}", a.remedy, "stop")


class Studio:
    def __init__(self, pipe: MvPipeline, store: SessionStore | None = None, root: Path = ROOT):
        self.pipe, self.store, self.root = pipe, store or SessionStore(), Path(root)

    # ---- capture -----------------------------------------------------------------------------------------
    def add_images(self, sid: str, paths) -> None:
        session = self.store.get(sid)
        for p in paths or []:
            session.items.append(Item(uuid.uuid4().hex[:8], str(p), Path(p).name))

    def set_face(self, sid: str, item_id: str, face: str) -> None:
        for item in self.store.get(sid).items:
            if item.id == item_id and face in FACE_CHOICES:
                item.face = face

    def set_kind(self, sid: str, item_id: str, kind: str) -> None:
        for item in self.store.get(sid).items:
            if item.id == item_id and kind in KIND_CHOICES:
                item.kind = kind

    def remove(self, sid: str, item_id: str) -> None:
        session = self.store.get(sid)
        session.items = [i for i in session.items if i.id != item_id]

    def load_examples(self, sid: str) -> None:
        session = self.store.get(sid)
        session.items = []
        for entry in json.loads((EXAMPLES / "examples.json").read_text()):
            session.items.append(Item(uuid.uuid4().hex[:8], str(EXAMPLES / entry["file"]), entry["file"],
                                      entry["face"], entry["kind"]))

    def coverage_html(self, sid: str) -> str:
        items = self.store.get(sid).items
        if not items:
            return chip("Add at least one image", "info")
        counts = {face: sum(1 for i in items if i.face == face) for face in S.FACES}
        auto = sum(1 for i in items if i.face == "auto")
        chips = []
        for canon, opposite in (("front", "back"), ("top", "bottom"), ("right", "left")):
            n, m = counts[canon], counts[opposite]
            if n or m:
                chips.append(chip(f"{canon} ✓{n}" + (f" · {opposite} ✓{m}" if m else ""), "ok"))
            else:
                chips.append(chip(f"{canon}: AI will draw it", "ai"))
        if auto:
            chips.append(chip(f"{auto} image(s) face: auto", "check"))
        return "".join(chips)

    # ---- review --------------------------------------------------------------------------------------------
    def analyze(self, sid: str, reference: str, ai: AiSettings) -> Review:
        session = self.store.get(sid)
        if not session.items:
            return Review(False, "capture", card("No images yet", "Drop sketches or photos, then Analyze.", "check"))
        if ai.randomize_seed:
            ai = ai.model_copy(update={"seed": random.randint(0, 2**31 - 1)})
        session.ai, session.reference = ai, reference
        images = []
        for item in session.items:
            try:
                data = Path(item.path).read_bytes()
            except OSError:
                return Review(False, "capture", card("Image missing", f"{item.name} is no longer available. Add it again.",
                                                     "stop"))
            images.append(ImageInput(data, None if item.face == "auto" else item.face,
                                     None if item.kind == "auto" else item.kind))
        pipe = self.pipe.configured(ai)
        observed = pipe.observe(images, None if reference in (None, "", "none") else reference)
        if isinstance(observed, S.MvAbstain):
            return Review(False, "capture", _abstain_card(observed), seed=ai.seed)
        session.observed, session.edits, session.rejected, session.part = observed, {}, (), None
        return self._review(session, pipe.fuse(observed, geometry=session.geometry))

    def redraw(self, sid: str) -> Review:
        session = self.store.get(sid)
        if session.observed is None:
            return Review(False, "capture", card("Nothing to redraw", "Analyze your images first.", "check"))
        session.ai = session.ai.model_copy(update={"seed": session.ai.seed + session.ai.attempts})
        return self._review(session, self._fuse(session))

    def _fuse(self, session):
        return self.pipe.configured(session.ai).fuse(session.observed, dict(session.edits),
                                                     rejected=session.rejected, geometry=session.geometry)

    def _review(self, session, res) -> Review:
        spec = None if isinstance(res, S.MvAbstain) else res
        abstain = res if isinstance(res, S.MvAbstain) else None
        session.spec = spec
        observed = session.observed
        known = ((abstain.partial or {}).get("known", {}) if abstain else {}) if not spec else {}
        suggested = ((abstain.partial or {}).get("suggested", {}) if abstain else {})
        sizes = {}
        for axis in AXES:
            path = f"envelope.{axis}_mm"
            if spec is not None:
                value, prov = getattr(spec.envelope, f"{axis}_mm"), spec.provenance[path]
            else:
                value, prov = known.get(path, session.edits.get(path)), "user_edited" if path in session.edits else None
            hint = suggested.get(path)
            sizes[axis] = {"value": "" if value is None else f"{value:g}",
                           "placeholder": f"suggested {hint:g}" if hint else "mm",
                           "info": f"from: {prov.replace('_', ' ')}" if prov else "Required: type it or use the suggestion",
                           "required": value is None}
            session.shown[path] = value
        rows, paths = [], []
        if spec is not None:
            for path, prov in spec.provenance.items():
                if path.startswith(("envelope.", "views.")):
                    continue
                paths.append(path)
                session.shown[path] = _value(spec, path)
                rows.append([field_label(spec, path), _value(spec, path), source_chip(prov)])
        session.row_paths = paths
        faces, ai_faces = [], []
        if spec is not None:
            for face in S.CANONICAL_FACES:
                ol = getattr(spec.views, face)
                a, b = S.face_size(face, spec.envelope)
                mask = 255 - outline_mask(ol.outer, ol.inner, a, b, px=256)
                who = observed.filled_by.get(face, ol.source)
                text, _ = FACE_BADGES.get(who, (who, "info"))
                merged = next((w.split(": ", 1)[1] for w in spec.warnings if w.startswith(f"{face}: merged")), "")
                faces.append((mask, f"{face}: {text}" + (f" · {merged}" if merged else "")))
                if who in ("qwen-image", "triposr"):
                    ai_faces.append(face)
        else:
            faces = [(255 - m, f"{face}: your image") for face, m in observed.masks.items()]
        warnings = spec.warnings if spec is not None else observed.warnings
        info = [w for w in warnings if ": merged " in w]
        check = [w for w in warnings if w not in info]
        reads = [[o.face, lv.reading.text, lv.reading.value_mm,
                  f"hole {lv.hole_index + 1}" if lv.hole_index is not None else f"{lv.axis} axis" if lv.axis else "not linked"]
                 for o in observed.observations for lv in o.values]
        unchecked = sum(1 for p in (spec.provenance.values() if spec else []) if p not in TRUSTED)
        message = (_abstain_card(abstain) if abstain else
                   card("Ready to build", "Check the amber values, reject any AI face you do not trust, then Build.", "ok"))
        return Review(spec is not None, "review", message, sizes, rows, faces, ai_faces,
                      bullet_html("Check", check, "check") + bullet_html("Info", info, "info"), reads,
                      session.ai.seed, unchecked)

    # ---- build ----------------------------------------------------------------------------------------------
    def build(self, sid: str, sizes: dict[str, str], rows, rejected, geometry: GeometrySettings) -> tuple[Review, Model]:
        session = self.store.get(sid)
        if session.observed is None:
            msg = card("Nothing to build", "Analyze your images first.", "check")
            return Review(False, "capture", msg), Model(False, msg)
        errors, edits = [], dict(session.edits)
        for axis in AXES:
            text = str(sizes.get(axis, "")).strip()
            path = f"envelope.{axis}_mm"
            if not text:
                continue
            value = parse_size(text)
            if value is None:
                errors.append(f"{AXIS_LABEL[axis]}: '{text}' is not a size in mm")
            elif session.shown.get(path) is None or abs(value - float(session.shown[path])) > 1e-9:
                edits[path] = value
        for path, row in zip(session.row_paths, rows or []):
            value = parse_size(row[1]) if len(row) > 1 else None
            shown = session.shown.get(path)
            if value is None:
                if str(row[1]).strip() not in ("", "None", str(shown)):
                    errors.append(f"{row[0]}: '{row[1]}' is not a positive number")
            elif shown is None or abs(value - float(shown)) > 1e-9:
                edits[path] = value
        if errors:
            msg = card("Please fix these values", " · ".join(errors), "stop")
            review = self._review(session, self._fuse(session))
            review.ok, review.message_html = False, msg
            return review, Model(False, msg)
        session.edits, session.rejected, session.geometry = edits, tuple(rejected or ()), geometry
        review = self._review(session, self._fuse(session))
        if not review.ok:
            return review, Model(False, review.message_html)
        return review, self._model(session)

    def rebuild_geometry(self, sid: str, geometry: GeometrySettings) -> Model:
        session = self.store.get(sid)
        if session.observed is None:
            return Model(False, card("Nothing to build", "Analyze your images first.", "check"))
        session.geometry = geometry
        res = self._fuse(session)
        if isinstance(res, S.MvAbstain):
            return Model(False, _abstain_card(res))
        session.spec = res
        return self._model(session)

    def _model(self, session) -> Model:
        part = build_part(session.spec, session.geometry, self.root)
        if isinstance(part, S.MvAbstain):
            session.part = None
            return Model(False, _abstain_card(part))
        session.part, session.exported = part, None
        masks = session.observed.masks
        views = []
        for face, mask in part.views.items():
            score = f" · match {iou(mask, masks[face]):.2f}" if face in masks else ""
            views.append((255 - mask, f"{face}{score}"))
        x, y, z = part.bbox_mm
        stats = stats_html([("Size", f"{x:.1f} × {y:.1f} × {z:.1f} mm"), ("Volume", f"{part.volume_mm3 / 1000:.2f} cm³"),
                            ("Solid mass, PLA", f"{part.volume_mm3 * 1.24 / 1000:.1f} g")])
        note = " ".join(part.warnings)
        return Model(True, card("Part built", note or "Rotate the part, then choose formats and export.", "ok"),
                      str(part.preview), views, stats)

    # ---- export ---------------------------------------------------------------------------------------------
    def export(self, sid: str, export: ExportSettings, mesh: MeshSettings, printing: PrintSettings) -> Exported:
        session = self.store.get(sid)
        if session.part is None:
            return Exported(card("Nothing to export", "Build the part first.", "check"))
        sweep(self.root)
        res = export_part(session.part, export.formats, mesh, printing, self.pipe.slicer, self.pipe.profile)
        session.exported = res
        settings = {"mesh": mesh.model_dump(), "printing": printing.model_dump(), "ai": session.ai.model_dump(),
                    "geometry": session.geometry.model_dump(), "formats": export.formats}
        zip_path = bundle(session.part, res, settings)
        x, y, z = session.part.bbox_mm
        items = [("Size", f"{x:.1f} × {y:.1f} × {z:.1f} mm"), ("Files", str(len(res.files))),
                 ("Download", f"{zip_path.stat().st_size / 1024:.0f} KB")]
        if res.print_time_s:
            items += [("Print time", f"{res.print_time_s / 3600:.0f} h {res.print_time_s % 3600 / 60:.0f} min"),
                      ("Filament", f"{res.filament_g or 0:.1f} g")]
        tone = "check" if res.warnings else "ok"
        body = " · ".join(res.warnings) or "Every file is in the zip, with a manifest of the values and their sources."
        return Exported(card("Files ready", body, tone), [str(p) for p in res.files.values()], str(zip_path),
                        stats_html(items))
```

- [ ] **Step 7: `app.py` and the launcher**

```python
# s2c/studio/app.py
"""The Studio layout: one Walkthrough, Capture -> Review -> Model & export. Spec 2026-09-23-studio section 8.
Run: uv run python app_mv_studio.py"""
from __future__ import annotations

import gradio as gr

from s2c.multiview.pipeline import MvPipeline, default_pipeline
from s2c.multiview.settings import (
    EDGE_LABELS,
    FORMATS,
    MATERIALS,
    NOZZLES,
    AiSettings,
    ExportSettings,
    GeometrySettings,
    MeshSettings,
    PrintSettings,
)
from s2c.studio.handlers import AXIS_LABEL, AXES, FACE_CHOICES, KIND_CHOICES, REFERENCES, Studio
from s2c.studio.status import header_html, provider_status
from s2c.studio.theme import CSS, THEME, card

NOTE = ("**Photos of real parts:** shoot straight on, the part lying flat on a plain surface, with a coin, a card or "
        "an A4 sheet in frame. **Sketches:** dark pen on white paper, one face per sheet, sizes in mm.")


def _size_update(size: dict) -> dict:
    return gr.update(value=size.get("value", ""), placeholder=size.get("placeholder", "mm"), info=size.get("info"),
                     elem_classes=["size", "required"] if size.get("required") else ["size"])


def build_app(pipe: MvPipeline | None = None, studio: Studio | None = None) -> gr.Blocks:
    studio = studio or Studio(pipe or default_pipeline())
    status = provider_status(studio.pipe)
    with gr.Blocks(title="Sketch-to-CAD Studio", delete_cache=(3600, 3600)) as app:
        sid = gr.State(studio.store.new, time_to_live=3600, delete_callback=studio.store.drop)
        version = gr.State(0)
        gr.HTML(header_html(status))
        with gr.Walkthrough(selected=0) as walk:
            # ---- 1 capture ----
            with gr.Step("Capture", id=0):
                capture_msg = gr.HTML()
                with gr.Row():
                    with gr.Column(scale=3):
                        drop = gr.File(label="Drop sketches or photos: one or more per face (JPEG or PNG)",
                                       file_count="multiple", file_types=["image"], height=120)

                        @gr.render(inputs=[sid, version], triggers=[version.change, app.load])
                        def cards(session_id, _v):
                            items = studio.store.get(session_id).items
                            if not items:
                                gr.Markdown("*No images yet. Drop them above, or press **Try an example**.*")
                                return
                            with gr.Row(equal_height=False):
                                for item in items:
                                    with gr.Column(min_width=170, variant="panel"):
                                        gr.Image(item.path, height=140, interactive=False, show_label=False,
                                                 key=f"img-{item.id}")
                                        face = gr.Dropdown(FACE_CHOICES, value=item.face, label="Face",
                                                           key=f"face-{item.id}")
                                        kind = gr.Radio(KIND_CHOICES, value=item.kind, label="Type",
                                                        key=f"kind-{item.id}")
                                        remove = gr.Button("Remove", size="sm", key=f"rm-{item.id}")
                                    face.input(lambda v, s, i=item.id: (studio.set_face(s, i, v),
                                                                        studio.coverage_html(s))[1],
                                               [face, sid], [coverage])
                                    kind.input(lambda v, s, i=item.id: studio.set_kind(s, i, v), [kind, sid], None)
                                    remove.click(lambda s, n, i=item.id: (studio.remove(s, i), n + 1,
                                                                          studio.coverage_html(s))[1:],
                                                 [sid, version], [version, coverage])
                    with gr.Column(scale=2, min_width=320):
                        coverage = gr.HTML()
                        reference = gr.Dropdown(REFERENCES, value="none", label="Reference object in the photos",
                                                info="Gives real millimetres from a photo")
                        gr.Markdown(NOTE)
                        example = gr.Button("Try an example", variant="secondary")
                        with gr.Accordion("Reading & AI", open=False):
                            use_reader = gr.Checkbox(True, label="Read handwriting with Qwen-VL",
                                                     info="Off: TrOCR only", interactive=status["Qwen-VL"])
                            use_qwen = gr.Checkbox(True, label="Draw missing faces with Qwen-Image",
                                                   interactive=status["Qwen-Image"])
                            use_rescue = gr.Checkbox(True, label="Rescue sketches with an open outline",
                                                     info="The redraw is marked 'cleaned'",
                                                     interactive=status["Qwen-Image"])
                            use_triposr = gr.Checkbox(True, label="TripoSR fallback for missing faces",
                                                      interactive=status["TripoSR"])
                            use_solaria = gr.Checkbox(True, label="Hole depth from photos (Solaria)",
                                                      info="Photos only; adds 60–180 s", interactive=status["Solaria"])
                            seed = gr.Number(7, precision=0, minimum=0, maximum=2**31 - 1, label="Seed",
                                             info="Same seed, same drawing")
                            randomize = gr.Checkbox(False, label="Randomize seed")
                            attempts = gr.Slider(1, 4, value=2, step=1, label="Attempts per face",
                                                 info="Each attempt uses the next seed, about 30 s")
                        analyze = gr.Button("Analyze →", variant="primary")
            # ---- 2 review ----
            with gr.Step("Review", id=1):
                review_msg = gr.HTML()
                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Row():
                            sizes = {a: gr.Textbox(label=AXIS_LABEL[a], max_lines=1, elem_classes=["size"])
                                     for a in AXES}
                        values = gr.Dataframe(headers=["Field", "Value (mm)", "Source"], type="array",
                                              datatype=["str", "number", "html"], interactive=True,
                                              static_columns=[0, 2], column_widths=["45%", "20%", "35%"],
                                              label="Values: edit any number")
                        with gr.Accordion("Handwriting read", open=False):
                            reads = gr.Dataframe(headers=["Face", "Text", "Value", "Linked to"], type="array",
                                                 interactive=False)
                    with gr.Column(scale=2, min_width=320):
                        faces = gr.Gallery(label="Faces the part is built from", columns=3, height=260,
                                           object_fit="contain")
                        rejected = gr.CheckboxGroup([], label="Reject an AI face (use a rectangle)")
                        redraw = gr.Button("Redraw AI faces (next seeds)", size="sm")
                        warnings = gr.HTML()
                        build = gr.Button("Build part →", variant="primary")
            # ---- 3 model & export ----
            with gr.Step("Model & export", id=2):
                model_msg = gr.HTML()
                with gr.Row():
                    with gr.Column(scale=3):
                        model = gr.Model3D(label="Part", height=520, clear_color=(0, 0, 0, 0))
                        views = gr.Gallery(label="Rendered views (match against your images)", columns=3,
                                           height=200, object_fit="contain")
                    with gr.Column(scale=2, min_width=320):
                        stats = gr.HTML()
                        with gr.Accordion("Geometry", open=True):
                            snap = gr.Checkbox(True, label="Snap estimated values to standard sizes",
                                               info="Your own numbers are never snapped")
                            clearance = gr.Radio(["fine", "medium", "coarse"], value="medium",
                                                 label="Hole clearance class (ISO 273)")
                            finish = gr.Radio(["none", "fillet", "chamfer"], value="none", label="Edge finish")
                            finish_mm = gr.Slider(0.2, 10, value=1.0, step=0.1, label="Finish size (mm)")
                            finish_edges = gr.Dropdown([(v, k) for k, v in EDGE_LABELS.items()],
                                                       value="all_vertical", label="Edges")
                        with gr.Accordion("Mesh & export", open=False):
                            quality = gr.Radio(["draft", "normal", "fine"], value="normal", label="Mesh quality",
                                               info="Chord error 0.1 / 0.02 / 0.005 mm")
                        with gr.Accordion("3D print", open=False):
                            material = gr.Dropdown(list(MATERIALS), value="PLA", label="Material",
                                                   info="PLA 210/60 · PETG 240/80 · ABS 250/100 · ASA 255/100 · TPU 225/50 °C")
                            nozzle = gr.Dropdown(list(NOZZLES), value=0.4, label="Nozzle (mm)")
                            layer = gr.Slider(0.05, 0.32, value=0.2, step=0.01, label="Layer height (mm)",
                                              info="At most 0.75 × nozzle")
                            infill = gr.Slider(0, 100, value=20, step=5, label="Infill (%)")
                            pattern = gr.Dropdown(["grid", "gyroid", "rectilinear", "honeycomb", "cubic", "lightning"],
                                                  value="grid", label="Infill pattern")
                            perimeters = gr.Slider(1, 8, value=3, step=1, label="Perimeters")
                            supports = gr.Radio(["off", "buildplate", "everywhere"], value="buildplate",
                                                label="Supports")
                            brim = gr.Slider(0, 10, value=0, step=1, label="Brim (mm)")
                            scale = gr.Number(100, minimum=50, maximum=200, label="Print scale (%)",
                                              info="G-code only; the CAD files stay true size")
                        formats = gr.CheckboxGroup([(v, k) for k, v in FORMATS.items()],
                                                   value=ExportSettings().formats, label="Formats")
                        export = gr.Button("Export selected", variant="primary")
                        export_msg = gr.HTML()
                        download = gr.DownloadButton("Download all (.zip)", visible=False, variant="primary")
                        files = gr.File(label="Files", file_count="multiple", interactive=False)

        review_outputs = [review_msg, *sizes.values(), values, faces, rejected, warnings, reads, seed, build]

        def show_review(r) -> list:
            label = f"Build anyway ({r.unchecked} unchecked) →" if r.unchecked else "Build part →"
            return [r.message_html, *(_size_update(r.sizes.get(a, {})) for a in AXES), r.rows, r.faces,
                    gr.update(choices=r.ai_faces, value=[]), r.warnings_html, r.reads, r.seed, gr.update(value=label)]

        def ai_settings(*v) -> AiSettings:
            return AiSettings(use_reader=v[0], use_qwen_image=v[1], use_rescue=v[2], use_triposr=v[3],
                              use_solaria=v[4], seed=int(v[5] or 0), randomize_seed=v[6], attempts=int(v[7]))

        def geometry_settings(*v) -> GeometrySettings:
            return GeometrySettings(snap=v[0], clearance=v[1], finish=v[2], finish_mm=float(v[3]), finish_edges=v[4])

        def on_upload(paths, s, n):
            studio.add_images(s, paths)
            return None, n + 1, studio.coverage_html(s)

        def on_example(s, n):
            studio.load_examples(s)
            return n + 1, studio.coverage_html(s)

        def on_analyze(s, ref, *v, progress=gr.Progress()):
            progress(0.1, desc="Reading your images, then drawing any missing faces…")
            r = studio.analyze(s, ref, ai_settings(*v))
            if r.stage == "capture":
                return [gr.update(), r.message_html, *[gr.update()] * len(review_outputs)]
            return [gr.Walkthrough(selected=1), "", *show_review(r)]

        def on_redraw(s):
            return show_review(studio.redraw(s))

        def on_build(s, x, y, z, rows, rej, *g):
            try:
                geometry = geometry_settings(*g)
            except ValueError as e:
                msg = card("Check the geometry settings", str(e), "stop")
                return [gr.update(), msg, *[gr.update()] * (len(review_outputs) - 1), msg, None, [], ""]
            r, m = studio.build(s, {"x": x, "y": y, "z": z}, rows, rej, geometry)
            step = gr.Walkthrough(selected=2) if m.ok else gr.update()
            return [step, *show_review(r), m.message_html, m.preview, m.views, m.stats_html]

        def on_geometry(s, *g):
            try:
                m = studio.rebuild_geometry(s, geometry_settings(*g))
            except ValueError as e:
                return card("Check the geometry settings", str(e), "stop"), gr.update(), gr.update(), gr.update()
            return m.message_html, m.preview, m.views, m.stats_html

        def on_export(s, fmts, q, mat, noz, lay, inf, pat, per, sup, br, sc):
            try:
                printing = PrintSettings(material=mat, nozzle_mm=float(noz), layer_mm=float(lay), infill_pct=int(inf),
                                         infill_pattern=pat, perimeters=int(per), supports=sup, brim_mm=float(br),
                                         scale_pct=float(sc or 100))
                export_settings = ExportSettings(formats=list(fmts or []))
            except ValueError as e:
                return card("Check the print settings", str(e).split("\n")[-1], "stop"), gr.update(), [], gr.update()
            if not export_settings.formats:
                return card("Choose at least one format", "Tick the files you want.", "check"), gr.update(), [], \
                    gr.update()
            e = studio.export(s, export_settings, MeshSettings(quality=q), printing)
            button = gr.DownloadButton(value=e.zip_path, visible=e.zip_path is not None)
            return e.message_html, button, e.files, e.stats_html or gr.update()

        ai_inputs = [use_reader, use_qwen, use_rescue, use_triposr, use_solaria, seed, randomize, attempts]
        geometry_inputs = [snap, clearance, finish, finish_mm, finish_edges]
        drop.upload(on_upload, [drop, sid, version], [drop, version, coverage])
        example.click(on_example, [sid, version], [version, coverage])
        app.load(studio.coverage_html, [sid], [coverage])
        analyze.click(on_analyze, [sid, reference, *ai_inputs], [walk, capture_msg, *review_outputs],
                      concurrency_id="models", concurrency_limit=2)
        redraw.click(on_redraw, [sid], review_outputs, concurrency_id="models", concurrency_limit=2)
        build.click(on_build, [sid, *sizes.values(), values, rejected, *geometry_inputs],
                    [walk, *review_outputs, model_msg, model, views, stats], concurrency_id="cad")
        gr.on([snap.input, clearance.input, finish.input, finish_mm.release, finish_edges.input], on_geometry,
              [sid, *geometry_inputs], [model_msg, model, views, stats], trigger_mode="always_last",
              concurrency_id="cad")
        export.click(on_export, [sid, formats, quality, material, nozzle, layer, infill, pattern, perimeters,
                                 supports, brim, scale], [export_msg, download, files, stats], concurrency_id="cad")
    return app


def launch() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    build_app().queue(max_size=20, default_concurrency_limit=1).launch(theme=THEME, css=CSS, max_file_size="20mb")
```

```python
# app_mv_studio.py
"""Sketch-to-CAD Studio. Run: uv run python app_mv_studio.py, then open http://localhost:7860"""
from s2c.studio.app import launch

if __name__ == "__main__":
    launch()
```

- [ ] **Step 8: Run the tests**

Run: `uv run pytest tests/test_studio_ui.py -v`
Expected: PASS (10 tests). A Gradio 6.28 keyword that differs from the UX advisor's checked list is fixed against `inspect.signature` and recorded as a ruling; the layout and handler behaviour stay as written.

- [ ] **Step 9: See it run**

Start the Studio in the preview pane (`.claude/launch.json` in the session root has an entry for `app_mv_studio.py`), take a screenshot of each step, and drive the flow through `gradio_client` (`/on_example`, `/on_analyze`, `/on_build`, `/on_export`) with the example sketches and sizes 50 / 30 / 20. Expected: the Review step shows the sizes, the part builds, and the export zip holds the chosen files. Fix any layout error the screenshots show.

- [ ] **Step 10: Controller commit**

```bash
git add s2c/studio app_mv_studio.py scripts/make_examples.py examples/mv/sketches tests/test_studio_ui.py
git commit -m "Add the Sketch-to-CAD Studio: guided capture, review, model and export"
```

---

### Task 10: API and CLI parity, docs, end-to-end

**Files:**
- Modify: `s2c/multiview/routes.py`, `scripts/mv.py`, `README.md`, `docs/models.md`, `docs/disclosure.md`, `CLAUDE.md` (one line under "Stack and commands")
- Create: `tests/test_studio_api.py`

**Interfaces:**
- Consumes: `artifacts.build_part/export_part/bundle/ROOT`, `settings.StudioSettings`.
- Produces: `POST /mv/export` (`{spec, settings}` → `{key, files: {fmt: url}, zip_url, print_time_s, filament_g, warnings}` or `{abstain}`), `GET /mv/artifacts/{key}/{path}`; `scripts/mv.py` flags `--format --quality --material --layer --infill --supports --scale --seed`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_studio_api.py
import json
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from s2c.multiview import artifacts, routes

SPEC = json.loads((Path(__file__).parents[1] / "examples" / "mv" / "l_bracket.json").read_text())


def client(tmp_path, monkeypatch):
    artifacts.clear_cache()
    monkeypatch.setattr(routes, "ARTIFACT_ROOT", tmp_path)
    monkeypatch.setattr("s2c.multiview.slice.find_slicer", lambda: None)
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def test_export_returns_downloadable_files_and_a_zip(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch)
    body = c.post("/mv/export", json={"spec": SPEC, "settings": {"export": {"formats": ["stl", "dxf"]}}}).json()
    assert set(body["files"]) == {"stl", "dxf"}
    assert c.get(body["files"]["stl"]).status_code == 200 and c.get(body["zip_url"]).status_code == 200


def test_bad_settings_and_paths_are_refused(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch)
    assert c.post("/mv/export", json={"spec": SPEC, "settings": {"printing": {"layer_mm": 0.9}}}).status_code == 422
    assert c.get("/mv/artifacts/" + "0" * 20 + "/..%2F..%2Fsecret").status_code == 404
    assert c.get("/mv/artifacts/nothex/part.stl").status_code == 404


def test_the_cli_exports_formats(tmp_path, monkeypatch):
    from scripts import mv_export
    monkeypatch.setattr("s2c.multiview.slice.find_slicer", lambda: None)
    monkeypatch.setattr(sys, "argv", ["mv_export", str(Path(__file__).parents[1] / "examples" / "mv" / "l_bracket.json"),
                                      "--format", "stl", "--format", "step", "--out", str(tmp_path)])
    mv_export.main()
    assert any(p.name == "part.stl" for p in tmp_path.rglob("*")) and list(tmp_path.glob("*.zip"))
```

Ruling built into this task: the CLI for exports is a new `scripts/mv_export.py` (spec file in, files out) rather than more flags on `scripts/mv.py`, which stays the images-in tool; `scripts/mv.py` only gains `--format` passthrough by calling the same function. Make `scripts/` importable with an empty `scripts/__init__.py`.

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_studio_api.py -v`
Expected: FAIL (404 on `/mv/export`; `ModuleNotFoundError: scripts.mv_export`).

- [ ] **Step 3: Routes**

Append to `s2c/multiview/routes.py` (add the imports at the top: `from s2c.multiview.artifacts import ROOT as ARTIFACT_ROOT, build_part, bundle, export_part` and `from s2c.multiview.settings import StudioSettings`):

```python
_KEY = re.compile(r"^[0-9a-f]{20}$")


class ExportBody(BaseModel):
    spec: MultiViewSpec
    settings: StudioSettings = StudioSettings()


@router.post("/export")
def export_files(body: ExportBody) -> dict:
    part = build_part(body.spec, body.settings.geometry, ARTIFACT_ROOT)
    if isinstance(part, MvAbstain):
        return {"abstain": part.model_dump()}
    s = body.settings
    res = export_part(part, s.export.formats, s.mesh, s.printing)
    zip_path = bundle(part, res, s.model_dump(mode="json"))
    base = f"/mv/artifacts/{part.key}"
    return {"key": part.key,
            "files": {f: f"{base}/{p.relative_to(part.folder).as_posix()}" for f, p in res.files.items()},
            "zip_url": f"{base}/{zip_path.name}", "print_time_s": res.print_time_s, "filament_g": res.filament_g,
            "warnings": res.warnings}


@router.get("/artifacts/{key}/{path:path}")
def artifact(key: str, path: str) -> FileResponse:
    parts = path.split("/")
    root = (ARTIFACT_ROOT / key).resolve()
    target = (root / path).resolve() if _KEY.match(key) and all(_NAME.match(p) for p in parts) else None
    if target is None or root not in target.parents or not target.is_file():
        raise HTTPException(404, "File not found or expired.")
    return FileResponse(target)
```

Use `ARTIFACT_ROOT` through the module attribute (`routes.ARTIFACT_ROOT`) in both functions so tests can redirect it: write `build_part(body.spec, body.settings.geometry, ARTIFACT_ROOT)` as `build_part(..., globals()["ARTIFACT_ROOT"])` only if the plain name does not pick up the monkeypatch; the plain module-level name is read at call time, so it does.

- [ ] **Step 4: CLI**

```python
# scripts/mv_export.py
"""A MultiViewSpec JSON -> every chosen format and a zip, with the Studio's settings.

  uv run python scripts/mv_export.py examples/mv/l_bracket.json --format stl --format step --format gcode \
      --quality fine --material PETG --layer 0.2 --infill 30 --supports off --scale 100 --out tmp/export"""
import argparse
import shutil
from pathlib import Path

from s2c.multiview.artifacts import build_part, bundle, export_part
from s2c.multiview.settings import FORMATS, ExportSettings, MeshSettings, PrintSettings
from s2c.multiview.spec import MultiViewSpec, MvAbstain


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec")
    ap.add_argument("--format", action="append", choices=list(FORMATS), help="repeat for several")
    ap.add_argument("--quality", choices=["draft", "normal", "fine"], default="normal")
    ap.add_argument("--material", default="PLA")
    ap.add_argument("--layer", type=float, default=0.2)
    ap.add_argument("--infill", type=int, default=20)
    ap.add_argument("--supports", choices=["off", "buildplate", "everywhere"], default="buildplate")
    ap.add_argument("--scale", type=float, default=100.0)
    ap.add_argument("--out", default="tmp/export")
    args = ap.parse_args()
    spec = MultiViewSpec.model_validate_json(Path(args.spec).read_text())
    formats = ExportSettings(formats=args.format or ExportSettings().formats).formats
    printing = PrintSettings(material=args.material, layer_mm=args.layer, infill_pct=args.infill,
                             supports=args.supports, scale_pct=args.scale)
    part = build_part(spec)
    if isinstance(part, MvAbstain):
        raise SystemExit(f"{part.stage}: {part.reason}. {part.remedy}")
    res = export_part(part, formats, MeshSettings(quality=args.quality), printing)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in res.files.values():
        shutil.copy2(path, out / path.name)
    zip_path = shutil.copy2(bundle(part, res), out)
    for fmt, path in res.files.items():
        print(f"{fmt:<6} {out / path.name}")
    print(f"zip    {zip_path}")
    for w in res.warnings:
        print("warning:", w)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_studio_api.py tests/test_mv_routes.py -v`
Expected: PASS.

- [ ] **Step 6: Docs**

- `README.md`: under "Multi-view path", make the first command `uv run python app_mv_studio.py   # the Studio on :7860 (guided flow, parameters, every export)`, add `uv run python scripts/mv_export.py examples/mv/l_bracket.json --format stl --format step --format pdf`, and add one paragraph: formats (STL, STEP, 3MF, OBJ, GLB, PLY, BREP, Blender, DXF/SVG/PDF drawing, G-code, zip with manifest), and "Blender: set `BLENDER_PATH`, or run `scripts/setup_blender.ps1`; without it the download is a Blender kit".
- `docs/models.md`: add rows for ezdxf (drawings), trimesh (OBJ/GLB/PLY), Blender/bpy 4.2 (optional, separate process).
- `docs/disclosure.md`: state that the Studio keeps builds under `tmp/mv_gradio/` for one hour after last use and the zip's `manifest.json` lists the values, their sources and the settings.
- `CLAUDE.md`, "Stack and commands": add `uv run python app_mv_studio.py` for the Studio.

- [ ] **Step 7: End-to-end check and full suite**

Run: `uv run pytest -q` then `uv run ruff check s2c tests scripts app_mv_studio.py`
Expected: all pass (network tests skipped); no new ruff rule classes beyond the accepted `except Exception` fallbacks.

Start the Studio in the preview pane and repeat Task 9 step 9 on the final tree, including a G-code export with the vendored PrusaSlicer. Expected: print time and filament appear in the stats.

- [ ] **Step 8: Controller commit**

```bash
git add s2c/multiview/routes.py scripts/mv_export.py scripts/__init__.py README.md docs CLAUDE.md tests/test_studio_api.py
git commit -m "Add export to the API and command line, and document the Studio"
```
