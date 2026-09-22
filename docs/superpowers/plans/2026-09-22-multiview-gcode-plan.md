# Multi-view Reconstruction and G-code Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn 1 to 6 images of a part (sketches, photos or drawings of its faces) into a validated `MultiViewSpec`, a CadQuery solid (the visual hull of three outlines), and STEP, STL and FDM G-code files.

**Architecture:** One package, `s2c/multiview/`, one file per stage. `spec.py` is the contract. Model calls (vision label, TrOCR, TripoSR) are injected callables, so every stage is testable with fakes. Geometry is deterministic CadQuery code; G-code comes from PrusaSlicer CLI with a profile stored in the repo.

**Tech Stack:** Python 3.11, uv, pydantic 2, CadQuery 2.4+, OpenCV headless, NumPy, FastAPI, pytest. Optional `ai` extra: torch (CUDA 12.4), transformers (TrOCR), rembg, scikit-image, trimesh, gradio_client, TripoSR from GitHub.

**Spec:** `docs/superpowers/specs/2026-09-22-multiview-gcode-design.md`

## Global Constraints

- All dimensions are millimetre floats. No inches, no unit strings. (spec 3, first spec 3)
- Global frame: X width, Y height, Z depth; envelope from `(0, 0, 0)` to `(x_mm, y_mm, z_mm)`. Face frames exactly as the table in spec section 3.
- The envelope `x_mm`, `y_mm`, `z_mm` needs provenance `user_written`, `measured` or `user_edited`; otherwise `MvAbstain(stage="dimensions", reason="missing_x|missing_y|missing_z")`. (spec 4.1)
- The model never writes code. Model output is JSON validated by pydantic, or a mesh that is only rendered. (spec 2)
- Model estimates only for non-envelope values, provenance `estimated` or `inferred`. (spec 2)
- Snapping only for `scaled`, `inferred`, `estimated`; never for `user_written`, `user_edited`, `measured`. (spec 4.5)
- Every stage failure is an `MvAbstain` or a `BuildError` with a reason slug and a one-sentence remedy. (spec 2)
- Do not modify `s2c/partspec/`, `s2c/builder.py`, `s2c/views.py`, `s2c/merge.py`. (spec 1)
- Rasterise each triangle with its own `cv2.fillPoly` call. (spec 6.6)
- Never hard-code a provider or model name for the vision model; read `VLM_BASE_URL`, `VLM_MODEL`, `VLM_API_KEY`. (CLAUDE.md)
- Commit messages: plain, no AI attribution, no co-author trailers. (CLAUDE.md, spec 5 of the first spec)
- Write the failing test first. (CLAUDE.md)

## Not in this plan

- Web app screens (spec 7.2): `web/` does not exist yet. They get their own plan once the integrator's web scaffold is on `main`.
- Mounting `/mv` in `s2c/api.py`: the integrator does that with one `include_router` line. Until then `s2c/multiview/app.py` serves the router alone.

## File structure

| Path | Responsibility |
| --- | --- |
| `pyproject.toml`, `.env.example`, `s2c/__init__.py`, `tests/__init__.py` | Scaffold, identical to the integrator's Task 1 plus the `ai` extra and pytest markers |
| `s2c/multiview/spec.py` | Contract models, face frames, mirror rule |
| `s2c/multiview/raster.py` | `Mesh`, per-face masks, mask to outline, IoU |
| `s2c/multiview/build.py` | `BuildError`, `build`, `export`, `volume` |
| `s2c/multiview/slice.py` | Print orientation, bed check, PrusaSlicer call, G-code stats |
| `profiles/fdm_default.ini` | The one FDM profile |
| `s2c/multiview/outline.py` | Image to pixel outline, openings, circles |
| `s2c/multiview/reference.py` | Coin, card, A4 scale |
| `s2c/multiview/ocr.py` | Text regions, value parsing, linking, TrOCR reader |
| `s2c/multiview/label.py` | Vision-model face label, env chat |
| `s2c/multiview/fuse.py` | Envelope gate, outlines, features, snapping, assembly |
| `s2c/multiview/complete.py` | Orientation search, predicted and assumed outlines |
| `s2c/multiview/hf3d.py` | TripoSR local and Space providers |
| `s2c/multiview/pipeline.py` | `MvPipeline`: observe, fuse, build |
| `s2c/multiview/routes.py`, `s2c/multiview/app.py` | FastAPI router and stand-alone app |
| `scripts/mv_build.py`, `scripts/mv.py`, `scripts/setup_triposr.ps1` | Command-line demos and TripoSR setup |
| `examples/mv/l_bracket.json` | Hand-written spec for the first demo |
| `tests/mv_helpers.py` and `tests/test_mv_*.py` | Tests |
| `tests/golden_mv/README.md` | Capture protocol for real golden cases |

---

### Task 0: Scaffold

**Files:**
- Create: `pyproject.toml`, `.env.example`, `s2c/__init__.py`, `tests/__init__.py`, `tests/test_smoke.py`
- Modify: `.gitignore` (append)

**Interfaces:**
- Produces: importable package `s2c`; `uv run pytest` works; extras `ai`; markers `gpu`, `slicer`.

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/test_smoke.py
def test_package_imports():
    import s2c
    assert s2c.__version__ == "0.1.0"
```

- [ ] **Step 2: Run it to confirm failure**

Run: `uv run pytest tests/test_smoke.py -v`
Expected: FAIL, no project or `ModuleNotFoundError: No module named 's2c'`.

- [ ] **Step 3: Create the scaffold**

```toml
# pyproject.toml
[project]
name = "s2c"
version = "0.1.0"
description = "Sketch or photo to parametric CAD"
requires-python = ">=3.11,<3.13"
dependencies = [
  "pydantic>=2.7",
  "openai>=1.40",
  "fastapi>=0.115",
  "uvicorn[standard]>=0.30",
  "python-multipart>=0.0.9",
  "gradio>=4.40",
  "cadquery>=2.4",
  "opencv-python-headless>=4.10",
  "numpy>=1.26",
  "pillow>=10",
  "python-dotenv>=1.0",
]

[project.optional-dependencies]
ai = [
  "torch>=2.4",
  "transformers>=4.44",
  "rembg[cpu]>=2.0.57",
  "scikit-image>=0.24",
  "trimesh>=4.4",
  "omegaconf>=2.3",
  "einops>=0.7",
  "imageio>=2.34",
  "gradio_client>=1.3",
]

[dependency-groups]
dev = ["pytest>=8", "ruff>=0.6", "httpx>=0.27"]

[tool.uv.sources]
torch = { index = "pytorch-cu124" }

[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["s2c"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
  "gpu: needs a CUDA GPU and the ai extra",
  "slicer: needs PrusaSlicer installed",
]

[tool.ruff]
line-length = 120
```

```python
# s2c/__init__.py
__version__ = "0.1.0"
```

`tests/__init__.py` is empty.

```dotenv
# .env.example
VLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
VLM_MODEL=replace-with-a-current-vision-model
VLM_API_KEY=replace-me
# multi-view path
TRIPOSR_SPACE=stabilityai/TripoSR
# SLICER_PATH=C:/Program Files/Prusa3D/PrusaSlicer/prusa-slicer-console.exe
# SLICER_PROFILE=profiles/fdm_default.ini
```

Append to `.gitignore`:

```text
logs/
.env
tmp/
vendor/
```

- [ ] **Step 4: Install and run**

Run: `uv sync && uv run pytest tests/test_smoke.py -v`
Expected: PASS. If `cadquery` fails to resolve, run `uv python install 3.11` then `uv sync --python 3.11`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock .env.example .gitignore s2c/__init__.py tests/__init__.py tests/test_smoke.py
git commit -m "Scaffold s2c package with uv, pytest and the optional ai extra"
```

Note for the merge with the integrator's scaffold: the dependency list is theirs verbatim; this task only adds the `ai` extra, the torch index and the pytest markers.

---

### Task 1: Contract, face frames and mirror rule

**Files:**
- Create: `s2c/multiview/__init__.py`, `s2c/multiview/spec.py`, `tests/mv_helpers.py`
- Test: `tests/test_mv_spec.py`

**Interfaces:**
- Produces: `Envelope(x_mm, y_mm, z_mm)` with `.length(axis)`; `Outline(outer, inner, source, confidence)`; `FaceHole`, `FaceSlot`, `Fillet`, `Chamfer`; `Views(front, top, right)`; `MvAbstain(stage, reason, remedy, partial)`; `MultiViewSpec`; constants `FACES`, `CANONICAL_FACES`, `TRUSTED`, `CANONICAL_OF`, `FACE_AXES`, `AXIS_NAMES`; functions `numeric_names(model) -> list[str]`, `numeric_field_paths(spec) -> list[str]`, `face_size(face, env) -> (a_len, b_len)`, `to_global(face, a, b, env) -> dict[str, float]`, `to_canonical(face, points, env) -> list[(a, b)]`.
- Produces (tests): `tests.mv_helpers.rect(w, h, x0=0, y0=0)`, `circle(cx, cy, r, n=180)`, `outline(pts, inner=(), source="observed", confidence=0.9) -> dict`, `make_spec(env_tuple, front=None, top=None, right=None, features=(), finishes=(), prov="user_written") -> MultiViewSpec`, `box_mesh(x, y, z) -> Mesh`.

- [ ] **Step 1: Write the test helpers and the failing tests**

```python
# tests/mv_helpers.py
"""Shared builders for the multi-view tests."""
import math

import numpy as np

from s2c.multiview.spec import Chamfer, FaceHole, FaceSlot, Fillet, MultiViewSpec, numeric_names


def rect(w, h, x0=0.0, y0=0.0):
    return [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]


def circle(cx, cy, r, n=180):
    return [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]


def outline(pts, inner=(), source="observed", confidence=0.9):
    return {"outer": list(pts), "inner": [list(p) for p in inner], "source": source, "confidence": confidence}


_FEATURES = {"hole": FaceHole, "slot": FaceSlot}
_FINISHES = {"fillet": Fillet, "chamfer": Chamfer}


def make_spec(env, front=None, top=None, right=None, features=(), finishes=(), prov="user_written"):
    x, y, z = env
    provenance = {"envelope.x_mm": prov, "envelope.y_mm": prov, "envelope.z_mm": prov,
                  "views.front.outer": prov, "views.top.outer": prov, "views.right.outer": prov}
    for i, f in enumerate(features):
        provenance.update({f"features[{i}].{n}": prov for n in numeric_names(_FEATURES[f["type"]](**f))})
    for i, f in enumerate(finishes):
        provenance.update({f"finishes[{i}].{n}": prov for n in numeric_names(_FINISHES[f["type"]](**f))})
    return MultiViewSpec.model_validate({
        "envelope": {"x_mm": x, "y_mm": y, "z_mm": z},
        "views": {"front": front or outline(rect(x, y)), "top": top or outline(rect(x, z)),
                  "right": right or outline(rect(z, y))},
        "features": list(features), "finishes": list(finishes), "provenance": provenance, "confidence": 0.9,
    })


def box_mesh(x, y, z):
    from s2c.multiview.raster import Mesh
    v = np.array([[0, 0, 0], [x, 0, 0], [x, y, 0], [0, y, 0], [0, 0, z], [x, 0, z], [x, y, z], [0, y, z]], float)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return Mesh(v, f)
```

```python
# tests/test_mv_spec.py
import pytest
from pydantic import ValidationError

from s2c.multiview.spec import (CANONICAL_OF, FACES, Envelope, MultiViewSpec, face_size, numeric_field_paths,
                                to_canonical, to_global)
from tests.mv_helpers import make_spec, outline, rect

ENV = Envelope(x_mm=60.0, y_mm=40.0, z_mm=20.0)
HOLE = {"type": "hole", "face": "front", "a_mm": 10.0, "b_mm": 10.0, "diameter_mm": 6.0}


def test_face_sizes_follow_the_face_table():
    assert face_size("front", ENV) == (60, 40) and face_size("back", ENV) == (60, 40)
    assert face_size("top", ENV) == (60, 20) and face_size("bottom", ENV) == (60, 20)
    assert face_size("right", ENV) == (20, 40) and face_size("left", ENV) == (20, 40)


def test_canonical_faces_map_to_global_axes():
    assert to_global("front", 10, 5, ENV) == {"x": 10, "y": 5}
    assert to_global("top", 10, 5, ENV) == {"x": 10, "z": 15}    # b runs from the front edge backwards
    assert to_global("right", 5, 7, ENV) == {"z": 15, "y": 7}    # a runs from the front edge backwards


@pytest.mark.parametrize("face", FACES)
def test_mirror_rule_keeps_the_same_global_point(face):
    for a, b in [(0.0, 0.0), (12.5, 3.0), (7.0, 19.0)]:
        (ca, cb), = to_canonical(face, [(a, b)], ENV)
        assert to_global(face, a, b, ENV) == pytest.approx(to_global(CANONICAL_OF[face], ca, cb, ENV))


def test_valid_spec_round_trips_through_json():
    s = make_spec((60.0, 40.0, 20.0), features=[HOLE])
    assert MultiViewSpec.model_validate_json(s.model_dump_json()) == s


def test_numeric_field_paths():
    s = make_spec((60.0, 40.0, 20.0), features=[HOLE])
    assert numeric_field_paths(s) == [
        "envelope.x_mm", "envelope.y_mm", "envelope.z_mm", "views.front.outer", "views.top.outer",
        "views.right.outer", "features[0].a_mm", "features[0].b_mm", "features[0].diameter_mm"]


def test_envelope_needs_a_trusted_source():
    with pytest.raises(ValidationError, match="trusted"):
        make_spec((60.0, 40.0, 20.0), prov="estimated")


def test_missing_provenance_is_rejected():
    data = make_spec((60.0, 40.0, 20.0)).model_dump()
    del data["provenance"]["envelope.z_mm"]
    with pytest.raises(ValidationError, match="envelope.z_mm"):
        MultiViewSpec.model_validate(data)


def test_outline_outside_the_envelope_is_rejected():
    with pytest.raises(ValidationError, match="outside"):
        make_spec((60.0, 40.0, 20.0), front=outline(rect(70.0, 40.0)))


def test_invented_fields_are_rejected():
    data = make_spec((60.0, 40.0, 20.0)).model_dump()
    data["envelope"]["w_mm"] = 3.0
    with pytest.raises(ValidationError):
        MultiViewSpec.model_validate(data)
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_mv_spec.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 's2c.multiview'`.

- [ ] **Step 3: Implement**

```python
# s2c/multiview/__init__.py
"""Multi-view path: 1 to 6 face images -> visual hull -> STEP, STL, G-code. Spec 2026-09-22."""
```

```python
# s2c/multiview/spec.py
"""Multi-view contract, face frames and the mirror rule. Spec 2026-09-22, sections 3 and 5.
Owned by the geometry owner. Does not touch s2c/partspec/."""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

Mm = Annotated[float, Field(gt=0, description="millimetres")]
Face = Literal["front", "back", "left", "right", "top", "bottom"]
MvProvenance = Literal["user_written", "measured", "user_edited", "scaled", "inferred", "estimated", "default"]
ViewSource = Literal["observed", "mirrored", "inferred", "assumed"]
EdgeSelector = Literal["all", "all_vertical", "top", "bottom"]
Point = tuple[float, float]

FACES = ("front", "back", "left", "right", "top", "bottom")
CANONICAL_FACES = ("front", "top", "right")
TRUSTED = frozenset({"user_written", "measured", "user_edited"})
CANONICAL_OF = {"front": "front", "back": "front", "top": "top", "bottom": "top", "right": "right", "left": "right"}
# face -> (global axis of a, global axis of b, axis the face looks along)
FACE_AXES = {
    "front": ("x", "y", "z"), "back": ("x", "y", "z"),
    "top": ("x", "z", "y"), "bottom": ("x", "z", "y"),
    "right": ("z", "y", "x"), "left": ("z", "y", "x"),
}
AXIS_NAMES = {"x": "width", "y": "height", "z": "depth"}
POINT_TOLERANCE_MM = 0.5


class _Strict(BaseModel):
    """extra="forbid" rejects fields a model invents, such as an estimated width."""
    model_config = ConfigDict(extra="forbid")


class Envelope(_Strict):
    x_mm: Mm
    y_mm: Mm
    z_mm: Mm

    def length(self, axis: str) -> float:
        return {"x": self.x_mm, "y": self.y_mm, "z": self.z_mm}[axis]


class Outline(_Strict):
    outer: list[Point] = Field(min_length=3)
    inner: list[list[Point]] = []
    source: ViewSource
    confidence: float = Field(ge=0, le=1)


class FaceHole(_Strict):
    type: Literal["hole"] = "hole"
    face: Face
    a_mm: float
    b_mm: float
    diameter_mm: Mm
    depth_mm: Mm | None = None  # None means through along the face axis


class FaceSlot(_Strict):
    type: Literal["slot"] = "slot"
    face: Face
    a_mm: float
    b_mm: float
    width_mm: Mm
    length_mm: Mm  # end to end
    angle_deg: float = 0.0
    depth_mm: Mm | None = None


class Fillet(_Strict):
    type: Literal["fillet"] = "fillet"
    edges: EdgeSelector = "all_vertical"
    radius_mm: Mm


class Chamfer(_Strict):
    type: Literal["chamfer"] = "chamfer"
    edges: EdgeSelector = "all_vertical"
    radius_mm: Mm


FaceFeature = Annotated[Union[FaceHole, FaceSlot], Field(discriminator="type")]
Finish = Annotated[Union[Fillet, Chamfer], Field(discriminator="type")]


class Views(_Strict):
    front: Outline
    top: Outline
    right: Outline


class MvAbstain(_Strict):
    stage: Literal["label", "outline", "dimensions", "complete", "build", "slice", "verify"]
    reason: str
    remedy: str
    partial: dict | None = None


class MultiViewSpec(_Strict):
    version: Literal["mv1"] = "mv1"
    envelope: Envelope
    views: Views
    features: list[FaceFeature] = []
    finishes: list[Finish] = []
    provenance: dict[str, MvProvenance]
    snapped: list[str] = []
    warnings: list[str] = []
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _check(self) -> "MultiViewSpec":
        missing = [p for p in numeric_field_paths(self) if p not in self.provenance]
        if missing:
            raise ValueError(f"missing provenance for {missing}")
        for axis in ("x", "y", "z"):
            if self.provenance[f"envelope.{axis}_mm"] not in TRUSTED:
                raise ValueError(f"envelope.{axis}_mm needs a trusted source: written, measured or typed")
        tol = POINT_TOLERANCE_MM
        for face in CANONICAL_FACES:
            a_len, b_len = face_size(face, self.envelope)
            outline = getattr(self.views, face)
            for a, b in outline.outer + [p for loop in outline.inner for p in loop]:
                if not (-tol <= a <= a_len + tol and -tol <= b <= b_len + tol):
                    raise ValueError(f"views.{face} point ({a:g}, {b:g}) lies outside the envelope")
        return self


def numeric_names(model: BaseModel) -> list[str]:
    """Numeric fields of a feature or finish that need provenance. None (through) needs none."""
    return [k for k, v in model.model_dump().items()
            if k != "type" and not isinstance(v, bool) and isinstance(v, (int, float))]


def numeric_field_paths(spec: MultiViewSpec) -> list[str]:
    paths = ["envelope.x_mm", "envelope.y_mm", "envelope.z_mm"]
    paths += [f"views.{face}.outer" for face in CANONICAL_FACES]
    for i, f in enumerate(spec.features):
        paths += [f"features[{i}].{n}" for n in numeric_names(f)]
    for i, f in enumerate(spec.finishes):
        paths += [f"finishes[{i}].{n}" for n in numeric_names(f)]
    return paths


def face_size(face: str, env: Envelope) -> tuple[float, float]:
    """Width and height of the envelope rectangle seen from `face`, in that face's (a, b) frame."""
    a_axis, b_axis, _ = FACE_AXES[face]
    return env.length(a_axis), env.length(b_axis)


def to_global(face: str, a: float, b: float, env: Envelope) -> dict[str, float]:
    """The two global coordinates a face-frame point fixes (spec section 3 table)."""
    x, z = env.x_mm, env.z_mm
    return {
        "front": {"x": a, "y": b}, "back": {"x": x - a, "y": b},
        "top": {"x": a, "z": z - b}, "bottom": {"x": a, "z": b},
        "right": {"z": z - a, "y": b}, "left": {"z": a, "y": b},
    }[face]


def to_canonical(face: str, points, env: Envelope) -> list[Point]:
    """Mirror rule: points in `face`'s frame -> the frame of its canonical face (front, top or right)."""
    a_len, b_len = face_size(face, env)
    if face in ("back", "left"):
        return [(float(a_len - a), float(b)) for a, b in points]
    if face == "bottom":
        return [(float(a), float(b_len - b)) for a, b in points]
    return [(float(a), float(b)) for a, b in points]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_mv_spec.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add s2c/multiview/__init__.py s2c/multiview/spec.py tests/mv_helpers.py tests/test_mv_spec.py
git commit -m "Add MultiViewSpec contract with face frames and the mirror rule"
```

---

### Task 2: Rasteriser, mask to outline, IoU

**Files:**
- Create: `s2c/multiview/raster.py`
- Test: `tests/test_mv_raster.py`

**Interfaces:**
- Consumes: `Envelope`, `face_size` from Task 1.
- Produces: `Mesh(vertices, faces)`; `face_coords(face, pts, env) -> (n, 2)`; `mm_to_px(points, s, px) -> (n, 2)`; `face_mask(mesh, face, env, px=512) -> (mask, s)`; `polygon_mask(outer, holes=(), shape=(512, 512))`; `outline_mask(outer_mm, inner_mm, a_len, b_len, px=256)`; `mask_to_mm(mask, s, px, min_opening_mm=3.0, eps_mm=0.5) -> (outer, inner)`; `normalize_mask(mask, px=512)`; `iou(a, b) -> float`; `solid_mesh(solid, tolerance=0.05, angular=0.2) -> Mesh`; constant `MARGIN = 8`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mv_raster.py
import cadquery as cq
import numpy as np
import pytest

from s2c.multiview.raster import face_mask, iou, mask_to_mm, normalize_mask, outline_mask, solid_mesh
from s2c.multiview.spec import FACES, Envelope, face_size
from tests.mv_helpers import box_mesh, circle, rect

ENV = Envelope(x_mm=60.0, y_mm=40.0, z_mm=5.0)


def plate_with_hole():
    return cq.Workplane("XY").box(60, 40, 5, centered=False).cut(
        cq.Workplane("XY").center(20, 20).circle(5).extrude(5))


def test_box_front_is_filled_although_front_and_back_triangles_overlap():
    mask, s = face_mask(box_mesh(60, 40, 5), "front", ENV, px=512)
    expected = (60 * s) * (40 * s) / 512 ** 2
    assert abs((mask == 255).mean() - expected) < 0.01


@pytest.mark.parametrize("face", FACES)
def test_every_face_of_a_box_is_its_envelope_rectangle(face):
    mask, _ = face_mask(box_mesh(60, 40, 5), face, ENV, px=512)
    a, b = face_size(face, ENV)
    assert iou(mask, outline_mask(rect(a, b), [], a, b, px=512)) > 0.97


def test_solid_with_a_hole_matches_the_analytic_front():
    mask, _ = face_mask(solid_mesh(plate_with_hole()), "front", ENV, px=512)
    assert iou(mask, outline_mask(rect(60, 40), [circle(20, 20, 5)], 60, 40, px=512)) > 0.98


def test_mask_to_mm_recovers_outline_and_opening():
    mask, s = face_mask(solid_mesh(plate_with_hole()), "front", ENV, px=512)
    outer, inner = mask_to_mm(mask, s, 512)
    xs, ys = zip(*outer)
    assert min(xs) == pytest.approx(0, abs=0.3) and max(xs) == pytest.approx(60, abs=0.3)
    assert min(ys) == pytest.approx(0, abs=0.3) and max(ys) == pytest.approx(40, abs=0.3)
    assert len(inner) == 1
    ixs, _ = zip(*inner[0])
    assert max(ixs) - min(ixs) == pytest.approx(10, abs=0.5)


def test_iou_and_normalize():
    a = np.zeros((100, 100), np.uint8)
    a[10:30, 10:50] = 255
    b = np.zeros((100, 100), np.uint8)
    b[50:90, 20:100] = 255  # the same shape at twice the size
    assert iou(a, b) == 0.0
    assert iou(normalize_mask(a, 128), normalize_mask(b, 128)) > 0.95
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_mv_raster.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 's2c.multiview.raster'`.

- [ ] **Step 3: Implement**

```python
# s2c/multiview/raster.py
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
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_mv_raster.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add s2c/multiview/raster.py tests/test_mv_raster.py
git commit -m "Rasterise meshes per face with one fill per triangle"
```

---

### Task 3: Visual-hull builder and export

**Files:**
- Create: `s2c/multiview/build.py`
- Test: `tests/test_mv_build.py`

**Interfaces:**
- Consumes: `MultiViewSpec`, `Envelope`, `Outline`, `FaceHole`, `Fillet`, `FACE_AXES`, `face_size` from Task 1.
- Produces: `BuildError(reason, remedy)` with `.reason`, `.remedy`; `build(spec) -> cq.Workplane`; `export(solid, out_dir) -> (step_path, stl_path)`; `volume(solid) -> float`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mv_build.py
import math

import cadquery as cq
import pytest

from s2c.multiview.build import BuildError, build, export, volume
from tests.mv_helpers import circle, make_spec, outline, rect


def region_volume(solid, lo, hi):
    box = cq.Workplane("XY").box(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], centered=False).translate(lo)
    return volume(solid.intersect(box))


def hole(face, a, b, d, depth=None):
    return {"type": "hole", "face": face, "a_mm": a, "b_mm": b, "diameter_mm": d, "depth_mm": depth}


def test_three_rectangles_make_the_envelope_box():
    solid = build(make_spec((60.0, 40.0, 5.0)))
    assert math.isclose(volume(solid), 60 * 40 * 5, rel_tol=0.005)
    bb = solid.val().BoundingBox()
    assert [round(v, 6) for v in (bb.xmin, bb.ymin, bb.zmin, bb.xmax, bb.ymax, bb.zmax)] == [0, 0, 0, 60, 40, 5]


def test_front_outline_orientation():
    solid = build(make_spec((40.0, 30.0, 20.0), front=outline([(0, 0), (40, 0), (0, 30)])))
    assert math.isclose(volume(solid), 600 * 20, rel_tol=0.005)
    assert math.isclose(region_volume(solid, (0, 0, 0), (20, 30, 20)), 9000, rel_tol=0.01)


def test_top_outline_orientation():
    solid = build(make_spec((40.0, 30.0, 20.0), top=outline([(0, 0), (40, 0), (0, 20)])))
    assert math.isclose(volume(solid), 400 * 30, rel_tol=0.005)
    # b = 0 is the front edge (Z = z_mm), so the wide end of the triangle sits at the front
    assert math.isclose(region_volume(solid, (0, 0, 10), (40, 30, 20)), 9000, rel_tol=0.01)


def test_right_outline_orientation():
    solid = build(make_spec((40.0, 30.0, 20.0), right=outline([(0, 0), (20, 0), (0, 30)])))
    assert math.isclose(volume(solid), 300 * 40, rel_tol=0.005)
    # a = 0 is the front edge (Z = z_mm), so the tall end of the triangle sits at the front
    assert math.isclose(region_volume(solid, (0, 0, 10), (40, 30, 20)), 9000, rel_tol=0.01)


def test_through_holes():
    solid = build(make_spec((60.0, 40.0, 5.0), features=[hole("front", 10.0, 10.0, 6.0), hole("front", 50.0, 30.0, 6.0)]))
    assert math.isclose(volume(solid), 60 * 40 * 5 - 2 * math.pi * 9 * 5, rel_tol=0.005)


def test_blind_hole_on_the_back_is_cut_from_the_back():
    solid = build(make_spec((60.0, 40.0, 5.0), features=[hole("back", 10.0, 10.0, 6.0, 2.0)]))
    assert math.isclose(volume(solid), 60 * 40 * 5 - math.pi * 9 * 2, rel_tol=0.005)
    assert math.isclose(region_volume(solid, (0, 0, 0), (60, 40, 2)), 60 * 40 * 2 - math.pi * 9 * 2, rel_tol=0.005)


def test_hole_on_the_right_face_runs_along_x():
    solid = build(make_spec((60.0, 40.0, 20.0), features=[hole("right", 10.0, 20.0, 6.0)]))
    assert math.isclose(volume(solid), 60 * 40 * 20 - math.pi * 9 * 60, rel_tol=0.005)


def test_slot():
    slot = {"type": "slot", "face": "front", "a_mm": 30.0, "b_mm": 20.0, "width_mm": 6.0, "length_mm": 20.0}
    solid = build(make_spec((60.0, 40.0, 5.0), features=[slot]))
    assert math.isclose(volume(solid), 60 * 40 * 5 - ((20 - 6) * 6 + math.pi * 9) * 5, rel_tol=0.005)


def test_spacer_from_a_circle_and_two_rectangles():
    s = make_spec((20.0, 20.0, 30.0), front=outline(circle(10, 10, 10), inner=[circle(10, 10, 4)]))
    assert math.isclose(volume(build(s)), math.pi * (100 - 16) * 30, rel_tol=0.005)


def test_l_bracket_profile():
    l_shape = [(0, 0), (50, 0), (50, 30), (47, 30), (47, 3), (0, 3)]
    s = make_spec((50.0, 30.0, 20.0), front=outline(l_shape))
    assert math.isclose(volume(build(s)), (50 * 3 + 3 * 27) * 20, rel_tol=0.005)


def test_flange():
    feats = [hole("front", 40.0, 40.0, 30.0)] + [hole("front", float(a), float(b), 6.0)
                                                  for a, b in [(70, 40), (10, 40), (40, 70), (40, 10)]]
    s = make_spec((80.0, 80.0, 6.0), front=outline(circle(40, 40, 40, n=360)), features=feats)
    expected = math.pi * (40 ** 2 - 15 ** 2) * 6 - 4 * math.pi * 9 * 6
    assert math.isclose(volume(build(s)), expected, rel_tol=0.005)


def test_disjoint_views_are_rejected():
    s = make_spec((60.0, 40.0, 5.0), front=outline(rect(10.0, 40.0)), top=outline(rect(10.0, 5.0, x0=50.0)))
    with pytest.raises(BuildError) as e:
        build(s)
    assert e.value.reason == "intersection_empty"


def test_feature_outside_the_part_is_rejected():
    with pytest.raises(BuildError) as e:
        build(make_spec((60.0, 40.0, 5.0), features=[hole("front", 70.0, 10.0, 6.0)]))
    assert e.value.reason == "feature_outside_part"


def test_fillet_reduces_volume_and_a_huge_fillet_fails():
    v0 = volume(build(make_spec((60.0, 40.0, 5.0))))
    small = make_spec((60.0, 40.0, 5.0), finishes=[{"type": "fillet", "edges": "all_vertical", "radius_mm": 2.0}])
    assert volume(build(small)) < v0
    huge = make_spec((60.0, 40.0, 5.0), finishes=[{"type": "fillet", "edges": "all_vertical", "radius_mm": 30.0}])
    with pytest.raises(BuildError) as e:
        build(huge)
    assert e.value.reason == "fillet_failed"


def test_export_writes_step_and_stl(tmp_path):
    step, stl = export(build(make_spec((60.0, 40.0, 5.0))), tmp_path)
    assert step.read_text(errors="ignore").startswith("ISO-10303-21")
    assert stl.stat().st_size > 84
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_mv_build.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 's2c.multiview.build'`.

- [ ] **Step 3: Implement**

```python
# s2c/multiview/build.py
"""MultiViewSpec -> CadQuery solid: intersection of three extruded outlines, then face features and finishes.
Spec section 6.4. Deterministic; no model output is ever executed here."""
from __future__ import annotations

from pathlib import Path

import cadquery as cq

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
    cq.exporters.export(solid, str(step))
    cq.exporters.export(solid, str(stl), tolerance=0.01, angularTolerance=0.1)
    return step, stl
```

If an orientation test fails, print `_prism(face, ...).val().BoundingBox()` for the failing face: it must span the whole envelope. If it spans a negative range, the plane normal is flipped.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_mv_build.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add s2c/multiview/build.py tests/test_mv_build.py
git commit -m "Build the visual hull of three outlines with face features and finishes"
```

---

### Task 4: Print orientation, slicing and the first demo

**Files:**
- Create: `s2c/multiview/slice.py`, `profiles/fdm_default.ini`, `scripts/mv_build.py`, `examples/mv/l_bracket.json`
- Test: `tests/test_mv_slice.py`, `tests/test_mv_examples.py`

**Interfaces:**
- Consumes: `build`, `export` from Task 3; `MvAbstain` from Task 1.
- Produces: `Profile(path, bed_x, bed_y, max_height)` with `.center`; `SliceResult(print_stl, gcode, print_time_s, filament_g, warnings)`; `load_profile(path=None)`; `choose_down(solid) -> str`; `orient_for_print(solid)`; `fits_bed(solid, profile)`; `find_slicer() -> Path | None`; `parse_duration(text)`; `parse_gcode_stats(text) -> (seconds, grams)`; `slice_solid(solid, out_dir, profile_path=None, slicer=None) -> SliceResult | MvAbstain`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mv_slice.py
import math

import pytest

from s2c.multiview.build import build
from s2c.multiview.slice import (choose_down, find_slicer, load_profile, orient_for_print, parse_duration,
                                 parse_gcode_stats, slice_solid)
from s2c.multiview.spec import MvAbstain
from tests.mv_helpers import make_spec, outline

L_SHAPE = [(0, 0), (50, 0), (50, 30), (47, 30), (47, 3), (0, 3)]


def test_default_profile():
    p = load_profile()
    assert (p.bed_x, p.bed_y, p.max_height) == (220, 220, 250) and p.center == (110, 110)


def test_a_standing_plate_is_laid_flat():
    solid = build(make_spec((5.0, 40.0, 60.0)))
    assert choose_down(solid) in ("-x", "+x")
    bb = orient_for_print(solid).val().BoundingBox()
    assert math.isclose(bb.zlen, 5, abs_tol=1e-6) and math.isclose(bb.zmin, 0, abs_tol=1e-6)


def test_an_l_bracket_prints_on_its_long_leg():
    solid = build(make_spec((50.0, 30.0, 20.0), front=outline(L_SHAPE)))
    assert choose_down(solid) == "-y"
    assert math.isclose(orient_for_print(solid).val().BoundingBox().zlen, 30, abs_tol=1e-6)


def test_too_big_for_the_bed(tmp_path):
    res = slice_solid(build(make_spec((300.0, 20.0, 10.0))), tmp_path)
    assert isinstance(res, MvAbstain) and res.reason == "too_big_for_bed"


def test_without_a_slicer_the_print_stl_is_still_written(tmp_path, monkeypatch):
    monkeypatch.setattr("s2c.multiview.slice.find_slicer", lambda: None)
    res = slice_solid(build(make_spec((60.0, 40.0, 5.0))), tmp_path)
    assert res.gcode is None and res.print_stl.exists()
    assert res.warnings == ["G-code unavailable: slicer not installed"]


def test_gcode_stats():
    text = "G1 X1\n; filament used [g] = 12.34\n; estimated printing time (normal mode) = 1h 2m 3s\n"
    assert parse_gcode_stats(text) == (3723.0, 12.34)
    assert parse_duration("1d 0h 0m 5s") == 86405.0


@pytest.mark.slicer
@pytest.mark.skipif(find_slicer() is None, reason="PrusaSlicer not installed")
def test_real_slice_produces_gcode(tmp_path):
    res = slice_solid(build(make_spec((60.0, 40.0, 5.0))), tmp_path)
    text = res.gcode.read_text()
    assert "G1" in text
    assert abs(text.count(";LAYER_CHANGE") - 25) <= 1
    assert res.print_time_s > 0 and res.filament_g > 0
```

```python
# tests/test_mv_examples.py
from pathlib import Path

import pytest

from s2c.multiview.build import build, volume
from s2c.multiview.spec import MultiViewSpec

EXAMPLES = sorted((Path(__file__).parents[1] / "examples" / "mv").glob("*.json"))


@pytest.mark.parametrize("path", EXAMPLES, ids=[p.stem for p in EXAMPLES])
def test_example_specs_build(path):
    assert volume(build(MultiViewSpec.model_validate_json(path.read_text()))) > 0
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_mv_slice.py tests/test_mv_examples.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 's2c.multiview.slice'`; the examples test collects no cases yet.

- [ ] **Step 3: Implement the slicer stage, the profile, the example and the script**

```python
# s2c/multiview/slice.py
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
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=SLICE_TIMEOUT_S)
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
```

```ini
# profiles/fdm_default.ini
# Generic FDM profile for Sketch-to-CAD: Marlin, 220 x 220 x 250 bed, 0.4 mm nozzle, PLA. PrusaSlicer 2.x keys.
printer_technology = FFF
gcode_flavor = marlin2
bed_shape = 0x0,220x0,220x220,0x220
max_print_height = 250
nozzle_diameter = 0.4
filament_diameter = 1.75
filament_type = PLA
filament_density = 1.24
temperature = 210
first_layer_temperature = 215
bed_temperature = 60
first_layer_bed_temperature = 60
layer_height = 0.2
first_layer_height = 0.2
perimeters = 3
top_solid_layers = 5
bottom_solid_layers = 4
fill_density = 20%
fill_pattern = grid
support_material = 1
support_material_auto = 1
support_material_buildplate_only = 1
support_material_threshold = 45
skirts = 1
start_gcode = G28 ; home all axes\nG1 Z5 F5000 ; lift nozzle
end_gcode = M104 S0 ; heater off\nM140 S0 ; bed off\nG28 X0 ; home x\nM84 ; motors off
```

```json
// examples/mv/l_bracket.json
{
  "version": "mv1",
  "envelope": {"x_mm": 50.0, "y_mm": 30.0, "z_mm": 20.0},
  "views": {
    "front": {"outer": [[0, 0], [50, 0], [50, 30], [47, 30], [47, 3], [0, 3]], "inner": [], "source": "observed", "confidence": 1.0},
    "top": {"outer": [[0, 0], [50, 0], [50, 20], [0, 20]], "inner": [], "source": "observed", "confidence": 1.0},
    "right": {"outer": [[0, 0], [20, 0], [20, 30], [0, 30]], "inner": [], "source": "observed", "confidence": 1.0}
  },
  "features": [
    {"type": "hole", "face": "top", "a_mm": 20.0, "b_mm": 10.0, "diameter_mm": 5.5},
    {"type": "hole", "face": "right", "a_mm": 10.0, "b_mm": 18.0, "diameter_mm": 5.5}
  ],
  "finishes": [],
  "provenance": {
    "envelope.x_mm": "user_written", "envelope.y_mm": "user_written", "envelope.z_mm": "user_written",
    "views.front.outer": "user_written", "views.top.outer": "user_written", "views.right.outer": "user_written",
    "features[0].a_mm": "user_written", "features[0].b_mm": "user_written", "features[0].diameter_mm": "user_written",
    "features[1].a_mm": "user_written", "features[1].b_mm": "user_written", "features[1].diameter_mm": "user_written"
  },
  "confidence": 1.0
}
```

(Drop the `//` line when saving; JSON has no comments.)

```python
# scripts/mv_build.py
"""Hand-written MultiViewSpec -> STEP, STL and G-code.
Usage: uv run python scripts/mv_build.py examples/mv/l_bracket.json --out tmp/mv_demo"""
import argparse
from pathlib import Path

from s2c.multiview.build import BuildError, build, export
from s2c.multiview.slice import slice_solid
from s2c.multiview.spec import MultiViewSpec, MvAbstain


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("spec")
    ap.add_argument("--out", default="tmp/mv_demo")
    args = ap.parse_args()
    spec = MultiViewSpec.model_validate_json(Path(args.spec).read_text())
    try:
        solid = build(spec)
    except BuildError as e:
        raise SystemExit(f"build: {e.reason}. {e.remedy}")
    step, stl = export(solid, args.out)
    print(f"STEP    {step}\nSTL     {stl}")
    res = slice_solid(solid, args.out)
    if isinstance(res, MvAbstain):
        raise SystemExit(f"slice: {res.reason}. {res.remedy}")
    print(f"print   {res.print_stl}\nG-code  {res.gcode or 'none'}")
    if res.print_time_s is not None:
        print(f"time    {res.print_time_s / 60:.0f} min, filament {res.filament_g or 0:.1f} g")
    for w in res.warnings:
        print("warning:", w)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests and the demo**

Run: `uv run pytest tests/test_mv_slice.py tests/test_mv_examples.py -v`
Expected: all PASS; `test_real_slice_produces_gcode` is skipped until PrusaSlicer is installed.

Install PrusaSlicer (Windows): `winget install --id Prusa3D.PrusaSlicer -e` needs administrator rights. Without them, unzip the portable `PrusaSlicer-<version>.zip` from the GitHub release into `vendor/`; `find_slicer()` looks there. Then run the slicer test again: PASS.

Run: `uv run python scripts/mv_build.py examples/mv/l_bracket.json --out tmp/mv_demo`
Expected: paths for STEP, STL, print STL and G-code, then print time and filament. Open `tmp/mv_demo/part.step` in FreeCAD: a 50 x 30 x 20 L-bracket with one 5.5 mm hole on each leg.

- [ ] **Step 5: Commit**

```bash
git add s2c/multiview/slice.py profiles/fdm_default.ini scripts/mv_build.py examples/mv/l_bracket.json tests/test_mv_slice.py tests/test_mv_examples.py
git commit -m "Orient for print, slice with PrusaSlicer and add the spec-to-G-code demo"
```

---

### Task 5: Outline extraction

**Files:**
- Create: `s2c/multiview/outline.py`
- Test: `tests/test_mv_outline.py`

**Interfaces:**
- Consumes: `MvAbstain` from Task 1.
- Produces: `PixelCircle(cx, cy, d)`; `PixelOutline(outer, inner, circles, bbox, circular, shape)`; `resize_long_side(image, long_side=1600)`; `circularity(contour) -> float`; `foreground(image_bgr, mask_out=()) -> uint8 mask`; `extract(image_bgr, mask_out=()) -> PixelOutline | MvAbstain`; `to_face_mm(points_px, bbox, sa, sb) -> list[(a, b)]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mv_outline.py
import cv2
import numpy as np

from s2c.multiview.outline import extract, resize_long_side, to_face_mm
from s2c.multiview.spec import MvAbstain


def page(h=1200, w=1600, color=255):
    return np.full((h, w, 3), color, np.uint8)


def test_sketch_rectangle_outline():
    img = page()
    cv2.rectangle(img, (400, 300), (1000, 700), (0, 0, 0), 4)
    o = extract(img)
    x, y, w, h = o.bbox
    assert abs(x - 400) <= 5 and abs(w - 600) <= 10 and abs(h - 400) <= 10
    assert o.circles == [] and o.inner == []


def test_drawn_circles_are_holes():
    img = page()
    cv2.rectangle(img, (400, 300), (1000, 700), (0, 0, 0), 4)
    cv2.circle(img, (500, 400), 40, (0, 0, 0), 3)
    cv2.circle(img, (900, 600), 40, (0, 0, 0), 3)
    o = extract(img)
    assert len(o.circles) == 2
    assert all(68 <= c.d <= 82 for c in o.circles)  # inner edge of the pen stroke around an 80 px circle


def test_photo_of_a_part_with_a_hole():
    img = page()
    cv2.rectangle(img, (400, 300), (1000, 700), (60, 60, 60), -1)
    cv2.circle(img, (500, 400), 30, (255, 255, 255), -1)
    o = extract(img)
    assert len(o.circles) == 1 and abs(o.circles[0].d - 60) <= 3


def test_a_slot_is_an_opening_not_a_circle():
    img = page()
    cv2.rectangle(img, (400, 300), (1000, 700), (60, 60, 60), -1)
    cv2.rectangle(img, (600, 480), (800, 520), (255, 255, 255), -1)
    o = extract(img)
    assert o.circles == [] and len(o.inner) == 1


def test_light_part_on_a_dark_background():
    img = page(color=30)
    cv2.rectangle(img, (400, 300), (1000, 700), (220, 220, 220), -1)
    assert abs(extract(img).bbox[2] - 600) <= 6


def test_blank_page_abstains():
    res = extract(page())
    assert isinstance(res, MvAbstain) and res.reason == "no_outline"


def test_masked_region_is_ignored():
    img = page()
    cv2.rectangle(img, (100, 300), (500, 700), (60, 60, 60), -1)    # the part
    cv2.rectangle(img, (900, 100), (1550, 1100), (60, 60, 60), -1)  # a bigger object, masked out
    assert abs(extract(img, mask_out=[(880, 80, 700, 1050)]).bbox[0] - 100) <= 3


def test_resize_and_face_millimetres():
    assert resize_long_side(page(600, 800)).shape[:2] == (1200, 1600)
    pts = to_face_mm(np.array([[100, 300], [700, 100]]), (100, 100, 601, 201), 0.1, 0.2)
    assert pts == [(0.0, 0.0), (60.0, 40.0)]
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_mv_outline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 's2c.multiview.outline'`.

- [ ] **Step 3: Implement**

```python
# s2c/multiview/outline.py
"""Image -> outer outline, openings and circles, in pixels. Spec section 6.2.
One rule covers pen sketches (a drawn ring) and photos (a filled part)."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from s2c.multiview.spec import MvAbstain

LONG_SIDE = 1600
MIN_OUTLINE_FRACTION = 0.02
MIN_OPENING_FRACTION = 0.0003
CIRCULARITY = 0.85
EDGE_BAND_PX = 31  # the inside of a drawn outline touches this band next to the edge; a real opening does not


@dataclass
class PixelCircle:
    cx: float
    cy: float
    d: float


@dataclass
class PixelOutline:
    outer: np.ndarray                                    # (n, 2) image pixels
    inner: list[np.ndarray] = field(default_factory=list)
    circles: list[PixelCircle] = field(default_factory=list)
    bbox: tuple[int, int, int, int] = (0, 0, 1, 1)        # x, y, w, h of the outer outline
    circular: bool = False                               # the outer outline is itself a circle
    shape: tuple[int, int] = (1, 1)                      # image height, width


def resize_long_side(image: np.ndarray, long_side: int = LONG_SIDE) -> np.ndarray:
    h, w = image.shape[:2]
    s = long_side / max(h, w)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, (round(w * s), round(h * s)), interpolation=interp)


def circularity(contour) -> float:
    perimeter = cv2.arcLength(contour, True)
    return 0.0 if perimeter == 0 else float(4 * np.pi * cv2.contourArea(contour) / perimeter ** 2)


def foreground(image_bgr: np.ndarray, mask_out=()) -> np.ndarray:
    """Ink or part pixels are 255. The polarity is chosen so the image border is background."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    background = int(np.median(np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])))
    for x, y, w, h in mask_out:
        gray[max(y, 0): y + h, max(x, 0): x + w] = background
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.concatenate([th[0], th[-1], th[:, 0], th[:, -1]]).mean() > 127:
        th = 255 - th
    return cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))


def extract(image_bgr: np.ndarray, mask_out=()) -> PixelOutline | MvAbstain:
    fg = foreground(image_bgr, mask_out)
    h, w = fg.shape
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outer = max(contours, key=cv2.contourArea) if contours else None
    if outer is None or cv2.contourArea(outer) < MIN_OUTLINE_FRACTION * h * w:
        return MvAbstain(stage="outline", reason="no_outline",
                         remedy="Retake on a plain background with the whole part in frame.")
    filled = np.zeros_like(fg)
    cv2.drawContours(filled, [outer], -1, 255, -1)
    band = cv2.subtract(filled, cv2.erode(filled, np.ones((EDGE_BAND_PX, EDGE_BAND_PX), np.uint8)))
    gaps = cv2.morphologyEx(cv2.bitwise_and(filled, cv2.bitwise_not(fg)), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(gaps, connectivity=4)
    inner, circles = [], []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_OPENING_FRACTION * h * w:
            continue
        comp = np.where(labels == i, 255, 0).astype(np.uint8)
        if cv2.countNonZero(cv2.bitwise_and(comp, band)):
            continue  # the inside of a drawn outline, not an opening
        c = max(cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
        if circularity(c) >= CIRCULARITY:
            m = cv2.moments(comp, binaryImage=True)
            circles.append(PixelCircle(m["m10"] / m["m00"], m["m01"] / m["m00"], float(2 * np.sqrt(area / np.pi))))
        else:
            inner.append(cv2.approxPolyDP(c, 2.0, True).reshape(-1, 2))
    return PixelOutline(outer=cv2.approxPolyDP(outer, 2.0, True).reshape(-1, 2), inner=inner, circles=circles,
                        bbox=tuple(int(v) for v in cv2.boundingRect(outer)),
                        circular=circularity(outer) >= CIRCULARITY, shape=(h, w))


def to_face_mm(points_px, bbox, sa: float, sb: float) -> list[tuple[float, float]]:
    """Image pixels -> face-frame millimetres: a from the left of the bbox, b up from its bottom row."""
    x, y, _, h = bbox
    pts = np.asarray(points_px, np.float64).reshape(-1, 2)
    return [(float((u - x) * sa), float((y + h - 1 - v) * sb)) for u, v in pts]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_mv_outline.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add s2c/multiview/outline.py tests/test_mv_outline.py
git commit -m "Extract part outlines, openings and circles from sketches and photos"
```

---

### Task 6: Reference objects

**Files:**
- Create: `s2c/multiview/reference.py`
- Test: `tests/test_mv_reference.py`

**Interfaces:**
- Consumes: `foreground`, `extract` (test only) from Task 5; `MvAbstain`.
- Produces: `REFERENCES` table; `RefScale(name, mm_per_px, bbox, image, confidence)`; `find_reference(image_bgr, name) -> RefScale | MvAbstain` (raises `ValueError` for an unknown name).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mv_reference.py
import cv2
import numpy as np
import pytest

from s2c.multiview.outline import extract
from s2c.multiview.reference import find_reference
from s2c.multiview.spec import MvAbstain


def scene():
    img = np.full((1200, 1600, 3), 255, np.uint8)
    cv2.rectangle(img, (200, 200), (900, 800), (60, 60, 60), -1)  # the part
    return img


def test_coin_gives_the_scale():
    img = scene()
    cv2.circle(img, (1300, 600), 100, (40, 40, 40), -1)  # a 1 TND coin, 200 px across
    r = find_reference(img, "1 TND")
    assert abs(r.mm_per_px - 0.125) / 0.125 < 0.02
    assert r.bbox[0] <= 1200 and r.bbox[0] + r.bbox[2] >= 1400


def test_tilted_coin_abstains():
    img = scene()
    cv2.ellipse(img, (1300, 600), (100, 60), 0, 0, 360, (40, 40, 40), -1)
    res = find_reference(img, "1 TND")
    assert isinstance(res, MvAbstain) and res.reason == "coin_tilted"


def test_missing_coin_abstains():
    res = find_reference(scene(), "2 EUR")
    assert isinstance(res, MvAbstain) and res.reason == "coin_not_found"


def test_card_gives_the_scale():
    img = scene()
    cv2.rectangle(img, (1000, 300), (1428, 570), (50, 50, 50), -1)  # 428 x 270 px, a card at 5 px/mm
    r = find_reference(img, "card")
    assert abs(r.mm_per_px - 0.2) < 0.004


def test_a4_sheet_is_rectified():
    img = np.full((1200, 1600, 3), 40, np.uint8)
    cv2.rectangle(img, (250, 150), (1350, 928), (245, 245, 245), -1)   # A4 at 3.704 px/mm
    cv2.rectangle(img, (600, 400), (822, 548), (30, 30, 30), -1)       # a 60 x 40 mm part on it
    r = find_reference(img, "a4")
    assert r.mm_per_px == pytest.approx(0.2)
    o = extract(r.image)
    assert abs(o.bbox[2] * r.mm_per_px - 60) < 1.5 and abs(o.bbox[3] * r.mm_per_px - 40) < 1.5


def test_unknown_reference_name():
    with pytest.raises(ValueError):
        find_reference(scene(), "banana")
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_mv_reference.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 's2c.multiview.reference'`.

- [ ] **Step 3: Implement**

```python
# s2c/multiview/reference.py
"""Reference objects in a photo: a coin, a card or an A4 sheet gives millimetres per pixel. Spec section 4.3.
Coins are the numbers owner's domain; this detector stands in until metrology.py lands."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from s2c.multiview.outline import foreground
from s2c.multiview.spec import MvAbstain

REFERENCES = {
    "1 TND": ("coin", 25.0), "1 EUR": ("coin", 23.25), "2 EUR": ("coin", 25.75),
    "card": ("rect", (85.60, 53.98)), "a4": ("rect", (297.0, 210.0)),
}
MAX_ECCENTRICITY = 0.30
ASPECT_TOLERANCE = 0.03
A4_PX_PER_MM = 5.0
COIN_REMEDY = "Lay the coin flat, shoot top-down, retake."


@dataclass
class RefScale:
    name: str
    mm_per_px: float
    bbox: tuple[int, int, int, int] | None  # region to mask out before outline extraction
    image: np.ndarray                       # the image to use from now on (rectified for A4)
    confidence: float


def find_reference(image_bgr: np.ndarray, name: str) -> RefScale | MvAbstain:
    if name not in REFERENCES:
        raise ValueError(f"unknown reference object {name!r}; choose one of {sorted(REFERENCES)}")
    kind, size = REFERENCES[name]
    if kind == "coin":
        return _coin(image_bgr, name, size)
    return _a4(image_bgr) if name == "a4" else _card(image_bgr, size)


def _padded(x, y, w, h, pad=6):
    return x - pad, y - pad, w + 2 * pad, h + 2 * pad


def _ellipse(contour) -> tuple[float, float, float]:
    """Fitted ellipse axes (major, minor) and the worst relative distance of a contour point from it."""
    (cx, cy), (d1, d2), angle = cv2.fitEllipse(contour)
    t = np.deg2rad(angle)
    pts = contour.reshape(-1, 2).astype(np.float64) - (cx, cy)
    u = (pts[:, 0] * np.cos(t) + pts[:, 1] * np.sin(t)) / (d1 / 2)
    v = (-pts[:, 0] * np.sin(t) + pts[:, 1] * np.cos(t)) / (d2 / 2)
    return max(d1, d2), min(d1, d2), float(np.abs(np.hypot(u, v) - 1).max())


def _coin(img: np.ndarray, name: str, diameter: float) -> RefScale | MvAbstain:
    """The smallest blob that is an ellipse is the coin; the part is assumed larger than the coin.
    Candidates are found by ellipse fit, not by circularity, so a tilted coin reaches the tilt gate."""
    contours, _ = cv2.findContours(foreground(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    min_area = 0.0005 * img.shape[0] * img.shape[1]
    candidates = []
    for c in contours:
        if len(c) >= 20 and cv2.contourArea(c) > min_area:
            major, minor, deviation = _ellipse(c)
            if deviation <= 0.08:
                candidates.append((cv2.contourArea(c), c, major, minor))
    if not candidates:
        return MvAbstain(stage="dimensions", reason="coin_not_found", remedy=COIN_REMEDY)
    _, coin, major, minor = min(candidates, key=lambda e: e[0])
    eccentricity = float(np.sqrt(1 - (minor / major) ** 2))
    if eccentricity > MAX_ECCENTRICITY:
        return MvAbstain(stage="dimensions", reason="coin_tilted", remedy=COIN_REMEDY)
    return RefScale(name, diameter / major, _padded(*cv2.boundingRect(coin)), img, 1.0 - eccentricity)


def _quads(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(contours, key=cv2.contourArea):
        if cv2.contourArea(c) < 0.002 * mask.size:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            yield approx.reshape(4, 2).astype(np.float32)


def _order(q: np.ndarray) -> np.ndarray:
    """Corners as top-left, top-right, bottom-right, bottom-left."""
    s, d = q.sum(axis=1), np.diff(q, axis=1).ravel()
    return np.array([q[np.argmin(s)], q[np.argmin(d)], q[np.argmax(s)], q[np.argmax(d)]], np.float32)


def _sides(q: np.ndarray) -> tuple[float, float]:
    tl, tr, br, bl = _order(q)
    horizontal = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    vertical = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    return max(horizontal, vertical), min(horizontal, vertical)


def _aspect_ok(q: np.ndarray, long_mm: float, short_mm: float) -> bool:
    long_px, short_px = _sides(q)
    want = long_mm / short_mm
    return abs(long_px / short_px - want) <= ASPECT_TOLERANCE * want


def _card(img: np.ndarray, size) -> RefScale | MvAbstain:
    """Smallest quadrilateral with the ID-1 aspect ratio."""
    long_mm, short_mm = size
    for q in _quads(foreground(img)):
        if _aspect_ok(q, long_mm, short_mm):
            long_px, short_px = _sides(q)
            x, y, w, h = cv2.boundingRect(q.astype(np.int32))
            return RefScale("card", (long_mm / long_px + short_mm / short_px) / 2, _padded(x, y, w, h), img, 0.9)
    return MvAbstain(stage="dimensions", reason="card_not_found",
                     remedy="Place the card flat next to the part, fully in frame, and retake.")


def _a4(img: np.ndarray) -> RefScale | MvAbstain:
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    for q in sorted(_quads(bright), key=lambda q: -cv2.contourArea(q)):
        if _aspect_ok(q, 297.0, 210.0):
            return _rectify(img, q)
    return MvAbstain(stage="dimensions", reason="sheet_not_found",
                     remedy="Put the part on an A4 sheet with all four corners in frame, and retake.")


def _rectify(img: np.ndarray, q: np.ndarray) -> RefScale:
    """Warp the sheet to 5 px/mm, which also removes perspective, then drop its border."""
    tl, tr, br, bl = _order(q)
    landscape = np.linalg.norm(tr - tl) >= np.linalg.norm(bl - tl)
    w_mm, h_mm = (297.0, 210.0) if landscape else (210.0, 297.0)
    w, h = round(w_mm * A4_PX_PER_MM), round(h_mm * A4_PX_PER_MM)
    m = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl]),
                                    np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32))
    inset = int(3 * A4_PX_PER_MM)
    warped = cv2.warpPerspective(img, m, (w, h))[inset:-inset, inset:-inset].copy()
    return RefScale("a4", 1 / A4_PX_PER_MM, None, warped, 0.95)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_mv_reference.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add s2c/multiview/reference.py tests/test_mv_reference.py
git commit -m "Scale photos from a coin, a card or an A4 sheet"
```

---

### Task 7: Handwritten values: regions, parsing, linking, TrOCR

**Files:**
- Create: `s2c/multiview/ocr.py`
- Test: `tests/test_mv_ocr.py`

**Interfaces:**
- Consumes: `PixelOutline`, `PixelCircle`, `foreground`, `extract` (test only) from Task 5.
- Produces: `Reader = Callable[[np.ndarray], tuple[str, float]]`; `Reading(value_mm, kind, bbox, confidence, text)`; `Linked(reading, axis, hole_index)` with `axis` in `"a" | "b" | "ab" | None`; `parse_value(text) -> (value, kind) | None`; `text_regions(image_bgr, outline) -> list[bbox]`; `read_values(image_bgr, outline, reader) -> list[Reading]`; `link(readings, outline) -> list[Linked]`; `trocr_reader(model_name=...) -> Reader`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mv_ocr.py
import cv2
import numpy as np
import pytest

from s2c.multiview.ocr import Reading, link, parse_value, read_values, text_regions
from s2c.multiview.outline import PixelCircle, PixelOutline, extract


def test_parse_value():
    assert parse_value("60") == (60.0, "linear")
    assert parse_value(" 12,5 mm") == (12.5, "linear")
    assert parse_value("Ø6") == (6.0, "diameter")
    assert parse_value("⌀ 8") == (8.0, "diameter")
    assert parse_value("o6") == (6.0, "diameter")
    assert parse_value("R3") == (3.0, "radius")
    assert parse_value("D10") == (10.0, "diameter")
    assert parse_value("1O") == (10.0, "linear")
    assert parse_value("hello") is None
    assert parse_value("0") is None


def outline_with_holes(circular=False):
    return PixelOutline(outer=np.array([[400, 300], [1000, 300], [1000, 700], [400, 700]]),
                        circles=[PixelCircle(500, 400, 60), PixelCircle(900, 600, 60)],
                        bbox=(400, 300, 601, 401), circular=circular, shape=(1200, 1600))


def reading(value, kind, cx, cy):
    return Reading(float(value), kind, (cx - 20, cy - 15, 40, 30), 0.9, str(value))


def test_link_by_position():
    linked = link([reading(60, "linear", 700, 760), reading(40, "linear", 330, 500),
                   reading(6, "diameter", 560, 380), reading(99, "linear", 700, 500)], outline_with_holes())
    assert [(lv.axis, lv.hole_index) for lv in linked] == [("a", None), ("b", None), (None, 0), (None, None)]


def test_diameter_outside_a_round_outline_is_its_envelope():
    assert link([reading(80, "diameter", 700, 760)], outline_with_holes(circular=True))[0].axis == "ab"


def sketch_with_values():
    img = np.full((1200, 1600, 3), 255, np.uint8)
    cv2.rectangle(img, (400, 300), (1000, 700), (0, 0, 0), 4)
    cv2.putText(img, "60", (660, 790), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    cv2.putText(img, "40", (250, 520), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    return img


def test_text_regions_find_values_written_outside_the_outline():
    img = sketch_with_values()
    boxes = text_regions(img, extract(img))
    centres = sorted((x + w // 2, y + h // 2) for x, y, w, h in boxes)
    assert len(boxes) == 2
    assert abs(centres[0][0] - 290) < 40 and abs(centres[1][1] - 770) < 40


def test_read_values_keeps_numbers_and_drops_words():
    img = sketch_with_values()
    o = extract(img)
    assert [r.value_mm for r in read_values(img, o, lambda crop: ("60", 0.9))] == [60.0, 60.0]
    assert read_values(img, o, lambda crop: ("sixty", 0.9)) == []


@pytest.mark.gpu
def test_trocr_reads_printed_digits():
    pytest.importorskip("transformers")
    from s2c.multiview.ocr import trocr_reader
    img = np.full((80, 160, 3), 255, np.uint8)
    cv2.putText(img, "60", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    text, confidence = trocr_reader()(img)
    assert parse_value(text) == (60.0, "linear") and confidence > 0.3
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_mv_ocr.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 's2c.multiview.ocr'`.

- [ ] **Step 3: Implement**

```python
# s2c/multiview/ocr.py
"""Handwritten dimension values: find the text, read it, link it to a face axis or a hole. Spec section 4.2.
The reader only reads what the user wrote; it never estimates a size."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Literal

import cv2
import numpy as np

from s2c.multiview.outline import PixelOutline, foreground

log = logging.getLogger(__name__)

Reader = Callable[[np.ndarray], tuple[str, float]]  # BGR crop -> (text, confidence)
_DIAMETER_SIGNS = "⌀ØøΦφ∅"
_VALUE = re.compile(r"^(⌀|D|R)?(\d+(?:\.\d+)?)$")
STROKE_PX = 25


@dataclass
class Reading:
    value_mm: float
    kind: Literal["linear", "diameter", "radius"]
    bbox: tuple[int, int, int, int]
    confidence: float
    text: str


@dataclass
class Linked:
    reading: Reading
    axis: Literal["a", "b", "ab"] | None  # face axis; "ab" is the diameter of a round outline
    hole_index: int | None


def parse_value(text: str) -> tuple[float, str] | None:
    t = text.strip().replace(" ", "").replace(",", ".")
    for sign in _DIAMETER_SIGNS:
        t = t.replace(sign, "⌀")
    if t.lower().endswith("mm"):
        t = t[:-2]
    t = t.rstrip(".")
    if len(t) > 1 and t[0] in "oO" and t[1].isdigit():
        t = "⌀" + t[1:]  # a handwritten ⌀ is often read as o
    if len(t) > 1:
        t = t[0] + t[1:].replace("O", "0").replace("o", "0")
    m = _VALUE.match(t)
    if not m:
        return None
    value = float(m.group(2))
    if not 0 < value < 2000:
        return None
    return value, {"⌀": "diameter", "D": "diameter", "R": "radius"}.get(m.group(1) or "", "linear")


def text_regions(image_bgr: np.ndarray, outline: PixelOutline) -> list[tuple[int, int, int, int]]:
    """Boxes of written text: ink left after erasing the outline, the openings and long straight lines."""
    ink = foreground(image_bgr)
    erase = np.zeros_like(ink)
    cv2.drawContours(erase, [outline.outer.reshape(-1, 1, 2).astype(np.int32)], -1, 255, STROKE_PX)
    for loop in outline.inner:
        cv2.drawContours(erase, [loop.reshape(-1, 1, 2).astype(np.int32)], -1, 255, STROKE_PX)
    for c in outline.circles:
        cv2.circle(erase, (round(c.cx), round(c.cy)), round(c.d / 2), 255, STROKE_PX)
    text = cv2.bitwise_and(ink, cv2.bitwise_not(erase))
    lines = cv2.bitwise_or(cv2.morphologyEx(text, cv2.MORPH_OPEN, np.ones((1, 60), np.uint8)),
                           cv2.morphologyEx(text, cv2.MORPH_OPEN, np.ones((60, 1), np.uint8)))
    words = cv2.dilate(cv2.subtract(text, lines), np.ones((9, 25), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(words)
    boxes = []
    for i in range(1, n):
        x, y, w, h = (int(v) for v in stats[i, :4])
        if 15 <= h <= 220 and w >= 12 and w / h <= 8:
            boxes.append((x, y, w, h))
    return boxes


def read_values(image_bgr: np.ndarray, outline: PixelOutline, reader: Reader) -> list[Reading]:
    out = []
    for x, y, w, h in text_regions(image_bgr, outline):
        crop = image_bgr[max(y - 6, 0): y + h + 6, max(x - 6, 0): x + w + 6]
        text, confidence = reader(crop)
        parsed = parse_value(text)
        if parsed is None:
            log.info("dropped OCR read %r at %s", text, (x, y, w, h))
            continue
        out.append(Reading(parsed[0], parsed[1], (x, y, w, h), float(confidence), text))
    return out


def link(readings: list[Reading], outline: PixelOutline) -> list[Linked]:
    """Below or above the outline: axis a. Left or right: axis b. Diameters: the nearest hole."""
    bx, by, bw, bh = outline.bbox
    out = []
    for r in readings:
        x, y, w, h = r.bbox
        cx, cy = x + w / 2, y + h / 2
        inside = bx <= cx <= bx + bw and by <= cy <= by + bh
        if r.kind != "linear":
            if outline.circles and (inside or not outline.circular):
                k = min(range(len(outline.circles)),
                        key=lambda i: (outline.circles[i].cx - cx) ** 2 + (outline.circles[i].cy - cy) ** 2)
                out.append(Linked(r, None, k))
            else:
                out.append(Linked(r, "ab" if outline.circular else None, None))
        elif cy < by or cy > by + bh:
            out.append(Linked(r, "a", None))
        elif cx < bx or cx > bx + bw:
            out.append(Linked(r, "b", None))
        else:
            out.append(Linked(r, None, None))  # written inside the part: kept for the lab view, not linked
    return out


@lru_cache(maxsize=1)
def _trocr(model_name: str):
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device).eval()
    return processor, model, device


def trocr_reader(model_name: str = "microsoft/trocr-base-handwritten") -> Reader:
    """Fallback reader until the numbers owner's reader lands. The model loads on the first read."""
    import transformers  # noqa: F401  fail early when the ai extra is missing

    def read(crop_bgr: np.ndarray) -> tuple[str, float]:
        import torch
        from PIL import Image
        processor, model, device = _trocr(model_name)
        image = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        pixels = processor(images=image, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            out = model.generate(pixels, max_new_tokens=10, num_beams=1, output_scores=True,
                                 return_dict_in_generate=True)
        text = processor.batch_decode(out.sequences, skip_special_tokens=True)[0]
        scores = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
        confidence = float(torch.exp(scores.mean()).item()) if scores.numel() else 0.0
        return text, confidence

    return read
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_mv_ocr.py -v`
Expected: all PASS; the `gpu` test is skipped without the `ai` extra.

- [ ] **Step 5: Commit**

```bash
git add s2c/multiview/ocr.py tests/test_mv_ocr.py
git commit -m "Find, read and link handwritten dimension values"
```

---

### Task 8: Fusion: envelope gate, outlines, features, snapping

**Files:**
- Create: `s2c/multiview/fuse.py`
- Test: `tests/test_mv_fuse.py`

**Interfaces:**
- Consumes: Task 1 spec module (as `S`), `Linked`, `Reading` (Task 7), `PixelOutline`, `PixelCircle`, `to_face_mm` (Task 5), `iou`, `outline_mask` (Task 2).
- Produces: `Observation(face, kind, outline, values=[], mm_per_px=None, blind={}, depth_estimates={}, confidence=0.9)`; `attach_label(obs, label)`; `fuse_envelope(observations, user_values=None) -> (Envelope, prov, warnings) | MvAbstain`; `observed_outline(obs, env) -> Outline`; `canonical_outlines(observations, env) -> (dict[face, (Outline, prov)], warnings)`; `features_from(observations, env) -> (list[dict], prov)`; `snap_diameter(d)`, `snap_coord(v, length)`, `snap(data)`; `assemble(env, env_prov, outlines, feats, feat_prov, warnings, user_values=None, accepted=()) -> MultiViewSpec` (raises `ValidationError` on bad values).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mv_fuse.py
import numpy as np
import pytest

from s2c.multiview.fuse import (Observation, assemble, canonical_outlines, features_from, fuse_envelope, snap_coord,
                                snap_diameter)
from s2c.multiview.ocr import Linked, Reading
from s2c.multiview.outline import PixelCircle, PixelOutline
from s2c.multiview.spec import Envelope, MvAbstain, Outline


def px_outline(x=400, y=300, w=601, h=401, circles=()):
    outer = np.array([[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]])
    return PixelOutline(outer=outer, circles=list(circles), bbox=(x, y, w, h), shape=(1200, 1600))


def written(value, axis=None, hole=None, kind="linear", conf=0.9):
    return Linked(Reading(float(value), kind, (0, 0, 10, 10), conf, str(value)), axis, hole)


def front_obs(**kw):
    return Observation(face="front", kind="sketch", outline=px_outline(**kw),
                       values=[written(60, "a"), written(40, "b")])


def test_gate_asks_for_the_missing_depth():
    res = fuse_envelope([front_obs()])
    assert isinstance(res, MvAbstain) and res.reason == "missing_z" and res.remedy == "Enter the depth in mm."
    assert res.partial["known"] == {"envelope.x_mm": 60, "envelope.y_mm": 40}
    assert res.partial["missing"] == ["envelope.z_mm"]


def test_a_top_view_without_a_depth_value_gives_a_suggestion():
    top = Observation(face="top", kind="sketch", outline=px_outline(h=101), values=[written(60, "a")])
    res = fuse_envelope([front_obs(), top])
    assert isinstance(res, MvAbstain) and res.partial["suggested"] == {"envelope.z_mm": 10.0}


def test_a_typed_value_passes_the_gate():
    env, prov, _ = fuse_envelope([front_obs()], {"envelope.z_mm": 5})
    assert (env.x_mm, env.y_mm, env.z_mm) == (60, 40, 5)
    assert prov == {"envelope.x_mm": "user_written", "envelope.y_mm": "user_written", "envelope.z_mm": "user_edited"}


def test_written_beats_measured_with_a_warning():
    photo = Observation(face="front", kind="photo", outline=px_outline(), mm_per_px=0.11, values=[written(60, "a")])
    env, prov, warnings = fuse_envelope([photo], {"envelope.z_mm": 5})
    assert env.x_mm == 60 and prov["envelope.x_mm"] == "user_written"
    assert env.y_mm == pytest.approx(44.0) and prov["envelope.y_mm"] == "measured"
    assert any("using the written value" in w for w in warnings)


def test_nothing_written_and_nothing_measured_abstains_on_x_first():
    res = fuse_envelope([Observation(face="front", kind="sketch", outline=px_outline())])
    assert isinstance(res, MvAbstain) and res.reason == "missing_x"


def test_a_back_view_is_mirrored_into_the_front():
    env = Envelope(x_mm=60, y_mm=40, z_mm=5)
    tri = PixelOutline(outer=np.array([[400, 700], [1000, 700], [400, 300]]), bbox=(400, 300, 601, 401),
                       shape=(1200, 1600))
    outlines, _ = canonical_outlines([Observation(face="back", kind="sketch", outline=tri)], env)
    ol, prov = outlines["front"]
    assert ol.source == "mirrored" and prov == "scaled"
    assert sorted(ol.outer) == sorted([(60.0, 0.0), (0.0, 0.0), (60.0, 40.0)])


def test_written_diameter_wins_and_a_through_hole_seen_twice_is_kept_once():
    env = Envelope(x_mm=60, y_mm=40, z_mm=5)
    front = Observation(face="front", kind="sketch", outline=px_outline(circles=[PixelCircle(500, 600, 58)]),
                        values=[written(6, None, 0, "diameter")])
    back = Observation(face="back", kind="sketch", outline=px_outline(circles=[PixelCircle(900, 600, 60)]))
    feats, prov = features_from([front, back], env)
    assert len(feats) == 1
    assert feats[0]["diameter_mm"] == 6 and prov["features[0].diameter_mm"] == "user_written"
    assert feats[0]["a_mm"] == pytest.approx(10) and feats[0]["b_mm"] == pytest.approx(10)
    assert prov["features[0].a_mm"] == "scaled"


def test_blind_hole_depth_defaults_to_half_the_axis():
    env = Envelope(x_mm=60, y_mm=40, z_mm=6)
    o = Observation(face="front", kind="sketch", outline=px_outline(circles=[PixelCircle(500, 600, 60)]),
                    blind={0: True})
    feats, prov = features_from([o], env)
    assert feats[0]["depth_mm"] == 3 and prov["features[0].depth_mm"] == "default"


def test_snapping_tables():
    assert snap_diameter(5.37) == 5.5 and snap_diameter(7.2) == 7.0
    assert snap_coord(0.3, 60) == 0.0 and snap_coord(59.8, 60) == 60
    assert snap_coord(2.85, 60) == 3.0 and snap_coord(57.1, 60) == 57.0
    assert snap_coord(31.26, 60) == 31.5


def test_assemble_snaps_only_untrusted_values_and_applies_edits():
    env = Envelope(x_mm=60, y_mm=40, z_mm=5)
    o = Observation(face="front", kind="sketch", outline=px_outline(circles=[PixelCircle(500, 600, 53.7)]))
    outlines, _ = canonical_outlines([o], env)
    outlines["top"] = (Outline(outer=[(0, 0), (60, 0), (60, 5), (0, 5)], source="assumed", confidence=0.3), "default")
    outlines["right"] = (Outline(outer=[(0, 0), (5, 0), (5, 40), (0, 40)], source="assumed", confidence=0.3), "default")
    feats, fprov = features_from([o], env)
    env_prov = {"envelope.x_mm": "user_written", "envelope.y_mm": "user_written", "envelope.z_mm": "user_edited"}
    spec = assemble(env, env_prov, outlines, feats, fprov, [], user_values={"features[0].b_mm": 12.3})
    assert spec.features[0].diameter_mm == 5.5 and "features[0].diameter_mm" in spec.snapped
    assert spec.features[0].b_mm == 12.3 and spec.provenance["features[0].b_mm"] == "user_edited"
    assert spec.confidence == 0.3
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_mv_fuse.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 's2c.multiview.fuse'`.

- [ ] **Step 3: Implement**

```python
# s2c/multiview/fuse.py
"""Fuse per-image observations into a MultiViewSpec. Spec sections 4 and 5.
Envelope trust: typed > written > measured. Nothing else passes the gate."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from s2c.multiview import spec as S
from s2c.multiview.ocr import Linked, Reading
from s2c.multiview.outline import PixelOutline, to_face_mm
from s2c.multiview.raster import iou, outline_mask

CLEARANCE_MM = (2.7, 3.4, 4.5, 5.5, 6.6, 9.0)
THICKNESS_MM = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
SNAPPABLE = frozenset({"scaled", "inferred", "estimated"})
DISAGREE = 0.05
_FEATURE_PATH = re.compile(r"features\[(\d+)\]\.(\w+)")


@dataclass
class Observation:
    face: str
    kind: str                                   # sketch | photo | drawing
    outline: PixelOutline
    values: list[Linked] = field(default_factory=list)
    mm_per_px: float | None = None              # reference-object scale, photos only
    blind: dict[int, bool] = field(default_factory=dict)             # circle index -> blind, from the label
    depth_estimates: dict[int, float] = field(default_factory=dict)  # circle index -> mm, from the label
    confidence: float = 0.9


def attach_label(obs: Observation, label) -> None:
    """Match the label's holes (u, v in the bounding box) to the detected circles."""
    x, y, w, h = obs.outline.bbox
    for i, hole in enumerate(label.holes):
        if not obs.outline.circles:
            return
        dist = [((c.cx - x) / w - hole.u) ** 2 + ((y + h - c.cy) / h - hole.v) ** 2 for c in obs.outline.circles]
        k = int(np.argmin(dist))
        if dist[k] <= 0.15 ** 2:
            obs.blind[k] = hole.blind
            estimate = label.estimates.get(f"holes[{i}].depth_mm")
            if estimate is not None:
                obs.depth_estimates[k] = estimate


def _value(r: Reading) -> float:
    return r.value_mm * 2 if r.kind == "radius" else r.value_mm


def _differs(a: float, b: float) -> bool:
    return abs(a - b) / max(a, b) > DISAGREE


# ---- envelope ---------------------------------------------------------------

@dataclass
class _Candidate:
    value: float
    prov: str
    confidence: float
    face: str


def _envelope_candidates(observations: list[Observation]) -> dict[str, list[_Candidate]]:
    cands: dict[str, list[_Candidate]] = {"x": [], "y": [], "z": []}
    for o in observations:
        a_axis, b_axis, _ = S.FACE_AXES[o.face]
        _, _, w, h = o.outline.bbox
        for which, axis, px in (("a", a_axis, w - 1), ("b", b_axis, h - 1)):
            readings = [lv.reading for lv in o.values
                        if (lv.axis == which and lv.reading.kind == "linear") or lv.axis == "ab"]
            if readings:
                r = max(readings, key=_value)
                cands[axis].append(_Candidate(_value(r), "user_written", r.confidence, o.face))
            if o.mm_per_px:
                cands[axis].append(_Candidate(px * o.mm_per_px, "measured", o.confidence, o.face))
    return cands


def fuse_envelope(observations: list[Observation], user_values: dict | None = None):
    user_values = user_values or {}
    cands = _envelope_candidates(observations)
    values: dict[str, float] = {}
    prov: dict[str, str] = {}
    warnings: list[str] = []
    for axis in "xyz":
        key, name = f"envelope.{axis}_mm", S.AXIS_NAMES[axis]
        if key in user_values:
            values[axis], prov[key] = float(user_values[key]), "user_edited"
            continue
        written = [c for c in cands[axis] if c.prov == "user_written"]
        measured = [c for c in cands[axis] if c.prov == "measured"]
        if written:
            best = max(written, key=lambda c: c.confidence)
            for c in written:
                if c is not best and _differs(c.value, best.value):
                    warnings.append(f"{name}: {c.face} says {c.value:g} mm, {best.face} says {best.value:g} mm; "
                                    f"using {best.value:g}")
            for c in measured:
                if _differs(c.value, best.value):
                    warnings.append(f"{name}: written {best.value:g} mm, measured {c.value:.1f} mm; "
                                    "using the written value")
            values[axis], prov[key] = best.value, "user_written"
        elif measured:
            best = max(measured, key=lambda c: c.confidence)
            values[axis], prov[key] = round(best.value, 2), "measured"
    missing = [a for a in "xyz" if a not in values]
    if missing:
        first = missing[0]
        return S.MvAbstain(
            stage="dimensions", reason=f"missing_{first}", remedy=f"Enter the {S.AXIS_NAMES[first]} in mm.",
            partial={"known": {f"envelope.{a}_mm": v for a, v in values.items()},
                     "missing": [f"envelope.{a}_mm" for a in missing],
                     "suggested": _suggest(observations, values, missing)})
    return S.Envelope(x_mm=values["x"], y_mm=values["y"], z_mm=values["z"]), prov, warnings


def _suggest(observations, values, missing) -> dict[str, float]:
    """A scaled pre-fill for each missing axis, from a view that shows it next to a known axis."""
    out = {}
    for axis in missing:
        for o in observations:
            a_axis, b_axis, _ = S.FACE_AXES[o.face]
            _, _, w, h = o.outline.bbox
            if a_axis == axis and b_axis in values:
                out[f"envelope.{axis}_mm"] = round(values[b_axis] * (w - 1) / (h - 1) * 2) / 2
                break
            if b_axis == axis and a_axis in values:
                out[f"envelope.{axis}_mm"] = round(values[a_axis] * (h - 1) / (w - 1) * 2) / 2
                break
    return out


# ---- outlines -----------------------------------------------------------------

def _scales(o: Observation, env: S.Envelope) -> tuple[float, float]:
    a_len, b_len = S.face_size(o.face, env)
    _, _, w, h = o.outline.bbox
    return a_len / max(w - 1, 1), b_len / max(h - 1, 1)


def _clamp(pts, a_len, b_len):
    return [(min(max(a, 0.0), a_len), min(max(b, 0.0), b_len)) for a, b in pts]


def _outline_prov(o: Observation) -> str:
    return "measured" if o.mm_per_px else "scaled"


def observed_outline(o: Observation, env: S.Envelope) -> S.Outline:
    sa, sb = _scales(o, env)
    a_len, b_len = S.face_size(o.face, env)
    bbox = o.outline.bbox

    def mm(points):
        return S.to_canonical(o.face, _clamp(to_face_mm(points, bbox, sa, sb), a_len, b_len), env)

    source = "observed" if o.face in S.CANONICAL_FACES else "mirrored"
    return S.Outline(outer=mm(o.outline.outer), inner=[mm(loop) for loop in o.outline.inner], source=source,
                     confidence=o.confidence)


def canonical_outlines(observations: list[Observation], env: S.Envelope):
    """Best observed or mirrored outline per canonical face, with its provenance, and warnings."""
    out, warnings = {}, []
    for o in observations:
        if o.kind == "photo":
            a_len, b_len = S.face_size(o.face, env)
            _, _, w, h = o.outline.bbox
            if _differs((w - 1) / (h - 1), a_len / b_len):
                warnings.append(f"{o.face}: photo is not square-on, retake it facing the part")
    for face in S.CANONICAL_FACES:
        group = [(o, observed_outline(o, env)) for o in observations if S.CANONICAL_OF[o.face] == face]
        if not group:
            continue
        best_o, best = max(group, key=lambda t: (t[0].confidence, t[1].source == "observed"))
        a_len, b_len = S.face_size(face, env)
        ref = outline_mask(best.outer, best.inner, a_len, b_len)
        for o, ol in group:
            if o is not best_o and iou(outline_mask(ol.outer, ol.inner, a_len, b_len), ref) < 0.9:
                warnings.append(f"{o.face} and {best_o.face} outlines disagree; using {best_o.face}")
        out[face] = (best, _outline_prov(best_o))
    return out, warnings


# ---- features -------------------------------------------------------------------

def _duplicate_through(feats: list[dict], face: str, a: float, b: float, env: S.Envelope) -> bool:
    here = S.to_global(face, a, b, env)
    for f in feats:
        if f["depth_mm"] is None and S.CANONICAL_OF[f["face"]] == S.CANONICAL_OF[face]:
            there = S.to_global(f["face"], f["a_mm"], f["b_mm"], env)
            if all(abs(here[k] - there[k]) <= 1.0 for k in here):
                return True
    return False


def features_from(observations: list[Observation], env: S.Envelope):
    """Every circle becomes a hole on its own face; a through hole seen from both sides is kept once."""
    feats: list[dict] = []
    prov: dict[str, str] = {}
    for o in observations:
        sa, sb = _scales(o, env)
        written = {lv.hole_index: _value(lv.reading) for lv in o.values if lv.hole_index is not None}
        axis_len = env.length(S.FACE_AXES[o.face][2])
        for i, c in enumerate(o.outline.circles):
            (a, b), = to_face_mm(np.array([[c.cx, c.cy]]), o.outline.bbox, sa, sb)
            if i in written:
                d, d_prov = written[i], "user_written"
            elif o.mm_per_px:
                d, d_prov = c.d * o.mm_per_px, "measured"
            else:
                d, d_prov = c.d * (sa + sb) / 2, "scaled"
            depth, depth_prov = None, None
            if o.blind.get(i):
                if i in o.depth_estimates:
                    depth, depth_prov = min(o.depth_estimates[i], axis_len), "estimated"
                else:
                    depth, depth_prov = axis_len / 2, "default"
            elif _duplicate_through(feats, o.face, a, b, env):
                continue
            k = len(feats)
            feats.append({"type": "hole", "face": o.face, "a_mm": a, "b_mm": b, "diameter_mm": float(d),
                          "depth_mm": depth})
            pos = _outline_prov(o)
            prov.update({f"features[{k}].a_mm": pos, f"features[{k}].b_mm": pos, f"features[{k}].diameter_mm": d_prov})
            if depth is not None:
                prov[f"features[{k}].depth_mm"] = depth_prov
    return feats, prov


# ---- snapping -----------------------------------------------------------------

def _grid(v: float) -> float:
    return round(v * 2) / 2


def snap_diameter(d: float) -> float:
    best = min(CLEARANCE_MM, key=lambda c: abs(c - d))
    return best if abs(best - d) <= 0.4 else _grid(d)


def snap_coord(v: float, length: float) -> float:
    """Envelope edges exactly, thin walls to standard thicknesses, everything else to 0.5 mm."""
    if v <= 0.5:
        return 0.0
    if length - v <= 0.5:
        return float(length)
    wall = min(v, length - v)
    if wall < 12:
        t = min(THICKNESS_MM, key=lambda t: abs(t - wall))
        if abs(t - wall) <= 0.3:
            return t if v < length / 2 else float(length - t)
    return min(max(_grid(v), 0.0), float(length))


def snap(data: dict) -> None:
    """Snap scaled, inferred and estimated values of a spec dict in place (spec 4.5)."""
    prov, snapped = data["provenance"], data.setdefault("snapped", [])
    for k, f in enumerate(data["features"]):
        for name in ("a_mm", "b_mm", "diameter_mm", "depth_mm", "width_mm", "length_mm"):
            path = f"features[{k}].{name}"
            if f.get(name) is None or prov.get(path) not in SNAPPABLE:
                continue
            new = snap_diameter(f[name]) if name == "diameter_mm" else _grid(f[name])
            if new > 0 and abs(new - f[name]) > 1e-9:
                f[name] = new
                snapped.append(path)
    lengths = {"x": data["envelope"]["x_mm"], "y": data["envelope"]["y_mm"], "z": data["envelope"]["z_mm"]}
    for face in S.CANONICAL_FACES:
        path = f"views.{face}.outer"
        if prov.get(path) not in SNAPPABLE:
            continue
        a_axis, b_axis, _ = S.FACE_AXES[face]
        outline = data["views"][face]
        new = [(snap_coord(a, lengths[a_axis]), snap_coord(b, lengths[b_axis])) for a, b in outline["outer"]]
        if len(set(new)) >= 3 and new != [tuple(p) for p in outline["outer"]]:
            outline["outer"] = new
            snapped.append(path)


def assemble(env: S.Envelope, env_prov: dict, outlines: dict, feats: list[dict], feat_prov: dict,
             warnings: list[str], user_values: dict | None = None, accepted=()) -> S.MultiViewSpec:
    """outlines: canonical face -> (Outline, provenance). Applies the user's edits, then snapping."""
    data = {
        "envelope": env.model_dump(),
        "views": {face: ol.model_dump() for face, (ol, _) in outlines.items()},
        "features": [dict(f) for f in feats],
        "finishes": [],
        "provenance": {**env_prov, **{f"views.{face}.outer": p for face, (_, p) in outlines.items()}, **feat_prov},
        "warnings": list(dict.fromkeys(warnings)),
        "confidence": round(min(ol.confidence for ol, _ in outlines.values()), 3),
    }
    for face in accepted:
        if f"views.{face}.outer" in data["provenance"]:
            data["provenance"][f"views.{face}.outer"] = "user_edited"
    for path, value in (user_values or {}).items():
        m = _FEATURE_PATH.fullmatch(path)
        if m and int(m.group(1)) < len(data["features"]):
            data["features"][int(m.group(1))][m.group(2)] = float(value)
            data["provenance"][path] = "user_edited"
    snap(data)
    return S.MultiViewSpec.model_validate(data)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_mv_fuse.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add s2c/multiview/fuse.py tests/test_mv_fuse.py
git commit -m "Fuse observations with the envelope gate, provenance and standard-size snapping"
```

---

### Task 9: Vision-model face labels

**Files:**
- Create: `s2c/multiview/label.py`
- Test: `tests/test_mv_label.py`

**Interfaces:**
- Consumes: `MvAbstain` from Task 1.
- Produces: `Chat = Callable[[list[dict]], str]`; `LabelHole(u, v, blind)`; `MvLabel(face, input_kind, holes, description, estimates, confidence)`; `label_image(image_bytes, chat, face_hint=None, kind_hint=None) -> MvLabel | MvAbstain`; `hint_label(face, kind="sketch") -> MvLabel`; `env_chat(log_path="logs/vlm.jsonl") -> Chat | None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mv_label.py
import json

from s2c.multiview.label import MvLabel, hint_label, label_image
from s2c.multiview.spec import MvAbstain

GOOD = json.dumps({"face": "front", "input_kind": "sketch", "holes": [{"u": 0.2, "v": 0.3, "blind": True}],
                   "description": "plate", "estimates": {"holes[0].depth_mm": 3.0, "envelope.x_mm": 60},
                   "confidence": 0.8})


def chat_returning(*outputs):
    calls = []

    def chat(messages):
        calls.append(messages)
        return outputs[len(calls) - 1]

    chat.calls = calls
    return chat


def test_valid_label_keeps_hole_depths_and_drops_envelope_estimates():
    label = label_image(b"jpeg", chat_returning(GOOD))
    assert label.face == "front" and label.holes[0].blind
    assert label.estimates == {"holes[0].depth_mm": 3.0}


def test_retry_once_with_the_validation_error():
    chat = chat_returning("not json", GOOD)
    assert isinstance(label_image(b"jpeg", chat), MvLabel)
    assert "failed validation" in chat.calls[1][-1]["content"]


def test_two_failures_abstain():
    res = label_image(b"jpeg", chat_returning("nope", "{}"))
    assert isinstance(res, MvAbstain) and res.reason == "label_invalid"


def test_the_users_face_tag_wins_and_unknown_abstains():
    unknown = GOOD.replace('"front"', '"unknown"')
    assert label_image(b"jpeg", chat_returning(unknown), face_hint="top").face == "top"
    res = label_image(b"jpeg", chat_returning(unknown))
    assert isinstance(res, MvAbstain) and res.reason == "face_unknown"


def test_code_fences_are_stripped():
    assert label_image(b"jpeg", chat_returning(f"```json\n{GOOD}\n```")).face == "front"


def test_hint_label_without_a_model():
    label = hint_label("right", "photo")
    assert (label.face, label.input_kind) == ("right", "photo")
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_mv_label.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 's2c.multiview.label'`.

- [ ] **Step 3: Implement**

```python
# s2c/multiview/label.py
"""Vision model labels each image: which face, which holes are blind, optional hole-depth guesses. Spec 6.1.
The model returns JSON only. It never sets the envelope and never returns code."""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from s2c.multiview.spec import MvAbstain

log = logging.getLogger(__name__)
Chat = Callable[[list[dict]], str]
_ESTIMATE_KEY = re.compile(r"^holes\[\d+\]\.depth_mm$")


class LabelHole(BaseModel):
    model_config = ConfigDict(extra="forbid")
    u: float = Field(ge=0, le=1)
    v: float = Field(ge=0, le=1)
    blind: bool = False


class MvLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    face: Literal["front", "back", "left", "right", "top", "bottom", "unknown"]
    input_kind: Literal["sketch", "photo", "drawing"]
    holes: list[LabelHole] = []
    description: str = ""
    estimates: dict[str, float] = {}
    confidence: float = Field(ge=0, le=1)


SYSTEM_PROMPT = """You label one image of a mechanical part for a CAD tool. Reply with JSON only, no prose, matching this schema:
{schema}
Rules:
- face: the side of the part the image shows (front, back, left, right, top, bottom), or unknown.
- input_kind: sketch (hand drawn), photo (real part) or drawing (clean printed drawing).
- holes: every round hole, as u, v fractions of the part's bounding box (u to the right, v upward); blind is true if it does not go through.
- estimates: optional hole depth guesses in millimetres, only with keys like "holes[0].depth_mm".
- Never estimate the overall width, height or depth. Never output code."""


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        t = t.rsplit("```", 1)[0]
    return t.strip()


def label_image(image_bytes: bytes, chat: Chat, face_hint: str | None = None,
                kind_hint: str | None = None) -> MvLabel | MvAbstain:
    b64 = base64.b64encode(image_bytes).decode()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(schema=json.dumps(MvLabel.model_json_schema()))},
        {"role": "user", "content": [
            {"type": "text", "text": "Label this image."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]},
    ]
    label = None
    for _ in range(2):
        raw = chat(messages)
        try:
            label = MvLabel.model_validate_json(_strip_fences(raw))
            break
        except (ValidationError, ValueError) as e:
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": f"Your previous output failed validation: {e}. "
                                                     "Return corrected JSON only."}]
    if label is None:
        return MvAbstain(stage="label", reason="label_invalid",
                         remedy="The model could not describe this photo. Try a cleaner photo.")
    dropped = [k for k in label.estimates if not _ESTIMATE_KEY.match(k)]
    if dropped:
        log.warning("discarded model estimates %s: only hole depths may be estimated", dropped)
    label = label.model_copy(update={
        "estimates": {k: v for k, v in label.estimates.items() if _ESTIMATE_KEY.match(k) and v > 0},
        "face": face_hint or label.face, "input_kind": kind_hint or label.input_kind})
    if label.face == "unknown":
        return MvAbstain(stage="label", reason="face_unknown", remedy="Tell us which face this photo shows.")
    return label


def hint_label(face: str, kind: str = "sketch") -> MvLabel:
    """A label from the user's tags alone, when no vision model is configured."""
    return MvLabel(face=face, input_kind=kind, confidence=0.9)


def env_chat(log_path: str | Path = "logs/vlm.jsonl") -> Chat | None:
    """OpenAI-compatible chat from VLM_BASE_URL, VLM_MODEL, VLM_API_KEY; None when not configured.
    Swap for the integrator's VLMClient when s2c/vision/ lands."""
    base, model, key = (os.environ.get(k) for k in ("VLM_BASE_URL", "VLM_MODEL", "VLM_API_KEY"))
    if not (base and model and key):
        return None
    from openai import OpenAI
    client, path = OpenAI(base_url=base, api_key=key), Path(log_path)

    def chat(messages: list[dict]) -> str:
        t0 = time.perf_counter()
        r = client.chat.completions.create(model=model, messages=messages, temperature=0)
        usage = r.usage
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"provider": base, "model": model, "stage": "mv_label",
                                "latency_ms": round((time.perf_counter() - t0) * 1000),
                                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                                "completion_tokens": getattr(usage, "completion_tokens", None)}) + "\n")
        return r.choices[0].message.content or ""

    return chat
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_mv_label.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add s2c/multiview/label.py tests/test_mv_label.py
git commit -m "Label face images with the vision model, validated, retried once"
```

---

### Task 10: Completing missing faces

**Files:**
- Create: `s2c/multiview/complete.py`
- Test: `tests/test_mv_complete.py`

**Interfaces:**
- Consumes: `Mesh`, `face_mask`, `iou`, `mask_to_mm`, `normalize_mask`, `outline_mask`, `solid_mesh` (Task 2); `build` (Task 3, tests only); `CANONICAL_FACES`, `Envelope`, `Outline`, `face_size` (Task 1).
- Produces: `MeshProvider = Callable[[np.ndarray], Mesh]`; `rotations() -> list[3x3]`; `orient(mesh, face, target_mask) -> (rotation, iou)`; `fit_to_envelope(mesh, r, env) -> Mesh`; `predicted_outline(mesh, face, env, confidence) -> Outline`; `assumed_outline(face, env) -> Outline`; `complete(outlines, env, target_face, target_mask, image, provider, mesh=None, rejected=()) -> (outlines, warnings, mesh)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mv_complete.py
import numpy as np

from s2c.multiview.build import build
from s2c.multiview.complete import complete, rotations
from s2c.multiview.raster import Mesh, face_mask, iou, outline_mask, solid_mesh
from s2c.multiview.spec import Envelope, Outline
from tests.mv_helpers import box_mesh, make_spec, outline

ENV = Envelope(x_mm=50.0, y_mm=30.0, z_mm=20.0)
FRONT_L = [(0, 0), (50, 0), (50, 30), (44, 30), (44, 6), (0, 6)]
TOP_TAPER = [(0, 5), (50, 0), (50, 20), (0, 15)]
IMAGE = np.zeros((4, 4, 3), np.uint8)


def true_mesh():
    return solid_mesh(build(make_spec((50.0, 30.0, 20.0), front=outline(FRONT_L), top=outline(TOP_TAPER))))


def front_only():
    return {"front": Outline.model_validate(outline(FRONT_L))}


def test_24_rotations():
    rs = rotations()
    assert len(rs) == 24 and len({r.tobytes() for r in rs}) == 24
    assert all(round(np.linalg.det(r)) == 1 for r in rs)


def test_the_predicted_top_matches_the_true_part_in_any_orientation():
    mesh = true_mesh()
    target, _ = face_mask(mesh, "front", ENV)
    turned = Mesh(mesh.vertices @ rotations()[7].T, mesh.faces)
    result, _, _ = complete(front_only(), ENV, "front", target, IMAGE, lambda img: turned)
    top = result["top"]
    assert top.source == "inferred" and result["right"].source == "inferred"
    assert iou(outline_mask(top.outer, top.inner, 50, 20), outline_mask(TOP_TAPER, [], 50, 20)) > 0.95


def test_a_failing_provider_falls_back_to_rectangles():
    target, _ = face_mask(true_mesh(), "front", ENV)

    def broken(img):
        raise RuntimeError("no GPU")

    result, warnings, _ = complete(front_only(), ENV, "front", target, IMAGE, broken)
    assert result["top"].source == "assumed" and result["right"].source == "assumed"
    assert "3D predictor unavailable" in warnings


def test_an_unlike_mesh_is_unreliable():
    target, _ = face_mask(true_mesh(), "front", ENV)
    result, warnings, _ = complete(front_only(), ENV, "front", target, IMAGE, lambda img: box_mesh(50, 30, 20))
    assert result["top"].source == "assumed" and "predicted view unreliable" in warnings


def test_a_rejected_face_is_assumed_and_the_mesh_is_reused():
    mesh = true_mesh()
    target, _ = face_mask(mesh, "front", ENV)
    calls = []

    def provider(img):
        calls.append(1)
        return mesh

    result, _, cached = complete(front_only(), ENV, "front", target, IMAGE, provider, rejected=("right",))
    assert result["right"].source == "assumed" and result["top"].source == "inferred"
    complete(front_only(), ENV, "front", target, IMAGE, provider, mesh=cached)
    assert calls == [1]
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_mv_complete.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 's2c.multiview.complete'`.

- [ ] **Step 3: Implement**

```python
# s2c/multiview/complete.py
"""Fill the canonical faces nobody photographed: predicted from a TripoSR mesh, else assumed rectangular.
The mirror rule already ran in fuse. Spec section 6.3. The mesh is only rendered, never exported."""
from __future__ import annotations

import itertools
import logging
from typing import Callable

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
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_mv_complete.py -v`
Expected: all PASS. The orientation search renders 24 small masks per call; the whole file runs in under a minute.

- [ ] **Step 5: Commit**

```bash
git add s2c/multiview/complete.py tests/test_mv_complete.py
git commit -m "Predict missing faces from a mesh with an orientation search, else assume rectangles"
```

---

### Task 11: TripoSR providers

**Files:**
- Create: `s2c/multiview/hf3d.py`, `scripts/setup_triposr.ps1`
- Test: `tests/test_mv_hf3d.py`

**Interfaces:**
- Consumes: `Mesh` from Task 2.
- Produces: `local_triposr(image_bgr) -> Mesh`; `space_triposr(image_bgr) -> Mesh`; `default_provider() -> MeshProvider` (local, then Space, raising `RuntimeError` with both errors); `REPO_DIR`; `_install_mcubes_shim(force=False)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mv_hf3d.py
import sys

import cv2
import numpy as np
import pytest

from s2c.multiview import hf3d
from s2c.multiview.raster import Mesh

IMAGE = np.zeros((8, 8, 3), np.uint8)


def test_the_provider_tries_local_then_the_space(monkeypatch):
    calls = []

    def local(img):
        calls.append("local")
        raise RuntimeError("CUDA out of memory")

    def space(img):
        calls.append("space")
        return Mesh(np.zeros((3, 3)), np.array([[0, 1, 2]]))

    monkeypatch.setattr(hf3d, "local_triposr", local)
    monkeypatch.setattr(hf3d, "space_triposr", space)
    assert hf3d.default_provider()(IMAGE).faces.shape == (1, 3)
    assert calls == ["local", "space"]


def test_the_provider_raises_when_both_fail(monkeypatch):
    def down(img):
        raise RuntimeError("down")

    monkeypatch.setattr(hf3d, "local_triposr", down)
    monkeypatch.setattr(hf3d, "space_triposr", down)
    with pytest.raises(RuntimeError, match="down"):
        hf3d.default_provider()(IMAGE)


def test_the_marching_cubes_shim_returns_torchmcubes_axis_order():
    torch = pytest.importorskip("torch")
    pytest.importorskip("skimage")
    hf3d._install_mcubes_shim(force=True)
    n = 32
    zz, yy, xx = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    dist2 = ((xx - 20.0) ** 2 + (yy - 16.0) ** 2 + (zz - 12.0) ** 2).astype(np.float32)
    v, f = sys.modules["torchmcubes"].marching_cubes(torch.from_numpy(25.0 - dist2), 0.0)
    assert np.allclose(v.numpy().mean(axis=0), [20, 16, 12], atol=0.5)  # (x, y, z) for a volume indexed [z][y][x]
    assert f.shape[1] == 3


@pytest.mark.gpu
def test_real_triposr_on_the_gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not hf3d.REPO_DIR.exists():
        pytest.skip("needs CUDA and vendor/TripoSR")
    img = np.full((512, 512, 3), 255, np.uint8)
    cv2.rectangle(img, (150, 200), (360, 320), (40, 40, 40), -1)
    assert len(hf3d.local_triposr(img).faces) > 100
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_mv_hf3d.py -v`
Expected: FAIL with `ImportError: cannot import name 'hf3d'`.

- [ ] **Step 3: Implement**

```python
# s2c/multiview/hf3d.py
"""Single image -> mesh with TripoSR, on the local GPU or through the Hugging Face Space. Spec section 6.3.
The mesh is only rendered to silhouettes by complete.py; it is never exported."""
from __future__ import annotations

import concurrent.futures as cf
import inspect
import logging
import os
import sys
import tempfile
import types
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from s2c.multiview.raster import Mesh

log = logging.getLogger(__name__)
LOCAL_TIMEOUT_S = 60
SPACE_TIMEOUT_S = 90
DEFAULT_SPACE = "stabilityai/TripoSR"
REPO_DIR = Path(os.environ.get("TRIPOSR_DIR", Path(__file__).resolve().parents[2] / "vendor" / "TripoSR"))
_pool = cf.ThreadPoolExecutor(max_workers=1)


def _install_mcubes_shim(force: bool = False) -> None:
    """TripoSR imports torchmcubes, which needs a CUDA compiler on Windows; scikit-image does the same on CPU."""
    if not force:
        try:
            import torchmcubes  # noqa: F401
            return
        except ImportError:
            pass
    import torch
    from skimage.measure import marching_cubes as sk_marching_cubes

    def marching_cubes(vol, thresh):
        verts, faces, _, _ = sk_marching_cubes(vol.detach().cpu().numpy(), level=thresh)
        verts = np.ascontiguousarray(verts[:, ::-1])  # torchmcubes returns (x, y, z) for a volume indexed [z][y][x]
        return (torch.from_numpy(verts.copy()).float().to(vol.device),
                torch.from_numpy(faces.astype(np.int64)).to(vol.device))

    module = types.ModuleType("torchmcubes")
    module.marching_cubes = marching_cubes
    sys.modules["torchmcubes"] = module


@lru_cache(maxsize=1)
def _rembg_session():
    import rembg
    return rembg.new_session()


def _prepare(image_bgr: np.ndarray):
    """Background removed, part centred at 85 percent of a square, composited on grey, as TripoSR expects."""
    import rembg
    from PIL import Image
    rgba = np.asarray(rembg.remove(Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)),
                                   session=_rembg_session()))
    ys, xs = np.nonzero(rgba[:, :, 3] > 127)
    if len(xs) == 0:
        raise RuntimeError("background removal left nothing")
    crop = rgba[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    side = int(max(crop.shape[:2]) / 0.85)
    canvas = np.zeros((side, side, 4), np.uint8)
    y0, x0 = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    canvas[y0: y0 + crop.shape[0], x0: x0 + crop.shape[1]] = crop
    rgb, alpha = canvas[:, :, :3] / 255.0, canvas[:, :, 3:4] / 255.0
    return Image.fromarray(((rgb * alpha + (1 - alpha) * 0.5) * 255).astype(np.uint8))


def _to_mesh(tm) -> Mesh:
    return Mesh(np.asarray(tm.vertices, np.float64), np.asarray(tm.faces, np.int64))


@lru_cache(maxsize=1)
def _local_model():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device")
    if not REPO_DIR.exists():
        raise RuntimeError(f"TripoSR not found at {REPO_DIR}; run scripts/setup_triposr.ps1")
    sys.path.insert(0, str(REPO_DIR))
    _install_mcubes_shim()
    from tsr.system import TSR
    model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(4096)  # smaller chunks fit a 6 GB GPU
    return model.to("cuda")


def _local_run(image_bgr: np.ndarray) -> Mesh:
    import torch
    model = _local_model()
    image = _prepare(image_bgr)
    with torch.no_grad():
        codes = model([image], device="cuda")
        kwargs = {"resolution": 256}
        if "has_vertex_color" in inspect.signature(model.extract_mesh).parameters:
            kwargs["has_vertex_color"] = False
        meshes = model.extract_mesh(codes, **kwargs)
    return _to_mesh(meshes[0])


def local_triposr(image_bgr: np.ndarray) -> Mesh:
    _local_model()  # loads outside the timeout: the first call downloads the weights
    return _pool.submit(_local_run, image_bgr).result(timeout=LOCAL_TIMEOUT_S)


def space_triposr(image_bgr: np.ndarray) -> Mesh:
    import trimesh
    from gradio_client import Client, handle_file

    def run() -> Mesh:
        client = Client(os.environ.get("TRIPOSR_SPACE", DEFAULT_SPACE))
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "input.png"
            _prepare(image_bgr).save(src)
            processed = client.predict(handle_file(str(src)), False, 0.85, api_name="/preprocess")
            result = client.predict(handle_file(processed), 256, api_name="/generate")
            path = result[0] if isinstance(result, (list, tuple)) else result
            return _to_mesh(trimesh.load(path, force="mesh"))

    with cf.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(run).result(timeout=SPACE_TIMEOUT_S)


def default_provider():
    """Local GPU first, the Hugging Face Space second. Raises when both fail; complete.py then assumes."""
    def provide(image_bgr: np.ndarray) -> Mesh:
        errors = []
        for name, fn in (("local", local_triposr), ("space", space_triposr)):
            try:
                return fn(image_bgr)
            except Exception as e:
                log.warning("TripoSR %s failed: %s", name, e)
                errors.append(f"{name}: {e}")
        raise RuntimeError("; ".join(errors))
    return provide
```

```powershell
# scripts/setup_triposr.ps1
# Clones TripoSR into vendor/ and installs the ai extra (torch with CUDA 12.4). Run from the repo root.
$ErrorActionPreference = "Stop"
if (-not (Test-Path vendor/TripoSR)) { git clone --depth 1 https://github.com/VAST-AI-Research/TripoSR vendor/TripoSR }
uv sync --extra ai
uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_mv_hf3d.py -v`
Expected: the two provider tests PASS; the shim and GPU tests are skipped without the `ai` extra.

Then, on the laptop with the RTX 3050: `powershell -File scripts/setup_triposr.ps1` and `uv run pytest tests/test_mv_hf3d.py -v -m "gpu or not gpu"`.
Expected: all four PASS. If `tsr` fails to import with a transformers error, pin `transformers==4.35.0` in the `ai` extra, the version TripoSR was released with.

- [ ] **Step 5: Commit**

```bash
git add s2c/multiview/hf3d.py scripts/setup_triposr.ps1 tests/test_mv_hf3d.py
git commit -m "Predict a mesh with TripoSR on the local GPU or the Hugging Face Space"
```

---

### Task 12: Pipeline

**Files:**
- Create: `s2c/multiview/pipeline.py`
- Test: `tests/test_mv_pipeline.py`

**Interfaces:**
- Consumes: everything from Tasks 1 to 11.
- Produces: `ImageInput(data, face=None, kind=None)`; `Observed(observations, images, masks, labels, warnings, mesh)`; `BuildResult(step, stl, print_stl, gcode, print_time_s, filament_g, views, iou, warnings)`; `input_mask(outline) -> mask`; `MvPipeline(chat=None, reader=None, mesh_provider=None, slicer=None, profile=None)` with `.observe(images, reference=None) -> Observed | MvAbstain`, `.fuse(observed, user_values=None, accepted=(), rejected=()) -> MultiViewSpec | MvAbstain`, `.build(spec, out_dir, masks=None) -> BuildResult | MvAbstain`; `default_pipeline() -> MvPipeline`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mv_pipeline.py
import cv2
import numpy as np
import pytest

from s2c.multiview import pipeline
from s2c.multiview.ocr import Reading
from s2c.multiview.pipeline import ImageInput, MvPipeline
from s2c.multiview.spec import MvAbstain


def sketch(w_px, h_px, circles=()):
    img = np.full((1200, 1600, 3), 255, np.uint8)
    x0, y0 = (1600 - w_px) // 2, (1200 - h_px) // 2
    cv2.rectangle(img, (x0, y0), (x0 + w_px, y0 + h_px), (0, 0, 0), 4)
    for cx, cy, r in circles:
        cv2.circle(img, (x0 + cx, y0 + cy), r, (0, 0, 0), 3)
    return cv2.imencode(".png", img)[1].tobytes()


def fake_reads(per_image):
    calls = iter(per_image)

    def read_values(bgr, outline, reader):
        x, y, w, h = outline.bbox
        out = []
        for value, where in next(calls):
            box = (x + w // 2 - 20, y + h + 30, 40, 30) if where == "below" else (x - 80, y + h // 2 - 15, 40, 30)
            out.append(Reading(float(value), "linear", box, 0.9, str(value)))
        return out

    return read_values


def test_front_and_top_sketches_to_a_built_part(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "read_values", fake_reads([[(60, "below"), (40, "left")], [(60, "below")]]))
    pipe = MvPipeline(reader=lambda crop: ("", 0.0))
    observed = pipe.observe([ImageInput(sketch(600, 400, circles=[(100, 300, 30)]), "front", "sketch"),
                             ImageInput(sketch(600, 100), "top", "sketch")])
    first = pipe.fuse(observed)
    assert isinstance(first, MvAbstain) and first.reason == "missing_z"
    assert abs(first.partial["suggested"]["envelope.z_mm"] - 10) <= 0.5
    spec = pipe.fuse(observed, {"envelope.z_mm": 10.0})
    assert spec.views.top.source == "observed" and spec.views.right.source == "assumed"
    assert spec.features[0].diameter_mm == pytest.approx(5.5)  # about 57 px -> 5.7 mm -> M5 clearance
    result = pipe.build(spec, tmp_path, observed.masks)
    assert result.stl.exists() and result.step.exists()
    assert result.iou["front"] > 0.85 and result.iou["top"] > 0.85


def test_an_untagged_image_without_a_model_abstains():
    res = MvPipeline().observe([ImageInput(sketch(600, 400))])
    assert isinstance(res, MvAbstain) and res.reason == "face_unknown"


def test_a_file_that_is_not_an_image_abstains():
    res = MvPipeline().observe([ImageInput(b"not an image", "front")])
    assert isinstance(res, MvAbstain) and res.reason == "bad_image"


def test_build_errors_become_abstentions(tmp_path):
    from s2c.multiview.spec import MultiViewSpec
    pipe = MvPipeline()
    observed = pipe.observe([ImageInput(sketch(600, 400), "front", "sketch")])
    spec = pipe.fuse(observed, {"envelope.x_mm": 60, "envelope.y_mm": 40, "envelope.z_mm": 5})
    data = spec.model_dump()
    data["features"] = [{"type": "hole", "face": "front", "a_mm": 70.0, "b_mm": 10.0, "diameter_mm": 6.0}]
    data["provenance"].update({"features[0].a_mm": "user_edited", "features[0].b_mm": "user_edited",
                               "features[0].diameter_mm": "user_edited"})
    res = pipe.build(MultiViewSpec.model_validate(data), tmp_path)
    assert isinstance(res, MvAbstain) and res.stage == "build" and res.reason == "feature_outside_part"
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_mv_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 's2c.multiview.pipeline'`.

- [ ] **Step 3: Implement**

```python
# s2c/multiview/pipeline.py
"""The multi-view path end to end. Model calls happen in observe(); fuse() calls TripoSR at most once per
request and caches the mesh; build() never calls a model. Spec sections 6 and 7."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from pydantic import ValidationError

from s2c.multiview import spec as S
from s2c.multiview.build import BuildError, export
from s2c.multiview.build import build as build_solid
from s2c.multiview.complete import MeshProvider, complete
from s2c.multiview.fuse import Observation, assemble, attach_label, canonical_outlines, features_from, fuse_envelope
from s2c.multiview.label import Chat, MvLabel, env_chat, hint_label, label_image
from s2c.multiview.ocr import Reader, link, read_values
from s2c.multiview.outline import PixelOutline, extract, resize_long_side
from s2c.multiview.raster import Mesh, face_mask, iou, normalize_mask, polygon_mask, solid_mesh
from s2c.multiview.reference import find_reference
from s2c.multiview.slice import slice_solid

log = logging.getLogger(__name__)
IOU_GREEN = 0.85


@dataclass
class ImageInput:
    data: bytes
    face: str | None = None
    kind: str | None = None


@dataclass
class Observed:
    """Everything that needed a model call, cached per request so merge never repeats it."""
    observations: list[Observation]
    images: list[np.ndarray]
    masks: dict[str, np.ndarray]
    labels: list[MvLabel]
    warnings: list[str] = field(default_factory=list)
    mesh: Mesh | None = None


@dataclass
class BuildResult:
    step: Path
    stl: Path
    print_stl: Path | None
    gcode: Path | None
    print_time_s: float | None
    filament_g: float | None
    views: dict[str, np.ndarray]
    iou: dict[str, float]
    warnings: list[str]


def input_mask(outline: PixelOutline) -> np.ndarray:
    """The input silhouette of one image: outer outline filled, openings and circles cut out, normalised."""
    holes = list(outline.inner)
    holes += [cv2.ellipse2Poly((round(c.cx), round(c.cy)), (round(c.d / 2), round(c.d / 2)), 0, 0, 360, 5)
              for c in outline.circles]
    return normalize_mask(polygon_mask(outline.outer, holes, outline.shape))


class MvPipeline:
    def __init__(self, chat: Chat | None = None, reader: Reader | None = None,
                 mesh_provider: MeshProvider | None = None, slicer: Path | None = None, profile: Path | None = None):
        self.chat, self.reader, self.mesh_provider = chat, reader, mesh_provider
        self.slicer, self.profile = slicer, profile

    def _label(self, item: ImageInput) -> MvLabel | S.MvAbstain:
        if self.chat is not None:
            return label_image(item.data, self.chat, item.face, item.kind)
        if item.face:
            return hint_label(item.face, item.kind or "sketch")
        return S.MvAbstain(stage="label", reason="face_unknown", remedy="Tell us which face this photo shows.")

    def observe(self, images: list[ImageInput], reference: str | None = None) -> Observed | S.MvAbstain:
        observed = Observed([], [], {}, [])
        if self.reader is None:
            observed.warnings.append("OCR unavailable: enter the dimensions by hand")
        for item in images:
            bgr = cv2.imdecode(np.frombuffer(item.data, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                return S.MvAbstain(stage="outline", reason="bad_image",
                                   remedy="The file is not an image. Upload a JPEG or PNG.")
            bgr = resize_long_side(bgr)
            label = self._label(item)
            if isinstance(label, S.MvAbstain):
                return label
            mm_per_px, mask_out = None, ()
            if reference and label.input_kind == "photo":
                ref = find_reference(bgr, reference)
                if isinstance(ref, S.MvAbstain):
                    return ref
                bgr, mm_per_px = ref.image, ref.mm_per_px
                mask_out = (ref.bbox,) if ref.bbox else ()
            outline = extract(bgr, mask_out)
            if isinstance(outline, S.MvAbstain):
                return outline
            values = []
            if self.reader is not None and label.input_kind != "photo":
                values = link(read_values(bgr, outline, self.reader), outline)
            obs = Observation(face=label.face, kind=label.input_kind, outline=outline, values=values,
                              mm_per_px=mm_per_px, confidence=label.confidence)
            attach_label(obs, label)
            observed.observations.append(obs)
            observed.images.append(bgr)
            observed.labels.append(label)
            observed.masks[label.face] = input_mask(outline)
        return observed

    def fuse(self, observed: Observed, user_values: dict | None = None, accepted=(),
             rejected=()) -> S.MultiViewSpec | S.MvAbstain:
        env_result = fuse_envelope(observed.observations, user_values)
        if isinstance(env_result, S.MvAbstain):
            return env_result
        env, env_prov, warnings = env_result
        outlines, more = canonical_outlines(observed.observations, env)
        warnings = observed.warnings + warnings + more
        best = max(range(len(observed.observations)), key=lambda i: observed.observations[i].confidence)
        target = observed.observations[best]
        full, more, observed.mesh = complete({f: ol for f, (ol, _) in outlines.items()}, env, target.face,
                                             observed.masks[target.face], observed.images[best],
                                             self.mesh_provider, observed.mesh, tuple(rejected))
        warnings += more
        with_prov = {f: (ol, outlines[f][1] if f in outlines else ("inferred" if ol.source == "inferred" else "default"))
                     for f, ol in full.items()}
        feats, feat_prov = features_from(observed.observations, env)
        try:
            return assemble(env, env_prov, with_prov, feats, feat_prov, warnings, user_values, accepted)
        except ValidationError as e:
            log.warning("spec rejected: %s", e)
            return S.MvAbstain(stage="dimensions", reason="invalid_value",
                               remedy="A value is out of range. Check the numbers you entered.")

    def build(self, spec: S.MultiViewSpec, out_dir: Path, masks: dict | None = None) -> BuildResult | S.MvAbstain:
        try:
            solid = build_solid(spec)
        except BuildError as e:
            return S.MvAbstain(stage="build", reason=e.reason, remedy=e.remedy)
        step, stl = export(solid, out_dir)
        sliced = slice_solid(solid, out_dir, self.profile, self.slicer)
        if isinstance(sliced, S.MvAbstain):
            return sliced
        mesh = solid_mesh(solid)
        views = {f: normalize_mask(face_mask(mesh, f, spec.envelope)[0]) for f in S.FACES}
        scores = {f: round(iou(views[f], m), 3) for f, m in (masks or {}).items()}
        warnings = list(spec.warnings) + sliced.warnings
        warnings += [f"Low confidence on {f}, check the dimensions." for f, s in scores.items() if s < IOU_GREEN]
        return BuildResult(step, stl, sliced.print_stl, sliced.gcode, sliced.print_time_s, sliced.filament_g,
                           views, scores, warnings)


def default_pipeline() -> MvPipeline:
    """Vision model from the environment, TrOCR and TripoSR when the ai extra is installed."""
    reader = provider = None
    try:
        from s2c.multiview.ocr import trocr_reader
        reader = trocr_reader()
    except Exception as e:  # transformers missing
        log.warning("TrOCR unavailable: %s", e)
    try:
        from s2c.multiview.hf3d import default_provider
        provider = default_provider()
    except Exception as e:
        log.warning("TripoSR unavailable: %s", e)
    return MvPipeline(chat=env_chat(), reader=reader, mesh_provider=provider)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_mv_pipeline.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add s2c/multiview/pipeline.py tests/test_mv_pipeline.py
git commit -m "Wire the multi-view pipeline from images to a verified build"
```

---

### Task 13: API router, stand-alone app and command-line tool

**Files:**
- Create: `s2c/multiview/routes.py`, `s2c/multiview/app.py`, `scripts/mv.py`
- Test: `tests/test_mv_routes.py`

**Interfaces:**
- Consumes: `MvPipeline`, `ImageInput`, `Observed`, `default_pipeline` (Task 12); `MultiViewSpec`, `MvAbstain` (Task 1).
- Produces: `router` (prefix `/mv`: `POST /analyze`, `POST /merge`, `POST /build`, `GET /files/{bid}/{name}`); `get_pipeline()` dependency; module variable `STORE_ROOT`; `app` in `s2c/multiview/app.py`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mv_routes.py
from fastapi import FastAPI
from fastapi.testclient import TestClient

from s2c.multiview import routes
from s2c.multiview.pipeline import MvPipeline
from tests.test_mv_pipeline import sketch


def client(tmp_path, monkeypatch):
    monkeypatch.setattr(routes, "STORE_ROOT", tmp_path)
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[routes.get_pipeline] = lambda: MvPipeline()
    return TestClient(app)


def test_analyze_merge_build_and_download(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch)
    files = [("files", ("front.png", sketch(600, 400), "image/png")), ("files", ("top.png", sketch(600, 100), "image/png"))]
    body = c.post("/mv/analyze", files=files, data={"faces": '["front", "top"]', "kinds": '["sketch", "sketch"]'}).json()
    assert body["abstain"]["reason"] == "missing_x"  # no OCR in this pipeline, so the gate asks
    values = {"envelope.x_mm": 60, "envelope.y_mm": 40, "envelope.z_mm": 10}
    spec = c.post("/mv/merge", json={"request_id": body["request_id"], "user_values": values}).json()["spec"]
    out = c.post("/mv/build", json={"spec": spec, "request_id": body["request_id"]}).json()
    assert out["iou"]["front"] > 0.85
    assert c.get(out["stl_url"]).status_code == 200
    assert c.get(out["step_url"]).status_code == 200
    assert c.get(out["views"]["front"]).status_code == 200


def test_unknown_requests_and_bad_file_names(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch)
    assert c.post("/mv/merge", json={"request_id": "nope"}).status_code == 404
    assert c.get("/mv/files/abc/part.stl").status_code == 404
    assert c.get("/mv/files/" + "0" * 32 + "/..%5Csecret").status_code == 404
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_mv_routes.py -v`
Expected: FAIL with `ImportError: cannot import name 'routes'`.

- [ ] **Step 3: Implement**

```python
# s2c/multiview/routes.py
"""HTTP surface of the multi-view path. The integrator mounts `router` in s2c/api.py. Spec section 7.1.
Files live under STORE_ROOT for one hour."""
from __future__ import annotations

import json
import re
import shutil
import time
import uuid
from functools import lru_cache
from pathlib import Path

import cv2
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from s2c.multiview.pipeline import ImageInput, MvPipeline, Observed, default_pipeline
from s2c.multiview.spec import MultiViewSpec, MvAbstain

router = APIRouter(prefix="/mv", tags=["multiview"])
STORE_ROOT = Path("tmp/mv")
TTL_S = 3600
_ID = re.compile(r"^[0-9a-f]{32}$")
_NAME = re.compile(r"^[\w.-]+$")
_requests: dict[str, tuple[float, Observed]] = {}


@lru_cache(maxsize=1)
def get_pipeline() -> MvPipeline:
    return default_pipeline()


def _sweep() -> None:
    now = time.time()
    for rid in [r for r, (t, _) in _requests.items() if now - t > TTL_S]:
        del _requests[rid]
    if STORE_ROOT.exists():
        for d in STORE_ROOT.iterdir():
            if d.is_dir() and now - d.stat().st_mtime > TTL_S:
                shutil.rmtree(d, ignore_errors=True)


def _result(res) -> dict:
    if isinstance(res, MvAbstain):
        return {"abstain": res.model_dump()}
    return {"spec": res.model_dump(mode="json")}


def _tag(items: list, i: int) -> str | None:
    value = items[i] if i < len(items) else None
    return None if value in (None, "", "auto") else value


@router.post("/analyze")
def analyze(files: list[UploadFile] = File(...), faces: str = Form("[]"), kinds: str = Form("[]"),
            reference: str | None = Form(None), pipe: MvPipeline = Depends(get_pipeline)) -> dict:
    _sweep()
    face_tags, kind_tags = json.loads(faces), json.loads(kinds)
    images = [ImageInput(f.file.read(), _tag(face_tags, i), _tag(kind_tags, i)) for i, f in enumerate(files)]
    rid = uuid.uuid4().hex
    observed = pipe.observe(images, reference or None)
    if isinstance(observed, MvAbstain):
        return {"request_id": rid, "abstain": observed.model_dump()}
    _requests[rid] = (time.time(), observed)
    return {"request_id": rid, **_result(pipe.fuse(observed)), "labels": [l.model_dump() for l in observed.labels]}


class MergeBody(BaseModel):
    request_id: str
    user_values: dict[str, float] = {}
    accepted: list[str] = []
    rejected: list[str] = []


@router.post("/merge")
def merge(body: MergeBody, pipe: MvPipeline = Depends(get_pipeline)) -> dict:
    entry = _requests.get(body.request_id)
    if entry is None:
        raise HTTPException(404, "Unknown or expired request. Analyze the images again.")
    return _result(pipe.fuse(entry[1], body.user_values, body.accepted, body.rejected))


class BuildBody(BaseModel):
    spec: MultiViewSpec
    request_id: str | None = None


@router.post("/build")
def build_part(body: BuildBody, pipe: MvPipeline = Depends(get_pipeline)) -> dict:
    _sweep()
    bid = uuid.uuid4().hex
    out = STORE_ROOT / bid
    entry = _requests.get(body.request_id or "")
    res = pipe.build(body.spec, out, entry[1].masks if entry else None)
    if isinstance(res, MvAbstain):
        return {"abstain": res.model_dump()}
    base = f"/mv/files/{bid}"
    views = {}
    for face, mask in res.views.items():
        cv2.imwrite(str(out / f"view_{face}.png"), mask)
        views[face] = f"{base}/view_{face}.png"
    return {"stl_url": f"{base}/part.stl", "step_url": f"{base}/part.step",
            "gcode_url": f"{base}/part.gcode" if res.gcode else None,
            "print_time_s": res.print_time_s, "filament_g": res.filament_g,
            "views": views, "iou": res.iou, "warnings": res.warnings}


@router.get("/files/{bid}/{name}")
def files(bid: str, name: str) -> FileResponse:
    path = STORE_ROOT / bid / name
    if not _ID.match(bid) or not _NAME.match(name) or not path.is_file():
        raise HTTPException(404, "File not found or expired.")
    return FileResponse(path)
```

```python
# s2c/multiview/app.py
"""Stand-alone server for the multi-view path until the integrator mounts the router in s2c/api.py.
Run: uv run uvicorn s2c.multiview.app:app --host 0.0.0.0 --port 8001"""
from dotenv import load_dotenv
from fastapi import FastAPI

from s2c.multiview.routes import router

load_dotenv()
app = FastAPI(title="Sketch-to-CAD multi-view")
app.include_router(router)
```

```python
# scripts/mv.py
"""Face images -> MultiViewSpec -> STEP, STL and G-code, from the command line.

  uv run python scripts/mv.py --image front.jpg@front@sketch --image top.jpg@top@sketch --out tmp/mv
  add --set envelope.z_mm=20 when it asks for a dimension; --no-ai skips TrOCR, TripoSR and the vision model."""
import argparse
from pathlib import Path

from dotenv import load_dotenv

from s2c.multiview.pipeline import ImageInput, MvPipeline, default_pipeline
from s2c.multiview.spec import MvAbstain


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", action="append", required=True,
                    help="PATH[@FACE[@KIND]]; FACE front|back|left|right|top|bottom, KIND sketch|photo|drawing")
    ap.add_argument("--reference", help="1 TND, 1 EUR, 2 EUR, card or a4")
    ap.add_argument("--set", action="append", default=[], help="FIELD=VALUE, for example envelope.z_mm=20")
    ap.add_argument("--reject", action="append", default=[], help="canonical face to replace by a rectangle")
    ap.add_argument("--out", default="tmp/mv")
    ap.add_argument("--no-ai", action="store_true")
    args = ap.parse_args()

    images = []
    for arg in args.image:
        path, face, kind = (arg.split("@") + [None, None])[:3]
        images.append(ImageInput(Path(path).read_bytes(), face, kind))
    values = {k: float(v) for k, v in (s.split("=", 1) for s in args.set)}
    pipe = MvPipeline() if args.no_ai else default_pipeline()

    observed = pipe.observe(images, args.reference)
    if isinstance(observed, MvAbstain):
        raise SystemExit(f"{observed.stage}: {observed.reason}. {observed.remedy}")
    spec = pipe.fuse(observed, values, rejected=args.reject)
    if isinstance(spec, MvAbstain):
        hint = (spec.partial or {}).get("suggested")
        raise SystemExit(f"{spec.stage}: {spec.reason}. {spec.remedy}" + (f" Suggested: {hint}" if hint else ""))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.json").write_text(spec.model_dump_json(indent=2))
    res = pipe.build(spec, out, observed.masks)
    if isinstance(res, MvAbstain):
        raise SystemExit(f"{res.stage}: {res.reason}. {res.remedy}")

    print(f"spec    {out / 'spec.json'}\nSTEP    {res.step}\nSTL     {res.stl}\nG-code  {res.gcode or 'none'}")
    for face in ("front", "top", "right"):
        print(f"{face:<7} {getattr(spec.views, face).source}")
    amber = {p: s for p, s in spec.provenance.items() if s in ("scaled", "inferred", "estimated", "default")}
    for path, source in amber.items():
        print(f"check   {path} ({source})")
    for face, score in res.iou.items():
        print(f"IoU     {face} {score:.2f}")
    for w in res.warnings:
        print("warning:", w)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests and the tool**

Run: `uv run pytest tests/test_mv_routes.py -v`
Expected: all PASS.

Make a test image: `uv run python -c "import cv2, numpy as np; img = np.full((1200, 1600, 3), 255, np.uint8); cv2.rectangle(img, (500, 400), (1100, 800), (0, 0, 0), 4); cv2.circle(img, (600, 700), 30, (0, 0, 0), 3); cv2.imwrite('tmp/front.png', img)"` (create `tmp/` first).

Run: `uv run python scripts/mv.py --no-ai --image tmp/front.png@front@sketch --set envelope.x_mm=60 --set envelope.y_mm=40 --set envelope.z_mm=5 --out tmp/mv`
Expected: spec, STEP, STL and G-code paths; `top assumed`, `right assumed`; the `check` lines list the scaled values.

- [ ] **Step 5: Commit**

```bash
git add s2c/multiview/routes.py s2c/multiview/app.py scripts/mv.py tests/test_mv_routes.py
git commit -m "Expose the multi-view path over HTTP and on the command line"
```

---

### Task 14: Golden multi-view set, full suite, docs

**Files:**
- Create: `tests/golden_mv/README.md`, `tests/test_mv_golden.py`
- Modify: `README.md` (add a section), `docs/models.md` (add a row group for the multi-view models)

**Interfaces:**
- Consumes: `MvPipeline`, `ImageInput`, `default_pipeline` (Task 12).
- Produces: the golden harness the geometry owner fills with real captures.

- [ ] **Step 1: Write the harness and the capture protocol**

```python
# tests/test_mv_golden.py
"""Real captures in tests/golden_mv/<name>/: face images plus expected.json. Skips while the folder is empty.
GOLDEN_AI=1 runs the full pipeline (vision model, TrOCR, TripoSR) without the typed user_values."""
import json
import os
from pathlib import Path

import pytest

from s2c.multiview.pipeline import ImageInput, MvPipeline, default_pipeline
from s2c.multiview.spec import MvAbstain

GOLDEN = Path(__file__).parent / "golden_mv"
CASES = sorted(p for p in GOLDEN.iterdir() if (p / "expected.json").exists()) if GOLDEN.exists() else []


def within(got: float, want: float) -> bool:
    return abs(got - want) <= max(0.05 * abs(want), 1.0)


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_golden_mv_case(case: Path, tmp_path):
    e = json.loads((case / "expected.json").read_text())
    ai = os.environ.get("GOLDEN_AI") == "1"
    pipe = default_pipeline() if ai else MvPipeline()
    images = [ImageInput((case / i["file"]).read_bytes(), i["face"], i["kind"]) for i in e["images"]]
    observed = pipe.observe(images, e.get("reference"))
    assert not isinstance(observed, MvAbstain), observed
    spec = pipe.fuse(observed, {} if ai else e.get("user_values", {}))
    assert not isinstance(spec, MvAbstain), spec
    for axis in "xyz":
        got, want = getattr(spec.envelope, f"{axis}_mm"), e["envelope"][f"{axis}_mm"]
        assert within(got, want), f"{axis}: got {got}, expected {want}"
    assert len(spec.features) == e["hole_count"]
    built = pipe.build(spec, tmp_path, observed.masks)
    assert not isinstance(built, MvAbstain), built
    assert min(built.iou.values()) >= 0.85, built.iou
```

```markdown
<!-- tests/golden_mv/README.md -->
# Golden multi-view cases

One folder per capture: the face images and `expected.json`. Five parts, each captured twice: once with a single face (`<name>_1view`) and once with three faces (`<name>_3view`).

Parts: plate with two holes, L-bracket with a hole on each leg, flange, spacer, wedge (a part whose side view is a triangle).

Capture rules:
- Sketches: dark pen on plain white paper, one face per sheet, dimensions in mm written outside the outline, with a gap between dimension lines and the part. Diameters with ⌀ or D.
- Photos: part flat on a plain background, phone parallel to the table, one reference object (a coin, a card or an A4 sheet under the part) fully in frame.
- Measure the real part with calipers; those values go in `expected.json`.

expected.json:

    {
      "images": [{"file": "front.jpg", "face": "front", "kind": "sketch"},
                 {"file": "top.jpg", "face": "top", "kind": "sketch"}],
      "reference": null,
      "user_values": {"envelope.x_mm": 60.0, "envelope.y_mm": 40.0, "envelope.z_mm": 5.0},
      "envelope": {"x_mm": 60.0, "y_mm": 40.0, "z_mm": 5.0},
      "hole_count": 2
    }

`user_values` is what a user would type when OCR is off. Without `GOLDEN_AI=1` the test uses it; with `GOLDEN_AI=1` the values must come from OCR or the reference object.

Run: `uv run pytest tests/test_mv_golden.py -v`, or `GOLDEN_AI=1 uv run pytest tests/test_mv_golden.py -v` with a `.env` and the ai extra.
```

- [ ] **Step 2: Run the harness and the whole suite**

Run: `uv run pytest -v`
Expected: every test PASS; the golden test collects no cases until captures are added; `gpu` tests skip without the ai extra; the slicer test runs once PrusaSlicer is installed.

- [ ] **Step 3: Document the path**

Append to `README.md`, after "Run it":

```markdown
## Multi-view path (pending team sign-off)

Give up to six face images; missing faces are mirrored, predicted with TripoSR, or assumed. The part is the intersection of the three extruded outlines, sliced to G-code for an FDM printer. Design: `docs/superpowers/specs/2026-09-22-multiview-gcode-design.md`.

    uv run python scripts/mv_build.py examples/mv/l_bracket.json --out tmp/mv_demo          # spec -> STEP, STL, G-code
    uv run python scripts/mv.py --image front.jpg@front@sketch --image top.jpg@top@sketch   # images -> the same
    uv run uvicorn s2c.multiview.app:app --port 8001                                        # /mv API
    powershell -File scripts/setup_triposr.ps1                                              # optional: TrOCR and TripoSR on the GPU

G-code needs PrusaSlicer: `winget install --id Prusa3D.PrusaSlicer -e` (needs admin), or unzip the portable zip from the PrusaSlicer GitHub release into `vendor/`. Without it you still get STL and STEP.
```

Append to `docs/models.md`:

```markdown
## Multi-view path

| Model | Source | Use | Runs |
| --- | --- | --- | --- |
| TripoSR | `stabilityai/TripoSR` (MIT) | Predicts a mesh from one image; only its silhouettes are used, for faces nobody photographed | Local CUDA, else the Hugging Face Space in `TRIPOSR_SPACE` |
| TrOCR base handwritten | `microsoft/trocr-base-handwritten` | Reads handwritten dimension values until the numbers owner's reader lands | Local, CUDA or CPU |
| rembg (u2net) | `rembg` | Removes the background before TripoSR | Local CPU |
| PrusaSlicer | prusa3d.com (AGPL) | Slices the STL to G-code with `profiles/fdm_default.ini` | Local CLI |
```

- [ ] **Step 4: Commit**

```bash
git add tests/golden_mv/README.md tests/test_mv_golden.py README.md docs/models.md
git commit -m "Add the golden multi-view harness and document the multi-view path"
```

---

## Self-review against the spec

- Spec 2 rules: no code execution path (label JSON validated, mesh only rendered); estimates only for hole depths (label drops others); abstentions everywhere. Tasks 3, 9, 10, 12.
- Spec 3 frames and mirror rule: Task 1 tests every face; build orientation tests in Task 3; `face_coords` in Task 2 uses the same table.
- Spec 4.1 minimum gate with `partial.suggested`: Task 8.
- Spec 4.2 OCR: regions, parsing, linking, TrOCR, reader hook: Task 7.
- Spec 4.3 reference objects, sketches never measured: Task 6; pipeline applies references to photos only (Task 12).
- Spec 4.4 fusion order, 4.5 snapping, 4.6 consistency checks: Task 8.
- Spec 5 contract: Task 1.
- Spec 6.1 label: Task 9. 6.2 outline: Task 5. 6.3 complete: Tasks 10 and 11. 6.4 build: Task 3. 6.5 slice: Task 4. 6.6 verify: Task 12 (`BuildResult.iou`, amber warnings).
- Spec 7.1 API: Task 13. Spec 7.2 web: out of this plan, stated at the top.
- Spec 8 testing: each stage has its test file; `gpu` and `slicer` markers; golden harness in Task 14.
- Spec 9 order of work: Tasks 1 to 4 give the spec-to-G-code demo before any model code.
- Type consistency: `Observation`, `Linked`, `Reading`, `PixelOutline`, `Mesh`, `Outline` and the `complete()` signature are used with the same fields in Tasks 5 to 13.
