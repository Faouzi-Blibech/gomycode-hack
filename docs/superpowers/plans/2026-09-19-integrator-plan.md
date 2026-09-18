# Integrator Implementation Plan (contracts, model layer, merge, API, UIs)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the frozen contracts, the provider-agnostic vision layer, the merge step with abstention gates, the FastAPI surface, the Gradio lab UI and the mobile web app, so a sketch photo becomes a downloadable STL end to end.

**Architecture:** Every stage produces a Pydantic model from `s2c/partspec/`. `s2c/pipeline.py` wires stage functions together and falls back to `s2c/fakes/` when a real module is missing, so this plan runs green before the geometry and numbers plans land. The API is stateless apart from a temp file store with a one-hour TTL.

**Tech Stack:** Python 3.11, uv, Pydantic v2, openai (OpenAI-compatible client), OpenCV headless, FastAPI, Gradio, pytest. Web: Vite, React, TypeScript, three.

**Spec:** `docs/superpowers/specs/2026-09-19-sketch-to-cad-design.md`

## Global Constraints

- The model never writes code. It returns Topology JSON only. (spec 2, Rule 1)
- No millimetre value may originate from the model. Topology has `extra="forbid"` so any extra field the model adds is rejected. (spec 2, Rule 2)
- Grammar is frozen: `plate`, `l_bracket`, `flange`, `spacer`, `profile_extrusion`; features `hole`, `slot`, `fillet`, `chamfer`. (spec 3)
- All dimensions are millimetre floats, no inches, no unit strings. (spec 3)
- Every abstention carries `stage`, `reason`, `remedy`, optional `partial`. (spec 2, Rule 4)
- Round-trip IoU threshold for green is 0.85. (spec 4.7)
- Provider config only through `VLM_BASE_URL`, `VLM_MODEL`, `VLM_API_KEY`. Never hard-code a model. (spec 4.4)
- Commit messages: plain, no AI attribution, no co-author trailers. (spec 5)
- Write the failing test first. (CLAUDE.md)

---

## File structure

| Path | Responsibility |
| --- | --- |
| `pyproject.toml` | Package `s2c`, dependencies, pytest and ruff config |
| `.env.example` | The three VLM variables |
| `s2c/partspec/models.py` | All contracts: PartSpec and its parts, Topology, Annotations, Measurements, Abstain |
| `s2c/partspec/schema.py` | JSON schema export used in prompts |
| `s2c/silhouette.py` | Input image to normalised mask, `normalize_mask`, `iou` |
| `s2c/vision/client.py` | `VLMClient` with injectable chat function, logging |
| `s2c/vision/prompts.py` | System and user prompts for the Topology call |
| `s2c/vision/topology.py` | `topology_from_image` with validate, retry once, abstain |
| `s2c/merge.py` | Fuse stage outputs into PartSpec, gates |
| `s2c/store.py` | Temp file store with TTL |
| `s2c/fakes/*.py` | Stand-ins for vision, ocr, metrology, builder, views |
| `s2c/pipeline.py` | `Pipeline` dataclass and `default_pipeline()` |
| `s2c/api.py` | FastAPI app |
| `app_gradio.py` | Lab UI |
| `web/` | Vite React app: capture, review, export |
| `tests/` | One test file per module |

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `.env.example`, `s2c/__init__.py`, `tests/__init__.py`, `tests/test_smoke.py`
- Modify: `.gitignore` (append `logs/`, `.env`, `tmp/`)

**Interfaces:**
- Produces: importable package `s2c`, `uv run pytest` works.

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/test_smoke.py
def test_package_imports():
    import s2c
    assert s2c.__version__ == "0.1.0"
```

- [ ] **Step 2: Run it to confirm failure**

Run: `uv run pytest tests/test_smoke.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 's2c'` (or uv complains there is no project yet).

- [ ] **Step 3: Create pyproject.toml**

```toml
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

[dependency-groups]
dev = ["pytest>=8", "ruff>=0.6", "httpx>=0.27"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["s2c"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
```

- [ ] **Step 4: Create the package and env example**

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
```

Append to `.gitignore`:

```text
logs/
.env
tmp/
```

- [ ] **Step 5: Install and run the test**

Run: `uv sync && uv run pytest tests/test_smoke.py -v`
Expected: PASS. If `cadquery` fails to resolve on your Python, run `uv python install 3.11` and `uv sync --python 3.11`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock .env.example .gitignore s2c/__init__.py tests/__init__.py tests/test_smoke.py
git commit -m "Scaffold s2c package with uv, pytest and ruff"
```

---

### Task 2: Contracts (PartSpec and parts)

**Files:**
- Create: `s2c/partspec/__init__.py`, `s2c/partspec/models.py`
- Test: `tests/test_partspec.py`

**Interfaces:**
- Produces: `PartSpec`, `Plate`, `LBracket`, `Flange`, `Spacer`, `ProfileExtrusion`, `Hole`, `Slot`, `Fillet`, `Chamfer`, `Provenance`, `SourceInput`, `numeric_field_paths(spec) -> list[str]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_partspec.py
import pytest
from pydantic import ValidationError
from s2c.partspec.models import PartSpec, Plate, Hole, numeric_field_paths


def plate_spec(**overrides):
    data = {
        "source_input": "sketch",
        "part": {"type": "plate", "width_mm": 60, "height_mm": 40, "thickness_mm": 5},
        "features": [{"type": "hole", "x_mm": 10, "y_mm": 10, "diameter_mm": 6}],
        "provenance": {
            "part.width_mm": "user_written",
            "part.height_mm": "user_written",
            "part.thickness_mm": "user_written",
            "part.corner_radius_mm": "default",
            "features[0].x_mm": "default",
            "features[0].y_mm": "default",
            "features[0].diameter_mm": "user_written",
        },
        "confidence": 0.9,
    }
    data.update(overrides)
    return data


def test_valid_plate_spec_round_trips():
    spec = PartSpec.model_validate(plate_spec())
    assert isinstance(spec.part, Plate)
    assert isinstance(spec.features[0], Hole)
    assert PartSpec.model_validate_json(spec.model_dump_json()) == spec


def test_numeric_field_paths_lists_every_number():
    spec = PartSpec.model_validate(plate_spec())
    assert numeric_field_paths(spec) == [
        "part.width_mm", "part.height_mm", "part.thickness_mm", "part.corner_radius_mm",
        "features[0].x_mm", "features[0].y_mm", "features[0].diameter_mm",
    ]


def test_missing_provenance_is_rejected():
    data = plate_spec()
    del data["provenance"]["part.thickness_mm"]
    with pytest.raises(ValidationError, match="part.thickness_mm"):
        PartSpec.model_validate(data)


def test_negative_dimension_is_rejected():
    data = plate_spec()
    data["part"]["width_mm"] = -1
    with pytest.raises(ValidationError):
        PartSpec.model_validate(data)


def test_unit_strings_are_rejected():
    data = plate_spec()
    data["part"]["width_mm"] = "60mm"
    with pytest.raises(ValidationError):
        PartSpec.model_validate(data)


def test_unknown_part_type_is_rejected():
    data = plate_spec()
    data["part"] = {"type": "gear", "teeth": 12}
    with pytest.raises(ValidationError):
        PartSpec.model_validate(data)


def test_flange_bore_must_fit_inside_bolt_circle():
    data = plate_spec(part={
        "type": "flange", "outer_diameter_mm": 80, "bore_diameter_mm": 70, "thickness_mm": 6,
        "bolt_circle_diameter_mm": 60, "bolt_count": 4, "bolt_hole_diameter_mm": 6,
    }, features=[], provenance={
        "part.outer_diameter_mm": "measured", "part.bore_diameter_mm": "measured",
        "part.thickness_mm": "user_edited", "part.bolt_circle_diameter_mm": "measured",
        "part.bolt_count": "measured", "part.bolt_hole_diameter_mm": "measured",
    })
    with pytest.raises(ValidationError, match="bore"):
        PartSpec.model_validate(data)
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_partspec.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 's2c.partspec'`.

- [ ] **Step 3: Write the models**

```python
# s2c/partspec/__init__.py
from .models import *  # noqa: F401,F403
```

```python
# s2c/partspec/models.py
"""Frozen contracts. Changing anything here needs a PR approved by all three owners."""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

Mm = Annotated[float, Field(gt=0, description="millimetres")]
NonNegMm = Annotated[float, Field(ge=0, description="millimetres")]
Provenance = Literal["measured", "user_written", "user_edited", "default"]
SourceInput = Literal["sketch", "photo", "drawing"]
EdgeSelector = Literal["all", "all_vertical", "top", "bottom"]


class _Strict(BaseModel):
    """extra="forbid" is what rejects a model-invented width_mm. Not strict mode: lists must
    still validate as tuples when specs arrive as Python dicts from json.loads."""
    model_config = ConfigDict(extra="forbid")


# ---- features -------------------------------------------------------------

class Hole(_Strict):
    type: Literal["hole"] = "hole"
    x_mm: float
    y_mm: float
    diameter_mm: Mm
    depth_mm: Mm | None = None  # None means through
    leg: Literal["a", "b"] | None = None  # l_bracket only


class Slot(_Strict):
    type: Literal["slot"] = "slot"
    x_mm: float
    y_mm: float
    width_mm: Mm
    length_mm: Mm
    angle_deg: float = 0.0
    depth_mm: Mm | None = None
    leg: Literal["a", "b"] | None = None


class Fillet(_Strict):
    type: Literal["fillet"] = "fillet"
    edges: EdgeSelector = "all_vertical"
    radius_mm: Mm


class Chamfer(_Strict):
    type: Literal["chamfer"] = "chamfer"
    edges: EdgeSelector = "all_vertical"
    radius_mm: Mm


# ---- part types -----------------------------------------------------------

class Plate(_Strict):
    type: Literal["plate"] = "plate"
    width_mm: Mm
    height_mm: Mm
    thickness_mm: Mm
    corner_radius_mm: NonNegMm = 0.0


class LBracket(_Strict):
    type: Literal["l_bracket"] = "l_bracket"
    leg_a_mm: Mm
    leg_b_mm: Mm
    width_mm: Mm
    thickness_mm: Mm
    angle_deg: float = 90.0


class Flange(_Strict):
    type: Literal["flange"] = "flange"
    outer_diameter_mm: Mm
    bore_diameter_mm: Mm
    thickness_mm: Mm
    bolt_circle_diameter_mm: Mm
    bolt_count: int = Field(ge=0)
    bolt_hole_diameter_mm: Mm

    @model_validator(mode="after")
    def _nested_diameters(self) -> "Flange":
        if not self.bore_diameter_mm < self.bolt_circle_diameter_mm < self.outer_diameter_mm:
            raise ValueError("bore must be inside the bolt circle, bolt circle inside the outer diameter")
        return self


class Spacer(_Strict):
    type: Literal["spacer"] = "spacer"
    outer_diameter_mm: Mm
    inner_diameter_mm: NonNegMm = 0.0  # 0 means solid
    length_mm: Mm

    @model_validator(mode="after")
    def _inner_inside_outer(self) -> "Spacer":
        if self.inner_diameter_mm >= self.outer_diameter_mm:
            raise ValueError("inner diameter must be smaller than outer diameter")
        return self


class ProfileExtrusion(_Strict):
    type: Literal["profile_extrusion"] = "profile_extrusion"
    points_mm: list[tuple[float, float]] = Field(min_length=3)
    depth_mm: Mm


Part = Annotated[Union[Plate, LBracket, Flange, Spacer, ProfileExtrusion], Field(discriminator="type")]
Feature = Annotated[Union[Hole, Slot], Field(discriminator="type")]
Finish = Annotated[Union[Fillet, Chamfer], Field(discriminator="type")]
PartType = Literal["plate", "l_bracket", "flange", "spacer", "profile_extrusion"]


# ---- PartSpec -------------------------------------------------------------

class PartSpec(_Strict):
    version: Literal["1"] = "1"
    source_input: SourceInput
    part: Part
    features: list[Feature] = []
    finishes: list[Finish] = []
    provenance: dict[str, Provenance]
    confidence: float = Field(ge=0, le=1)
    warnings: list[str] = []

    @model_validator(mode="after")
    def _every_number_has_provenance(self) -> "PartSpec":
        missing = [p for p in numeric_field_paths(self) if p not in self.provenance]
        if missing:
            raise ValueError(f"missing provenance for {missing}")
        return self


def _numeric_names(model: BaseModel) -> list[str]:
    names = []
    for name, value in model.model_dump().items():
        if name == "type":
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) or name == "points_mm":
            names.append(name)
    return names


def numeric_field_paths(spec: PartSpec) -> list[str]:
    """Every field that must carry provenance, in a stable order."""
    paths = [f"part.{n}" for n in _numeric_names(spec.part)]
    for i, feature in enumerate(spec.features):
        paths += [f"features[{i}].{n}" for n in _numeric_names(feature)]
    for i, finish in enumerate(spec.finishes):
        paths += [f"finishes[{i}].{n}" for n in _numeric_names(finish)]
    return paths
```

Note: `"60mm"` fails because it is not parseable as a number, and `"gear"` fails the discriminator. Do not enable Pydantic strict mode; it would reject `[x, y]` lists for tuple fields in every JSON-derived dict.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_partspec.py -v`
Expected: all 7 PASS.

- [ ] **Step 5: Commit**

```bash
git add s2c/partspec tests/test_partspec.py
git commit -m "Add PartSpec contracts with provenance validation"
```

---

### Task 3: Contracts (Topology, Annotations, Measurements, Abstain) and JSON schema export

**Files:**
- Modify: `s2c/partspec/models.py` (append)
- Create: `s2c/partspec/schema.py`
- Test: `tests/test_stage_contracts.py`

**Interfaces:**
- Produces: `Topology`, `TopoHole`, `TopoSlot`, `Annotations`, `Annotation`, `LinkedTo`, `Measurements`, `Coin`, `Abstain`, `topology_json_schema() -> str`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_stage_contracts.py
import json
import pytest
from pydantic import ValidationError
from s2c.partspec.models import Topology, Annotations, Measurements, Abstain
from s2c.partspec.schema import topology_json_schema


def test_topology_accepts_normalised_positions():
    t = Topology.model_validate({
        "part_type": "plate", "holes": [{"u": 0.2, "v": 0.5, "kind": "through"}],
        "confidence": 0.8,
    })
    assert t.holes[0].u == 0.2
    assert t.view == "unknown"


def test_topology_rejects_any_dimension_field():
    with pytest.raises(ValidationError):
        Topology.model_validate({"part_type": "plate", "width_mm": 60, "confidence": 0.8})


def test_topology_rejects_position_outside_unit_square():
    with pytest.raises(ValidationError):
        Topology.model_validate({"part_type": "plate", "holes": [{"u": 1.5, "v": 0}], "confidence": 0.8})


def test_annotations_and_measurements_validate():
    a = Annotations.model_validate({"items": [
        {"value_mm": 60, "kind": "linear", "bbox_px": [1, 2, 30, 12], "linked_to": "width", "confidence": 0.9}
    ], "confidence": 0.9})
    assert a.items[0].hole_index is None
    m = Measurements.model_validate({
        "mm_per_px": 0.1, "coin": {"name": "1 TND", "pixel_diameter": 250, "eccentricity": 0.05, "confidence": 0.95},
        "outer_contour_mm": [[0, 0], [60, 0], [60, 40], [0, 40]], "bbox_mm": {"width": 60, "height": 40},
        "circles_mm": [{"x": 10, "y": 10, "diameter": 6}], "confidence": 0.9,
    })
    assert m.bbox_mm.width == 60


def test_abstain_carries_reason_and_remedy():
    a = Abstain(stage="metrology", reason="coin_tilted", remedy="Lay the coin flat, shoot top-down, retake.")
    assert a.partial is None


def test_topology_schema_is_valid_json_without_mm_fields():
    schema = json.loads(topology_json_schema())
    assert "mm" not in json.dumps(schema)
    assert schema["properties"]["part_type"]["enum"][0] == "plate"
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_stage_contracts.py -v`
Expected: FAIL with `ImportError: cannot import name 'Topology'`.

- [ ] **Step 3: Append the stage models**

Append to `s2c/partspec/models.py`:

```python
# ---- stage outputs --------------------------------------------------------

Unit = Annotated[float, Field(ge=0, le=1)]


class TopoHole(_Strict):
    u: Unit
    v: Unit
    kind: Literal["through", "blind", "unknown"] = "unknown"


class TopoSlot(_Strict):
    u: Unit
    v: Unit
    orientation: Literal["horizontal", "vertical", "diagonal"] = "horizontal"


class Topology(_Strict):
    """What the vision model may say. No absolute numbers, ever."""
    part_type: Literal["plate", "l_bracket", "flange", "spacer", "profile_extrusion", "unsupported"]
    unsupported_reason: str | None = None
    view: Literal["front", "top", "side", "isometric", "unknown"] = "unknown"
    holes: list[TopoHole] = []
    slots: list[TopoSlot] = []
    rounded_corners: bool = False
    symmetric: bool = False
    bolt_count: int | None = None
    annotation_count: int | None = None
    confidence: Unit
    notes: str = ""


LinkedTo = Literal[
    "width", "height", "thickness", "depth", "length", "leg_a", "leg_b", "corner_radius",
    "hole_diameter", "hole_x", "hole_y", "slot_length", "slot_width",
    "outer_diameter", "inner_diameter", "bolt_circle_diameter", "bolt_hole_diameter", "unknown",
]


class Annotation(_Strict):
    value_mm: Mm
    kind: Literal["linear", "diameter", "radius"]
    bbox_px: tuple[float, float, float, float]
    linked_to: LinkedTo = "unknown"
    hole_index: int | None = None
    confidence: Unit


class Annotations(_Strict):
    items: list[Annotation] = []
    confidence: Unit


class Coin(_Strict):
    name: str
    pixel_diameter: float
    eccentricity: float
    confidence: Unit


class Bbox(_Strict):
    width: Mm
    height: Mm


class Circle(_Strict):
    x: float
    y: float
    diameter: Mm


class Measurements(_Strict):
    mm_per_px: float = Field(gt=0)
    coin: Coin
    outer_contour_mm: list[tuple[float, float]]
    bbox_mm: Bbox
    circles_mm: list[Circle] = []
    confidence: Unit


class Abstain(_Strict):
    stage: Literal["metrology", "ocr", "vision", "merge", "build", "verify"]
    reason: str
    remedy: str
    partial: dict | None = None
```

```python
# s2c/partspec/schema.py
import json
from s2c.partspec.models import Topology


def topology_json_schema() -> str:
    return json.dumps(Topology.model_json_schema(), indent=2)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_stage_contracts.py tests/test_partspec.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add s2c/partspec tests/test_stage_contracts.py
git commit -m "Add Topology, Annotations, Measurements and Abstain contracts"
```

---

### Task 4: Silhouette extraction and IoU

**Files:**
- Create: `s2c/silhouette.py`
- Test: `tests/test_silhouette.py`

**Interfaces:**
- Produces: `input_silhouette(image_bgr: np.ndarray, px: int = 512) -> np.ndarray`, `normalize_mask(mask: np.ndarray, px: int = 512) -> np.ndarray`, `iou(a: np.ndarray, b: np.ndarray) -> float`. Masks are `uint8`, values 0 or 255, square `px` by `px`, content cropped to its bounding box and padded to square.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_silhouette.py
import numpy as np
import cv2
from s2c.silhouette import input_silhouette, normalize_mask, iou


def sketch_of_rect(w=300, h=200, thickness=3):
    img = np.full((600, 800, 3), 255, np.uint8)
    cv2.rectangle(img, (250, 200), (250 + w, 200 + h), (0, 0, 0), thickness)
    return img


def test_input_silhouette_fills_a_drawn_rectangle():
    mask = input_silhouette(sketch_of_rect(), px=256)
    assert mask.shape == (256, 256)
    assert set(np.unique(mask)) <= {0, 255}
    # a 3:2 rectangle padded to square fills 2/3 of the rows fully
    assert 0.6 < (mask == 255).mean() < 0.7


def test_iou_is_scale_invariant():
    small = input_silhouette(sketch_of_rect(150, 100), px=256)
    big = input_silhouette(sketch_of_rect(450, 300), px=256)
    assert iou(small, big) > 0.95


def test_iou_of_disjoint_shapes_is_low():
    a = np.zeros((256, 256), np.uint8); a[:128] = 255
    b = np.zeros((256, 256), np.uint8); b[128:] = 255
    assert iou(normalize_mask(a), normalize_mask(b)) > 0.9  # both normalise to a full band
    assert iou(a, b) == 0.0
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_silhouette.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# s2c/silhouette.py
"""Input image to a normalised binary silhouette. Shared by the API and views."""
import cv2
import numpy as np


def normalize_mask(mask: np.ndarray, px: int = 512) -> np.ndarray:
    """Crop to the bounding box of nonzero pixels, pad to a square, resize to px."""
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


def input_silhouette(image_bgr: np.ndarray, px: int = 512) -> np.ndarray:
    """Largest closed outline in the image, filled. Works for pen sketches and top-down photos."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("no outline found")
    outline = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.fillPoly(mask, [outline], 255)
    return normalize_mask(mask, px)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a_on, b_on = a > 127, b > 127
    union = np.logical_or(a_on, b_on).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a_on, b_on).sum() / union)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_silhouette.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add s2c/silhouette.py tests/test_silhouette.py
git commit -m "Add silhouette extraction, mask normalisation and IoU"
```

---

### Task 5: Vision client with injectable transport and logging

**Files:**
- Create: `s2c/vision/__init__.py`, `s2c/vision/client.py`
- Test: `tests/test_vision_client.py`

**Interfaces:**
- Produces: `VLMClient(chat: Callable[[list[dict]], str] | None = None, log_path: str | Path = "logs/vlm.jsonl")`, `VLMClient.complete_json(system: str, user: str, image_bytes: bytes, mime: str = "image/jpeg") -> str`, `VLMClient.from_env()`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_vision_client.py
import json
from s2c.vision.client import VLMClient


def test_complete_json_sends_image_and_returns_text(tmp_path):
    seen = {}

    def fake_chat(messages):
        seen["messages"] = messages
        return '{"ok": true}'

    client = VLMClient(chat=fake_chat, log_path=tmp_path / "vlm.jsonl", model="fake-model")
    out = client.complete_json("SYS", "USER", b"\xff\xd8bytes")
    assert out == '{"ok": true}'
    assert seen["messages"][0] == {"role": "system", "content": "SYS"}
    parts = seen["messages"][1]["content"]
    assert parts[0] == {"type": "text", "text": "USER"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_every_call_is_logged(tmp_path):
    client = VLMClient(chat=lambda m: "{}", log_path=tmp_path / "vlm.jsonl", model="fake-model")
    client.complete_json("s", "u", b"x")
    client.complete_json("s", "u", b"x")
    lines = (tmp_path / "vlm.jsonl").read_text().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["model"] == "fake-model"
    assert "latency_ms" in rec
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_vision_client.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`s2c/vision/__init__.py` is empty.

```python
# s2c/vision/client.py
"""One OpenAI-compatible client. Provider chosen by env vars only."""
from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Callable

Chat = Callable[[list[dict]], str]


class VLMClient:
    def __init__(
        self,
        chat: Chat | None = None,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        log_path: str | Path = "logs/vlm.jsonl",
        temperature: float = 0.0,
    ):
        self.model = model or os.environ.get("VLM_MODEL", "unset")
        self.base_url = base_url or os.environ.get("VLM_BASE_URL", "unset")
        self.log_path = Path(log_path)
        self.temperature = temperature
        self._chat = chat or self._openai_chat(api_key or os.environ.get("VLM_API_KEY", ""))

    @classmethod
    def from_env(cls) -> "VLMClient":
        return cls()

    def _openai_chat(self, api_key: str) -> Chat:
        from openai import OpenAI

        client = OpenAI(base_url=self.base_url, api_key=api_key or "missing")

        def chat(messages: list[dict]) -> str:
            resp = client.chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature, max_tokens=800,
            )
            usage = getattr(resp, "usage", None)
            self._last_usage = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
            }
            return resp.choices[0].message.content or ""

        return chat

    _last_usage: dict = {}

    def complete_json(self, system: str, user: str, image_bytes: bytes, mime: str = "image/jpeg") -> str:
        data_url = f"data:{mime};base64," + base64.b64encode(image_bytes).decode()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": user},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
        ]
        t0 = time.perf_counter()
        text = self._chat(messages)
        self._log({
            "ts": time.time(), "provider": self.base_url, "model": self.model,
            "latency_ms": round((time.perf_counter() - t0) * 1000),
            "chars_out": len(text), **self._last_usage,
        })
        return text

    def _log(self, record: dict) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_vision_client.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add s2c/vision tests/test_vision_client.py
git commit -m "Add provider-agnostic VLM client with call logging"
```

---

### Task 6: Topology prompt, validation, retry and abstain

**Files:**
- Create: `s2c/vision/prompts.py`, `s2c/vision/topology.py`
- Test: `tests/test_topology.py`

**Interfaces:**
- Consumes: `VLMClient.complete_json`, `Topology`, `Abstain`, `topology_json_schema`.
- Produces: `topology_from_image(image_bytes: bytes, client: VLMClient, input_kind: SourceInput) -> Topology | Abstain`, `extract_json(text: str) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_topology.py
from s2c.partspec.models import Topology, Abstain
from s2c.vision.client import VLMClient
from s2c.vision.topology import topology_from_image, extract_json

GOOD = '{"part_type": "plate", "holes": [{"u": 0.2, "v": 0.5, "kind": "through"}], "confidence": 0.9}'
WITH_DIMS = '{"part_type": "plate", "width_mm": 60, "confidence": 0.9}'


def client_returning(*replies, log_path):
    it = iter(replies)
    calls = []

    def chat(messages):
        calls.append(messages)
        return next(it)

    return VLMClient(chat=chat, log_path=log_path, model="fake"), calls


def test_extract_json_strips_code_fences():
    fence = "`" * 3
    assert extract_json(fence + "json\n{\"a\": 1}\n" + fence) == '{"a": 1}'
    assert extract_json("Sure! {\"a\": 1} done") == '{"a": 1}'


def test_valid_reply_becomes_topology(tmp_path):
    client, calls = client_returning(GOOD, log_path=tmp_path / "l")
    out = topology_from_image(b"img", client, "sketch")
    assert isinstance(out, Topology)
    assert out.holes[0].u == 0.2
    assert len(calls) == 1
    assert "never output a length" in calls[0][0]["content"].lower()


def test_dimension_in_reply_triggers_one_retry_with_error(tmp_path):
    client, calls = client_returning(WITH_DIMS, GOOD, log_path=tmp_path / "l")
    out = topology_from_image(b"img", client, "sketch")
    assert isinstance(out, Topology)
    assert len(calls) == 2
    retry_text = calls[1][1]["content"][0]["text"]
    assert "failed validation" in retry_text and "width_mm" in retry_text


def test_two_failures_abstain(tmp_path):
    client, calls = client_returning(WITH_DIMS, "not json at all", log_path=tmp_path / "l")
    out = topology_from_image(b"img", client, "sketch")
    assert isinstance(out, Abstain)
    assert out.stage == "vision" and out.reason == "schema_failed"
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_topology.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement prompts**

```python
# s2c/vision/prompts.py
from s2c.partspec.schema import topology_json_schema

SYSTEM = """You describe the TOPOLOGY of a single mechanical part seen in an image.
You NEVER output a length, width, height, thickness, diameter, radius, depth or any measurement.
Positions are fractions of the part's bounding box: u from left 0 to right 1, v from bottom 0 to top 1.

The only part types are:
- plate: a flat rectangular or rounded-rectangular plate, possibly with holes or slots
- l_bracket: two flat legs joined at an angle
- flange: a round disc with a central bore and a ring of bolt holes
- spacer: a plain cylinder or tube
- profile_extrusion: any other closed flat outline extruded straight
- unsupported: anything that is not a flat profile extruded straight (curved surfaces, assemblies, threads, organic shapes)

Return ONLY a JSON object matching this schema, no prose, no code fences:
""" + topology_json_schema()

USER_BY_KIND = {
    "sketch": "This is a hand-drawn sketch on paper. Written numbers are dimensions; count them in annotation_count but do NOT report their values. Describe the topology.",
    "photo": "This is a top-down photo of a real part next to a coin. Ignore the coin. Describe the topology of the part.",
    "drawing": "This is a technical drawing. Count dimension labels in annotation_count but do NOT report their values. Describe the topology.",
}

RETRY_SUFFIX = "\n\nYour previous output failed validation:\n{error}\nReturn corrected JSON only. Remember: no measurements of any kind."
```

- [ ] **Step 4: Implement topology extraction**

```python
# s2c/vision/topology.py
from __future__ import annotations

import re

from pydantic import ValidationError

from s2c.partspec.models import Abstain, SourceInput, Topology
from s2c.vision.client import VLMClient
from s2c.vision.prompts import RETRY_SUFFIX, SYSTEM, USER_BY_KIND

_FENCE = re.compile(r"`{3}(?:json)?\s*(.*?)`{3}", re.S)  # code fences some models wrap JSON in


def extract_json(text: str) -> str:
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    start, end = text.find("{"), text.rfind("}")
    return text[start: end + 1] if start != -1 and end != -1 else text.strip()


def topology_from_image(image_bytes: bytes, client: VLMClient, input_kind: SourceInput) -> Topology | Abstain:
    user = USER_BY_KIND[input_kind]
    raw = client.complete_json(SYSTEM, user, image_bytes)
    try:
        return Topology.model_validate_json(extract_json(raw))
    except (ValidationError, ValueError) as first:
        raw2 = client.complete_json(SYSTEM, user + RETRY_SUFFIX.format(error=str(first)[:600]), image_bytes)
        try:
            return Topology.model_validate_json(extract_json(raw2))
        except (ValidationError, ValueError):
            return Abstain(
                stage="vision", reason="schema_failed",
                remedy="The model could not describe this part. Try a cleaner, better-lit photo.",
            )
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_topology.py -v`
Expected: PASS.

- [ ] **Step 6: Live smoke test on a free provider (manual, not in CI)**

Create `.env` from `.env.example` with a Gemini or Groq key and a current vision model name, take a phone photo of a hand-drawn plate with two holes, save it as `tmp/plate.jpg`, then run:

```bash
uv run python -c "
from dotenv import load_dotenv; load_dotenv()
from s2c.vision.client import VLMClient
from s2c.vision.topology import topology_from_image
print(topology_from_image(open('tmp/plate.jpg','rb').read(), VLMClient.from_env(), 'sketch'))
"
```

Expected: a `Topology` with `part_type='plate'` and two holes. Record the model name and date in `docs/models.md`.

- [ ] **Step 7: Commit**

```bash
git add s2c/vision tests/test_topology.py docs/models.md
git commit -m "Add topology extraction with schema validation, retry and abstain"
```

---

### Task 7: Fakes for every stage

**Files:**
- Create: `s2c/fakes/__init__.py`, `s2c/fakes/vision.py`, `s2c/fakes/ocr.py`, `s2c/fakes/metrology.py`, `s2c/fakes/builder.py`, `s2c/fakes/views.py`
- Test: `tests/test_fakes.py`

**Interfaces:**
- Produces, matching the real modules the other owners will ship:
  - `fakes.vision.chat(messages) -> str` (a `Chat` for `VLMClient`)
  - `fakes.ocr.read_annotations(image_bgr, topology) -> Annotations`
  - `fakes.metrology.measure(image_bgr) -> Measurements | Abstain`
  - `fakes.builder.build(spec) -> object`, `fakes.builder.export(solid, out_dir) -> tuple[Path, Path]`
  - `fakes.views.silhouettes(solid, px=512) -> dict[str, np.ndarray]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fakes.py
import numpy as np
from s2c.partspec.models import Topology, Annotations, Measurements, PartSpec
from s2c.fakes import vision, ocr, metrology, builder, views


def test_fake_stages_produce_valid_contracts(tmp_path):
    img = np.zeros((10, 10, 3), np.uint8)
    topo = Topology.model_validate_json(vision.chat([]))
    assert topo.part_type == "plate"
    assert isinstance(ocr.read_annotations(img, topo), Annotations)
    assert isinstance(metrology.measure(img), Measurements)
    spec = PartSpec.model_validate_json(open("tests/data/plate_60x40x5.json").read())
    solid = builder.build(spec)
    step, stl = builder.export(solid, tmp_path)
    assert step.exists() and stl.exists() and stl.read_text().startswith("solid")
    masks = views.silhouettes(solid)
    assert set(masks) == {"front", "back", "left", "right", "top", "bottom"}
    assert masks["front"].shape == (512, 512)
```

- [ ] **Step 2: Add the shared test spec**

```json
// tests/data/plate_60x40x5.json
{
  "version": "1",
  "source_input": "sketch",
  "part": {"type": "plate", "width_mm": 60.0, "height_mm": 40.0, "thickness_mm": 5.0, "corner_radius_mm": 0.0},
  "features": [
    {"type": "hole", "x_mm": 10.0, "y_mm": 10.0, "diameter_mm": 6.0, "depth_mm": null, "leg": null},
    {"type": "hole", "x_mm": 50.0, "y_mm": 30.0, "diameter_mm": 6.0, "depth_mm": null, "leg": null}
  ],
  "finishes": [],
  "provenance": {
    "part.width_mm": "user_written", "part.height_mm": "user_written", "part.thickness_mm": "user_written",
    "part.corner_radius_mm": "default",
    "features[0].x_mm": "user_written", "features[0].y_mm": "user_written", "features[0].diameter_mm": "user_written",
    "features[1].x_mm": "user_written", "features[1].y_mm": "user_written", "features[1].diameter_mm": "user_written"
  },
  "confidence": 0.9,
  "warnings": []
}
```

(Remove the `//` comment line when saving; JSON has no comments.)

- [ ] **Step 3: Run to confirm failure**

Run: `uv run pytest tests/test_fakes.py -v`
Expected: FAIL with `ModuleNotFoundError: s2c.fakes`.

- [ ] **Step 4: Implement the fakes**

`s2c/fakes/__init__.py`:

```python
"""Stand-ins for modules owned by other people. Dev and test only."""
```

```python
# s2c/fakes/vision.py
def chat(messages: list[dict]) -> str:
    return ('{"part_type": "plate", "view": "front", '
            '"holes": [{"u": 0.17, "v": 0.25, "kind": "through"}, {"u": 0.83, "v": 0.75, "kind": "through"}], '
            '"annotation_count": 4, "confidence": 0.9, "notes": "fake"}')
```

```python
# s2c/fakes/ocr.py
import numpy as np
from s2c.partspec.models import Annotation, Annotations, Topology


def read_annotations(image_bgr: np.ndarray, topology: Topology) -> Annotations:
    def a(value, kind, linked_to, hole_index=None):
        return Annotation(value_mm=value, kind=kind, bbox_px=(0.0, 0.0, 10.0, 10.0),
                          linked_to=linked_to, hole_index=hole_index, confidence=0.9)
    return Annotations(items=[
        a(60.0, "linear", "width"), a(40.0, "linear", "height"), a(5.0, "linear", "thickness"),
        a(6.0, "diameter", "hole_diameter"),
    ], confidence=0.9)
```

```python
# s2c/fakes/metrology.py
import numpy as np
from s2c.partspec.models import Abstain, Bbox, Circle, Coin, Measurements


def measure(image_bgr: np.ndarray) -> Measurements | Abstain:
    return Measurements(
        mm_per_px=0.1,
        coin=Coin(name="1 TND", pixel_diameter=250.0, eccentricity=0.05, confidence=0.95),
        outer_contour_mm=[(0.0, 0.0), (60.0, 0.0), (60.0, 40.0), (0.0, 40.0)],
        bbox_mm=Bbox(width=60.0, height=40.0),
        circles_mm=[Circle(x=10.0, y=10.0, diameter=6.0), Circle(x=50.0, y=30.0, diameter=6.0)],
        confidence=0.9,
    )
```

```python
# s2c/fakes/builder.py
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
```

```python
# s2c/fakes/views.py
import numpy as np

VIEWS = ("front", "back", "left", "right", "top", "bottom")


def silhouettes(solid, px: int = 512) -> dict[str, np.ndarray]:
    """A filled rectangle with the aspect ratio of the front view, holes drawn if the spec has them."""
    import cv2
    from s2c.silhouette import normalize_mask
    spec = getattr(solid, "spec", None)
    w, h = 60.0, 40.0
    if spec is not None and spec.part.type == "plate":
        w, h = spec.part.width_mm, spec.part.height_mm
    scale = 400 / max(w, h)
    mask = np.zeros((int(h * scale) + 2, int(w * scale) + 2), np.uint8)
    mask[1:-1, 1:-1] = 255
    if spec is not None:
        for f in spec.features:
            if f.type == "hole":
                cv2.circle(mask, (int(f.x_mm * scale), int((h - f.y_mm) * scale)), int(f.diameter_mm * scale / 2), 0, -1)
    front = normalize_mask(mask, px)
    return {v: front for v in VIEWS}
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_fakes.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add s2c/fakes tests/test_fakes.py tests/data/plate_60x40x5.json
git commit -m "Add fake stages so the pipeline runs before real modules land"
```

---

### Task 8: Merge, sketch path

**Files:**
- Create: `s2c/merge.py`
- Test: `tests/test_merge_sketch.py`

**Interfaces:**
- Consumes: `Topology`, `Annotations`, `Measurements`, `PartSpec`, `Abstain`.
- Produces: `merge(topology, *, source_input, annotations=None, measurements=None, user_values=None) -> PartSpec | Abstain`. `user_values` keys are annotation keys (`width`, `thickness`, `leg_a`, ...) and are recorded as `user_edited`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_merge_sketch.py
from s2c.partspec.models import Topology, Annotations, Annotation, PartSpec, Abstain
from s2c.merge import merge


def ann(value, kind="linear", linked_to="unknown", hole_index=None, conf=0.9):
    return Annotation(value_mm=value, kind=kind, bbox_px=(0.0, 0.0, 1.0, 1.0),
                      linked_to=linked_to, hole_index=hole_index, confidence=conf)


def topo(**kw):
    base = dict(part_type="plate", holes=[{"u": 0.2, "v": 0.25}], confidence=0.9)
    base.update(kw)
    return Topology.model_validate(base)


def test_linked_annotations_fill_the_plate():
    a = Annotations(items=[ann(60.0, linked_to="width"), ann(40.0, linked_to="height"),
                           ann(5.0, linked_to="thickness"), ann(6.0, "diameter", "hole_diameter")], confidence=0.9)
    spec = merge(topo(), source_input="sketch", annotations=a)
    assert isinstance(spec, PartSpec)
    assert spec.part.width_mm == 60 and spec.part.height_mm == 40 and spec.part.thickness_mm == 5
    assert spec.provenance["part.width_mm"] == "user_written"
    hole = spec.features[0]
    assert (hole.x_mm, hole.y_mm, hole.diameter_mm) == (12.0, 10.0, 6.0)
    assert spec.provenance["features[0].x_mm"] == "default"
    assert any("hole position" in w for w in spec.warnings)


def test_unlinked_values_are_assigned_largest_first():
    a = Annotations(items=[ann(5.0), ann(60.0), ann(40.0)], confidence=0.8)
    spec = merge(topo(holes=[]), source_input="sketch", annotations=a)
    assert (spec.part.width_mm, spec.part.height_mm, spec.part.thickness_mm) == (60, 40, 5)


def test_missing_thickness_abstains_with_partial():
    a = Annotations(items=[ann(60.0, linked_to="width"), ann(40.0, linked_to="height")], confidence=0.8)
    out = merge(topo(holes=[]), source_input="sketch", annotations=a)
    assert isinstance(out, Abstain)
    assert out.stage == "merge" and out.reason == "missing_thickness"
    assert out.partial == {"width": 60.0, "height": 40.0}
    assert "thickness" in out.remedy


def test_user_values_fill_the_gap_as_user_edited():
    a = Annotations(items=[ann(60.0, linked_to="width"), ann(40.0, linked_to="height")], confidence=0.8)
    spec = merge(topo(holes=[]), source_input="sketch", annotations=a, user_values={"thickness": 4.0})
    assert spec.part.thickness_mm == 4.0
    assert spec.provenance["part.thickness_mm"] == "user_edited"


def test_unsupported_topology_abstains():
    out = merge(topo(part_type="unsupported", unsupported_reason="curved surface", holes=[]),
                source_input="sketch", annotations=Annotations(confidence=0.5))
    assert isinstance(out, Abstain) and out.reason == "unsupported_geometry"
    assert "curved surface" in out.remedy


def test_profile_extrusion_from_sketch_abstains():
    out = merge(topo(part_type="profile_extrusion", holes=[]), source_input="sketch",
                annotations=Annotations(confidence=0.5))
    assert isinstance(out, Abstain) and out.reason == "profile_needs_photo"


def test_spacer_uses_diameter_annotations():
    a = Annotations(items=[ann(20.0, "diameter"), ann(8.0, "diameter"), ann(30.0, linked_to="length")], confidence=0.9)
    spec = merge(topo(part_type="spacer", holes=[]), source_input="sketch", annotations=a)
    assert (spec.part.outer_diameter_mm, spec.part.inner_diameter_mm, spec.part.length_mm) == (20, 8, 30)


def test_confidence_is_the_minimum_of_stages():
    a = Annotations(items=[ann(60.0, linked_to="width"), ann(40.0, linked_to="height"),
                           ann(5.0, linked_to="thickness")], confidence=0.6)
    spec = merge(topo(holes=[]), source_input="sketch", annotations=a)
    assert spec.confidence == 0.6
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_merge_sketch.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement merge**

```python
# s2c/merge.py
"""Fuse Topology + Annotations + Measurements into a PartSpec. The only place numbers meet topology."""
from __future__ import annotations

from pydantic import ValidationError

from s2c.partspec.models import (
    Abstain, Annotations, Measurements, PartSpec, Provenance, SourceInput, Topology,
)

# annotation key -> PartSpec part field, per type
FIELD_MAP: dict[str, dict[str, str]] = {
    "plate": {"width": "width_mm", "height": "height_mm", "thickness": "thickness_mm",
              "corner_radius": "corner_radius_mm"},
    "l_bracket": {"leg_a": "leg_a_mm", "leg_b": "leg_b_mm", "width": "width_mm", "thickness": "thickness_mm"},
    "spacer": {"outer_diameter": "outer_diameter_mm", "inner_diameter": "inner_diameter_mm",
               "length": "length_mm"},
    "flange": {"outer_diameter": "outer_diameter_mm", "inner_diameter": "bore_diameter_mm",
               "thickness": "thickness_mm", "bolt_circle_diameter": "bolt_circle_diameter_mm",
               "bolt_hole_diameter": "bolt_hole_diameter_mm"},
    "profile_extrusion": {"depth": "depth_mm"},
}
REQUIRED: dict[str, list[str]] = {
    "plate": ["width", "height", "thickness"],
    "l_bracket": ["leg_a", "leg_b", "width", "thickness"],
    "spacer": ["outer_diameter", "length"],
    "flange": ["outer_diameter", "inner_diameter", "thickness", "bolt_circle_diameter", "bolt_hole_diameter"],
    "profile_extrusion": ["depth"],
}
# where unlinked values go, largest first
LINEAR_ORDER: dict[str, list[str]] = {
    "plate": ["width", "height", "thickness"],
    "l_bracket": ["leg_a", "leg_b", "width", "thickness"],
    "spacer": ["length"],
    "flange": ["thickness"],
    "profile_extrusion": ["depth"],
}
DIAMETER_ORDER: dict[str, list[str]] = {
    "plate": [], "l_bracket": [], "profile_extrusion": [],
    "spacer": ["outer_diameter", "inner_diameter"],
    "flange": ["outer_diameter", "bolt_circle_diameter", "inner_diameter", "bolt_hole_diameter"],
}

Values = dict[str, tuple[float, Provenance]]


def merge(
    topology: Topology,
    *,
    source_input: SourceInput,
    annotations: Annotations | None = None,
    measurements: Measurements | None = None,
    user_values: dict[str, float] | None = None,
) -> PartSpec | Abstain:
    if topology.part_type == "unsupported":
        return Abstain(stage="merge", reason="unsupported_geometry",
                       remedy=f"Unsupported geometry: {topology.unsupported_reason or 'not a flat profile'}.")
    if topology.part_type == "profile_extrusion" and measurements is None:
        return Abstain(stage="merge", reason="profile_needs_photo",
                       remedy="Free-form outlines need a top-down photo with a coin for scale.")

    kind = topology.part_type
    values: Values = {}
    hole_values: dict[int | None, dict[str, float]] = {}
    warnings: list[str] = []
    confidences = [topology.confidence]

    if measurements is not None:
        values.update(_values_from_measurements(measurements, kind))
        confidences.append(measurements.confidence)
    if annotations is not None:
        ann_values, hole_values = _values_from_annotations(annotations, kind)
        for k, v in ann_values.items():
            values.setdefault(k, v)
        confidences.append(annotations.confidence)
    for k, v in (user_values or {}).items():
        values[k] = (float(v), "user_edited")

    missing = [k for k in REQUIRED[kind] if k not in values]
    if missing:
        field = missing[0]
        return Abstain(
            stage="merge", reason=f"missing_{field}",
            remedy=f"We could not read the {field.replace('_', ' ')}. Enter it below.",
            partial={k: v for k, (v, _) in values.items()},
        )

    part, provenance = _assemble_part(kind, values, measurements, topology)
    features, feat_prov, feat_warn = _assemble_holes(kind, topology, values, hole_values, measurements)
    provenance.update(feat_prov)
    warnings += feat_warn

    try:
        return PartSpec(
            source_input=source_input, part=part, features=features, finishes=[],
            provenance=provenance, confidence=min(confidences), warnings=warnings,
        )
    except ValidationError as e:
        return Abstain(stage="merge", reason="inconsistent_dimensions",
                       remedy=f"The dimensions do not fit together: {e.errors()[0]['msg']}.",
                       partial={k: v for k, (v, _) in values.items()})


def _values_from_annotations(ann: Annotations, kind: str) -> tuple[Values, dict]:
    values: Values = {}
    hole_values: dict[int | None, dict[str, float]] = {}
    unknown_linear: list[float] = []
    unknown_diam: list[float] = []
    for a in sorted(ann.items, key=lambda a: -a.confidence):
        if a.linked_to in ("hole_diameter", "hole_x", "hole_y"):
            hole_values.setdefault(a.hole_index, {}).setdefault(a.linked_to, a.value_mm)
        elif a.linked_to != "unknown" and a.linked_to in FIELD_MAP[kind]:
            values.setdefault(a.linked_to, (a.value_mm, "user_written"))
        else:  # unknown, or linked to a field this part type does not have: never drop a value
            if a.kind == "diameter":
                unknown_diam.append(a.value_mm)
            elif a.kind == "radius":
                if kind == "plate":
                    values.setdefault("corner_radius", (a.value_mm, "user_written"))
            else:
                unknown_linear.append(a.value_mm)
    for key, val in zip([k for k in LINEAR_ORDER[kind] if k not in values], sorted(unknown_linear, reverse=True)):
        values[key] = (val, "user_written")
    for key, val in zip([k for k in DIAMETER_ORDER[kind] if k not in values], sorted(unknown_diam, reverse=True)):
        values[key] = (val, "user_written")
    if kind in ("plate", "l_bracket") and unknown_diam and None not in hole_values:
        hole_values[None] = {"hole_diameter": max(unknown_diam)}
    return values, hole_values


def _values_from_measurements(m: Measurements, kind: str) -> Values:
    w, h = m.bbox_mm.width, m.bbox_mm.height
    circles = sorted(m.circles_mm, key=lambda c: -c.diameter)
    v: Values = {}
    if kind == "plate":
        v["width"], v["height"] = (w, "measured"), (h, "measured")
    elif kind == "l_bracket":
        v["leg_a"], v["leg_b"] = (w, "measured"), (h, "measured")
    elif kind == "spacer":
        v["outer_diameter"] = (max(w, h), "measured")
        v["inner_diameter"] = ((circles[0].diameter if circles else 0.0), "measured")
    elif kind == "flange":
        v["outer_diameter"] = (max(w, h), "measured")
        if circles:
            v["inner_diameter"] = (circles[0].diameter, "measured")
        bolts = circles[1:]
        if bolts:
            cx, cy = w / 2, h / 2
            radius = sum(((c.x - cx) ** 2 + (c.y - cy) ** 2) ** 0.5 for c in bolts) / len(bolts)
            v["bolt_circle_diameter"] = (2 * radius, "measured")
            v["bolt_hole_diameter"] = (sum(c.diameter for c in bolts) / len(bolts), "measured")
    return v


def _assemble_part(kind: str, values: Values, m: Measurements | None, topology: Topology):
    fields: dict = {"type": kind}
    prov: dict[str, Provenance] = {}
    for key, field in FIELD_MAP[kind].items():
        if key in values:
            value, source = values[key]
            fields[field], prov[f"part.{field}"] = float(value), source
    if kind == "plate" and "corner_radius_mm" not in fields:
        fields["corner_radius_mm"], prov["part.corner_radius_mm"] = 0.0, "default"
    if kind == "l_bracket":
        fields["angle_deg"], prov["part.angle_deg"] = 90.0, "default"
    if kind == "spacer" and "inner_diameter_mm" not in fields:
        fields["inner_diameter_mm"], prov["part.inner_diameter_mm"] = 0.0, "default"
    if kind == "flange":
        if m is not None and m.circles_mm:
            count, source = max(len(m.circles_mm) - 1, 0), "measured"
        elif topology.bolt_count:
            count, source = topology.bolt_count, "default"  # a count is topology, not a dimension
        else:
            count, source = 0, "default"
        fields["bolt_count"], prov["part.bolt_count"] = count, source
    if kind == "profile_extrusion":
        fields["points_mm"] = _simplify(m.outer_contour_mm) if m is not None else []
        prov["part.points_mm"] = "measured"
    return fields, prov


def _assemble_holes(kind, topology: Topology, values: Values, hole_values: dict, m: Measurements | None):
    if kind not in ("plate", "l_bracket"):
        return [], {}, []
    if kind == "plate":
        w, h = values["width"][0], values["height"][0]
    else:
        w, h = values["leg_a"][0], values["width"][0]
    features, prov, warnings = [], {}, []
    shared_d = hole_values.get(None, {}).get("hole_diameter")

    if m is not None and m.circles_mm:
        for i, c in enumerate(m.circles_mm):
            features.append({"type": "hole", "x_mm": c.x, "y_mm": c.y, "diameter_mm": c.diameter})
            prov.update({f"features[{i}].x_mm": "measured", f"features[{i}].y_mm": "measured",
                         f"features[{i}].diameter_mm": "measured"})
        return features, prov, warnings

    for i, th in enumerate(topology.holes):
        own = hole_values.get(i, {})
        d = own.get("hole_diameter", shared_d)
        if d is None:
            warnings.append(f"hole {i + 1}: no diameter written, defaulting to 5 mm")
            d, d_prov = 5.0, "default"
        else:
            d_prov = "user_written"
        if "hole_x" in own and "hole_y" in own:
            x, y, p_prov = own["hole_x"], own["hole_y"], "user_written"
        else:
            x, y, p_prov = round(th.u * w, 1), round(th.v * h, 1), "default"
            warnings.append(f"hole {i + 1}: hole position estimated from the sketch, confirm it")
        features.append({"type": "hole", "x_mm": x, "y_mm": y, "diameter_mm": d, "leg": "a" if kind == "l_bracket" else None})
        prov.update({f"features[{i}].x_mm": p_prov, f"features[{i}].y_mm": p_prov, f"features[{i}].diameter_mm": d_prov})
    return features, prov, warnings


def _simplify(points: list[tuple[float, float]], epsilon_mm: float = 0.5) -> list[tuple[float, float]]:
    import cv2
    import numpy as np
    arr = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(arr, epsilon_mm, True).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in approx]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_merge_sketch.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add s2c/merge.py tests/test_merge_sketch.py
git commit -m "Add merge for the sketch path with abstention gates"
```

---

### Task 9: Merge, photo path

**Files:**
- Modify: `s2c/merge.py` (already handles measurements; this task tests it and fixes gaps)
- Test: `tests/test_merge_photo.py`

**Interfaces:**
- Consumes: `Measurements` from `s2c.fakes.metrology.measure`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_merge_photo.py
import numpy as np
from s2c.partspec.models import Topology, Measurements, Bbox, Circle, Coin, PartSpec, Abstain
from s2c.merge import merge
from s2c.fakes.metrology import measure


def topo(**kw):
    base = dict(part_type="plate", holes=[], confidence=0.9)
    base.update(kw)
    return Topology.model_validate(base)


def test_photo_plate_needs_thickness_then_builds():
    m = measure(np.zeros((1, 1, 3), np.uint8))
    out = merge(topo(), source_input="photo", measurements=m)
    assert isinstance(out, Abstain) and out.reason == "missing_thickness"
    spec = merge(topo(), source_input="photo", measurements=m, user_values={"thickness": 5})
    assert isinstance(spec, PartSpec)
    assert spec.provenance["part.width_mm"] == "measured"
    assert len(spec.features) == 2 and spec.provenance["features[0].x_mm"] == "measured"


def test_photo_flange_derives_bolt_circle():
    m = Measurements(
        mm_per_px=0.1, coin=Coin(name="1 TND", pixel_diameter=250, eccentricity=0.02, confidence=0.9),
        outer_contour_mm=[(0, 0), (80, 0), (80, 80), (0, 80)], bbox_mm=Bbox(width=80, height=80),
        circles_mm=[Circle(x=40, y=40, diameter=30), Circle(x=70, y=40, diameter=6), Circle(x=10, y=40, diameter=6),
                    Circle(x=40, y=70, diameter=6), Circle(x=40, y=10, diameter=6)],
        confidence=0.85,
    )
    spec = merge(topo(part_type="flange"), source_input="photo", measurements=m, user_values={"thickness": 8})
    assert isinstance(spec, PartSpec)
    assert spec.part.bolt_count == 4
    assert abs(spec.part.bolt_circle_diameter_mm - 60) < 0.01
    assert spec.part.bore_diameter_mm == 30


def test_photo_profile_extrusion_simplifies_contour():
    m = Measurements(
        mm_per_px=0.1, coin=Coin(name="1 TND", pixel_diameter=250, eccentricity=0.02, confidence=0.9),
        outer_contour_mm=[(0, 0), (30, 0), (60, 0), (60, 20), (30, 40), (0, 40)], bbox_mm=Bbox(width=60, height=40),
        confidence=0.85,
    )
    spec = merge(topo(part_type="profile_extrusion"), source_input="photo", measurements=m, user_values={"depth": 10})
    assert isinstance(spec, PartSpec)
    assert (30.0, 0.0) not in spec.part.points_mm  # collinear point removed
    assert len(spec.part.points_mm) == 5
```

- [ ] **Step 2: Run to confirm failure or pass**

Run: `uv run pytest tests/test_merge_photo.py -v`
Expected: some tests FAIL. Fix `merge.py` until they pass without breaking `tests/test_merge_sketch.py`. Likely fixes: strict float coercion (`float(...)` on every value from measurements), and `bolt_count` provenance.

- [ ] **Step 3: Run the full suite**

Run: `uv run pytest -v`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add s2c/merge.py tests/test_merge_photo.py
git commit -m "Cover the photo path in merge"
```

---

### Task 10: Temp file store

**Files:**
- Create: `s2c/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Produces: `FileStore(root: Path | None = None, ttl_s: int = 3600)`, `.put(data: bytes, suffix: str) -> str`, `.path(file_id: str) -> Path | None`, `.sweep() -> int`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_store.py
import os
import time
from s2c.store import FileStore


def test_put_and_get(tmp_path):
    s = FileStore(tmp_path, ttl_s=3600)
    fid = s.put(b"hello", ".stl")
    assert fid.endswith(".stl") and "/" not in fid and ".." not in fid
    assert s.path(fid).read_bytes() == b"hello"


def test_unknown_or_traversal_ids_return_none(tmp_path):
    s = FileStore(tmp_path)
    assert s.path("nope.stl") is None
    assert s.path("../pyproject.toml") is None


def test_sweep_removes_expired(tmp_path):
    s = FileStore(tmp_path, ttl_s=1)
    fid = s.put(b"x", ".png")
    old = time.time() - 10
    os.utime(s.path(fid), (old, old))
    assert s.sweep() == 1
    assert s.path(fid) is None
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# s2c/store.py
"""Temp files with a TTL. Nothing here outlives an hour."""
import re
import tempfile
import time
import uuid
from pathlib import Path

_SAFE = re.compile(r"^[a-f0-9]{32}\.[a-z0-9]{1,5}$")


class FileStore:
    def __init__(self, root: Path | None = None, ttl_s: int = 3600):
        self.root = Path(root) if root else Path(tempfile.gettempdir()) / "s2c_store"
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl_s = ttl_s

    def put(self, data: bytes, suffix: str) -> str:
        file_id = uuid.uuid4().hex + suffix
        (self.root / file_id).write_bytes(data)
        return file_id

    def path(self, file_id: str) -> Path | None:
        if not _SAFE.match(file_id):
            return None
        p = self.root / file_id
        return p if p.exists() else None

    def sweep(self) -> int:
        now, removed = time.time(), 0
        for p in self.root.iterdir():
            if now - p.stat().st_mtime > self.ttl_s:
                p.unlink(missing_ok=True)
                removed += 1
        return removed
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_store.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add s2c/store.py tests/test_store.py
git commit -m "Add temp file store with one-hour TTL"
```

---

### Task 11: Pipeline wiring with fallback to fakes

**Files:**
- Create: `s2c/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Produces: `Pipeline` dataclass with fields `topology(image_bytes, input_kind)`, `annotations(image_bgr, topology)`, `measure(image_bgr)`, `build(spec)`, `export(solid, out_dir)`, `silhouettes(solid, px)`; `default_pipeline(client: VLMClient | None = None) -> Pipeline`; `fake_pipeline() -> Pipeline`; `Pipeline.analyze(image_bytes, input_kind, user_values=None) -> AnalyzeResult`; `Pipeline.build_and_verify(spec, input_mask) -> BuildResult | Abstain`.
- `AnalyzeResult` fields: `partspec: PartSpec | None`, `abstain: Abstain | None`, `topology: Topology | None`, `annotations: Annotations | None`, `measurements: Measurements | None`, `input_mask: np.ndarray`.
- `BuildResult` fields: `step_path: Path`, `stl_path: Path`, `iou: float`, `views: dict[str, np.ndarray]`, `warnings: list[str]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pipeline.py
import cv2
import numpy as np
from s2c.pipeline import fake_pipeline, default_pipeline
from s2c.partspec.models import PartSpec


def sketch_bytes():
    img = np.full((600, 800, 3), 255, np.uint8)
    cv2.rectangle(img, (250, 200), (550, 400), (0, 0, 0), 3)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def test_fake_pipeline_runs_sketch_end_to_end(tmp_path):
    p = fake_pipeline()
    res = p.analyze(sketch_bytes(), "sketch")
    assert res.abstain is None and isinstance(res.partspec, PartSpec)
    assert res.input_mask.shape == (512, 512)
    built = p.build_and_verify(res.partspec, res.input_mask, tmp_path)
    assert built.stl_path.exists()
    assert 0.0 <= built.iou <= 1.0


def test_default_pipeline_falls_back_to_fakes_when_modules_missing(caplog):
    p = default_pipeline(client=None)
    assert p.build is not None and p.measure is not None
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# s2c/pipeline.py
"""Wires the stages together. Real modules when present, fakes otherwise."""
from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from s2c.merge import merge
from s2c.partspec.models import Abstain, Annotations, Measurements, PartSpec, SourceInput, Topology
from s2c.silhouette import input_silhouette, iou, normalize_mask
from s2c.vision.client import VLMClient
from s2c.vision.topology import topology_from_image

log = logging.getLogger(__name__)
IOU_GREEN = 0.85


@dataclass
class AnalyzeResult:
    input_mask: np.ndarray
    partspec: PartSpec | None = None
    abstain: Abstain | None = None
    topology: Topology | None = None
    annotations: Annotations | None = None
    measurements: Measurements | None = None


@dataclass
class BuildResult:
    step_path: Path
    stl_path: Path
    iou: float
    views: dict[str, np.ndarray]
    warnings: list[str] = field(default_factory=list)


@dataclass
class Pipeline:
    topology: Callable[[bytes, SourceInput], Topology | Abstain]
    annotations: Callable[[np.ndarray, Topology], Annotations]
    measure: Callable[[np.ndarray], Measurements | Abstain]
    build: Callable[[PartSpec], object]
    export: Callable[[object, Path], tuple[Path, Path]]
    silhouettes: Callable[..., dict[str, np.ndarray]]

    def analyze(self, image_bytes: bytes, input_kind: SourceInput,
                user_values: dict[str, float] | None = None) -> AnalyzeResult:
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return AnalyzeResult(input_mask=np.zeros((512, 512), np.uint8),
                                 abstain=Abstain(stage="vision", reason="unreadable_image",
                                                 remedy="The image could not be decoded. Use JPEG or PNG."))
        try:
            mask = input_silhouette(image)
        except ValueError:
            return AnalyzeResult(input_mask=np.zeros((512, 512), np.uint8),
                                 abstain=Abstain(stage="vision", reason="no_outline",
                                                 remedy="No part outline found. Use a plain background and good light."))
        res = AnalyzeResult(input_mask=mask)
        topo = self.topology(image_bytes, input_kind)
        if isinstance(topo, Abstain):
            res.abstain = topo
            return res
        res.topology = topo
        if input_kind == "photo":
            m = self.measure(image)
            if isinstance(m, Abstain):
                res.abstain = m
                return res
            res.measurements = m
        else:
            res.annotations = self.annotations(image, topo)
        out = merge(topo, source_input=input_kind, annotations=res.annotations,
                    measurements=res.measurements, user_values=user_values)
        if isinstance(out, Abstain):
            res.abstain = out
        else:
            res.partspec = out
        return res

    def remerge(self, topology: Topology, *, source_input: SourceInput, annotations=None,
                measurements=None, user_values=None) -> PartSpec | Abstain:
        return merge(topology, source_input=source_input, annotations=annotations,
                     measurements=measurements, user_values=user_values)

    def build_and_verify(self, spec: PartSpec, input_mask: np.ndarray, out_dir: Path) -> BuildResult | Abstain:
        try:
            solid = self.build(spec)
        except Exception as e:  # BuildError from the geometry owner carries reason and remedy
            reason = getattr(e, "reason", "build_failed")
            remedy = getattr(e, "remedy", "The part could not be built. Check the dimensions.")
            return Abstain(stage="build", reason=reason, remedy=remedy)
        step, stl = self.export(solid, Path(out_dir))
        views = {k: normalize_mask(v) for k, v in self.silhouettes(solid, px=512).items()}
        score = iou(input_mask, views["front"])
        warnings = list(spec.warnings)
        if score < IOU_GREEN:
            warnings.append(f"Low confidence (IoU {score:.2f}), check the dimensions.")
        return BuildResult(step_path=step, stl_path=stl, iou=score, views=views, warnings=warnings)


def _load(module: str, attr: str, fallback):
    try:
        return getattr(importlib.import_module(module), attr)
    except (ImportError, AttributeError):
        log.warning("using fake for %s.%s", module, attr)
        return fallback


def fake_pipeline() -> Pipeline:
    from s2c.fakes import builder, metrology, ocr, views, vision
    client = VLMClient(chat=vision.chat, model="fake", log_path="logs/vlm_fake.jsonl")
    return Pipeline(
        topology=lambda b, k: topology_from_image(b, client, k),
        annotations=ocr.read_annotations, measure=metrology.measure,
        build=builder.build, export=builder.export, silhouettes=views.silhouettes,
    )


def default_pipeline(client: VLMClient | None = None) -> Pipeline:
    from s2c.fakes import builder, metrology, ocr, views, vision
    if client is None:
        import os
        client = VLMClient.from_env() if os.environ.get("VLM_API_KEY") else VLMClient(
            chat=vision.chat, model="fake", log_path="logs/vlm_fake.jsonl")
    return Pipeline(
        topology=lambda b, k: topology_from_image(b, client, k),
        annotations=_load("s2c.ocr", "read_annotations", ocr.read_annotations),
        measure=_load("s2c.metrology", "measure", metrology.measure),
        build=_load("s2c.builder", "build", builder.build),
        export=_load("s2c.builder", "export", builder.export),
        silhouettes=_load("s2c.views", "silhouettes", views.silhouettes),
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add s2c/pipeline.py tests/test_pipeline.py
git commit -m "Wire the pipeline with fallback to fake stages"
```

---

### Task 12: FastAPI surface

**Files:**
- Create: `s2c/api.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `Pipeline`, `FileStore`.
- Produces: `app` with `POST /analyze`, `POST /merge`, `POST /build`, `GET /files/{file_id}`, `GET /health`. Response shapes are in the tests below and are what `web/` consumes.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_api.py
import cv2
import numpy as np
from fastapi.testclient import TestClient
from s2c.api import app
from s2c.pipeline import fake_pipeline
from s2c.store import FileStore


def sketch_bytes():
    img = np.full((600, 800, 3), 255, np.uint8)
    cv2.rectangle(img, (250, 200), (550, 400), (0, 0, 0), 3)
    return cv2.imencode(".jpg", img)[1].tobytes()


def client(tmp_path):
    app.state.pipeline = fake_pipeline()
    app.state.store = FileStore(tmp_path)
    return TestClient(app)


def test_health(tmp_path):
    assert client(tmp_path).get("/health").json()["ok"] is True


def test_analyze_then_build(tmp_path):
    c = client(tmp_path)
    r = c.post("/analyze", files={"image": ("s.jpg", sketch_bytes(), "image/jpeg")}, data={"input_kind": "sketch"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["abstain"] is None
    assert body["partspec"]["part"]["type"] == "plate"
    assert body["silhouette_id"].endswith(".png")
    assert body["topology"]["part_type"] == "plate"

    r2 = c.post("/build", json={"partspec": body["partspec"], "silhouette_id": body["silhouette_id"]})
    assert r2.status_code == 200, r2.text
    b = r2.json()
    assert b["abstain"] is None
    assert 0 <= b["iou"] <= 1
    assert set(b["views"]) == {"front", "back", "left", "right", "top", "bottom"}
    stl = c.get(b["stl_url"])
    assert stl.status_code == 200 and stl.content.startswith(b"solid")


def test_merge_endpoint_applies_user_values(tmp_path):
    c = client(tmp_path)
    r = c.post("/analyze", files={"image": ("s.jpg", sketch_bytes(), "image/jpeg")}, data={"input_kind": "sketch"})
    body = r.json()
    r2 = c.post("/merge", json={
        "topology": body["topology"], "annotations": body["annotations"], "measurements": None,
        "source_input": "sketch", "user_values": {"thickness": 9.0},
    })
    assert r2.status_code == 200, r2.text
    assert r2.json()["partspec"]["part"]["thickness_mm"] == 9.0
    assert r2.json()["partspec"]["provenance"]["part.thickness_mm"] == "user_edited"


def test_bad_file_id_is_404(tmp_path):
    assert client(tmp_path).get("/files/../pyproject.toml").status_code == 404
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# s2c/api.py
"""HTTP surface. Stateless apart from the temp file store."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from s2c.partspec.models import Abstain, Annotations, Measurements, PartSpec, SourceInput, Topology
from s2c.pipeline import default_pipeline
from s2c.store import FileStore

app = FastAPI(title="Sketch-to-CAD")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.state.pipeline = None
app.state.store = None


def _pipeline(request: Request):
    if request.app.state.pipeline is None:
        request.app.state.pipeline = default_pipeline()
    return request.app.state.pipeline


def _store(request: Request) -> FileStore:
    if request.app.state.store is None:
        request.app.state.store = FileStore()
    request.app.state.store.sweep()
    return request.app.state.store


def _png(mask) -> bytes:
    return cv2.imencode(".png", mask)[1].tobytes()


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/analyze")
async def analyze(request: Request, image: UploadFile = File(...), input_kind: SourceInput = Form(...)):
    p, store = _pipeline(request), _store(request)
    res = p.analyze(await image.read(), input_kind)
    return {
        "partspec": res.partspec.model_dump() if res.partspec else None,
        "abstain": res.abstain.model_dump() if res.abstain else None,
        "topology": res.topology.model_dump() if res.topology else None,
        "annotations": res.annotations.model_dump() if res.annotations else None,
        "measurements": res.measurements.model_dump() if res.measurements else None,
        "silhouette_id": store.put(_png(res.input_mask), ".png"),
    }


class MergeBody(BaseModel):
    topology: Topology
    annotations: Annotations | None = None
    measurements: Measurements | None = None
    source_input: SourceInput
    user_values: dict[str, float] | None = None


@app.post("/merge")
def remerge(request: Request, body: MergeBody):
    out = _pipeline(request).remerge(body.topology, source_input=body.source_input,
                                     annotations=body.annotations, measurements=body.measurements,
                                     user_values=body.user_values)
    if isinstance(out, Abstain):
        return {"partspec": None, "abstain": out.model_dump()}
    return {"partspec": out.model_dump(), "abstain": None}


class BuildBody(BaseModel):
    partspec: PartSpec
    silhouette_id: str


@app.post("/build")
def build(request: Request, body: BuildBody):
    p, store = _pipeline(request), _store(request)
    mask_path = store.path(body.silhouette_id)
    if mask_path is None:
        raise HTTPException(404, "silhouette expired, analyze again")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    with tempfile.TemporaryDirectory() as tmp:
        out = p.build_and_verify(body.partspec, mask, Path(tmp))
        if isinstance(out, Abstain):
            return {"abstain": out.model_dump(), "iou": None, "views": {}, "stl_url": None, "step_url": None, "warnings": []}
        stl_id = store.put(out.stl_path.read_bytes(), ".stl")
        step_id = store.put(out.step_path.read_bytes(), ".step")
    views = {k: f"/files/{store.put(_png(v), '.png')}" for k, v in out.views.items()}
    return {"abstain": None, "iou": out.iou, "views": views, "warnings": out.warnings,
            "stl_url": f"/files/{stl_id}", "step_url": f"/files/{step_id}"}


@app.get("/files/{file_id}")
def files(request: Request, file_id: str):
    path = _store(request).path(file_id)
    if path is None:
        raise HTTPException(404)
    return FileResponse(path)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_api.py -v`
Expected: PASS. If `/files/../pyproject.toml` is normalised by the test client to `/pyproject.toml`, a 404 is still the expected result.

- [ ] **Step 5: Run the server once by hand**

Run: `uv run uvicorn s2c.api:app --reload --host 0.0.0.0 --port 8000` and open `http://localhost:8000/docs`. Upload a sketch to `/analyze`.

- [ ] **Step 6: Commit**

```bash
git add s2c/api.py tests/test_api.py
git commit -m "Add FastAPI surface with analyze, merge, build and file endpoints"
```

---

### Task 13: Gradio lab UI

**Files:**
- Create: `app_gradio.py`

**Interfaces:**
- Consumes: `default_pipeline`, `BuildResult`.

- [ ] **Step 1: Write the app**

```python
# app_gradio.py
"""Lab view: every stage output side by side. Run: uv run python app_gradio.py"""
import json
import tempfile
from pathlib import Path

import cv2
import gradio as gr
from dotenv import load_dotenv

from s2c.partspec.models import Abstain, PartSpec
from s2c.pipeline import default_pipeline

load_dotenv()
PIPE = default_pipeline()
OUT = Path(tempfile.mkdtemp(prefix="s2c_lab_"))


def run(image_path, input_kind, user_values_json):
    image_bytes = Path(image_path).read_bytes()
    user_values = json.loads(user_values_json) if user_values_json.strip() else None
    res = PIPE.analyze(image_bytes, input_kind, user_values=user_values)
    stage = {
        "topology": res.topology.model_dump() if res.topology else None,
        "annotations": res.annotations.model_dump() if res.annotations else None,
        "measurements": res.measurements.model_dump() if res.measurements else None,
    }
    if res.abstain:
        return stage, res.abstain.model_dump(), None, "abstained", None, None, []
    built = PIPE.build_and_verify(res.partspec, res.input_mask, OUT)
    if isinstance(built, Abstain):
        return stage, built.model_dump(), res.partspec.model_dump(), "build abstained", None, None, []
    views = [cv2.cvtColor(v, cv2.COLOR_GRAY2RGB) for v in built.views.values()]
    status = f"IoU {built.iou:.2f} " + ("green" if built.iou >= 0.85 else "amber") + "\n" + "\n".join(built.warnings)
    return stage, None, res.partspec.model_dump(), status, str(built.stl_path), str(built.step_path), views


def rebuild(partspec_json, image_path):
    spec = PartSpec.model_validate(partspec_json)
    res = PIPE.analyze(Path(image_path).read_bytes(), spec.source_input)
    built = PIPE.build_and_verify(spec, res.input_mask, OUT)
    if isinstance(built, Abstain):
        return built.model_dump(), "build abstained", None, None, []
    views = [cv2.cvtColor(v, cv2.COLOR_GRAY2RGB) for v in built.views.values()]
    return None, f"IoU {built.iou:.2f}", str(built.stl_path), str(built.step_path), views


with gr.Blocks(title="Sketch-to-CAD lab") as demo:
    gr.Markdown("# Sketch-to-CAD lab\nEvery stage output, side by side.")
    with gr.Row():
        image = gr.Image(type="filepath", label="Sketch, photo or drawing")
        with gr.Column():
            kind = gr.Radio(["sketch", "photo", "drawing"], value="sketch", label="Input kind")
            user_values = gr.Textbox(label="User values JSON, e.g. {\"thickness\": 5}", value="")
            go = gr.Button("Run pipeline", variant="primary")
    with gr.Row():
        stage_out = gr.JSON(label="Stage outputs")
        abstain_out = gr.JSON(label="Abstain")
        spec_out = gr.JSON(label="PartSpec (editable, then Rebuild)")
    status = gr.Textbox(label="Status")
    with gr.Row():
        stl = gr.File(label="STL")
        step = gr.File(label="STEP")
    gallery = gr.Gallery(label="Six views", columns=6)
    rebuild_btn = gr.Button("Rebuild from edited PartSpec")
    go.click(run, [image, kind, user_values], [stage_out, abstain_out, spec_out, status, stl, step, gallery])
    rebuild_btn.click(rebuild, [spec_out, image], [abstain_out, status, stl, step, gallery])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
```

- [ ] **Step 2: Run it by hand**

Run: `uv run python app_gradio.py`, open `http://localhost:7860`, upload the test sketch, click Run. Expected: stage JSONs, a PartSpec, an IoU status, STL and STEP downloads, six identical rectangles from the fake views. Then edit `width_mm` in the PartSpec JSON and click Rebuild. Expected: a new STL and status.

- [ ] **Step 3: Commit**

```bash
git add app_gradio.py
git commit -m "Add Gradio lab UI showing every pipeline stage"
```

---

### Task 14: Web app scaffold and API client

**Files:**
- Create: `web/` via Vite, `web/src/api.ts`, `web/src/types.ts`, `web/.env.example`
- Test: `web/src/api.test.ts` (Vitest)

**Interfaces:**
- Produces: `analyze(file: File, kind: InputKind): Promise<AnalyzeResponse>`, `remerge(body: MergeBody): Promise<MergeResponse>`, `build(partspec: PartSpec, silhouetteId: string): Promise<BuildResponse>`, `fileUrl(path: string): string`. Types mirror the API JSON from Task 12.

- [ ] **Step 1: Scaffold**

```bash
cd web 2>/dev/null || npm create vite@latest web -- --template react-ts
cd web && npm install && npm install three && npm install -D @types/three vitest
```

Add to `web/package.json` scripts: `"test": "vitest run"`. Create `web/.env.example` with `VITE_API_BASE=http://localhost:8000`.

- [ ] **Step 2: Write types**

```ts
// web/src/types.ts
export type InputKind = "sketch" | "photo" | "drawing";
export type Provenance = "measured" | "user_written" | "user_edited" | "default";

export interface Abstain { stage: string; reason: string; remedy: string; partial: Record<string, number> | null }

export interface PartSpec {
  version: "1";
  source_input: InputKind;
  part: Record<string, unknown> & { type: string };
  features: Array<Record<string, unknown> & { type: string }>;
  finishes: Array<Record<string, unknown> & { type: string }>;
  provenance: Record<string, Provenance>;
  confidence: number;
  warnings: string[];
}

export interface AnalyzeResponse {
  partspec: PartSpec | null;
  abstain: Abstain | null;
  topology: unknown | null;
  annotations: unknown | null;
  measurements: unknown | null;
  silhouette_id: string;
}

export interface MergeBody {
  topology: unknown; annotations: unknown | null; measurements: unknown | null;
  source_input: InputKind; user_values: Record<string, number> | null;
}
export interface MergeResponse { partspec: PartSpec | null; abstain: Abstain | null }

export interface BuildResponse {
  abstain: Abstain | null; iou: number | null; views: Record<string, string>;
  stl_url: string | null; step_url: string | null; warnings: string[];
}
```

- [ ] **Step 3: Write the failing test**

```ts
// web/src/api.test.ts
import { describe, it, expect, vi } from "vitest";
import { analyze, fileUrl } from "./api";

describe("api", () => {
  it("posts multipart to /analyze", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ silhouette_id: "x.png" }) });
    vi.stubGlobal("fetch", fetchMock);
    const res = await analyze(new File(["a"], "s.jpg", { type: "image/jpeg" }), "sketch");
    expect(res.silhouette_id).toBe("x.png");
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toMatch(/\/analyze$/);
    expect(init.body).toBeInstanceOf(FormData);
    expect(init.body.get("input_kind")).toBe("sketch");
  });
  it("builds absolute file urls", () => {
    expect(fileUrl("/files/a.stl")).toMatch(/^https?:\/\/.+\/files\/a\.stl$/);
  });
});
```

- [ ] **Step 4: Run to confirm failure**

Run: `cd web && npm test`
Expected: FAIL, cannot find module `./api`.

- [ ] **Step 5: Implement the client**

```ts
// web/src/api.ts
import type { AnalyzeResponse, BuildResponse, InputKind, MergeBody, MergeResponse, PartSpec } from "./types";

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "http://localhost:8000";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json() as Promise<T>;
}

export async function analyze(file: File, kind: InputKind): Promise<AnalyzeResponse> {
  const body = new FormData();
  body.append("image", file);
  body.append("input_kind", kind);
  return json(await fetch(`${BASE}/analyze`, { method: "POST", body }));
}

export async function remerge(body: MergeBody): Promise<MergeResponse> {
  return json(await fetch(`${BASE}/merge`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  }));
}

export async function build(partspec: PartSpec, silhouetteId: string): Promise<BuildResponse> {
  return json(await fetch(`${BASE}/build`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ partspec, silhouette_id: silhouetteId }),
  }));
}

export function fileUrl(path: string): string {
  return `${BASE}${path}`;
}
```

- [ ] **Step 6: Run tests**

Run: `cd web && npm test`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add web
git commit -m "Scaffold mobile web app with typed API client"
```

---

### Task 15: Capture screen and app state

**Files:**
- Create: `web/src/screens/Capture.tsx`, `web/src/state.ts`
- Modify: `web/src/App.tsx`

**Interfaces:**
- Produces: `AppState` with `screen: "capture" | "review" | "export"`, `kind`, `file`, `analysis: AnalyzeResponse | null`, `partspec`, `buildResult`. `Capture` calls `analyze` and moves to `review`, or shows the abstain card.

- [ ] **Step 1: Write state**

```ts
// web/src/state.ts
import { useState } from "react";
import type { AnalyzeResponse, BuildResponse, InputKind, PartSpec } from "./types";

export type Screen = "capture" | "review" | "export";

export function useAppState() {
  const [screen, setScreen] = useState<Screen>("capture");
  const [kind, setKind] = useState<InputKind>("sketch");
  const [file, setFile] = useState<File | null>(null);
  const [analysis, setAnalysis] = useState<AnalyzeResponse | null>(null);
  const [partspec, setPartspec] = useState<PartSpec | null>(null);
  const [buildResult, setBuildResult] = useState<BuildResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  return { screen, setScreen, kind, setKind, file, setFile, analysis, setAnalysis,
           partspec, setPartspec, buildResult, setBuildResult, busy, setBusy, error, setError };
}
export type AppState = ReturnType<typeof useAppState>;
```

- [ ] **Step 2: Write the Capture screen**

```tsx
// web/src/screens/Capture.tsx
import { analyze } from "../api";
import type { AppState } from "../state";
import type { InputKind } from "../types";

const KINDS: Array<{ id: InputKind; label: string; hint: string }> = [
  { id: "sketch", label: "Sketch", hint: "Write the dimensions on the paper in mm." },
  { id: "photo", label: "Real part", hint: "Lay a coin flat next to the part. Shoot straight down." },
  { id: "drawing", label: "2D drawing", hint: "A clean orthographic view with dimensions." },
];

export function Capture({ s }: { s: AppState }) {
  async function onFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    s.setFile(f); s.setBusy(true); s.setError(null);
    try {
      const res = await analyze(f, s.kind);
      s.setAnalysis(res);
      s.setPartspec(res.partspec);
      s.setScreen("review");
    } catch (err) {
      s.setError(String(err));
    } finally {
      s.setBusy(false);
    }
  }
  const hint = KINDS.find(k => k.id === s.kind)!.hint;
  return (
    <main className="screen">
      <h1>Sketch to CAD</h1>
      <p>Point your phone at a sketch or a broken part. Get a printable model back.</p>
      <div className="segmented">
        {KINDS.map(k => (
          <button key={k.id} className={k.id === s.kind ? "on" : ""} onClick={() => s.setKind(k.id)}>{k.label}</button>
        ))}
      </div>
      <p className="hint">{hint}</p>
      <label className="capture-btn">
        {s.busy ? "Analysing…" : "Open camera"}
        <input type="file" accept="image/*" capture="environment" onChange={onFile} disabled={s.busy} hidden />
      </label>
      {s.error && <div className="card error">{s.error}</div>}
    </main>
  );
}
```

- [ ] **Step 3: Wire App.tsx**

```tsx
// web/src/App.tsx
import { useAppState } from "./state";
import { Capture } from "./screens/Capture";
import { Review } from "./screens/Review";
import { Export } from "./screens/Export";
import "./App.css";

export default function App() {
  const s = useAppState();
  if (s.screen === "review") return <Review s={s} />;
  if (s.screen === "export") return <Export s={s} />;
  return <Capture s={s} />;
}
```

Create placeholder `Review.tsx` and `Export.tsx` that render `<main>Review</main>` and `<main>Export</main>` so the app compiles; Tasks 16 and 17 replace them.

- [ ] **Step 4: Minimal CSS**

```css
/* web/src/App.css */
:root { font-family: system-ui, sans-serif; color-scheme: light dark; }
body { margin: 0; }
.screen { max-width: 480px; margin: 0 auto; padding: 16px; }
.segmented { display: flex; gap: 4px; }
.segmented button { flex: 1; padding: 10px; border: 1px solid #888; background: transparent; border-radius: 8px; }
.segmented button.on { background: #2563eb; color: white; border-color: #2563eb; }
.hint { opacity: 0.7; font-size: 14px; }
.capture-btn { display: block; text-align: center; padding: 18px; background: #2563eb; color: white; border-radius: 12px; font-size: 18px; margin-top: 16px; }
.card { border: 1px solid #888; border-radius: 12px; padding: 12px; margin: 12px 0; }
.card.error { border-color: #dc2626; }
.card.amber { border-color: #d97706; }
.card.green { border-color: #16a34a; }
.badge { font-size: 11px; padding: 2px 6px; border-radius: 6px; background: #8883; margin-left: 6px; }
.field { display: grid; grid-template-columns: 1fr 80px; gap: 8px; align-items: center; margin: 8px 0; }
.field input[type=range] { width: 100%; }
.viewer { width: 100%; height: 260px; border-radius: 12px; overflow: hidden; background: #111; }
.row { display: flex; gap: 8px; }
.row > * { flex: 1; }
button.primary { padding: 14px; background: #2563eb; color: white; border: 0; border-radius: 12px; font-size: 16px; }
```

- [ ] **Step 5: Run the dev server on your phone**

Run: `cd web && npm run dev -- --host` with the API running. Open the LAN URL on the phone, take a photo of a sketch. Expected: the screen switches to the Review placeholder.

- [ ] **Step 6: Commit**

```bash
git add web/src
git commit -m "Add capture screen with camera input and app state"
```

---

### Task 16: Review screen with provenance sliders, abstain card and 3D viewer

**Files:**
- Create: `web/src/components/SpecForm.tsx`, `web/src/components/Viewer.tsx`, `web/src/components/AbstainCard.tsx`
- Modify: `web/src/screens/Review.tsx`
- Test: `web/src/components/SpecForm.test.tsx`

**Interfaces:**
- Consumes: `build`, `remerge`, `fileUrl`.
- Produces: `numericPaths(spec): Array<{ path: string; label: string; value: number }>`, `setPath(spec, path, value): PartSpec` (pure helpers, tested), `SpecForm`, `Viewer({ stlUrl })`, `AbstainCard({ abstain, onSubmit })`.

- [ ] **Step 1: Write the failing helper test**

```tsx
// web/src/components/SpecForm.test.tsx
import { describe, it, expect } from "vitest";
import { numericPaths, setPath } from "./SpecForm";
import type { PartSpec } from "../types";

const spec: PartSpec = {
  version: "1", source_input: "sketch",
  part: { type: "plate", width_mm: 60, height_mm: 40, thickness_mm: 5, corner_radius_mm: 0 },
  features: [{ type: "hole", x_mm: 10, y_mm: 10, diameter_mm: 6, depth_mm: null, leg: null }],
  finishes: [],
  provenance: { "part.width_mm": "user_written", "part.height_mm": "user_written", "part.thickness_mm": "user_written",
    "part.corner_radius_mm": "default", "features[0].x_mm": "default", "features[0].y_mm": "default", "features[0].diameter_mm": "user_written" },
  confidence: 0.9, warnings: [],
};

describe("SpecForm helpers", () => {
  it("lists every numeric path with provenance", () => {
    const paths = numericPaths(spec);
    expect(paths.map(p => p.path)).toEqual([
      "part.width_mm", "part.height_mm", "part.thickness_mm", "part.corner_radius_mm",
      "features[0].x_mm", "features[0].y_mm", "features[0].diameter_mm",
    ]);
  });
  it("setPath updates the value and marks it user_edited", () => {
    const next = setPath(spec, "features[0].x_mm", 12);
    expect(next.features[0].x_mm).toBe(12);
    expect(next.provenance["features[0].x_mm"]).toBe("user_edited");
    expect(spec.features[0].x_mm).toBe(10);
  });
});
```

- [ ] **Step 2: Run to confirm failure**

Run: `cd web && npm test`
Expected: FAIL, cannot find module `./SpecForm`.

- [ ] **Step 3: Implement SpecForm**

```tsx
// web/src/components/SpecForm.tsx
import type { PartSpec, Provenance } from "../types";

export interface NumericPath { path: string; label: string; value: number; provenance: Provenance }

function numbersOf(obj: Record<string, unknown>, prefix: string, spec: PartSpec): NumericPath[] {
  return Object.entries(obj)
    .filter(([k, v]) => k !== "type" && typeof v === "number")
    .map(([k, v]) => ({ path: `${prefix}.${k}`, label: k.replace(/_mm$|_deg$/, "").replace(/_/g, " "),
                       value: v as number, provenance: spec.provenance[`${prefix}.${k}`] ?? "default" }));
}

export function numericPaths(spec: PartSpec): NumericPath[] {
  return [
    ...numbersOf(spec.part, "part", spec),
    ...spec.features.flatMap((f, i) => numbersOf(f, `features[${i}]`, spec)),
    ...spec.finishes.flatMap((f, i) => numbersOf(f, `finishes[${i}]`, spec)),
  ];
}

export function setPath(spec: PartSpec, path: string, value: number): PartSpec {
  const next: PartSpec = structuredClone(spec);
  const m = path.match(/^(part|features|finishes)(?:\[(\d+)\])?\.(\w+)$/);
  if (!m) return spec;
  const [, root, idx, key] = m;
  if (root === "part") (next.part as Record<string, unknown>)[key] = value;
  else (next[root as "features" | "finishes"][Number(idx)] as Record<string, unknown>)[key] = value;
  next.provenance[path] = "user_edited";
  return next;
}

const BADGE: Record<Provenance, string> = { measured: "measured", user_written: "written", user_edited: "edited", default: "guess" };

export function SpecForm({ spec, onChange }: { spec: PartSpec; onChange: (s: PartSpec) => void }) {
  return (
    <div>
      {numericPaths(spec).map(p => {
        const max = Math.max(10, p.value * 2);
        return (
          <div className="field" key={p.path}>
            <label>
              {p.label}<span className="badge">{BADGE[p.provenance]}</span>
              <input type="range" min={0} max={max} step={0.5} value={p.value}
                     onChange={e => onChange(setPath(spec, p.path, Number(e.target.value)))} />
            </label>
            <input type="number" step={0.1} value={p.value}
                   onChange={e => onChange(setPath(spec, p.path, Number(e.target.value)))} />
          </div>
        );
      })}
    </div>
  );
}
```

- [ ] **Step 4: Implement the Viewer**

```tsx
// web/src/components/Viewer.tsx
import { useEffect, useRef } from "react";
import * as THREE from "three";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

export function Viewer({ stlUrl }: { stlUrl: string | null }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el || !stlUrl) return;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, el.clientWidth / el.clientHeight, 0.1, 5000);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(el.clientWidth, el.clientHeight);
    el.replaceChildren(renderer.domElement);
    scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.2));
    const dir = new THREE.DirectionalLight(0xffffff, 0.8); dir.position.set(1, 2, 3); scene.add(dir);
    const controls = new OrbitControls(camera, renderer.domElement);
    let frame = 0;
    new STLLoader().load(stlUrl, geom => {
      geom.computeBoundingBox();
      const box = geom.boundingBox!; const size = new THREE.Vector3(); box.getSize(size);
      const center = new THREE.Vector3(); box.getCenter(center);
      geom.translate(-center.x, -center.y, -center.z);
      const mesh = new THREE.Mesh(geom, new THREE.MeshStandardMaterial({ color: 0x3b82f6, metalness: 0.1, roughness: 0.6 }));
      scene.add(mesh);
      const d = Math.max(size.x, size.y, size.z) * 2;
      camera.position.set(d, d, d); camera.lookAt(0, 0, 0); controls.update();
      const loop = () => { frame = requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); };
      loop();
    });
    return () => { cancelAnimationFrame(frame); renderer.dispose(); };
  }, [stlUrl]);
  return <div className="viewer" ref={ref} />;
}
```

- [ ] **Step 5: Implement the AbstainCard**

```tsx
// web/src/components/AbstainCard.tsx
import { useState } from "react";
import type { Abstain } from "../types";

export function AbstainCard({ abstain, onSubmit }: { abstain: Abstain; onSubmit?: (values: Record<string, number>) => void }) {
  const missing = abstain.reason.startsWith("missing_") ? abstain.reason.slice("missing_".length) : null;
  const [value, setValue] = useState("");
  return (
    <div className="card error">
      <strong>{abstain.stage}: {abstain.reason.replace(/_/g, " ")}</strong>
      <p>{abstain.remedy}</p>
      {missing && onSubmit && (
        <div className="row">
          <input type="number" placeholder={`${missing.replace(/_/g, " ")} in mm`} value={value} onChange={e => setValue(e.target.value)} />
          <button className="primary" onClick={() => onSubmit({ [missing]: Number(value) })}>Use it</button>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 6: Implement the Review screen**

```tsx
// web/src/screens/Review.tsx
import { useEffect, useRef } from "react";
import { build, fileUrl, remerge } from "../api";
import { AbstainCard } from "../components/AbstainCard";
import { SpecForm } from "../components/SpecForm";
import { Viewer } from "../components/Viewer";
import type { AppState } from "../state";

export function Review({ s }: { s: AppState }) {
  const timer = useRef<number | undefined>(undefined);

  async function rebuild() {
    if (!s.partspec || !s.analysis) return;
    s.setBusy(true);
    try { s.setBuildResult(await build(s.partspec, s.analysis.silhouette_id)); }
    catch (e) { s.setError(String(e)); }
    finally { s.setBusy(false); }
  }

  useEffect(() => {
    if (!s.partspec) return;
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(rebuild, 400);
    return () => window.clearTimeout(timer.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s.partspec]);

  async function fillMissing(values: Record<string, number>) {
    if (!s.analysis) return;
    const prev = s.analysis.abstain?.partial ?? {};
    const res = await remerge({ topology: s.analysis.topology, annotations: s.analysis.annotations,
      measurements: s.analysis.measurements, source_input: s.kind, user_values: { ...prev, ...values } });
    s.setAnalysis({ ...s.analysis, partspec: res.partspec, abstain: res.abstain });
    s.setPartspec(res.partspec);
  }

  const abstain = s.analysis?.abstain ?? s.buildResult?.abstain ?? null;
  const iou = s.buildResult?.iou ?? null;
  return (
    <main className="screen">
      <button onClick={() => s.setScreen("capture")}>← Retake</button>
      {abstain && <AbstainCard abstain={abstain} onSubmit={fillMissing} />}
      {s.partspec && (
        <>
          <Viewer stlUrl={s.buildResult?.stl_url ? fileUrl(s.buildResult.stl_url) : null} />
          {iou !== null && (
            <div className={`card ${iou >= 0.85 ? "green" : "amber"}`}>
              Round-trip match {Math.round(iou * 100)}%{iou < 0.85 && ": low confidence, check the dimensions."}
            </div>
          )}
          {s.buildResult?.warnings.map(w => <p className="hint" key={w}>{w}</p>)}
          <SpecForm spec={s.partspec} onChange={s.setPartspec} />
          <button className="primary" disabled={!s.buildResult?.stl_url || s.busy} onClick={() => s.setScreen("export")}>
            {s.busy ? "Building…" : "Export"}
          </button>
        </>
      )}
    </main>
  );
}
```

- [ ] **Step 7: Run tests and try on the phone**

Run: `cd web && npm test && npm run build`
Expected: tests PASS, build succeeds. On the phone: after capture, the viewer shows the fake cube, sliders show badges, moving a slider rebuilds after 400 ms and the IoU card updates.

- [ ] **Step 8: Commit**

```bash
git add web/src
git commit -m "Add review screen with provenance sliders, abstain card and 3D viewer"
```

---

### Task 17: Export screen

**Files:**
- Modify: `web/src/screens/Export.tsx`

- [ ] **Step 1: Implement**

```tsx
// web/src/screens/Export.tsx
import { fileUrl } from "../api";
import type { AppState } from "../state";

export function Export({ s }: { s: AppState }) {
  const b = s.buildResult;
  if (!b?.stl_url || !b.step_url) return <main className="screen"><p>Nothing built yet.</p></main>;
  return (
    <main className="screen">
      <button onClick={() => s.setScreen("review")}>← Back to review</button>
      <h2>Your part is ready</h2>
      <p>Round-trip match {Math.round((b.iou ?? 0) * 100)}%. Every number came from your sketch or your edits, never from the model.</p>
      <div className="row">
        <a className="primary" href={fileUrl(b.stl_url)} download="part.stl">Download STL</a>
        <a className="primary" href={fileUrl(b.step_url)} download="part.step">Download STEP</a>
      </div>
      <h3>Six views</h3>
      <div className="row" style={{ flexWrap: "wrap" }}>
        {Object.entries(b.views).map(([name, url]) => (
          <figure key={name} style={{ width: "30%", margin: 4 }}>
            <img src={fileUrl(url)} alt={name} style={{ width: "100%", borderRadius: 8 }} />
            <figcaption className="hint">{name}</figcaption>
          </figure>
        ))}
      </div>
      <button onClick={() => { s.setScreen("capture"); s.setBuildResult(null); s.setPartspec(null); s.setAnalysis(null); }}>
        Start another part
      </button>
    </main>
  );
}
```

Add `a.primary { display: block; text-align: center; padding: 14px; background: #2563eb; color: white; border-radius: 12px; text-decoration: none; }` to `App.css`.

- [ ] **Step 2: Verify on the phone**

Expected: both downloads work and the six view images appear.

- [ ] **Step 3: Commit**

```bash
git add web/src
git commit -m "Add export screen with downloads and six views"
```

---

### Task 18: Golden-set test harness

**Files:**
- Create: `tests/test_golden.py`, `tests/golden/README.md`

**Interfaces:**
- Consumes: `default_pipeline`, golden folders `tests/golden/<name>/image.jpg` + `expected.json` (an `expected.json` holds `{"input_kind": "sketch", "part": {...}, "features": [...]}` with the same field names as PartSpec). The geometry owner fills the folders.

- [ ] **Step 1: Write the harness**

```python
# tests/test_golden.py
"""Runs the full pipeline on every golden folder. Skips when no VLM key is set (CI) unless GOLDEN_FAKE=1."""
import json
import os
from pathlib import Path

import pytest

from s2c.pipeline import default_pipeline, fake_pipeline

GOLDEN = Path(__file__).parent / "golden"
CASES = sorted(p for p in GOLDEN.iterdir() if (p / "expected.json").exists()) if GOLDEN.exists() else []


def within(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= max(1.0, 0.05 * abs(expected))


@pytest.fixture(scope="module")
def pipe():
    if os.environ.get("GOLDEN_FAKE") == "1":
        return fake_pipeline()
    if not os.environ.get("VLM_API_KEY"):
        pytest.skip("set VLM_API_KEY (or GOLDEN_FAKE=1) to run the golden set")
    return default_pipeline()


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_golden_case(case: Path, pipe, tmp_path):
    expected = json.loads((case / "expected.json").read_text())
    image = next(p for p in case.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    res = pipe.analyze(image.read_bytes(), expected["input_kind"], user_values=expected.get("user_values"))
    assert res.abstain is None, res.abstain
    part = res.partspec.part.model_dump()
    for key, val in expected["part"].items():
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            assert within(part[key], val), f"{key}: got {part[key]}, expected {val}"
        else:
            assert part[key] == val
    assert len(res.partspec.features) == len(expected.get("features", []))
    built = pipe.build_and_verify(res.partspec, res.input_mask, tmp_path)
    assert built.iou >= 0.85, f"IoU {built.iou:.2f}"
```

```markdown
<!-- tests/golden/README.md -->
One folder per case: `image.jpg` and `expected.json`.

expected.json:
{
  "input_kind": "sketch",
  "user_values": {"thickness": 5.0},
  "part": {"type": "plate", "width_mm": 60.0, "height_mm": 40.0, "thickness_mm": 5.0, "corner_radius_mm": 0.0},
  "features": [{"type": "hole", "x_mm": 10.0, "y_mm": 10.0, "diameter_mm": 6.0}]
}

Run: `uv run pytest tests/test_golden.py -v` with a `.env`, or `GOLDEN_FAKE=1 uv run pytest tests/test_golden.py` to check the harness.
```

- [ ] **Step 2: Run the harness with fakes**

Run: `GOLDEN_FAKE=1 uv run pytest tests/test_golden.py -v` (PowerShell: `$env:GOLDEN_FAKE=1; uv run pytest tests/test_golden.py -v`)
Expected: no cases yet, so pytest reports "no tests ran" or the parametrised test is empty. Once the geometry owner adds folders, cases appear.

- [ ] **Step 3: Commit**

```bash
git add tests/test_golden.py tests/golden/README.md
git commit -m "Add golden-set harness with tolerance and IoU checks"
```

---

### Task 19: CI, README and disclosure skeleton

**Files:**
- Create: `.github/workflows/ci.yml`, `README.md` (replace), `docs/disclosure.md`

- [ ] **Step 1: CI**

```yaml
# .github/workflows/ci.yml
name: ci
on: [push, pull_request]
jobs:
  python:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv python install 3.11
      - run: uv sync
      - run: uv run ruff check .
      - run: uv run pytest -q
  web:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: 20 }
      - run: cd web && npm ci && npm test && npm run build
```

- [ ] **Step 2: README**

```markdown
# Sketch-to-CAD

Point your phone at a broken part or a sketch of one. Get a parametric STEP + STL back.

## Run it
    cp .env.example .env   # add a vision model key
    uv sync && uv run uvicorn s2c.api:app --host 0.0.0.0 --port 8000
    cd web && npm install && npm run dev -- --host

Lab view: `uv run python app_gradio.py`

## How it works
Classical CV measures. The model only understands topology. Our own deterministic builder makes the geometry. A round-trip silhouette check scores every result. Full design in `docs/superpowers/specs/`.

## Numbers (updated as tests land)
| Metric | Value |
| --- | --- |
| Golden sketches passing | pending |
| Coin scale error | pending |
| OCR value accuracy | pending |
| Median sketch-to-STL latency | pending |
```

- [ ] **Step 3: Disclosure skeleton**

```markdown
# Tool disclosure

| Category | Used | Notes |
| --- | --- | --- |
| Vision model (event) | pending, NVIDIA Build | topology only, never dimensions |
| Vision model (pre-event testing) | pending | see docs/models.md |
| OCR | pending | trained model or VLM-assisted, see numbers plan |
| Classical CV | OpenCV | coin scale, contours, silhouettes |
| CAD kernel | CadQuery (OpenCascade) | deterministic geometry |
| Coding assistants | pending | list what the team used |
| Datasets | our own golden set, no external data | |
| Generated assets | none | |

Privacy: images are processed in memory; one silhouette PNG and the exported files live in a temp folder for one hour.
Pre-event work: the pipeline was scaffolded and tested in the week before the event; event day was integration on NVIDIA Build, polish and the demo.
```

- [ ] **Step 4: Commit and push**

```bash
git add .github README.md docs/disclosure.md
git commit -m "Add CI, README and tool disclosure skeleton"
git push origin main
```

---

## Self-review against the spec

- Rule 1 (no code from the model): Task 6 prompts ask for JSON only; Topology is the only thing parsed. Covered.
- Rule 2 (no model numbers): Topology `extra="forbid"` in Task 3, retry test in Task 6. Covered.
- Rule 3 (profile + features): grammar enforced by the discriminated unions in Task 2. Covered.
- Rule 4 gates: coin gate lives in the numbers plan; grammar, missing dimension, schema and round-trip gates are in Tasks 6, 8, 11. Covered.
- Contracts 4.2: Tasks 2 and 3. Merge rules 4.3: Tasks 8 and 9. Provider layer 4.4: Task 5. API 4.5: Task 12. UI 4.6: Tasks 13, 15 to 17. Verification 4.7: Tasks 4, 11, 18. Workflow 5: Task 19 CI. Judging map 7: README and disclosure in Task 19.
- Gap: `slot` features are in the grammar but `merge.py` never emits them from Topology slots. Deliberate for version 1: slots are added by the user through the review form only. Add a "+ slot" button to `SpecForm` if time allows on 26 September; not required for the demo.
- Type consistency: `Pipeline.build_and_verify(spec, input_mask, out_dir)` is used with three arguments in Tasks 11, 12, 13 and 18. `analyze` returns `AnalyzeResult` everywhere. `BuildResponse.views` is a map of view name to `/files/...` path in Task 12 and consumed as such in Task 17.
