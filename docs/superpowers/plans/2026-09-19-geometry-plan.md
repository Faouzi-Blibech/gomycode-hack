# Geometry Implementation Plan (builder, views, golden set)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a validated PartSpec into a CadQuery solid, STEP and STL files, and six orthographic silhouettes, and produce the golden set of ground-truth parts the whole team tests against.

**Architecture:** `s2c/builder.py` has one function per part type and one per feature, all pure functions from Pydantic models to CadQuery workplanes. `s2c/views.py` tessellates the solid and rasterises the triangles along each axis. Both are replaced into the pipeline automatically: `s2c/pipeline.py` imports them if they exist and falls back to `s2c/fakes/` otherwise.

**Tech Stack:** Python 3.11, CadQuery 2.x (OpenCascade), NumPy, OpenCV headless, pytest.

**Spec:** `docs/superpowers/specs/2026-09-19-sketch-to-cad-design.md` (sections 3, 4.1, 4.7). Contracts are in `s2c/partspec/models.py`; read that file before starting.

## Global Constraints

- Grammar is frozen: `plate`, `l_bracket`, `flange`, `spacer`, `profile_extrusion`; features `hole`, `slot`, `fillet`, `chamfer`. Do not add types. (spec 3)
- All dimensions are millimetre floats. (spec 3)
- Coordinate convention: front view is the XY plane, extrusion along +Z. Origin at the bottom-left of the front-view bounding box for plate, l_bracket and profile_extrusion. Round parts (spacer, flange) are centred on the origin. (role brief)
- A valid spec that cannot be built raises `BuildError(reason, remedy)`; never return a wrong solid. (spec Rule 4)
- Self round-trip IoU of a built part against its own silhouette must be above 0.98. (spec 4.7)
- Volume tests within 0.5 percent for every type. (role brief)
- Commit messages: plain, no AI attribution, no co-author trailers. (spec 5)
- Write the failing test first. (CLAUDE.md)

Prerequisite: the integrator's Tasks 1 to 4 and 7 are merged (package, contracts, `s2c/silhouette.py`, `s2c/fakes/`). Until then, run against the branch that has them.

---

## File structure

| Path | Responsibility |
| --- | --- |
| `s2c/builder.py` | `BuildError`, `build(spec)`, `export(solid, out_dir)`, one private function per part type and feature |
| `s2c/views.py` | `silhouettes(solid, px)` |
| `tests/test_builder.py` | Volume and export tests |
| `tests/test_views.py` | Self round-trip tests |
| `tests/golden/<name>/` | `image.jpg` + `expected.json` per case |
| `tests/test_golden_files.py` | Every golden folder is well-formed |
| `scripts/render_golden.py` | Renders each golden expected part to STL for checking in a viewer |

---

### Task 1: Plate and spacer with volume tests

**Files:**
- Create: `s2c/builder.py`
- Test: `tests/test_builder.py`

**Interfaces:**
- Consumes: `PartSpec`, `Plate`, `Spacer` from `s2c.partspec.models`.
- Produces: `class BuildError(Exception)` with `.reason: str` and `.remedy: str`; `build(spec: PartSpec) -> cadquery.Workplane`; `volume(solid) -> float` (test helper, exported for the golden script).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_builder.py
import math
import pytest
from s2c.partspec.models import PartSpec
from s2c.builder import build, volume, BuildError


def spec(part: dict, features=(), finishes=(), source="sketch") -> PartSpec:
    """Build a PartSpec with provenance filled in automatically. Test helper only."""
    data = {"source_input": source, "part": part, "features": list(features), "finishes": list(finishes),
            "provenance": {}, "confidence": 0.9}
    def prov(prefix, obj):
        for k, v in obj.items():
            if k != "type" and isinstance(v, (int, float)) and not isinstance(v, bool) or k == "points_mm":
                data["provenance"][f"{prefix}.{k}"] = "user_written"
    prov("part", part)
    for i, f in enumerate(features):
        prov(f"features[{i}]", f)
    for i, f in enumerate(finishes):
        prov(f"finishes[{i}]", f)
    return PartSpec.model_validate(data)


def test_plate_volume():
    s = spec({"type": "plate", "width_mm": 60.0, "height_mm": 40.0, "thickness_mm": 5.0, "corner_radius_mm": 0.0})
    assert math.isclose(volume(build(s)), 60 * 40 * 5, rel_tol=0.005)


def test_plate_origin_is_bottom_left():
    s = spec({"type": "plate", "width_mm": 60.0, "height_mm": 40.0, "thickness_mm": 5.0, "corner_radius_mm": 0.0})
    bb = build(s).val().BoundingBox()
    assert (round(bb.xmin), round(bb.ymin), round(bb.zmin)) == (0, 0, 0)
    assert (round(bb.xmax), round(bb.ymax), round(bb.zmax)) == (60, 40, 5)


def test_rounded_plate_has_less_volume():
    flat = spec({"type": "plate", "width_mm": 60.0, "height_mm": 40.0, "thickness_mm": 5.0, "corner_radius_mm": 0.0})
    rounded = spec({"type": "plate", "width_mm": 60.0, "height_mm": 40.0, "thickness_mm": 5.0, "corner_radius_mm": 5.0})
    expected_removed = (4 - math.pi) * 5 ** 2 * 5
    assert math.isclose(volume(build(flat)) - volume(build(rounded)), expected_removed, rel_tol=0.01)


def test_spacer_tube_volume():
    s = spec({"type": "spacer", "outer_diameter_mm": 20.0, "inner_diameter_mm": 8.0, "length_mm": 30.0})
    expected = math.pi * (10 ** 2 - 4 ** 2) * 30
    assert math.isclose(volume(build(s)), expected, rel_tol=0.005)


def test_solid_spacer():
    s = spec({"type": "spacer", "outer_diameter_mm": 20.0, "inner_diameter_mm": 0.0, "length_mm": 30.0})
    assert math.isclose(volume(build(s)), math.pi * 100 * 30, rel_tol=0.005)


def test_corner_radius_too_big_raises_build_error():
    s = spec({"type": "plate", "width_mm": 20.0, "height_mm": 10.0, "thickness_mm": 5.0, "corner_radius_mm": 6.0})
    with pytest.raises(BuildError) as e:
        build(s)
    assert e.value.reason == "fillet_failed"
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_builder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 's2c.builder'`.

- [ ] **Step 3: Implement**

```python
# s2c/builder.py
"""PartSpec -> CadQuery solid. Deterministic. No model output is ever executed here."""
from __future__ import annotations

from pathlib import Path

import cadquery as cq

from s2c.partspec.models import (
    Chamfer, Fillet, Flange, Hole, LBracket, PartSpec, Plate, ProfileExtrusion, Slot, Spacer,
)

BIG = 10_000.0  # longer than any part, used for through cuts


class BuildError(Exception):
    def __init__(self, reason: str, remedy: str):
        super().__init__(f"{reason}: {remedy}")
        self.reason, self.remedy = reason, remedy


def volume(solid: cq.Workplane) -> float:
    return float(solid.val().Volume())


def build(spec: PartSpec) -> cq.Workplane:
    part = spec.part
    solid = _PART_BUILDERS[part.type](part)
    for feature in spec.features:
        solid = _apply_feature(solid, part, feature)
    for finish in spec.finishes:
        solid = _apply_finish(solid, finish)
    return solid


# ---- part types -----------------------------------------------------------

def _plate(p: Plate) -> cq.Workplane:
    solid = cq.Workplane("XY").box(p.width_mm, p.height_mm, p.thickness_mm, centered=False)
    if p.corner_radius_mm > 0:
        if 2 * p.corner_radius_mm >= min(p.width_mm, p.height_mm):
            raise BuildError("fillet_failed", "Reduce the corner radius; it is larger than half the shortest side.")
        try:
            solid = solid.edges("|Z").fillet(p.corner_radius_mm)
        except Exception as e:  # OCC raises StdFail_NotDone
            raise BuildError("fillet_failed", "Reduce the corner radius.") from e
    return solid


def _spacer(p: Spacer) -> cq.Workplane:
    solid = cq.Workplane("XY").circle(p.outer_diameter_mm / 2).extrude(p.length_mm)
    if p.inner_diameter_mm > 0:
        solid = solid.faces(">Z").workplane().hole(p.inner_diameter_mm)
    return solid


_PART_BUILDERS = {"plate": _plate, "spacer": _spacer}


def _apply_feature(solid, part, feature):
    raise BuildError("unsupported_feature", "Features land in Task 3.")


def _apply_finish(solid, finish):
    raise BuildError("unsupported_finish", "Finishes land in Task 3.")
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_builder.py -v`
Expected: PASS. If CadQuery fillet raises a different exception type on your platform, widen the `except`.

- [ ] **Step 5: Commit**

```bash
git add s2c/builder.py tests/test_builder.py
git commit -m "Build plate and spacer from PartSpec with volume tests"
```

---

### Task 2: Export to STEP and STL

**Files:**
- Modify: `s2c/builder.py`
- Test: `tests/test_builder.py` (append)

**Interfaces:**
- Produces: `export(solid: cq.Workplane, out_dir: Path) -> tuple[Path, Path]` returning `(step_path, stl_path)`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_builder.py
from s2c.builder import export


def test_export_writes_step_and_stl(tmp_path):
    s = spec({"type": "plate", "width_mm": 60.0, "height_mm": 40.0, "thickness_mm": 5.0, "corner_radius_mm": 0.0})
    step, stl = export(build(s), tmp_path)
    assert step.suffix == ".step" and stl.suffix == ".stl"
    assert step.read_text(errors="ignore").startswith("ISO-10303-21")
    head = stl.read_bytes()[:80]
    assert head.startswith(b"solid") or len(stl.read_bytes()) > 84  # ascii or binary STL
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_builder.py::test_export_writes_step_and_stl -v`
Expected: FAIL with `ImportError: cannot import name 'export'`.

- [ ] **Step 3: Implement**

```python
# append to s2c/builder.py
def export(solid: cq.Workplane, out_dir: Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    step, stl = out_dir / "part.step", out_dir / "part.stl"
    cq.exporters.export(solid, str(step))
    cq.exporters.export(solid, str(stl), tolerance=0.01, angularTolerance=0.1)
    return step, stl
```

- [ ] **Step 4: Run tests and open the files**

Run: `uv run pytest tests/test_builder.py -v`
Expected: PASS. Then open the STL and STEP in FreeCAD (or your CAD tool) and confirm a 60 by 40 by 5 plate.

- [ ] **Step 5: Commit**

```bash
git add s2c/builder.py tests/test_builder.py
git commit -m "Export solids to STEP and STL"
```

---

### Task 3: Flange, L-bracket, profile extrusion, features and finishes

**Files:**
- Modify: `s2c/builder.py`
- Test: `tests/test_builder.py` (append)

**Interfaces:**
- Produces: full `build()` coverage of the grammar.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_builder.py

def test_flange_volume():
    s = spec({"type": "flange", "outer_diameter_mm": 80.0, "bore_diameter_mm": 30.0, "thickness_mm": 6.0,
              "bolt_circle_diameter_mm": 60.0, "bolt_count": 4, "bolt_hole_diameter_mm": 6.0})
    expected = math.pi * (40 ** 2 - 15 ** 2) * 6 - 4 * math.pi * 3 ** 2 * 6
    assert math.isclose(volume(build(s)), expected, rel_tol=0.005)


def test_l_bracket_volume():
    s = spec({"type": "l_bracket", "leg_a_mm": 50.0, "leg_b_mm": 30.0, "width_mm": 20.0, "thickness_mm": 3.0, "angle_deg": 90.0})
    expected = 50 * 20 * 3 + (30 - 3) * 20 * 3
    assert math.isclose(volume(build(s)), expected, rel_tol=0.005)


def test_l_bracket_non_right_angle_abstains():
    s = spec({"type": "l_bracket", "leg_a_mm": 50.0, "leg_b_mm": 30.0, "width_mm": 20.0, "thickness_mm": 3.0, "angle_deg": 60.0})
    with pytest.raises(BuildError) as e:
        build(s)
    assert e.value.reason == "unsupported_angle"


def test_profile_extrusion_volume():
    s = spec({"type": "profile_extrusion", "points_mm": [(0.0, 0.0), (60.0, 0.0), (60.0, 20.0), (30.0, 40.0), (0.0, 40.0)], "depth_mm": 10.0})
    area = 60 * 40 - 0.5 * 30 * 20
    assert math.isclose(volume(build(s)), area * 10, rel_tol=0.005)


def test_through_and_blind_holes():
    plate = {"type": "plate", "width_mm": 60.0, "height_mm": 40.0, "thickness_mm": 5.0, "corner_radius_mm": 0.0}
    through = spec(plate, [{"type": "hole", "x_mm": 10.0, "y_mm": 10.0, "diameter_mm": 6.0}])
    blind = spec(plate, [{"type": "hole", "x_mm": 10.0, "y_mm": 10.0, "diameter_mm": 6.0, "depth_mm": 2.0}])
    assert math.isclose(volume(build(through)), 60 * 40 * 5 - math.pi * 9 * 5, rel_tol=0.005)
    assert math.isclose(volume(build(blind)), 60 * 40 * 5 - math.pi * 9 * 2, rel_tol=0.005)


def test_hole_on_l_bracket_leg_b():
    br = {"type": "l_bracket", "leg_a_mm": 50.0, "leg_b_mm": 30.0, "width_mm": 20.0, "thickness_mm": 3.0, "angle_deg": 90.0}
    s = spec(br, [{"type": "hole", "x_mm": 20.0, "y_mm": 10.0, "diameter_mm": 6.0, "leg": "b"}])
    base = 50 * 20 * 3 + 27 * 20 * 3
    assert math.isclose(volume(build(s)), base - math.pi * 9 * 3, rel_tol=0.005)


def test_slot_volume():
    plate = {"type": "plate", "width_mm": 60.0, "height_mm": 40.0, "thickness_mm": 5.0, "corner_radius_mm": 0.0}
    s = spec(plate, [{"type": "slot", "x_mm": 30.0, "y_mm": 20.0, "width_mm": 6.0, "length_mm": 20.0, "angle_deg": 0.0}])
    slot_area = (20 - 6) * 6 + math.pi * 9
    assert math.isclose(volume(build(s)), 60 * 40 * 5 - slot_area * 5, rel_tol=0.005)


def test_fillet_and_chamfer_reduce_volume():
    plate = {"type": "plate", "width_mm": 60.0, "height_mm": 40.0, "thickness_mm": 5.0, "corner_radius_mm": 0.0}
    v0 = volume(build(spec(plate)))
    vf = volume(build(spec(plate, finishes=[{"type": "fillet", "edges": "all_vertical", "radius_mm": 2.0}])))
    vc = volume(build(spec(plate, finishes=[{"type": "chamfer", "edges": "top", "radius_mm": 1.0}])))
    assert vf < v0 and vc < v0


def test_hole_outside_part_raises():
    plate = {"type": "plate", "width_mm": 60.0, "height_mm": 40.0, "thickness_mm": 5.0, "corner_radius_mm": 0.0}
    s = spec(plate, [{"type": "hole", "x_mm": 70.0, "y_mm": 10.0, "diameter_mm": 6.0}])
    with pytest.raises(BuildError) as e:
        build(s)
    assert e.value.reason == "feature_outside_part"
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_builder.py -v`
Expected: the new tests FAIL with `BuildError` or `KeyError`.

- [ ] **Step 3: Implement the remaining types**

Replace `_PART_BUILDERS`, `_apply_feature` and `_apply_finish` in `s2c/builder.py` with:

```python
def _flange(p: Flange) -> cq.Workplane:
    solid = cq.Workplane("XY").circle(p.outer_diameter_mm / 2).extrude(p.thickness_mm)
    solid = solid.faces(">Z").workplane().hole(p.bore_diameter_mm)
    if p.bolt_count > 0:
        solid = (solid.faces(">Z").workplane()
                 .polygon(p.bolt_count, p.bolt_circle_diameter_mm, forConstruction=True)
                 .vertices().hole(p.bolt_hole_diameter_mm))
    return solid


def _l_bracket(p: LBracket) -> cq.Workplane:
    if abs(p.angle_deg - 90.0) > 0.01:
        raise BuildError("unsupported_angle", "Only 90 degree brackets are supported in this version.")
    if p.leg_b_mm <= p.thickness_mm:
        raise BuildError("inconsistent_dimensions", "Leg b must be longer than the thickness.")
    leg_a = cq.Workplane("XY").box(p.leg_a_mm, p.width_mm, p.thickness_mm, centered=False)
    leg_b = (cq.Workplane("XY").box(p.thickness_mm, p.width_mm, p.leg_b_mm, centered=False)
             .translate((p.leg_a_mm - p.thickness_mm, 0, 0)))
    return leg_a.union(leg_b)


def _profile(p: ProfileExtrusion) -> cq.Workplane:
    pts = [tuple(pt) for pt in p.points_mm]
    if pts[0] == pts[-1]:
        pts = pts[:-1]
    try:
        return cq.Workplane("XY").polyline(pts).close().extrude(p.depth_mm)
    except Exception as e:
        raise BuildError("bad_profile", "The outline crosses itself. Retake the photo on a plain background.") from e


_PART_BUILDERS = {"plate": _plate, "spacer": _spacer, "flange": _flange, "l_bracket": _l_bracket,
                  "profile_extrusion": _profile}


# ---- features -------------------------------------------------------------

def _footprint(part) -> tuple[float, float, float]:
    """(width, height, thickness) of the face features are placed on."""
    if isinstance(part, Plate):
        return part.width_mm, part.height_mm, part.thickness_mm
    if isinstance(part, LBracket):
        return part.leg_a_mm, part.width_mm, part.thickness_mm
    if isinstance(part, ProfileExtrusion):
        xs, ys = zip(*part.points_mm)
        return max(xs) - min(xs), max(ys) - min(ys), part.depth_mm
    if isinstance(part, (Spacer, Flange)):
        d = part.outer_diameter_mm
        t = part.length_mm if isinstance(part, Spacer) else part.thickness_mm
        return d, d, t
    raise BuildError("unsupported_feature", "Features are not supported on this part type.")


def _check_inside(part, x, y, radius):
    w, h, _ = _footprint(part)
    if isinstance(part, (Spacer, Flange)):
        if (x ** 2 + y ** 2) ** 0.5 + radius > w / 2:
            raise BuildError("feature_outside_part", "A hole or slot lies outside the part. Check its position.")
        return
    if not (radius <= x <= w - radius and radius <= y <= h - radius):
        raise BuildError("feature_outside_part", "A hole or slot lies outside the part. Check its position.")


def _shape_on(wp: cq.Workplane, feature: Hole | Slot) -> cq.Workplane:
    """Hole or slot outline on an already-centred workplane."""
    if isinstance(feature, Hole):
        return wp.circle(feature.diameter_mm / 2)
    return wp.slot2D(feature.length_mm, feature.width_mm, feature.angle_deg)


def _apply_feature(solid: cq.Workplane, part, feature: Hole | Slot) -> cq.Workplane:
    radius = feature.diameter_mm / 2 if isinstance(feature, Hole) else feature.length_mm / 2
    if isinstance(part, LBracket) and feature.leg == "b":
        # Leg b occupies x in [leg_a - t, leg_a], y in [0, width], z in [0, leg_b].
        # On leg b, feature.x_mm runs up the leg (global z) and feature.y_mm runs along the width (global y).
        if not (radius <= feature.x_mm <= part.leg_b_mm - radius and radius <= feature.y_mm <= part.width_mm - radius):
            raise BuildError("feature_outside_part", "A hole on leg b lies outside the leg. Check its position.")
        wp = cq.Workplane("YZ").center(feature.y_mm, feature.x_mm)  # YZ plane: normal is +X
        shape = _shape_on(wp, feature)
        if feature.depth_mm is None:
            cutter = shape.extrude(part.thickness_mm + 2).translate((part.leg_a_mm - part.thickness_mm - 1, 0, 0))
        else:
            cutter = shape.extrude(feature.depth_mm).translate((part.leg_a_mm - feature.depth_mm, 0, 0))
        return solid.cut(cutter)
    _check_inside(part, feature.x_mm, feature.y_mm, radius)
    _, _, thickness = _footprint(part)
    shape = _shape_on(cq.Workplane("XY").center(feature.x_mm, feature.y_mm), feature)
    if feature.depth_mm is None:
        cutter = shape.extrude(thickness + 2).translate((0, 0, -1))
    else:  # blind: cut down from the top face
        cutter = shape.extrude(feature.depth_mm).translate((0, 0, thickness - feature.depth_mm))
    return solid.cut(cutter)


# ---- finishes -------------------------------------------------------------

_EDGE_SELECTORS = {"all": None, "all_vertical": "|Z", "top": ">Z", "bottom": "<Z"}


def _apply_finish(solid: cq.Workplane, finish: Fillet | Chamfer) -> cq.Workplane:
    selector = _EDGE_SELECTORS[finish.edges]
    edges = solid.edges() if selector is None else solid.edges(selector)
    try:
        if isinstance(finish, Fillet):
            return edges.fillet(finish.radius_mm)
        return edges.chamfer(finish.radius_mm)
    except Exception as e:
        kind = "fillet" if isinstance(finish, Fillet) else "chamfer"
        raise BuildError(f"{kind}_failed", f"Reduce the {kind} radius.") from e
```

If `test_hole_on_l_bracket_leg_b` fails, print `cutter.val().BoundingBox()`: it must span `x` in `[leg_a - t - 1, leg_a + 1]`, `y` around `y_mm`, `z` around `x_mm`. If `y` and `z` are swapped, the `center()` argument order on the YZ workplane is the culprit.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_builder.py -v`
Expected: all PASS. The full suite with `uv run pytest -q` must also stay green (the pipeline now picks up the real builder instead of the fake).

- [ ] **Step 5: Commit**

```bash
git add s2c/builder.py tests/test_builder.py
git commit -m "Build every grammar type with holes, slots, fillets and chamfers"
```

---

### Task 4: Six-view silhouettes

**Files:**
- Create: `s2c/views.py`
- Test: `tests/test_views.py`

**Interfaces:**
- Consumes: `normalize_mask`, `iou` from `s2c.silhouette`.
- Produces: `silhouettes(solid: cq.Workplane, px: int = 512) -> dict[str, np.ndarray]` with keys `front, back, left, right, top, bottom`, each a normalised uint8 mask.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_views.py
import cv2
import numpy as np
from s2c.builder import build
from s2c.silhouette import iou, normalize_mask
from s2c.views import silhouettes
from tests.test_builder import spec


def analytic_plate_mask(w, h, holes, px=512):
    scale = 8
    m = np.zeros((int(h * scale), int(w * scale)), np.uint8)
    m[:] = 255
    for x, y, d in holes:
        cv2.circle(m, (int(x * scale), int((h - y) * scale)), int(d * scale / 2), 0, -1)
    return normalize_mask(m, px)


def test_front_view_matches_analytic_plate_with_holes():
    holes = [(10.0, 10.0, 6.0), (50.0, 30.0, 6.0)]
    s = spec({"type": "plate", "width_mm": 60.0, "height_mm": 40.0, "thickness_mm": 5.0, "corner_radius_mm": 0.0},
             [{"type": "hole", "x_mm": x, "y_mm": y, "diameter_mm": d} for x, y, d in holes])
    views = silhouettes(build(s))
    assert set(views) == {"front", "back", "left", "right", "top", "bottom"}
    assert iou(views["front"], analytic_plate_mask(60, 40, holes)) > 0.98


def test_top_view_of_plate_is_a_thin_bar():
    s = spec({"type": "plate", "width_mm": 60.0, "height_mm": 40.0, "thickness_mm": 5.0, "corner_radius_mm": 0.0})
    top = silhouettes(build(s))["top"]
    assert 0.06 < (top == 255).mean() < 0.11  # 60:5 aspect padded to square


def test_flange_front_view_shows_bore_and_bolt_holes():
    s = spec({"type": "flange", "outer_diameter_mm": 80.0, "bore_diameter_mm": 30.0, "thickness_mm": 6.0,
              "bolt_circle_diameter_mm": 60.0, "bolt_count": 4, "bolt_hole_diameter_mm": 6.0})
    front = silhouettes(build(s))["front"]
    filled = (front == 255).mean()
    expected = (np.pi * (40 ** 2 - 15 ** 2) - 4 * np.pi * 9) / 80 ** 2
    assert abs(filled - expected) < 0.02
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_views.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 's2c.views'`.

- [ ] **Step 3: Implement**

```python
# s2c/views.py
"""Solid -> six orthographic silhouettes. Rasterises the tessellation along each axis."""
from __future__ import annotations

import cv2
import numpy as np

from s2c.silhouette import normalize_mask

# view -> (horizontal axis, vertical axis, mirror horizontal)
_VIEWS = {
    "front": (0, 1, False), "back": (0, 1, True),
    "top": (0, 2, False), "bottom": (0, 2, True),
    "right": (1, 2, False), "left": (1, 2, True),
}


def silhouettes(solid, px: int = 512) -> dict[str, np.ndarray]:
    verts, tris = solid.val().tessellate(0.05, 0.2)
    pts = np.array([[v.x, v.y, v.z] for v in verts], dtype=np.float64)
    faces = np.array(tris, dtype=np.int64)
    out: dict[str, np.ndarray] = {}
    margin = 10
    for name, (i, j, mirror) in _VIEWS.items():
        p2 = pts[:, [i, j]].copy()
        if mirror:
            p2[:, 0] = -p2[:, 0]
        lo = p2.min(axis=0)
        span = float((p2.max(axis=0) - lo).max()) or 1.0
        scale = (px - 2 * margin) / span
        img = np.zeros((px, px), np.uint8)
        polys = (p2[faces] - lo) * scale + margin  # (n, 3, 2)
        polys[:, :, 1] = px - polys[:, :, 1]
        cv2.fillPoly(img, np.round(polys).astype(np.int32), 255)
        out[name] = normalize_mask(img, px)
    return out
```

Why this shows holes: the wall triangles of a hole are parallel to the view axis, so they project to zero-area slivers, and the top and bottom faces are tessellated with the hole cut out. If a hole appears filled, the tessellation tolerance is too coarse; lower the first argument of `tessellate`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_views.py tests/test_builder.py -v`
Expected: PASS. If the front-view IoU is between 0.95 and 0.98, the cause is usually the 1 px outline `fillPoly` draws for degenerate triangles; erase them with `cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))` before `normalize_mask`.

- [ ] **Step 5: Commit**

```bash
git add s2c/views.py tests/test_views.py
git commit -m "Render six orthographic silhouettes from a solid"
```

---

### Task 5: Golden set, ten sketches and five photos

**Files:**
- Create: `tests/golden/<name>/expected.json` and `image.jpg` for each case, `tests/test_golden_files.py`, `scripts/render_golden.py`

**Interfaces:**
- Produces: folders the integrator's `tests/test_golden.py` and the numbers owner's accuracy tests consume. `expected.json` schema is in `tests/golden/README.md`.

- [ ] **Step 1: Write the well-formedness test**

```python
# tests/test_golden_files.py
import json
from pathlib import Path

import pytest

from s2c.partspec.models import PartSpec

GOLDEN = Path(__file__).parent / "golden"
CASES = sorted(p for p in GOLDEN.iterdir() if p.is_dir()) if GOLDEN.exists() else []


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_golden_folder_is_well_formed(case: Path):
    expected = json.loads((case / "expected.json").read_text())
    assert expected["input_kind"] in ("sketch", "photo", "drawing")
    images = [p for p in case.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    assert len(images) == 1, "exactly one image per case"
    # the expected part must itself be a buildable PartSpec
    prov = {f"part.{k}": "user_written" for k, v in expected["part"].items()
            if k != "type" and (isinstance(v, (int, float)) and not isinstance(v, bool) or k == "points_mm")}
    for i, f in enumerate(expected.get("features", [])):
        prov.update({f"features[{i}].{k}": "user_written" for k, v in f.items()
                     if k != "type" and isinstance(v, (int, float)) and not isinstance(v, bool)})
    PartSpec.model_validate({"source_input": expected["input_kind"], "part": expected["part"],
                             "features": expected.get("features", []), "provenance": prov, "confidence": 1.0})
```

- [ ] **Step 2: Write the render script**

```python
# scripts/render_golden.py
"""Builds every golden expected part to tmp/golden/<name>.stl so you can eyeball them."""
import json
from pathlib import Path

from s2c.builder import build, export
from tests.test_golden_files import CASES
from tests.test_builder import spec

out = Path("tmp/golden")
for case in CASES:
    e = json.loads((case / "expected.json").read_text())
    solid = build(spec(e["part"], e.get("features", []), source=e["input_kind"]))
    step, stl = export(solid, out / case.name)
    print(case.name, stl)
```

- [ ] **Step 3: Model and sketch the ten parts**

Name each folder `<type>_<key dims>`. Required mix:

| Folder | Type | Dimensions (mm) | Features |
| --- | --- | --- | --- |
| `plate_60x40x5_2holes` | plate | 60 x 40 x 5 | two 6 mm holes at (10,10) and (50,30) |
| `plate_80x50x4_r8` | plate | 80 x 50 x 4, corner radius 8 | four 5 mm holes 10 mm in from each corner |
| `plate_100x30x6_slot` | plate | 100 x 30 x 6 | one slot 30 x 8 at (50,15) |
| `plate_40x40x3` | plate | 40 x 40 x 3 | one 10 mm hole at centre |
| `spacer_20x8x30` | spacer | OD 20, ID 8, length 30 | none |
| `spacer_12x0x15` | spacer | OD 12, solid, length 15 | none |
| `flange_80_30_60_4` | flange | OD 80, bore 30, 6 thick, bolt circle 60, 4 x 6 mm | none |
| `l_bracket_50x30x20x3` | l_bracket | legs 50 and 30, width 20, thickness 3 | one 6 mm hole on each leg |
| `l_bracket_40x40x25x4` | l_bracket | legs 40 and 40, width 25, thickness 4 | none |
| `profile_60x40_notch` | profile_extrusion | points (0,0) (60,0) (60,20) (30,40) (0,40), depth 10 | none, photo only |

For each: write `expected.json`, run `uv run python scripts/render_golden.py` and check the STL looks right. Then draw the part by hand on plain white paper with a dark pen: one front view, dimensions written next to arrows in mm, a small side view with the thickness written, diameter values prefixed with the ⌀ symbol or "D". Photograph it top-down in good light with the phone. Save as `image.jpg` in the folder. Add `"user_values"` to `expected.json` for anything a sketch cannot carry (none for sketches; `{"thickness": ...}` for photos).

- [ ] **Step 4: Photograph five real parts**

Pick five of the ten that you can find or print: at minimum one plate with holes, one spacer, one flange or washer, one bracket, one free-form outline. Photograph each top-down next to a coin from the table in `docs/roles/numbers-owner.md`, plain background, phone parallel to the table. Folder names get the suffix `_photo`, `input_kind` is `photo`, and `user_values` carries the thickness. Measure the real part with calipers and put the true values in `expected.json`.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_golden_files.py -v && GOLDEN_FAKE=1 uv run pytest tests/test_golden.py -v`
Expected: well-formedness PASS for every folder. The fake golden run will fail on most cases because the fakes always return the 60 x 40 plate; that is expected until the real OCR lands.

- [ ] **Step 6: Commit**

```bash
git add tests/golden tests/test_golden_files.py scripts/render_golden.py
git commit -m "Add golden set of ten sketches and five photos with ground truth"
```

---

### Task 6: Viewer polish (with the integrator)

**Files:**
- Modify: `web/src/components/Viewer.tsx`

- [ ] **Step 1: Add a ground grid and millimetre scale**

In `Viewer.tsx`, after the lights, add:

```ts
scene.add(new THREE.GridHelper(200, 20, 0x444444, 0x222222));
```

and after loading the geometry, position the mesh so its lowest point sits on the grid: `mesh.position.y = -box.min.z` after rotating the mesh with `mesh.rotation.x = -Math.PI / 2` so the extrusion axis (Z in CAD) points up (Y in three).

- [ ] **Step 2: Keep the camera between rebuilds**

Store `camera.position` and `controls.target` in a `useRef` and reapply them when a new STL loads, so moving a slider does not reset the view.

- [ ] **Step 3: Check on the phone, then commit**

```bash
git add web/src/components/Viewer.tsx
git commit -m "Keep camera between rebuilds and add a ground grid to the viewer"
```

---

## Self-review against the spec

- Grammar (spec 3): every type and feature has a builder and a volume test in Tasks 1 and 3.
- Coordinate convention: tested in `test_plate_origin_is_bottom_left`.
- BuildError for unbuildable specs (Rule 4): corner radius, angle, hole outside part, self-crossing profile, fillet failure.
- Views and IoU above 0.98 (spec 4.7): Task 4.
- Golden set of ten sketches and five photos (spec 4.7): Task 5.
- Type consistency: `build(spec) -> Workplane`, `export(solid, out_dir) -> (step, stl)`, `silhouettes(solid, px)` match the fakes in the integrator plan Task 7 and the `Pipeline` fields in Task 11.
- Known simplification: L-bracket angle other than 90 degrees abstains. Slots on leg b are placed with the same transform as holes.
