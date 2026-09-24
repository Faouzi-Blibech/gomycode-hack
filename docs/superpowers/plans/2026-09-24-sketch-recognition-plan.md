# Hand-Sketch Recognition Implementation Plan (must-have scope)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn one photo of a hand-drawn multi-view sheet into a validated `SketchReading` JSON: views, shapes with line types, every dimension linked to what it measures, exact sizes from a cross-view solve, predictions with badges for what is missing or unclear, and holes inferred from hidden lines.

**Architecture:** A pipeline of small deterministic stages in `s2c/sketch/` (capture, text, views, vectorize, classify, link, solve, features), with handwriting reading behind a pluggable `Reader` protocol (two readers must agree). Sizes come from one least-squares solve per global axis over "reference positions"; a model never produces a coordinate.

**Tech Stack:** Python 3.11-3.12, uv, NumPy, SciPy, OpenCV (headless), scikit-image, Pydantic 2, pytest, ruff. Optional: PaddleOCR 3 (PP-OCRv5), transformers (TrOCR), the team's OpenAI-compatible `VLMClient`.

**Spec:** `docs/superpowers/specs/2026-09-24-sketch-recognition-design.md`. This plan covers the spec's "Must" row (section 15). The "Should" row (angles and chamfers, prediction ranks 3 and 4, meaning check, `resolve()` with edits, reader head-to-head, Brev script, correction records) gets a second plan once this one is green.

## Global Constraints

- Python `>=3.11,<3.13` (from `pyproject.toml`); run everything through `uv run`.
- Ruff line length 100; `uv run ruff check .` and `uv run pytest -q` must pass after every task (CI runs both).
- Every Pydantic model in `s2c/sketch/models.py` uses `extra="forbid"`.
- Units are millimetres. Badges are exactly: `written`, `edited`, `derived`, `uncertain`, `predicted`, `conflict`.
- A model never produces a coordinate or a size; readers only transcribe text.
- No model name or provider is hard-coded in logic; readers are chosen by `SKETCH_READERS` (default `paddle,vlm`), the vision model by `VLM_BASE_URL`, `VLM_MODEL`, `VLM_API_KEY`. Local model ids have env overrides with documented defaults.
- Tolerances (spec): reference merge 1.5 percent of the view extent, at least 4 px; conflict residual 0.5 percent of the value, at least 0.05 mm; reader confidence below 0.6 is `uncertain`; decimal trap at a scale ratio above 3; outline gap closing under 3 percent of the view extent.
- Work on the rectified sheet at a longest side of 1600 px; report `px` in original photo pixels.
- Never drop a value that was read: unlinked values become `unplaced` dimensions with an issue.
- Commit messages are plain, in the team's voice. **No `Co-Authored-By`, no "Generated with", no AI attribution** (repo `CLAUDE.md` overrides any default). **Never push**; commits stay local on `sketch/recognition-design`.
- Tests contain no photos of people and no personal data.

## Review Focus

The five inputs most likely to bite a real user that the spec implies but does not spell out. Each has a test in the owning task.

1. **Sheet fills the frame edge to edge** (no table visible around the paper): the capture must fall back to the full frame, not abstain. Test in Task 3.
2. **A shadow across half the sheet** (phone held over the paper): binarisation must keep the lines in the dark half and not turn the shadow into ink. Test in Task 3.
3. **A small hole circle that looks like the letter O or digit 0** is found as a text box: it must stay in the geometry ink, because only boxes that parse as a dimension or a label are erased. Test in Task 5.
4. **A vertical dimension written sideways** (text rotated 90 degrees along a vertical dimension line): tall boxes are also read rotated by plus and minus 90 degrees, and the parseable reading wins. Test in Task 5.
5. **Two views drawn close together** so their dimension lines almost touch: they must come out as two views, not one merged blob. Test in Task 6.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `s2c/sketch/__init__.py` | exports `read_sketch` |
| `s2c/sketch/models.py` | the `SketchReading` contract (spec 10.1) |
| `s2c/sketch/grammar.py` | pure text parsing: dimension grammar and view-label vocabulary |
| `s2c/sketch/capture.py` | stage 1: find the sheet, rectify, remove shadows, binarise, quality gates |
| `s2c/sketch/readers.py` | `Crop`, `ReaderResult`, `Reader`; `VlmReader`, `PaddleReader`, `TrocrReader`; `readers_from_env` |
| `s2c/sketch/text.py` | stage 2: text boxes, two-reader agreement, roles |
| `s2c/sketch/views.py` | stage 3: split the ink into named views |
| `s2c/sketch/vectorize.py` | stage 4: skeleton tracing and primitive fitting |
| `s2c/sketch/classify.py` | stage 5: arrowheads, hidden and centre chains, dimension, extension and leader lines |
| `s2c/sketch/link.py` | stage 6: reference positions and value-to-geometry links |
| `s2c/sketch/solve.py` | stages 7 and 8 (ranks 1, 2, 5): cross-view solve, badges, conflicts, decimal trap, proposals for uncertain values |
| `s2c/sketch/features.py` | stage 9: holes from circles and hidden pairs |
| `s2c/sketch/debug.py` | debug overlays |
| `s2c/sketch/pipeline.py` | `read_sketch(image_bytes, readers=None) -> SketchReading` |
| `tests/sketch/synth.py` | draws synthetic sheets for tests |
| `tests/sketch/test_*.py` | one test file per stage |
| `tests/golden_sketch/README.md` | how to add a real golden sheet |
| `scripts/sketch_accuracy.py` | accuracy table over the golden sheets |

The spec's section 13 lists `text.py` for detection and grammar together; the grammar is split into `grammar.py` because it is pure and heavily tested. `predict.py` (ranks 3 and 4) and `corrections.py` arrive with the second plan.

---

### Task 1: Package, dependencies and the `SketchReading` contract

**Files:**
- Modify: `pyproject.toml`
- Create: `s2c/sketch/__init__.py`, `s2c/sketch/models.py`, `tests/sketch/__init__.py`
- Test: `tests/sketch/test_models.py`

**Interfaces:**
- Produces: `Badge`, `Evidence`, `ViewName` type aliases; models `View`, `Entity`, `Reading`, `Dimension`, `Feature`, `Size`, `Issue`, `SketchAbstain`, `SketchReading`, exactly as spec 10.1. `_Strict` base with `extra="forbid"`.

- [ ] **Step 1: Install uv and Python 3.11 (once per machine)**

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv python install 3.11
```

Then from the repo root: `uv sync`. Expected: a `.venv` is created and `uv run pytest -q` passes on the existing suite.

- [ ] **Step 2: Add the dependencies**

In `pyproject.toml`, add to `dependencies`:

```toml
  "scikit-image>=0.24",
  "scipy>=1.13",
```

and add after the `[dependency-groups]` block:

```toml
[project.optional-dependencies]
ocr = ["paddleocr>=3.0", "paddlepaddle>=3.0"]
trocr = ["transformers>=4.44", "torch>=2.3"]
```

Run: `uv sync`. Expected: success. The optional groups are installed only with `uv sync --extra ocr` or `--extra trocr`.

- [ ] **Step 3: Write the failing tests**

```python
# tests/sketch/test_models.py
import pytest
from pydantic import ValidationError

from s2c.sketch.models import Dimension, Issue, Reading, SketchAbstain, SketchReading, Size, View


def minimal(**over):
    data = dict(
        image_size_px=(1200, 1600),
        views=[View(name="front", label_text="FRONT", bbox_px=(0, 0, 10, 10), size_mm=(40.0, 20.0))],
        entities=[], dimensions=[], features=[],
        envelope={"x": Size(value=40.0, badge="written", evidence="reader")},
        issues=[], timings_ms={"total": 1.0},
    )
    data.update(over)
    return SketchReading(**data)


def test_round_trip_json():
    r = minimal()
    again = SketchReading.model_validate_json(r.model_dump_json())
    assert again == r
    assert again.version == "sketch-1" and again.units == "mm"


def test_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        View(name="front", label_text=None, bbox_px=(0, 0, 1, 1), size_mm=(1, 1), width_mm=3)


def test_badge_is_a_closed_set():
    with pytest.raises(ValidationError):
        Size(value=1.0, badge="guessed", evidence="reader")


def test_dimension_keeps_both_readings():
    d = Dimension(id="d1", view="front", kind="linear", value=1.5, text_raw="1.50",
                  readings=[Reading(reader="paddle", text="1.50", confidence=0.9),
                            Reading(reader="vlm", text="1.5O", confidence=0.7)],
                  measures=["r1", "r2"], axis="a", badge="written", evidence="reader",
                  bbox_px=(1, 2, 3, 4))
    assert len(d.readings) == 2 and d.candidates == []


def test_abstain_and_issue():
    a = SketchAbstain(stage="capture", reason="sheet_not_found", remedy="Retake.")
    r = minimal(abstain=a, issues=[Issue(severity="red", kind="unit", message="mm?", targets=[])])
    assert r.abstain.reason == "sheet_not_found"
```

- [ ] **Step 4: Run to confirm failure**

Run: `uv run pytest tests/sketch/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 's2c.sketch'`.

- [ ] **Step 5: Implement**

```python
# s2c/sketch/__init__.py
"""Hand-sketch recognition: one photo of a multi-view sheet -> SketchReading JSON."""
```

```python
# tests/sketch/__init__.py
```

```python
# s2c/sketch/models.py
"""The SketchReading contract (spec section 10.1). The team builds against this; change it only by
agreement."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Badge = Literal["written", "edited", "derived", "uncertain", "predicted", "conflict"]
Evidence = Literal["reader", "user", "arithmetic", "cross_view", "symmetry", "pattern",
                   "standard", "proportion", "geometry"]
ViewName = Literal["front", "top", "right", "left", "bottom", "back"]
Unit = float
Box = tuple[float, float, float, float]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class View(_Strict):
    name: ViewName
    label_text: str | None
    bbox_px: Box
    size_mm: tuple[float, float]


class Entity(_Strict):
    id: str
    view: ViewName
    type: Literal["line", "arc", "circle", "curve"]
    line_type: Literal["visible", "hidden", "centre"]
    mm: dict
    px: dict
    confidence: float = Field(ge=0, le=1)


class Reading(_Strict):
    reader: str
    text: str
    confidence: float = Field(ge=0, le=1)


class Dimension(_Strict):
    id: str
    view: ViewName | None
    kind: Literal["linear", "diameter", "radius", "angle", "chamfer", "thread"]
    value: float | None
    text_raw: str
    readings: list[Reading]
    candidates: list[float] = []
    implied: float | None = None
    tolerance: str | None = None
    measures: list[str]
    axis: Literal["a", "b"] | None = None
    badge: Badge
    evidence: Evidence
    bbox_px: Box


class Feature(_Strict):
    id: str
    type: Literal["hole", "slot", "other"]
    axis: Literal["x", "y", "z"]
    position_mm: tuple[float, float, float]
    diameter: float | None = None
    width: float | None = None
    length: float | None = None
    through: bool | None = None
    depth: float | None = None
    evidence: list[str]
    badge: Badge


class Size(_Strict):
    value: float
    badge: Badge
    evidence: Evidence


class Issue(_Strict):
    severity: Literal["red", "amber"]
    kind: Literal["conflict", "uncertain", "predicted", "unplaced", "decimal", "unit", "label",
                  "meaning_check", "not_in_3d", "gap_closed"]
    message: str
    targets: list[str]


class SketchAbstain(_Strict):
    stage: Literal["capture", "text", "views", "vectorize", "link", "solve"]
    reason: str
    remedy: str


class SketchReading(_Strict):
    version: Literal["sketch-1"] = "sketch-1"
    units: Literal["mm"] = "mm"
    image_size_px: tuple[int, int]
    views: list[View]
    entities: list[Entity]
    dimensions: list[Dimension]
    features: list[Feature]
    envelope: dict[Literal["x", "y", "z"], Size]
    issues: list[Issue]
    abstain: SketchAbstain | None = None
    timings_ms: dict[str, float]
```

- [ ] **Step 6: Run tests and lint**

Run: `uv run pytest tests/sketch/test_models.py -v` then `uv run ruff check .`
Expected: 5 passed; no lint errors.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock s2c/sketch tests/sketch
git commit -m "Add the sketch package and the SketchReading contract"
```

---

### Task 2: Dimension grammar and view labels

**Files:**
- Create: `s2c/sketch/grammar.py`
- Test: `tests/sketch/test_grammar.py`

**Interfaces:**
- Produces: `Parsed` frozen dataclass `(kind, value, count=1, tolerance=None, through=False, depth=None, angle=None)`; `parse_text(raw: str) -> Parsed | None`; `match_label(raw: str) -> ViewName | None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/sketch/test_grammar.py
import pytest

from s2c.sketch.grammar import match_label, parse_text


@pytest.mark.parametrize("raw, kind, value", [
    ("60", "linear", 60.0),
    ("4.00", "linear", 4.0),
    (".50", "linear", 0.5),
    ("12,5", "linear", 12.5),
    ("1.5O", "linear", 1.5),        # O read in a digit position is 0
    ("5O", "linear", 50.0),
    ("Ø12", "diameter", 12.0),
    ("⌀.50", "diameter", 0.5),
    ("∅6", "diameter", 6.0),
    ("o6", "diameter", 6.0),        # a handwritten Ø often reads as o
    ("D 8", "diameter", 8.0),
    ("R5", "radius", 5.0),
    ("r 2.5", "radius", 2.5),
    ("45°", "angle", 45.0),
    ("C2", "chamfer", 2.0),
    ("M6", "thread", 6.0),
    ("40 mm", "linear", 40.0),
])
def test_values_and_kinds(raw, kind, value):
    p = parse_text(raw)
    assert p is not None, raw
    assert p.kind == kind and p.value == pytest.approx(value)


def test_count_prefix():
    p = parse_text("2xØ6")
    assert p.kind == "diameter" and p.value == 6.0 and p.count == 2
    assert parse_text("4 × Ø 5").count == 4


def test_chamfer_size_by_angle():
    p = parse_text("2x45°")
    assert p.kind == "chamfer" and p.value == 2.0 and p.angle == 45.0


def test_tolerance_is_kept_as_text():
    p = parse_text("40±0.1")
    assert p.value == 40.0 and p.tolerance == "±0.1"
    assert parse_text("25 +0.2/-0.1").tolerance == "+0.2/-0.1"


def test_hole_callouts():
    assert parse_text("Ø6 THRU").through is True
    p = parse_text("Ø6 ↧10")
    assert p.depth == 10.0 and p.through is False
    assert parse_text("Ø6 DEEP 12").depth == 12.0


@pytest.mark.parametrize("raw", ["", "hello", "FRONT", "0", "-5", "..", "Ø", "12.5.3", "99999"])
def test_rejects_non_dimensions(raw):
    assert parse_text(raw) is None


@pytest.mark.parametrize("raw, name", [
    ("TOP", "top"), ("Top view", "top"), ("DESSUS", "top"), ("vue de dessus", "top"),
    ("FRONT", "front"), ("FACE", "front"), ("Vue de face", "front"),
    ("SIDE", "right"), ("RIGHT", "right"), ("CÔTÉ", "right"), ("droite", "right"),
    ("PROFIL", "right"), ("LEFT", "left"), ("gauche", "left"), ("côté gauche", "left"),
    ("BOTTOM", "bottom"), ("DESSOUS", "bottom"), ("BACK", "back"), ("ARRIÈRE", "back"),
])
def test_labels(raw, name):
    assert match_label(raw) == name


def test_label_rejects_numbers_and_noise():
    assert match_label("4.00") is None and match_label("stop") is None
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/sketch/test_grammar.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# s2c/sketch/grammar.py
"""Pure parsing of what a reader transcribed: dimension grammar (spec 5.3) and view labels."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from s2c.sketch.models import ViewName

Kind = Literal["linear", "diameter", "radius", "angle", "chamfer", "thread"]
MAX_VALUE = 5000.0


@dataclass(frozen=True)
class Parsed:
    kind: Kind
    value: float
    count: int = 1
    tolerance: str | None = None
    through: bool = False
    depth: float | None = None
    angle: float | None = None


_NUM = r"(?:\d+(?:\.\d+)?|\.\d+)"
_TOL = rf"(?P<tol>±{_NUM}|\+{_NUM}/-{_NUM})"
_MAIN = re.compile(rf"^(?P<prefix>Ø|D|R|C|M)?(?P<num>{_NUM})(?P<deg>°)?(?:MM)?{_TOL}?$")
_CHAMFER = re.compile(rf"^(?P<size>{_NUM})X(?P<ang>{_NUM})°$")
_COUNT = re.compile(r"^(?P<n>\d+)X(?=[ØRCDM])")
_DEPTH = re.compile(rf"(?:↧|DP|DEEP)(?P<d>{_NUM})$")
_DIGIT_FIXES = str.maketrans({"O": "0", "o": "0", "l": "1", "L": "1", "I": "1", "|": "1"})


def _normalise(raw: str) -> str:
    t = raw.strip()
    for sign in "⌀∅øΦφ":
        t = t.replace(sign, "Ø")
    t = t.replace("×", "x").replace(",", ".").replace(" ", "")
    if len(t) > 1 and t[0] in "oO" and (t[1].isdigit() or t[1] == "."):
        t = "Ø" + t[1:]
    return t


def _fix_digits(t: str) -> str:
    """Map look-alike letters to digits, but only after the first character (the prefix slot)."""
    if not t:
        return t
    head, tail = t[0], t[1:]
    if head in "OolLI|":
        head = head.translate(_DIGIT_FIXES)
    return head + tail.translate(_DIGIT_FIXES)


def parse_text(raw: str) -> Parsed | None:
    t = _normalise(raw)
    if not t:
        return None
    upper = t.upper()
    through = upper.endswith("THRU")
    if through:
        upper = upper[: -len("THRU")]
    upper = upper.removesuffix("TYP")
    depth = None
    m = _DEPTH.search(upper)
    if m:
        depth = float(m.group("d"))
        upper = upper[: m.start()]
    m = _CHAMFER.match(_fix_digits(upper))
    if m:
        return Parsed("chamfer", float(m.group("size")), angle=float(m.group("ang")))
    count = 1
    m = _COUNT.match(upper)
    if m:
        count, upper = int(m.group("n")), upper[m.end():]
    m = _MAIN.match(_fix_digits(upper))
    if not m:
        return None
    value = float(m.group("num"))
    prefix, deg = m.group("prefix"), m.group("deg")
    kind: Kind = ("angle" if deg else {"Ø": "diameter", "D": "diameter", "R": "radius",
                                       "C": "chamfer", "M": "thread"}.get(prefix or "", "linear"))
    limit = 360.0 if kind == "angle" else MAX_VALUE
    if not 0 < value < limit:
        return None
    return Parsed(kind, value, count=count, tolerance=m.group("tol"), through=through, depth=depth)


# order matters: the more specific words first ("COTE GAUCHE" is left, not right)
_LABELS: list[tuple[str, ViewName]] = [
    ("GAUCHE", "left"), ("LEFT", "left"),
    ("DESSOUS", "bottom"), ("BOTTOM", "bottom"),
    ("ARRIERE", "back"), ("BACK", "back"),
    ("DESSUS", "top"), ("TOP", "top"),
    ("FACE", "front"), ("FRONT", "front"),
    ("DROITE", "right"), ("RIGHT", "right"), ("SIDE", "right"), ("COTE", "right"),
    ("PROFIL", "right"),
]


def match_label(raw: str) -> ViewName | None:
    t = unicodedata.normalize("NFD", raw.upper())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    words = set(re.findall(r"[A-Z]+", t))
    for word, name in _LABELS:
        if word in words:
            return name
    return None
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/sketch/test_grammar.py -v` then `uv run ruff check .`
Expected: all passed. If `"1.5O"` fails, check `_fix_digits` runs before `_MAIN.match`; if `"C2"` parses as linear, check `C` is kept as the prefix (not translated by `_fix_digits`, which only maps `O o l I |`).

- [ ] **Step 5: Commit**

```bash
git add s2c/sketch/grammar.py tests/sketch/test_grammar.py
git commit -m "Parse dimension text and view labels"
```

---

### Task 3: Capture: find the sheet, rectify, flatten shadows, binarise

**Files:**
- Create: `s2c/sketch/capture.py`
- Test: `tests/sketch/test_capture.py`

**Interfaces:**
- Produces: `LONG_SIDE = 1600`; dataclass `Captured(sheet: np.ndarray BGR, gray: np.ndarray uint8, ink: np.ndarray uint8 with 255 = ink, to_original: np.ndarray 3x3 float, stroke_px: float)`; `capture(image_bgr: np.ndarray) -> Captured | SketchAbstain`; `sheet_to_original(pts: np.ndarray, H: np.ndarray) -> np.ndarray` (N x 2 points).

- [ ] **Step 1: Write the failing tests**

```python
# tests/sketch/test_capture.py
import cv2
import numpy as np

from s2c.sketch.capture import LONG_SIDE, Captured, capture, sheet_to_original
from s2c.sketch.models import SketchAbstain


def page(h=1130, w=800):
    img = np.full((h, w, 3), 245, np.uint8)
    cv2.rectangle(img, (200, 300), (600, 700), (20, 20, 20), 4)
    cv2.line(img, (100, 900), (700, 900), (20, 20, 20), 4)
    return img


def on_table(pg, corners):
    h, w = pg.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, np.float32(corners))
    canvas = cv2.warpPerspective(pg, M, (1600, 1200), borderValue=(70, 70, 70))
    return canvas


def test_rectifies_a_tilted_sheet():
    tl = (300, 150)
    img = on_table(page(), [tl, (1150, 190), (1120, 1150), (270, 1100)])
    out = capture(img)
    assert isinstance(out, Captured)
    assert max(out.sheet.shape[:2]) == LONG_SIDE
    frac = (out.ink > 0).mean()
    assert 0.002 < frac < 0.1
    back = sheet_to_original(np.float32([[0, 0]]), out.to_original)[0]
    assert abs(back[0] - tl[0]) < 8 and abs(back[1] - tl[1]) < 8
    assert 2 <= out.stroke_px <= 12


def test_full_frame_sheet_falls_back_to_whole_image():
    img = cv2.resize(page(), (1131, 1600))
    out = capture(img)
    assert isinstance(out, Captured)
    back = sheet_to_original(np.float32([[500, 700]]), out.to_original)[0]
    assert np.allclose(back, [500 * 1131 / out.sheet.shape[1], 700 * 1600 / out.sheet.shape[0]], atol=3)


def test_shadow_keeps_lines_and_ignores_the_shadow():
    img = cv2.resize(page(), (1131, 1600)).astype(np.float32)
    gradient = np.linspace(0.45, 1.0, img.shape[1])[None, :, None]
    img = (img * gradient).astype(np.uint8)
    out = capture(img)
    assert isinstance(out, Captured)
    ink = out.ink > 0
    sx = out.sheet.shape[1] / 1131
    # the rectangle's left side at x=200*1131/800 sits in the dark half
    x = int(200 * 1131 / 800 * sx)
    assert ink[int(500 * 1600 / 1130 * sx), x - 6: x + 6].any()
    blank = ink[int(150 * sx): int(350 * sx), int(40 * sx): int(240 * sx)]
    assert blank.mean() < 0.01


def test_dark_photo_abstains():
    out = capture(np.full((1200, 1600, 3), 25, np.uint8))
    assert isinstance(out, SketchAbstain) and out.reason == "image_quality"


def test_no_sheet_abstains():
    img = np.full((1200, 1600, 3), 110, np.uint8)
    cv2.circle(img, (800, 600), 200, (40, 40, 40), -1)
    out = capture(img)
    assert isinstance(out, SketchAbstain) and out.reason == "sheet_not_found"


def test_blurry_photo_abstains():
    img = cv2.GaussianBlur(cv2.resize(page(), (1131, 1600)), (61, 61), 20)
    out = capture(img)
    assert isinstance(out, SketchAbstain) and out.reason == "image_quality"


def test_large_photo_is_resized():
    img = cv2.resize(page(), (2830, 4000))
    out = capture(img)
    assert isinstance(out, Captured) and max(out.sheet.shape[:2]) == LONG_SIDE
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/sketch/test_capture.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# s2c/sketch/capture.py
"""Stage 1: find the sheet, rectify it, flatten shadows, binarise. Pure OpenCV and scikit-image."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.filters import threshold_sauvola
from skimage.morphology import skeletonize

from s2c.sketch.models import SketchAbstain

LONG_SIDE = 1600
MIN_SHEET_AREA = 0.2      # of the photo
FULL_FRAME_AREA = 0.95    # a "sheet" this big is the whole photo
BRIGHT_PAPER = 150
TOO_DARK = 50
BLUR_VAR = 8.0            # variance of the Laplacian; set on synthetic tests, re-check on golden photos

RETAKE_FRAME = "Put the whole sheet in the frame on a darker surface and retake."
RETAKE_LIGHT = "Retake in good light and hold the phone steady."


@dataclass
class Captured:
    sheet: np.ndarray
    gray: np.ndarray
    ink: np.ndarray
    to_original: np.ndarray
    stroke_px: float


def sheet_to_original(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def _order(pts: np.ndarray) -> np.ndarray:
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.float32([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])


def _find_sheet(gray: np.ndarray) -> np.ndarray | None:
    """Corners (tl, tr, br, bl) of the paper in `gray` pixels, or None."""
    f = 800 / max(gray.shape)
    small = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c) / th.size
    if area < MIN_SHEET_AREA or area > FULL_FRAME_AREA:
        return None
    approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True).reshape(-1, 2)
    if len(approx) != 4:
        rect = cv2.boxPoints(cv2.minAreaRect(c))
        if cv2.contourArea(c) / max(cv2.contourArea(rect), 1) < 0.85:
            return None
        approx = rect
    if not _edges_are_sharp(blur, approx.astype(np.float32)):
        return None  # a shadow boundary, not a paper edge
    return _order(approx.astype(np.float32) / f)


def _edges_are_sharp(img: np.ndarray, quad: np.ndarray, min_step: float = 30.0) -> bool:
    """A paper edge is a brightness step; a shadow boundary is a slow ramp. Sides lying on the
    photo border are skipped."""
    h, w = img.shape
    centre = quad.mean(0)
    for a, b in zip(quad, np.roll(quad, -1, 0)):
        if all(min(p[0], p[1], w - 1 - p[0], h - 1 - p[1]) < 3 for p in (a, b)):
            continue
        d = (b - a) / (np.linalg.norm(b - a) + 1e-9)
        n = np.array([-d[1], d[0]])
        if np.dot(n, (a + b) / 2 - centre) < 0:
            n = -n
        steps = []
        for t in np.linspace(0.1, 0.9, 15):
            p = a + t * (b - a)
            q_in, q_out = p - 4 * n, p + 4 * n
            if all(0 <= q[0] < w and 0 <= q[1] < h for q in (q_in, q_out)):
                steps.append(float(img[int(q_in[1]), int(q_in[0])])
                             - float(img[int(q_out[1]), int(q_out[0])]))
        if steps and np.median(steps) < min_step:
            return False
    return True


def _rectify(image: np.ndarray, corners: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Rectified sheet at LONG_SIDE and the homography from sheet pixels back to the photo."""
    h, w = image.shape[:2]
    if corners is None:
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    tl, tr, br, bl = corners
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    k = LONG_SIDE / max(width, height)
    W, Hh = round(width * k), round(height * k)
    dst = np.float32([[0, 0], [W, 0], [W, Hh], [0, Hh]])
    M = cv2.getPerspectiveTransform(corners, dst)
    sheet = cv2.warpPerspective(image, M, (W, Hh), flags=cv2.INTER_AREA)
    return sheet, np.linalg.inv(M)


def _flatten(gray: np.ndarray) -> np.ndarray:
    """Divide by the estimated paper brightness: shadows and uneven light disappear."""
    bg = cv2.medianBlur(cv2.dilate(gray, np.ones((7, 7), np.uint8)), 41)
    return cv2.divide(gray, bg, scale=255)


def _binarise(flat: np.ndarray) -> np.ndarray:
    thresh = threshold_sauvola(flat, window_size=31, k=0.2)
    ink = ((flat < thresh) & (flat < 225)).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    small = np.where(stats[:, cv2.CC_STAT_AREA] < 12)[0]
    ink[np.isin(labels, small[small > 0])] = 0
    return ink


def _stroke_px(ink: np.ndarray) -> float:
    if not ink.any():
        return 3.0
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    widths = 2 * dist[skeletonize(ink > 0)]
    return float(np.median(widths)) if widths.size else 3.0


def capture(image_bgr: np.ndarray) -> Captured | SketchAbstain:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    corners = _find_sheet(gray)
    if corners is None:
        mean = float(gray.mean())
        if mean < TOO_DARK:
            return SketchAbstain(stage="capture", reason="image_quality", remedy=RETAKE_LIGHT)
        if mean < BRIGHT_PAPER:
            return SketchAbstain(stage="capture", reason="sheet_not_found", remedy=RETAKE_FRAME)
    sheet, to_original = _rectify(image_bgr, corners)
    sheet_gray = cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY)
    # a blank page has no contrast at all and must reach the views stage (no_views_found);
    # a blurred drawing still has some contrast but no sharp edges
    contrast = float(np.percentile(sheet_gray, 99.5) - np.percentile(sheet_gray, 0.5))
    blurry = contrast >= 10 and cv2.Laplacian(sheet_gray, cv2.CV_64F).var() < BLUR_VAR
    if float(sheet_gray.mean()) < TOO_DARK or blurry:
        return SketchAbstain(stage="capture", reason="image_quality", remedy=RETAKE_LIGHT)
    flat = _flatten(sheet_gray)
    ink = _binarise(flat)
    return Captured(sheet=sheet, gray=flat, ink=ink, to_original=to_original, stroke_px=_stroke_px(ink))
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/sketch/test_capture.py -v` then `uv run ruff check .`
Expected: 7 passed. Tuning notes if a test fails: the shadow test depends on the median kernel of `_flatten` (41) being larger than the stroke width; the blur test depends on `BLUR_VAR`: print `cv2.Laplacian(sheet_gray, cv2.CV_64F).var()` for the sharp and blurred page and set the threshold between them, closer to the blurred value.

- [ ] **Step 5: Commit**

```bash
git add s2c/sketch/capture.py tests/sketch/test_capture.py
git commit -m "Find, rectify and binarise the sketch sheet with quality gates"
```

---

### Task 4: Pluggable readers

**Files:**
- Create: `s2c/sketch/readers.py`
- Test: `tests/sketch/test_readers.py`

**Interfaces:**
- Consumes: `s2c.vision.client.VLMClient` (`complete_json(system, user, image_bytes, mime) -> str`).
- Produces: dataclass `Crop(image: np.ndarray BGR, box: tuple[int, int, int, int])` (box is `x, y, w, h` on the sheet); `ReaderResult(text: str, confidence: float)` Pydantic model; `Reader` protocol with `name: str` and `read(crops: list[Crop]) -> list[ReaderResult] | None` (one result per crop, same order; `None` means the reader failed); `VlmReader(client, batch=24)`; `PaddleReader(model_name=None)`; `TrocrReader(model_id=None, device=None)`; `tile_grid(crops) -> bytes` (PNG); `readers_from_env(spec: str | None = None) -> list[Reader]`; constant `SYSTEM` (the VLM prompt).

- [ ] **Step 1: Write the failing tests**

```python
# tests/sketch/test_readers.py
import json
import os

import cv2
import numpy as np
import pytest

from s2c.sketch.readers import SYSTEM, Crop, VlmReader, readers_from_env, tile_grid
from s2c.vision.client import VLMClient


def crops(n):
    return [Crop(np.full((30 + i, 60, 3), 255, np.uint8), (i, i, 60, 30 + i)) for i in range(n)]


def client(chat, tmp_path):
    return VLMClient(chat=chat, model="fake", log_path=tmp_path / "vlm.jsonl")


def test_tile_grid_is_a_png_holding_every_crop():
    png = tile_grid(crops(5))
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.shape[0] > 0 and img.shape[1] > 0


def test_one_result_per_crop_in_order(tmp_path):
    def chat(messages):
        return json.dumps({"reads": [{"i": 2, "text": "Ø6", "confidence": 0.8},
                                     {"i": 1, "text": ".50", "confidence": 0.9}]})
    out = VlmReader(client(chat, tmp_path)).read(crops(3))
    assert [r.text for r in out] == [".50", "Ø6", ""]
    assert out[2].confidence == 0.0


def test_prompt_transcribes_and_forbids_estimates():
    s = SYSTEM.lower()
    assert "transcribe" in s and "never estimate" in s and ".50" in SYSTEM


def test_retries_once_then_gives_up(tmp_path):
    calls = []
    def chat(messages):
        calls.append(messages)
        return "not json"
    assert VlmReader(client(chat, tmp_path)).read(crops(2)) is None
    assert len(calls) == 2
    assert "invalid" in calls[1][1]["content"][0]["text"]


def test_fenced_json_is_accepted(tmp_path):
    reply = '```json\n{"reads": [{"i": 1, "text": "4.00", "confidence": 0.9}]}\n```'
    out = VlmReader(client(lambda m: reply, tmp_path)).read(crops(1))
    assert out[0].text == "4.00"


def test_transport_error_returns_none(tmp_path):
    def chat(messages):
        raise TimeoutError("slow")
    assert VlmReader(client(chat, tmp_path)).read(crops(1)) is None


def test_large_sets_are_batched(tmp_path):
    calls = []
    def chat(messages):
        calls.append(1)
        n = int(messages[1]["content"][0]["text"].split(" tiles")[0].split()[-1])
        return json.dumps({"reads": [{"i": i, "text": "1", "confidence": 0.9} for i in range(1, n + 1)]})
    out = VlmReader(client(chat, tmp_path), batch=24).read(crops(50))
    assert len(out) == 50 and len(calls) == 3


def test_empty_input_needs_no_call(tmp_path):
    assert VlmReader(client(lambda m: 1 / 0, tmp_path)).read([]) == []


def test_unknown_reader_names_are_skipped(monkeypatch):
    monkeypatch.setenv("SKETCH_READERS", "nope")
    assert readers_from_env() == []


@pytest.mark.skipif(os.environ.get("SKETCH_MODEL_TESTS") != "1", reason="downloads a model")
def test_paddle_reader_reads_printed_digits():
    from s2c.sketch.readers import PaddleReader
    img = np.full((60, 160, 3), 255, np.uint8)
    cv2.putText(img, "40", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    out = PaddleReader().read([Crop(img, (0, 0, 160, 60))])
    assert out is not None and "40" in out[0].text


@pytest.mark.skipif(os.environ.get("SKETCH_MODEL_TESTS") != "1", reason="downloads a model")
def test_trocr_reader_reads_printed_digits():
    from s2c.sketch.readers import TrocrReader
    img = np.full((60, 160, 3), 255, np.uint8)
    cv2.putText(img, "40", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    out = TrocrReader().read([Crop(img, (0, 0, 160, 60))])
    assert out is not None and "40" in out[0].text
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/sketch/test_readers.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# s2c/sketch/readers.py
"""Readers transcribe handwriting in crops. They never estimate a size. Swappable by env var."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
from pydantic import BaseModel, Field, ValidationError

from s2c.vision.client import VLMClient

log = logging.getLogger(__name__)


@dataclass
class Crop:
    image: np.ndarray
    box: tuple[int, int, int, int]


class ReaderResult(BaseModel):
    text: str
    confidence: float = Field(ge=0, le=1)


class Reader(Protocol):
    name: str

    def read(self, crops: list[Crop]) -> list[ReaderResult] | None: ...


SYSTEM = (
    "You transcribe handwriting cropped from an engineering sketch. Each numbered tile is one crop. "
    'Return JSON only: {"reads": [{"i": 1, "text": "...", "confidence": 0.0}]}, one entry per tile. '
    "Copy exactly the characters written. Use Ø for a diameter sign, R for a radius, ° for degrees. "
    "Keep a leading decimal point exactly as written, for example .50. "
    "Return an empty text when a tile is unreadable. "
    "Never estimate, measure or guess a value that is not written."
)


class _Read(BaseModel):
    i: int
    text: str
    confidence: float = Field(default=0.8, ge=0, le=1)


class _Reply(BaseModel):
    reads: list[_Read]


def _json_block(raw: str) -> str:
    start, end = raw.find("{"), raw.rfind("}")
    return raw[start: end + 1] if start >= 0 and end > start else raw


def tile_grid(crops: list[Crop], cols: int = 4, cell_h: int = 80, max_w: int = 320) -> bytes:
    """One image with every crop in a numbered cell; the number sits in a header, not on the ink."""
    header = 26
    tiles = []
    for c in crops:
        h, w = c.image.shape[:2]
        k = cell_h / max(h, 1)
        tile = cv2.resize(c.image, (min(max_w, max(1, round(w * k))), cell_h))
        tiles.append(tile)
    cell_w = max_w + 16
    rows = (len(tiles) + cols - 1) // cols
    grid = np.full((rows * (cell_h + header + 12), cols * cell_w, 3), 255, np.uint8)
    for i, tile in enumerate(tiles):
        r, col = divmod(i, cols)
        y, x = r * (cell_h + header + 12), col * cell_w
        cv2.putText(grid, f"#{i + 1}", (x + 4, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 0), 2)
        grid[y + header: y + header + cell_h, x + 8: x + 8 + tile.shape[1]] = tile
        cv2.rectangle(grid, (x + 4, y + header - 2), (x + cell_w - 4, y + header + cell_h + 2),
                      (180, 180, 180), 1)
    ok, buf = cv2.imencode(".png", grid)
    return buf.tobytes()


class VlmReader:
    name = "vlm"

    def __init__(self, client: VLMClient, batch: int = 24):
        self.client, self.batch = client, batch

    def read(self, crops: list[Crop]) -> list[ReaderResult] | None:
        out: list[ReaderResult] = []
        for start in range(0, len(crops), self.batch):
            part = self._read_batch(crops[start: start + self.batch])
            if part is None:
                return None
            out += part
        return out

    def _read_batch(self, crops: list[Crop]) -> list[ReaderResult] | None:
        n = len(crops)
        image = tile_grid(crops)
        user = f"There are {n} tiles, numbered 1 to {n}."
        reply = None
        for _ in range(2):
            try:
                raw = self.client.complete_json(SYSTEM, user, image, mime="image/png")
            except Exception as exc:  # transport errors must not break the pipeline
                log.warning("vlm reader failed: %s", exc)
                return None
            try:
                reply = _Reply.model_validate_json(_json_block(raw))
                break
            except ValidationError as exc:
                user += f"\nYour previous reply was invalid ({exc.error_count()} errors). Return JSON only."
        if reply is None:
            return None
        by_i = {r.i: r for r in reply.reads}
        return [ReaderResult(text=by_i[i].text, confidence=by_i[i].confidence) if i in by_i
                else ReaderResult(text="", confidence=0.0) for i in range(1, n + 1)]


class PaddleReader:
    """PP-OCRv5 text recognition. Optional dependency: `uv sync --extra ocr`.
    Checked against PaddleOCR 3.x (`TextRecognition(...).predict(...)` returning `rec_text`, `rec_score`)."""
    name = "paddle"

    def __init__(self, model_name: str | None = None):
        from paddleocr import TextRecognition

        self._model = TextRecognition(
            model_name=model_name or os.environ.get("SKETCH_PADDLE_MODEL", "PP-OCRv5_server_rec"))

    def read(self, crops: list[Crop]) -> list[ReaderResult] | None:
        try:
            out = []
            for c in crops:
                res = list(self._model.predict(input=c.image, batch_size=1))[0]
                data = res.json.get("res", res.json) if hasattr(res, "json") else dict(res)
                out.append(ReaderResult(text=str(data.get("rec_text", "")),
                                        confidence=float(np.clip(data.get("rec_score", 0.0), 0, 1))))
            return out
        except Exception as exc:
            log.warning("paddle reader failed: %s", exc)
            return None


class TrocrReader:
    """TrOCR handwritten. Optional dependency: `uv sync --extra trocr`."""
    name = "trocr"

    def __init__(self, model_id: str | None = None, device: str | None = None):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        mid = model_id or os.environ.get("SKETCH_TROCR_MODEL", "microsoft/trocr-base-handwritten")
        self._proc = TrOCRProcessor.from_pretrained(mid)
        self._model = VisionEncoderDecoderModel.from_pretrained(mid)
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device).eval()

    def read(self, crops: list[Crop]) -> list[ReaderResult] | None:
        if not crops:
            return []
        try:
            import torch
            from PIL import Image

            images = [Image.fromarray(cv2.cvtColor(c.image, cv2.COLOR_BGR2RGB)) for c in crops]
            pixels = self._proc(images=images, return_tensors="pt").pixel_values.to(self._device)
            with torch.no_grad():
                gen = self._model.generate(pixels, max_new_tokens=16, output_scores=True,
                                           return_dict_in_generate=True)
            texts = self._proc.batch_decode(gen.sequences, skip_special_tokens=True)
            scores = self._model.compute_transition_scores(gen.sequences, gen.scores,
                                                           normalize_logits=True)
            conf = torch.exp(scores.mean(dim=1)).clamp(0, 1).tolist()
            return [ReaderResult(text=t.strip(), confidence=float(c)) for t, c in zip(texts, conf)]
        except Exception as exc:
            log.warning("trocr reader failed: %s", exc)
            return None


def readers_from_env(spec: str | None = None) -> list[Reader]:
    names = [n.strip() for n in (spec or os.environ.get("SKETCH_READERS", "paddle,vlm")).split(",")]
    out: list[Reader] = []
    for name in filter(None, names):
        try:
            if name == "vlm":
                out.append(VlmReader(VLMClient.from_env()))
            elif name == "paddle":
                out.append(PaddleReader())
            elif name == "trocr":
                out.append(TrocrReader())
            else:
                log.warning("unknown reader %r skipped", name)
        except Exception as exc:  # a missing optional dependency must not stop the others
            log.warning("reader %s unavailable: %s", name, exc)
    return out
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/sketch/test_readers.py -v` then `uv run ruff check .`
Expected: all pass except the two model tests, which are skipped. With `uv sync --extra ocr` and `SKETCH_MODEL_TESTS=1`, run the paddle test once and adjust `PaddleReader.read` to the installed result format if it differs (print `res.json` for one crop).

- [ ] **Step 5: Commit**

```bash
git add s2c/sketch/readers.py tests/sketch/test_readers.py
git commit -m "Add swappable handwriting readers: vision model, PaddleOCR, TrOCR"
```

---

### Task 5: Synthetic sheets, text boxes and two-reader agreement

**Files:**
- Create: `tests/sketch/synth.py`, `s2c/sketch/text.py`
- Test: `tests/sketch/test_text.py`

**Interfaces:**
- Consumes: `Crop`, `ReaderResult`, `Reader` (Task 4); `parse_text`, `match_label`, `Parsed` (Task 2); `Reading`, `SketchAbstain` (Task 1).
- Produces:
  - `tests/sketch/synth.py`: class `Sheet(w=1600, h=1131)` with `line`, `dashed`, `circle`, `arrowhead`, `text(s, centre, scale=0.8, t=2, rotate=False)`, `hdim(x1, x2, y_obj, y_line, text)` (x1 < x2), `vdim(y1, y2, x_obj, x_line, text, rotate=False)` (y1 < y2), `leader(tip, tail, text)`, `bgr()`, `ink()`, and attribute `texts: list[tuple[str, box]]`; function `bridge_block(sheet, labels=True)` drawing TOP, FRONT, SIDE of the reference part at 4 px per mm with FRONT's bottom-left at (200, 900), TOP's at (200, 520), SIDE's at (800, 900); class `TruthReader(texts, name="truth", confidence=0.95, swap=None, upright_only=False)`.
  - `s2c/sketch/text.py`: dataclass `TextItem(id, box, readings: list[Reading], parsed: Parsed | None, label: ViewName | None, role: "label" | "dimension" | "other", badge: "written" | "uncertain" | None, confidence: float, candidates: list[float])`; `find_text_boxes(ink, stroke_px) -> list[tuple[int, int, int, int]]`; `read_texts(sheet_bgr, boxes, readers) -> list[TextItem] | SketchAbstain`; `erase_mask(texts, shape, pad=2) -> np.ndarray` (255 where label and dimension texts sit).

The reference part (the "bridge block"), in mm: overall 100 wide (X), 50 high (Y), 25 deep (Z). FRONT outline `(0,0) (37.5,0) (37.5,12.5) (62.5,12.5) (62.5,0) (100,0) (100,12.5) (75,12.5) (75,50) (62.5,50) (62.5,25) (37.5,25) (37.5,50) (25,50) (25,12.5) (0,12.5)`. Two vertical Ø12.5 through holes in the base at X 12.5 and 87.5, Z 12.5; one horizontal Ø12.5 through hole along X in both uprights at Y 37.5, Z 12.5. Written values: FRONT `100`, `37.5` twice (bottom), chain `12.5 25 12.5` (top), `37.5` (left), chain `25 12.5 12.5` (right); TOP `2xØ12.5` leader, `12.5` (hole X), `12.5` (hole from the back edge); SIDE `Ø12.5` leader, chain `12.5 12.5` (top), `12.5` (hole from the top), `50`.

- [ ] **Step 1: Write the synthetic sheet helper**

```python
# tests/sketch/synth.py
"""Synthetic sheets for tests: white page, pen lines, dashed lines, arrows, dimension text."""
from __future__ import annotations

import cv2
import numpy as np

from s2c.sketch.readers import ReaderResult

INK = (25, 25, 25)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _i(p):
    return int(round(p[0])), int(round(p[1]))


class Sheet:
    def __init__(self, w: int = 1600, h: int = 1131):
        self.img = np.full((h, w, 3), 250, np.uint8)
        self.texts: list[tuple[str, tuple[int, int, int, int]]] = []

    def line(self, p, q, t=3):
        cv2.line(self.img, _i(p), _i(q), INK, t, cv2.LINE_AA)

    def dashed(self, p, q, dash=14, gap=9, t=2):
        p, q = np.float64(p), np.float64(q)
        length = np.linalg.norm(q - p)
        d = (q - p) / length
        s = 0.0
        while s < length:
            e = min(s + dash, length)
            self.line(p + d * s, p + d * e, t)
            s = e + gap

    def circle(self, c, r, t=3):
        cv2.circle(self.img, _i(c), int(round(r)), INK, t, cv2.LINE_AA)

    def arrowhead(self, tip, direction, length=16, half=5):
        tip, d = np.float64(tip), np.float64(direction)
        d /= np.linalg.norm(d)
        n = np.array([-d[1], d[0]])
        base = tip - d * length
        cv2.fillPoly(self.img, [np.int32([tip, base + n * half, base - n * half])], INK, cv2.LINE_AA)

    def text(self, s, centre, scale=0.8, t=2, rotate=False):
        """Draw `s` centred on `centre`. Ø is drawn as a slashed circle (Hershey fonts have none)."""
        parts = s.split("Ø")
        (_, th), base = cv2.getTextSize("0", FONT, scale, t)
        widths = [cv2.getTextSize(p, FONT, scale, t)[0][0] if p else 0 for p in parts]
        glyph = int(th * 1.1)
        W = sum(widths) + glyph * (len(parts) - 1) + 8
        H = th + base + 8
        tile = np.full((H, W, 3), 250, np.uint8)
        x = 4
        for k, part in enumerate(parts):
            if part:
                cv2.putText(tile, part, (x, th + 4), FONT, scale, INK, t, cv2.LINE_AA)
                x += widths[k]
            if k < len(parts) - 1:
                c, r = (x + glyph // 2, 4 + th // 2), int(th * 0.45)
                cv2.circle(tile, c, r, INK, t, cv2.LINE_AA)
                cv2.line(tile, (c[0] - r, c[1] + r), (c[0] + r, c[1] - r), INK, t, cv2.LINE_AA)
                x += glyph
        if rotate:
            tile = cv2.rotate(tile, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h, w = tile.shape[:2]
        x0, y0 = int(centre[0] - w / 2), int(centre[1] - h / 2)
        region = self.img[y0:y0 + h, x0:x0 + w]
        np.minimum(region, tile, out=region)
        self.texts.append((s, (x0, y0, w, h)))

    def hdim(self, x1, x2, y_obj, y_line, text):
        sgn = 1 if y_line > y_obj else -1
        for x in (x1, x2):
            self.line((x, y_obj + sgn * 6), (x, y_line + sgn * 8), 1)
        self.line((x1, y_line), (x2, y_line), 1)
        self.arrowhead((x1, y_line), (-1, 0))
        self.arrowhead((x2, y_line), (1, 0))
        self.text(text, ((x1 + x2) / 2, y_line - 16))

    def vdim(self, y1, y2, x_obj, x_line, text, rotate=False):
        sgn = 1 if x_line > x_obj else -1
        for y in (y1, y2):
            self.line((x_obj + sgn * 6, y), (x_line + sgn * 8, y), 1)
        self.line((x_line, y1), (x_line, y2), 1)
        self.arrowhead((x_line, y1), (0, -1))
        self.arrowhead((x_line, y2), (0, 1))
        self.text(text, (x_line + sgn * (20 if rotate else 36), (y1 + y2) / 2), rotate=rotate)

    def leader(self, tip, tail, text):
        self.line(tail, tip, 1)
        self.arrowhead(tip, np.float64(tip) - np.float64(tail))
        self.text(text, (tail[0], tail[1] - 16))

    def bgr(self):
        return self.img.copy()

    def ink(self):
        return (cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY) < 128).astype(np.uint8) * 255


S = 4.0  # px per mm


def F(x, y):
    return 200 + S * x, 900 - S * y


def T(x, b):  # TOP: b is the distance from the front face (the view's bottom edge)
    return 200 + S * x, 520 - S * b


def R(a, y):  # SIDE: a is the distance from the front face (the view's left edge)
    return 800 + S * a, 900 - S * y


FRONT_OUTLINE = [(0, 0), (37.5, 0), (37.5, 12.5), (62.5, 12.5), (62.5, 0), (100, 0), (100, 12.5),
                 (75, 12.5), (75, 50), (62.5, 50), (62.5, 25), (37.5, 25), (37.5, 50), (25, 50),
                 (25, 12.5), (0, 12.5)]


def bridge_block(sh: Sheet, labels: bool = True) -> Sheet:
    # FRONT
    for p, q in zip(FRONT_OUTLINE, FRONT_OUTLINE[1:] + FRONT_OUTLINE[:1]):
        sh.line(F(*p), F(*q))
    for x in (6.25, 18.75, 81.25, 93.75):
        sh.dashed(F(x, 0), F(x, 12.5))
    for x0, x1 in ((25, 37.5), (62.5, 75)):
        for y in (31.25, 43.75):
            sh.dashed(F(x0, y), F(x1, y))
    bottom, top, left, right = F(0, 0)[1], F(0, 50)[1], F(0, 0)[0], F(100, 0)[0]
    sh.hdim(F(0, 0)[0], F(100, 0)[0], bottom, bottom + 90, "100")
    sh.hdim(F(0, 0)[0], F(37.5, 0)[0], bottom, bottom + 45, "37.5")
    sh.hdim(F(62.5, 0)[0], F(100, 0)[0], bottom, bottom + 45, "37.5")
    for x0, x1, t in ((25, 37.5, "12.5"), (37.5, 62.5, "25"), (62.5, 75, "12.5")):
        sh.hdim(F(x0, 0)[0], F(x1, 0)[0], top, top - 40, t)
    sh.vdim(F(0, 50)[1], F(0, 12.5)[1], left, left - 50, "37.5")
    for y0, y1, t in ((50, 25, "25"), (25, 12.5, "12.5"), (12.5, 0, "12.5")):
        sh.vdim(F(0, y0)[1], F(0, y1)[1], right, right + 50, t)
    # TOP
    corners = [T(0, 0), T(100, 0), T(100, 25), T(0, 25)]
    for p, q in zip(corners, corners[1:] + corners[:1]):
        sh.line(p, q)
    for x in (25, 37.5, 62.5, 75):
        sh.line(T(x, 0), T(x, 25))
    for x in (12.5, 87.5):
        sh.circle(T(x, 12.5), S * 6.25)
    for x0, x1 in ((25, 37.5), (62.5, 75)):
        for b in (6.25, 18.75):
            sh.dashed(T(x0, b), T(x1, b))
    tip = np.float64(T(87.5, 12.5)) + S * 6.25 * np.array([0.707, -0.707])
    sh.leader(tip, (tip[0] + 45, tip[1] - 45), "2xØ12.5")
    sh.hdim(T(0, 0)[0], T(12.5, 0)[0], T(0, 25)[1], T(0, 25)[1] - 35, "12.5")
    sh.vdim(T(0, 25)[1], T(0, 12.5)[1], T(0, 0)[0], T(0, 0)[0] - 50, "12.5")
    # SIDE
    corners = [R(0, 0), R(25, 0), R(25, 50), R(0, 50)]
    for p, q in zip(corners, corners[1:] + corners[:1]):
        sh.line(p, q)
    sh.line(R(0, 12.5), R(25, 12.5))
    sh.dashed(R(0, 25), R(25, 25))
    for a in (6.25, 18.75):
        sh.dashed(R(a, 0), R(a, 12.5))
    sh.circle(R(12.5, 37.5), S * 6.25)
    tip = np.float64(R(12.5, 37.5)) + S * 6.25 * np.array([-0.707, -0.707])
    sh.leader(tip, (tip[0] - 45, tip[1] - 60), "Ø12.5")
    stop, sright = R(0, 50)[1], R(25, 0)[0]
    for a0, a1 in ((0, 12.5), (12.5, 25)):
        sh.hdim(R(a0, 0)[0], R(a1, 0)[0], stop, stop - 40, "12.5")
    sh.vdim(R(0, 50)[1], R(0, 37.5)[1], sright, sright + 50, "12.5")
    sh.vdim(R(0, 50)[1], R(0, 0)[1], sright, sright + 110, "50")
    if labels:
        sh.text("TOP", (400, 575), scale=1.0)
        sh.text("FRONT", (400, 1060), scale=1.0)
        sh.text("SIDE", (850, 1060), scale=1.0)
    return sh


def _overlap(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    return ix * iy / max(bw * bh, 1)


class TruthReader:
    """Reads a crop by looking up the synthetic text whose box it overlaps most.
    `swap` maps a true text to what this reader "sees"; `upright_only` reads only crops that are
    wider than tall, like a reader that cannot handle sideways text."""

    def __init__(self, texts, name="truth", confidence=0.95, swap=None, upright_only=False):
        self.texts, self.name, self.confidence = texts, name, confidence
        self.swap, self.upright_only = swap or {}, upright_only

    def read(self, crops):
        out = []
        for c in crops:
            best, score = "", 0.0
            for s, b in self.texts:
                ov = _overlap(c.box, b)
                if ov > score:
                    best, score = s, ov
            h, w = c.image.shape[:2]
            if score < 0.3 or (self.upright_only and h > w):
                best = ""
            best = self.swap.get(best, best)
            out.append(ReaderResult(text=best, confidence=self.confidence if best else 0.0))
        return out
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/sketch/test_text.py
import numpy as np

from s2c.sketch.models import SketchAbstain
from s2c.sketch.text import erase_mask, find_text_boxes, read_texts
from tests.sketch.synth import Sheet, TruthReader, _overlap, bridge_block


def sheet():
    return bridge_block(Sheet())


def test_boxes_cover_every_written_text():
    sh = sheet()
    boxes = find_text_boxes(sh.ink(), 3.0)
    for s, b in sh.texts:
        assert any(_overlap(box, b) > 0.6 for box in boxes), s
    for box in boxes:  # no box swallows two texts
        assert sum(_overlap(box, b) > 0.6 for _, b in sh.texts) <= 1


def test_two_agreeing_readers_give_written_values_and_labels():
    sh = sheet()
    items = read_texts(sh.bgr(), find_text_boxes(sh.ink(), 3.0),
                       [TruthReader(sh.texts, "a"), TruthReader(sh.texts, "b")])
    dims = [t for t in items if t.role == "dimension"]
    labels = {t.label for t in items if t.role == "label"}
    assert labels == {"top", "front", "right"}
    assert len(dims) == 18
    assert all(t.badge == "written" for t in dims)
    assert sorted(t.parsed.value for t in dims).count(12.5) == 11


def test_disagreement_is_uncertain_with_both_candidates():
    sh = sheet()
    items = read_texts(sh.bgr(), find_text_boxes(sh.ink(), 3.0),
                       [TruthReader(sh.texts, "a"), TruthReader(sh.texts, "b", swap={"100": "700"})])
    t = next(t for t in items if t.role == "dimension" and 100.0 in t.candidates)
    assert t.badge == "uncertain" and t.candidates == [100.0, 700.0]
    assert [r.reader for r in t.readings] == ["a", "b"]


def test_a_single_reader_is_never_trusted_alone():
    sh = sheet()
    items = read_texts(sh.bgr(), find_text_boxes(sh.ink(), 3.0), [TruthReader(sh.texts)])
    assert all(t.badge == "uncertain" for t in items if t.role == "dimension")


def test_a_small_circle_read_as_O_stays_in_the_geometry():
    sh = Sheet()
    sh.circle((600, 500), 14)
    sh.texts.append(("O", (582, 482, 36, 36)))  # what a reader would say about the circle
    sh.text("40", (900, 500))
    items = read_texts(sh.bgr(), find_text_boxes(sh.ink(), 3.0),
                       [TruthReader(sh.texts, "a"), TruthReader(sh.texts, "b")])
    mask = erase_mask(items, sh.img.shape[:2])
    assert mask[500, 586] == 0 and mask[486, 600] == 0  # the circle is not erased
    assert mask[500, 900] == 255                         # the number is


def test_sideways_vertical_text_is_read():
    sh = Sheet()
    sh.line((400, 300), (400, 700), 1)
    sh.text("37.5", (380, 500), rotate=True)
    readers = [TruthReader(sh.texts, "a", upright_only=True),
               TruthReader(sh.texts, "b", upright_only=True)]
    items = read_texts(sh.bgr(), find_text_boxes(sh.ink(), 3.0), readers)
    dims = [t for t in items if t.role == "dimension"]
    assert len(dims) == 1 and dims[0].parsed.value == 37.5 and dims[0].badge == "written"


def test_readers_down_abstains():
    class Down:
        name = "down"

        def read(self, crops):
            return None
    sh = sheet()
    out = read_texts(sh.bgr(), find_text_boxes(sh.ink(), 3.0), [Down(), Down()])
    assert isinstance(out, SketchAbstain) and out.reason == "readers_unavailable"


def test_no_text_needs_no_reader():
    assert read_texts(np.full((100, 100, 3), 255, np.uint8), [], []) == []
```

- [ ] **Step 3: Run to confirm failure**

Run: `uv run pytest tests/sketch/test_text.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 's2c.sketch.text'`.

- [ ] **Step 4: Implement**

```python
# s2c/sketch/text.py
"""Stage 2: find text boxes, read them with every reader, decide what each text is."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from s2c.sketch.grammar import Parsed, match_label, parse_text
from s2c.sketch.models import Reading, SketchAbstain, ViewName
from s2c.sketch.readers import Crop, Reader

MIN_CONFIDENCE = 0.6
PAD = 4


@dataclass
class TextItem:
    id: str
    box: tuple[int, int, int, int]
    readings: list[Reading]
    parsed: Parsed | None
    label: ViewName | None
    role: Literal["label", "dimension", "other"]
    badge: Literal["written", "uncertain"] | None
    confidence: float
    candidates: list[float] = field(default_factory=list)


def _line_like(w: int, h: int, stroke: float) -> bool:
    return min(w, h) <= 1.8 * stroke and max(w, h) >= 3 * min(w, h)


def _dash_members(stats: np.ndarray, idx: list[int], stroke: float) -> set[int]:
    """Line-like pieces with at least two similar pieces on the same line: pieces of a dashed line."""
    pieces = [i for i in idx if _line_like(stats[i, 2], stats[i, 3], stroke)]
    out = set()
    for i in pieces:
        xi, yi, wi, hi = stats[i, :4]
        horizontal = wi > hi
        near = 0
        for j in pieces:
            if j == i:
                continue
            xj, yj, wj, hj = stats[j, :4]
            if (wj > hj) != horizontal:
                continue
            if horizontal and abs((yi + hi / 2) - (yj + hj / 2)) < 2 * stroke \
                    and abs(xi - xj) < 3 * max(wi, wj):
                near += 1
            if not horizontal and abs((xi + wi / 2) - (xj + wj / 2)) < 2 * stroke \
                    and abs(yi - yj) < 3 * max(hi, hj):
                near += 1
        if near >= 2:
            out.add(i)
    return out


def find_text_boxes(ink: np.ndarray, stroke_px: float) -> list[tuple[int, int, int, int]]:
    h, w = ink.shape
    long_side = max(h, w)
    L = max(40, int(0.04 * long_side))
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, L), np.uint8)) | \
        cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((L, 1), np.uint8))
    rest = cv2.bitwise_and(ink, cv2.bitwise_not(cv2.dilate(lines, np.ones((3, 3), np.uint8))))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(rest, connectivity=8)
    char_max = 0.06 * long_side
    idx = [i for i in range(1, n) if stats[i, 4] >= 10 and stats[i, 3] <= char_max
           and stats[i, 2] <= 1.5 * char_max]
    dashes = _dash_members(stats, idx, stroke_px)
    glyphs = np.isin(labels, [i for i in idx if i not in dashes]).astype(np.uint8) * 255
    heights = [stats[i, 3] for i in idx if i not in dashes]
    k = max(5, int(0.6 * np.median(heights))) if heights else 5
    glued = cv2.dilate(glyphs, np.ones((k, k), np.uint8))
    m, _, st, _ = cv2.connectedComponentsWithStats(glued, connectivity=8)
    boxes = []
    for i in range(1, m):
        x, y, bw, bh = (int(v) for v in st[i, :4])
        x, y, bw, bh = x + k // 2, y + k // 2, bw - k, bh - k  # undo the dilation margin
        if bw < 6 or bh < 6 or max(bw, bh) / max(min(bw, bh), 1) > 10:
            continue
        boxes.append((max(0, x - PAD), max(0, y - PAD), bw + 2 * PAD, bh + 2 * PAD))
    return boxes


def _crops(sheet: np.ndarray, box) -> list[Crop]:
    x, y, w, h = box
    img = sheet[y:y + h, x:x + w]
    out = [Crop(img, box)]
    if h > 1.3 * w:  # maybe sideways text on a vertical dimension
        out += [Crop(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), box),
                Crop(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), box)]
    return out


def _best(results) -> tuple[str, float]:
    """Among one reader's readings of the orientations of one box: parseable first, then labels,
    then the most confident."""
    ranked = sorted(results, key=lambda r: (parse_text(r.text) is not None,
                                            match_label(r.text) is not None, r.confidence),
                    reverse=True)
    return ranked[0].text, ranked[0].confidence


def read_texts(sheet_bgr: np.ndarray, boxes, readers: list[Reader]) -> list[TextItem] | SketchAbstain:
    if not boxes:
        return []
    groups = [_crops(sheet_bgr, b) for b in boxes]
    flat = [c for g in groups for c in g]
    per_reader = []
    for r in readers:
        res = r.read(flat)
        if res is not None:
            per_reader.append((r.name, res))
    if not per_reader:
        return SketchAbstain(stage="text", reason="readers_unavailable",
                             remedy="Reading service unavailable. Retry in a minute.")
    items: list[TextItem] = []
    start = 0
    for n, (box, group) in enumerate(zip(boxes, groups)):
        readings = []
        for name, res in per_reader:
            text, conf = _best(res[start:start + len(group)])
            readings.append(Reading(reader=name, text=text, confidence=conf))
        start += len(group)
        items.append(_decide(f"t{n}", box, readings, len(per_reader)))
    return items


def _decide(tid: str, box, readings: list[Reading], n_readers: int) -> TextItem:
    parses = [(r, parse_text(r.text)) for r in readings]
    parsed = [(r, p) for r, p in parses if p is not None]
    if parsed:
        values: list[float] = []
        for _, p in parsed:
            if p.value not in values:
                values.append(p.value)
        kinds = {p.kind for _, p in parsed}
        agree = (len(parsed) == n_readers >= 2 and len(values) == 1 and len(kinds) == 1
                 and all(r.confidence >= MIN_CONFIDENCE for r, _ in parsed))
        return TextItem(tid, box, readings, parsed[0][1], None, "dimension",
                        "written" if agree else "uncertain",
                        max(r.confidence for r, _ in parsed), values)
    for r in readings:
        name = match_label(r.text)
        if name:
            return TextItem(tid, box, readings, None, name, "label", None, r.confidence)
    return TextItem(tid, box, readings, None, None, "other", None,
                    max((r.confidence for r in readings), default=0.0))


def erase_mask(texts: list[TextItem], shape, pad: int = 2) -> np.ndarray:
    mask = np.zeros(shape[:2], np.uint8)
    for t in texts:
        if t.role in ("label", "dimension"):
            x, y, w, h = t.box
            mask[max(0, y - pad): y + h + pad, max(0, x - pad): x + w + pad] = 255
    return mask
```

- [ ] **Step 5: Run tests and lint**

Run: `uv run pytest tests/sketch/test_text.py -v` then `uv run ruff check .`
Expected: 8 passed. Tuning notes: if two neighbouring chain values merge into one box, lower the glue factor 0.6; if a number splits into two boxes, raise it. If arrowheads glue onto numbers, that is acceptable (the readers still read the number); if a box then fails `test_boxes_cover_every_written_text` because it is much larger than the text, strip isolated filled triangles before gluing.

- [ ] **Step 6: Commit**

```bash
git add tests/sketch/synth.py s2c/sketch/text.py tests/sketch/test_text.py
git commit -m "Find and read dimension text with two readers that must agree"
```

---

### Task 6: Split the sheet into named views

**Files:**
- Create: `s2c/sketch/views.py`
- Test: `tests/sketch/test_views.py`

**Interfaces:**
- Consumes: `TextItem`, `erase_mask` (Task 5); `Issue`, `SketchAbstain`, `ViewName` (Task 1).
- Produces: dataclass `ViewRegion(name: ViewName, label_text: str | None, bbox: tuple[int, int, int, int], ink: np.ndarray, texts: list[TextItem], named_by: "label" | "layout")` where `ink` is a sheet-sized mask holding only this view's geometry (texts erased) and `texts` are the dimension and other texts that belong to it; `split_views(ink, texts, stroke_px) -> tuple[list[ViewRegion], list[Issue]] | SketchAbstain`.

Method: every view has a closed outline. Closed outlines (components that enclose a hole) are the seeds; seeds whose boxes overlap are one view; every other ink pixel goes to the nearest seed, within a reach of 10 percent of the sheet. Seeding on closed outlines rather than on dilated blobs is what keeps two close views apart (Review Focus 5).

- [ ] **Step 1: Write the failing tests**

```python
# tests/sketch/test_views.py
import numpy as np

from s2c.sketch.models import SketchAbstain
from s2c.sketch.text import find_text_boxes, read_texts
from s2c.sketch.views import split_views
from tests.sketch.synth import F, Sheet, TruthReader, bridge_block


def run(sh):
    items = read_texts(sh.bgr(), find_text_boxes(sh.ink(), 3.0),
                       [TruthReader(sh.texts, "a"), TruthReader(sh.texts, "b")])
    return split_views(sh.ink(), items, 3.0)


def test_bridge_block_has_three_views_named_by_their_labels():
    views, issues = run(bridge_block(Sheet()))
    by = {v.name: v for v in views}
    assert set(by) == {"top", "front", "right"}
    assert all(v.named_by == "label" for v in views) and issues == []
    x, y = (int(round(c)) for c in F(0, 0))
    assert by["front"].ink[y - 2:y + 3, x - 2:x + 3].any()
    assert not by["top"].ink[y - 2:y + 3, x - 2:x + 3].any()
    assert len([t for t in by["front"].texts if t.role == "dimension"]) == 10
    assert len([t for t in by["right"].texts if t.role == "dimension"]) == 5


def test_views_without_labels_are_named_from_the_third_angle_layout():
    views, issues = run(bridge_block(Sheet(), labels=False))
    assert {v.name for v in views} == {"top", "front", "right"}
    assert all(v.named_by == "layout" for v in views)
    assert len(issues) == 3 and all(i.kind == "label" for i in issues)


def test_two_close_views_stay_apart():
    sh = Sheet()
    for x0 in (200, 560):
        sh.line((x0, 300), (x0 + 300, 300))
        sh.line((x0 + 300, 300), (x0 + 300, 600))
        sh.line((x0 + 300, 600), (x0, 600))
        sh.line((x0, 600), (x0, 300))
    sh.vdim(300, 600, 500, 520, "75", rotate=True)   # A's dimension fills the 60 px gap
    sh.hdim(560, 860, 600, 645, "75")
    sh.text("FRONT", (350, 700), scale=1.0)
    sh.text("SIDE", (710, 720), scale=1.0)
    views, _ = run(sh)
    assert sorted(v.name for v in views) == ["front", "right"]


def test_blank_sheet_abstains():
    out = split_views(np.zeros((1131, 1600), np.uint8), [], 3.0)
    assert isinstance(out, SketchAbstain) and out.reason == "no_views_found"
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/sketch/test_views.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# s2c/sketch/views.py
"""Stage 3: split the sheet into views and name them from their labels or the third-angle layout."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from scipy import ndimage

from s2c.sketch.models import Issue, SketchAbstain, ViewName
from s2c.sketch.text import TextItem, erase_mask

REACH = 0.10          # of the sheet's long side: how far a view's annotations may sit
MIN_HOLE = 0.03       # side of the smallest enclosed area of an outline, of the long side
THIRD_ANGLE = {"above": "top", "below": "bottom", "right": "right", "left": "left"}


@dataclass
class ViewRegion:
    name: ViewName
    label_text: str | None
    bbox: tuple[int, int, int, int]
    ink: np.ndarray
    texts: list[TextItem]
    named_by: Literal["label", "layout"]


def _seeds(geo: np.ndarray) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    """Closed outlines: components that enclose an area. Overlapping ones are merged."""
    long_side = max(geo.shape)
    closed = cv2.morphologyEx(geo, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    min_hole = (MIN_HOLE * long_side) ** 2
    boxes = []
    for i, c in enumerate(contours):
        child = hierarchy[0][i][2]
        if hierarchy[0][i][3] != -1 or child == -1:
            continue
        holes = []
        while child != -1:
            holes.append(cv2.contourArea(contours[child]))
            child = hierarchy[0][child][0]
        if max(holes) >= min_hole:
            boxes.append(list(cv2.boundingRect(c)))
    merged = True
    while merged:  # merge overlapping boxes: inner outlines belong to the outer one
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]:
                    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
                    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
                    boxes[i] = [x0, y0, x1 - x0, y1 - y0]
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    out = []
    for x, y, w, h in boxes:
        mask = np.zeros(geo.shape, bool)
        mask[y:y + h, x:x + w] = geo[y:y + h, x:x + w] > 0
        out.append((mask, (x, y, w, h)))
    return out


def _relation(a, b) -> str | None:
    """Where box b sits relative to box a."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x_overlap = min(ax + aw, bx + bw) - max(ax, bx) > 0.3 * min(aw, bw)
    y_overlap = min(ay + ah, by + bh) - max(ay, by) > 0.3 * min(ah, bh)
    if x_overlap and by + bh <= ay + 0.2 * ah:
        return "above"
    if x_overlap and by >= ay + 0.8 * ah:
        return "below"
    if y_overlap and bx >= ax + 0.8 * aw:
        return "right"
    if y_overlap and bx + bw <= ax + 0.2 * aw:
        return "left"
    return None


def _layout_names(boxes, known: dict[int, ViewName]) -> dict[int, ViewName]:
    names = dict(known)
    front = next((i for i, n in names.items() if n == "front"), None)
    if front is None:
        def score(i):
            rel = [_relation(boxes[i], boxes[j]) for j in range(len(boxes)) if j != i]
            x, y, w, h = boxes[i]
            return (sum(r is not None for r in rel), y + h, -x)
        front = max((i for i in range(len(boxes)) if i not in names), key=score, default=None)
        if front is None:
            return names
        names[front] = "front"
    taken = set(names.values())
    rights = sorted((i for i in range(len(boxes)) if i not in names
                     and _relation(boxes[front], boxes[i]) == "right"), key=lambda i: boxes[i][0])
    for k, i in enumerate(rights):
        name = "right" if k == 0 and "right" not in taken else "back"
        if name not in taken:
            names[i] = name
            taken.add(name)
    for i in range(len(boxes)):
        if i in names:
            continue
        rel = _relation(boxes[front], boxes[i])
        name = THIRD_ANGLE.get(rel) if rel else None
        if name and name not in taken:
            names[i] = name
            taken.add(name)
    return names


def _distance_to_box(point, box) -> float:
    x, y, w, h = box
    dx = max(x - point[0], 0, point[0] - (x + w))
    dy = max(y - point[1], 0, point[1] - (y + h))
    return float(np.hypot(dx, dy))


def split_views(ink, texts: list[TextItem], stroke_px: float):
    geo = cv2.bitwise_and(ink, cv2.bitwise_not(erase_mask(texts, ink.shape)))
    seeds = _seeds(geo)
    if not seeds:
        return SketchAbstain(stage="views", reason="no_views_found",
                             remedy="Draw the views with a dark pen and retake.")
    markers = np.zeros(geo.shape, np.int32)
    for k, (mask, _) in enumerate(seeds):
        markers[mask] = k + 1
    dist, (iy, ix) = ndimage.distance_transform_edt(markers == 0, return_indices=True)
    nearest = markers[iy, ix]
    reach = REACH * max(geo.shape)
    boxes = [box for _, box in seeds]

    labels = [t for t in texts if t.role == "label"]
    known: dict[int, ViewName] = {}
    label_text: dict[int, str] = {}
    for t in sorted(labels, key=lambda t: t.confidence, reverse=True):
        centre = (t.box[0] + t.box[2] / 2, t.box[1] + t.box[3] / 2)
        free = [k for k in range(len(boxes)) if k not in known]
        if not free:
            break
        k = min(free, key=lambda k: _distance_to_box(centre, boxes[k]))
        if t.label not in known.values():
            known[k] = t.label
            label_text[k] = t.readings[0].text if t.readings else t.label
    names = _layout_names(boxes, known)

    issues: list[Issue] = []
    views: list[ViewRegion] = []
    for k in range(len(boxes)):
        if k not in names:
            issues.append(Issue(severity="amber", kind="label", targets=[],
                                message="A drawing on the sheet could not be named as a view."))
            continue
        mine = (geo > 0) & (nearest == k + 1) & (dist <= reach)
        ys, xs = np.nonzero(mine)
        bbox = (int(xs.min()), int(ys.min()), int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1)
        by_label = k in known
        if not by_label:
            issues.append(Issue(severity="amber", kind="label", targets=[names[k]],
                                message=f"View named {names[k]} from its position; check it."))
        views.append(ViewRegion(names[k], label_text.get(k), bbox,
                                mine.astype(np.uint8) * 255, [], "label" if by_label else "layout"))
    for t in texts:
        if t.role == "label" or not views:
            continue
        centre = (t.box[0] + t.box[2] / 2, t.box[1] + t.box[3] / 2)
        min(views, key=lambda v: _distance_to_box(centre, v.bbox)).texts.append(t)
    return views, issues
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/sketch/test_views.py -v` then `uv run ruff check .`
Expected: 4 passed. Tuning notes: if a view's outline is not seen as closed because hand-drawn corners leave gaps, raise the closing kernel (7) slightly; if FRONT dimension texts land in SIDE, check that their distance is measured to the view's final bbox (which includes its dimension lines), not the seed box.

- [ ] **Step 5: Commit**

```bash
git add s2c/sketch/views.py tests/sketch/test_views.py
git commit -m "Split the sheet into views named by label or third-angle layout"
```

---

### Task 7: Vectorize: skeleton tracing and primitive fitting

**Files:**
- Create: `s2c/sketch/vectorize.py`
- Test: `tests/sketch/test_vectorize.py`

**Interfaces:**
- Produces: dataclass `Prim(id: str, kind: "line" | "circle" | "arc" | "curve", pts: np.ndarray (N x 2, sheet px, x then y), width: float, center: tuple[float, float] | None = None, radius: float | None = None)` with properties `p0`, `p1` (np arrays), `length`, and method `direction() -> np.ndarray` (unit vector from `p0` to `p1`); `trace(mask) -> list[np.ndarray]` (skeleton paths between endpoints and junctions, plus closed loops); `circle_fit(pts) -> tuple[np.ndarray, float, float]` (centre, radius, rms); `vectorize(ink, stroke_px, view) -> list[Prim]` with ids `"{view}-p{k}"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/sketch/test_vectorize.py
import numpy as np

from s2c.sketch.vectorize import circle_fit, trace, vectorize
from tests.sketch.synth import Sheet


def prims_of(draw):
    sh = Sheet()
    draw(sh)
    return vectorize(sh.ink(), 3.0, "front")


def near(p, q, tol=5):
    return np.linalg.norm(np.asarray(p, float) - np.asarray(q, float)) <= tol


def test_rectangle_gives_four_lines_at_the_corners():
    def draw(sh):
        for p, q in [((200, 200), (600, 200)), ((600, 200), (600, 450)),
                     ((600, 450), (200, 450)), ((200, 450), (200, 200))]:
            sh.line(p, q)
    prims = prims_of(draw)
    lines = [p for p in prims if p.kind == "line"]
    assert len(lines) == 4
    corners = [(200, 200), (600, 200), (600, 450), (200, 450)]
    for c in corners:
        assert sum(near(l.p0, c) or near(l.p1, c) for l in lines) == 2


def test_circle():
    prims = prims_of(lambda sh: sh.circle((500, 400), 40))
    assert len(prims) == 1 and prims[0].kind == "circle"
    assert near(prims[0].center, (500, 400), 2) and abs(prims[0].radius - 40) < 3


def test_arc():
    import cv2

    def draw(sh):
        cv2.ellipse(sh.img, (500, 400), (80, 80), 0, 0, 120, (25, 25, 25), 3, cv2.LINE_AA)
    prims = prims_of(draw)
    assert len(prims) == 1 and prims[0].kind == "arc" and abs(prims[0].radius - 80) < 4


def test_dashed_line_becomes_short_collinear_pieces():
    prims = prims_of(lambda sh: sh.dashed((200, 300), (600, 300)))
    assert len(prims) >= 10
    assert all(p.kind == "line" and p.length < 20 for p in prims)
    assert all(abs(p.p0[1] - 300) < 3 and abs(p.p1[1] - 300) < 3 for p in prims)


def test_t_junction_splits_into_three_lines():
    def draw(sh):
        sh.line((200, 300), (600, 300))
        sh.line((400, 300), (400, 500))
    prims = prims_of(draw)
    assert len([p for p in prims if p.kind == "line" and p.length > 50]) == 3


def test_smooth_free_curve_is_a_curve():
    def draw(sh):
        xs = np.linspace(200, 800, 200)
        pts = np.stack([xs, 400 + 60 * np.sin((xs - 200) / 600 * 2 * np.pi)], 1)
        for p, q in zip(pts[:-1], pts[1:]):
            sh.line(p, q)
    prims = prims_of(draw)
    assert [p.kind for p in prims] == ["curve"]


def test_circle_fit_is_exact_on_a_circle():
    t = np.linspace(0, 2 * np.pi, 50)
    c, r, rms = circle_fit(np.stack([10 + 5 * np.cos(t), 20 + 5 * np.sin(t)], 1))
    assert np.allclose(c, [10, 20]) and abs(r - 5) < 1e-9 and rms < 1e-9


def test_trace_of_empty_mask_is_empty():
    assert trace(np.zeros((50, 50), np.uint8)) == []
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/sketch/test_vectorize.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# s2c/sketch/vectorize.py
"""Stage 4: skeleton -> paths -> lines, circles, arcs and free curves. Deterministic geometry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from skimage.morphology import skeletonize

_OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
SMOOTH_TURN_DEG = 35.0


@dataclass
class Prim:
    id: str
    kind: Literal["line", "circle", "arc", "curve"]
    pts: np.ndarray
    width: float
    center: tuple[float, float] | None = None
    radius: float | None = None

    @property
    def p0(self) -> np.ndarray:
        return self.pts[0]

    @property
    def p1(self) -> np.ndarray:
        return self.pts[-1]

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.pts, axis=0), axis=1).sum())

    def direction(self) -> np.ndarray:
        d = self.p1 - self.p0
        return d / (np.linalg.norm(d) + 1e-9)


def trace(mask: np.ndarray) -> list[np.ndarray]:
    sk = skeletonize(mask > 0)
    if not sk.any():
        return []
    h, w = sk.shape
    count = cv2.filter2D(sk.astype(np.uint8), -1, np.ones((3, 3), np.float32),
                         borderType=cv2.BORDER_CONSTANT) - sk.astype(np.uint8)
    node = sk & (count != 2)
    visited = np.zeros_like(sk)

    def neighbours(y, x):
        for dy, dx in _OFFS:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and sk[yy, xx]:
                yield yy, xx

    paths = []
    for y, x in zip(*np.nonzero(node)):
        for ny, nx in neighbours(y, x):
            if node[ny, nx]:
                if (y, x) < (ny, nx):
                    paths.append([(y, x), (ny, nx)])
                continue
            if visited[ny, nx]:
                continue
            path, prev, cur = [(y, x), (ny, nx)], (y, x), (ny, nx)
            visited[cur] = True
            while True:
                nxt = None
                for q in neighbours(*cur):
                    if q == prev:
                        continue
                    if node[q]:
                        nxt = q
                        break
                    if not visited[q]:
                        nxt = q
                if nxt is None:
                    break
                path.append(nxt)
                if node[nxt]:
                    break
                visited[nxt] = True
                prev, cur = cur, nxt
            paths.append(path)
    for y, x in zip(*np.nonzero(sk & ~node & ~visited)):  # pure loops: circles, closed outlines
        if visited[y, x]:
            continue
        path, cur = [(y, x)], (y, x)
        visited[cur] = True
        while True:
            nxt = next((q for q in neighbours(*cur) if not visited[q]), None)
            if nxt is None:
                break
            path.append(nxt)
            visited[nxt] = True
            cur = nxt
        path.append((y, x))
        paths.append(path)
    return [np.array([(x, y) for y, x in p], float) for p in paths if len(p) >= 2]


def circle_fit(pts: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Algebraic (Kasa) circle fit: centre, radius, rms distance to the circle."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.stack([x, y, np.ones_like(x)], 1)
    b = x * x + y * y
    (a, bb, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = a / 2, bb / 2
    r = float(np.sqrt(max(c + cx * cx + cy * cy, 0.0)))
    rms = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2)))
    return np.array([cx, cy]), r, rms


def _turns_deg(poly: np.ndarray) -> np.ndarray:
    d = np.diff(poly, axis=0)
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    return np.abs((np.diff(ang) + 180) % 360 - 180)


def _span_deg(pts: np.ndarray, c: np.ndarray) -> float:
    ang = np.unwrap(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
    return float(np.degrees(ang.max() - ang.min()))


def vectorize(ink: np.ndarray, stroke_px: float, view: str) -> list[Prim]:
    dist = cv2.distanceTransform((ink > 0).astype(np.uint8), cv2.DIST_L2, 3)
    prims: list[Prim] = []

    def add(kind, pts, width, center=None, radius=None):
        prims.append(Prim(f"{view}-p{len(prims)}", kind, np.asarray(pts, float), width,
                          center, radius))

    for path in trace(ink):
        length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        if length < max(3.0, stroke_px):
            continue
        ij = np.clip(path.round().astype(int), 0, [ink.shape[1] - 1, ink.shape[0] - 1])
        width = float(2 * dist[ij[:, 1], ij[:, 0]].mean())
        closed = np.linalg.norm(path[0] - path[-1]) <= 2 * stroke_px and length > 6 * stroke_px
        if closed:
            c, r, rms = circle_fit(path)
            if r > 1.5 * stroke_px and rms / r < 0.08:
                add("circle", path, width, (float(c[0]), float(c[1])), r)
                continue
            # start the loop at a corner (the point farthest from the centroid), so RDP does not
            # split one edge in two where the loop happened to start
            k = int(np.argmax(np.linalg.norm(path - path.mean(0), axis=1)))
            path = np.vstack([path[k:-1], path[:k + 1]])
        eps = max(2.0, 0.8 * stroke_px)
        simp = cv2.approxPolyDP(path.astype(np.float32).reshape(-1, 1, 2), eps, False).reshape(-1, 2)
        if len(simp) >= 4 and length > 8 * stroke_px and not closed:
            c, r, rms = circle_fit(path)
            if rms / max(r, 1e-9) < 0.05 and r < 5 * length and _span_deg(path, c) > 30:
                add("arc", path, width, (float(c[0]), float(c[1])), r)
                continue
        if len(simp) >= 5 and _turns_deg(simp).max() < SMOOTH_TURN_DEG:
            add("curve", simp, width)
            continue
        for a, b in zip(simp[:-1], simp[1:]):
            if np.linalg.norm(b - a) >= max(2.0, 0.5 * stroke_px):
                add("line", [a, b], width)
    return prims
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/sketch/test_vectorize.py -v` then `uv run ruff check .`
Expected: 8 passed. Tuning notes: if the rectangle gives more than four lines, the corner pixels of the skeleton create tiny node-to-node paths; they are dropped by the length filter, and RDP `eps` merges wobble; raise `eps` a little if needed. If the sine turns into lines, lower the RDP `eps` for curves or raise `SMOOTH_TURN_DEG`.

- [ ] **Step 5: Commit**

```bash
git add s2c/sketch/vectorize.py tests/sketch/test_vectorize.py
git commit -m "Vectorize view ink into lines, circles, arcs and free curves"
```

---

### Task 8: Classify strokes: arrowheads, hidden and centre lines, dimension, extension and leader lines

**Files:**
- Create: `s2c/sketch/classify.py`
- Test: `tests/sketch/test_classify.py`

**Interfaces:**
- Consumes: `Prim` (Task 7), `TextItem` (Task 5).
- Produces: dataclasses `Arrow(tip: np.ndarray, direction: np.ndarray, prim_id: str)` (direction points toward the tip); `DimLine(id: str, p0: np.ndarray, p1: np.ndarray, arrows: list[Arrow], prim_ids: list[str])` with property `horizontal: bool`; `Leader(id: str, tip: np.ndarray, tail: np.ndarray, prim_ids: list[str])`; `Classified(visible: list[Prim], hidden: list[Prim], centre: list[Prim], dimlines: list[DimLine], extensions: dict[str, Prim], leaders: list[Leader])` where `extensions` maps `f"{dimline_id}:{0|1}"` (the end of the dimension line) to its extension line; `classify(prims, ink, stroke_px, texts) -> Classified`.

Rules (spec 6.1):
- **Filled arrowhead**: opening the view ink with a disk slightly wider than a line leaves only filled blobs; a small compact blob touching a line end is an arrowhead. Its tip is the narrowest cross-section of the blob on the far side (this also handles two arrowheads meeting tip to tip in a chain).
- **Open arrowhead** (a hand-drawn V): two short strokes leaving a line end on opposite sides, each 15 to 60 degrees off the line.
- **Hidden** / **centre**: chains of three or more short collinear pieces; a chain whose piece lengths vary a lot (coefficient of variation above 0.6) is a centre line.
- **Dimension line**: a line carrying an arrowhead at one or both ends. **Leader**: a line with one arrowhead whose other end is near a dimension text. **Extension line**: a line roughly perpendicular to a dimension line passing through one of its arrow tips.
- **Visible**: everything else, except specks.

- [ ] **Step 1: Write the failing tests**

```python
# tests/sketch/test_classify.py
import cv2
import numpy as np

from s2c.sketch.classify import classify
from s2c.sketch.text import erase_mask, find_text_boxes, read_texts
from s2c.sketch.vectorize import vectorize
from tests.sketch.synth import Sheet, TruthReader


def run(draw):
    sh = Sheet()
    draw(sh)
    ink = sh.ink()
    texts = read_texts(sh.bgr(), find_text_boxes(ink, 3.0),
                       [TruthReader(sh.texts, "a"), TruthReader(sh.texts, "b")])
    geo = cv2.bitwise_and(ink, cv2.bitwise_not(erase_mask(texts, ink.shape)))
    return classify(vectorize(geo, 3.0, "front"), geo, 3.0, texts)


def box(sh, x0, y0, x1, y1):
    for p, q in [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]:
        sh.line(p, q)


def test_horizontal_dimension_with_extension_lines():
    def draw(sh):
        box(sh, 300, 300, 700, 500)
        sh.hdim(300, 700, 500, 560, "100")
    c = run(draw)
    assert len(c.dimlines) == 1
    d = c.dimlines[0]
    assert d.horizontal and len(d.arrows) == 2
    tips = sorted(a.tip[0] for a in d.arrows)
    assert abs(tips[0] - 300) < 5 and abs(tips[1] - 700) < 5
    assert set(c.extensions) == {f"{d.id}:0", f"{d.id}:1"}
    assert len([p for p in c.visible if p.kind == "line"]) == 4


def test_dashed_line_is_one_hidden_line():
    c = run(lambda sh: (box(sh, 300, 300, 700, 500), sh.dashed((400, 300), (400, 500))))
    assert len(c.hidden) == 1
    h = c.hidden[0]
    assert abs(h.p0[0] - 400) < 4 and abs(h.p1[0] - 400) < 4
    assert abs(min(h.p0[1], h.p1[1]) - 300) < 12 and abs(max(h.p0[1], h.p1[1]) - 500) < 12


def test_leader_to_a_circle():
    def draw(sh):
        box(sh, 300, 300, 700, 500)
        sh.circle((500, 400), 30)
        tip = np.array([500, 400]) + 30 * np.array([0.707, -0.707])
        sh.leader(tip, (tip[0] + 60, tip[1] - 60), "Ø15")
    c = run(draw)
    assert len(c.leaders) == 1
    tip = c.leaders[0].tip
    assert abs(np.hypot(tip[0] - 500, tip[1] - 400) - 30) < 6
    assert any(p.kind == "circle" for p in c.visible)


def test_chained_dimensions_share_their_middle_tip():
    def draw(sh):
        box(sh, 300, 300, 700, 500)
        sh.hdim(300, 450, 300, 250, "30")
        sh.hdim(450, 700, 300, 250, "50")
    c = run(draw)
    assert len(c.dimlines) == 2
    tips = sorted(round(a.tip[0]) for d in c.dimlines for a in d.arrows)
    assert len(tips) == 4 and abs(tips[1] - 450) < 6 and abs(tips[2] - 450) < 6


def test_open_v_arrowheads_count():
    def draw(sh):
        box(sh, 300, 300, 700, 500)
        for x in (300, 700):
            sh.line((x, 506), (x, 570), 1)
        sh.line((300, 560), (700, 560), 1)
        for tip, s in (((300, 560), 1), ((700, 560), -1)):
            sh.line(tip, (tip[0] + s * 18, tip[1] - 7), 1)
            sh.line(tip, (tip[0] + s * 18, tip[1] + 7), 1)
        sh.text("100", (500, 540))
    c = run(draw)
    assert len(c.dimlines) == 1 and len(c.dimlines[0].arrows) == 2


def test_a_lone_short_edge_is_visible_not_hidden():
    c = run(lambda sh: box(sh, 300, 300, 330, 320))
    assert c.hidden == [] and len(c.visible) == 4
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/sketch/test_classify.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# s2c/sketch/classify.py
"""Stage 5: what each stroke is. Pure geometry rules; no public model exists for hand-drawn
line types (see the research report)."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from s2c.sketch.text import TextItem
from s2c.sketch.vectorize import Prim, circle_fit

CHAIN_MIN = 3
ANGLE_TOL = np.radians(20)


@dataclass
class Arrow:
    tip: np.ndarray
    direction: np.ndarray
    prim_id: str


@dataclass
class DimLine:
    id: str
    p0: np.ndarray
    p1: np.ndarray
    arrows: list[Arrow]
    prim_ids: list[str]

    @property
    def horizontal(self) -> bool:
        d = self.p1 - self.p0
        return abs(d[0]) >= abs(d[1])


@dataclass
class Leader:
    id: str
    tip: np.ndarray
    tail: np.ndarray
    prim_ids: list[str]


@dataclass
class Classified:
    visible: list[Prim] = field(default_factory=list)
    hidden: list[Prim] = field(default_factory=list)
    centre: list[Prim] = field(default_factory=list)
    dimlines: list[DimLine] = field(default_factory=list)
    extensions: dict[str, Prim] = field(default_factory=dict)
    leaders: list[Leader] = field(default_factory=list)


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def _angle(u, v) -> float:
    """Unsigned angle between two directions, ignoring orientation (0..pi/2)."""
    c = abs(float(np.dot(_unit(u), _unit(v))))
    return float(np.arccos(min(1.0, c)))


def _point_line_dist(p, a, b) -> float:
    d = _unit(b - a)
    v = p - a
    return float(abs(v[0] * d[1] - v[1] * d[0]))


def _ends(lines: list[Prim]):
    """(prim, end index, end point, outward direction) for every line end."""
    for p in lines:
        yield p, 0, p.p0, _unit(p.p0 - p.p1)
        yield p, 1, p.p1, _unit(p.p1 - p.p0)


def _filled_arrows(lines, ink, stroke) -> dict[tuple[str, int], Arrow]:
    k = max(4, int(round(1.8 * stroke)) + 1)
    core = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(core, connectivity=8)
    blobs = [i for i in range(1, n) if 6 <= stats[i, 4] <= 60 * stroke * stroke
             and max(stats[i, 2], stats[i, 3]) <= 12 * stroke]
    out: dict[tuple[str, int], Arrow] = {}
    for i in blobs:
        ys, xs = np.nonzero(labels == i)
        pts = np.stack([xs, ys], 1).astype(float)
        best = None
        for p, end, e, d in _ends(lines):
            dist = np.min(np.linalg.norm(pts - e, axis=1))
            if dist <= 3 * stroke + 2 and (best is None or dist < best[0]):
                best = (dist, p, end, e, d)
        if best is None:
            continue
        _, p, end, e, d = best
        proj = (pts - e) @ d
        if proj.max() - proj.min() < 2:
            continue
        bins = np.round(proj).astype(int)
        lo = bins.min() + 0.3 * (bins.max() - bins.min())
        widths = {b: int((bins == b).sum()) for b in range(bins.min(), bins.max() + 1)}
        cand = [b for b in widths if b >= lo]
        tip_bin = min(cand, key=lambda b: (widths[b], -b)) if cand else bins.max()
        tip = e + d * (tip_bin + 1)
        out[(p.id, end)] = Arrow(tip=tip, direction=d, prim_id=p.id)
        # the second line end touching the same blob (chains): its own narrowest point
        for q, end2, e2, d2 in _ends(lines):
            if (q.id, end2) in out or q.id == p.id:
                continue
            if np.min(np.linalg.norm(pts - e2, axis=1)) <= 3 * stroke + 2 and np.dot(d2, d) < -0.9:
                proj2 = np.round((pts - e2) @ d2).astype(int)
                lo2 = proj2.min() + 0.3 * (proj2.max() - proj2.min())
                w2 = {b: int((proj2 == b).sum()) for b in range(proj2.min(), proj2.max() + 1)}
                c2 = [b for b in w2 if b >= lo2]
                b2 = min(c2, key=lambda b: (w2[b], -b)) if c2 else proj2.max()
                out[(q.id, end2)] = Arrow(tip=e2 + d2 * (b2 + 1), direction=d2, prim_id=q.id)
    return out


def _open_arrows(lines, stroke) -> tuple[dict[tuple[str, int], Arrow], set[str]]:
    out, barbs = {}, set()
    short = [p for p in lines if p.length <= 8 * stroke]
    for p, end, e, d in _ends(lines):
        if p.length <= 8 * stroke:
            continue
        sides = []
        for q in short:
            for qe, qo in ((q.p0, q.p1), (q.p1, q.p0)):
                if np.linalg.norm(qe - e) <= 2 * stroke + 1:
                    v = _unit(qo - qe)
                    ang = np.degrees(np.arccos(np.clip(np.dot(v, -d), -1, 1)))
                    if 15 <= ang <= 60:
                        sides.append((np.sign(d[0] * v[1] - d[1] * v[0]), q.id))
        signs = {s for s, _ in sides}
        if {1.0, -1.0} <= signs:
            out[(p.id, end)] = Arrow(tip=e.copy(), direction=d, prim_id=p.id)
            barbs |= {qid for _, qid in sides}
    return out, barbs


def _chains(pieces: list[Prim], stroke: float) -> list[list[Prim]]:
    if not pieces:
        return []
    med = float(np.median([p.length for p in pieces]))
    gap_max = max(3 * stroke, 2.5 * med)
    parent = list(range(len(pieces)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(pieces):
        for j in range(i + 1, len(pieces)):
            b = pieces[j]
            if _angle(a.direction(), b.direction()) > ANGLE_TOL:
                continue
            ends = [(ea, eb) for ea in (a.p0, a.p1) for eb in (b.p0, b.p1)]
            ea, eb = min(ends, key=lambda t: np.linalg.norm(t[0] - t[1]))
            gap = np.linalg.norm(ea - eb)
            if gap > gap_max or gap < 1e-6:
                continue
            if _angle(eb - ea, a.direction()) > ANGLE_TOL:
                continue
            parent[find(i)] = find(j)
    groups: dict[int, list[Prim]] = {}
    for i, p in enumerate(pieces):
        groups.setdefault(find(i), []).append(p)
    return [g for g in groups.values() if len(g) >= CHAIN_MIN]


def _merge_chain(chain: list[Prim], kind_id: str) -> Prim:
    pts = np.vstack([p.pts for p in chain])
    centre = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - centre)
    d = vt[0]
    t = (pts - centre) @ d
    resid = np.abs((pts - centre) @ np.array([-d[1], d[0]]))
    width = float(np.mean([p.width for p in chain]))
    if resid.max() <= 4:
        return Prim(kind_id, "line", np.array([centre + d * t.min(), centre + d * t.max()]), width)
    c, r, _ = circle_fit(pts)
    order = np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
    return Prim(kind_id, "arc", pts[order], width, (float(c[0]), float(c[1])), r)


def classify(prims: list[Prim], ink: np.ndarray, stroke_px: float, texts: list[TextItem]) -> Classified:
    out = Classified()
    s = stroke_px
    lines = [p for p in prims if p.kind == "line"]
    others = [p for p in prims if p.kind != "line"]

    arrows = _filled_arrows(lines, ink, s)
    open_arrows, barbs = _open_arrows(lines, s)
    for key, a in open_arrows.items():
        arrows.setdefault(key, a)
    lines = [p for p in lines if p.id not in barbs]

    dash_max = 12 * s
    candidates = [p for p in lines if p.length <= dash_max
                  and (p.id, 0) not in arrows and (p.id, 1) not in arrows]
    in_chain: set[str] = set()
    for k, chain in enumerate(_chains(candidates, s)):
        lengths = np.array([p.length for p in chain])
        merged = _merge_chain(chain, f"{chain[0].id}-h{k}")
        (out.centre if lengths.std() / lengths.mean() > 0.6 else out.hidden).append(merged)
        in_chain |= {p.id for p in chain}
    lines = [p for p in lines if p.id not in in_chain]

    dim_texts = [t for t in texts if t.role == "dimension"]
    carriers = [p for p in lines if (p.id, 0) in arrows or (p.id, 1) in arrows]
    used = {p.id for p in carriers}
    for p in carriers:
        ends = [e for e in (0, 1) if (p.id, e) in arrows]
        if len(ends) == 1:
            tail = p.p1 if ends[0] == 0 else p.p0
            near_text = any(np.hypot(tail[0] - (t.box[0] + t.box[2] / 2),
                                     tail[1] - (t.box[1] + t.box[3] / 2)) <= 1.5 * max(t.box[2], t.box[3])
                            for t in dim_texts)
            if near_text:
                out.leaders.append(Leader(f"{p.id}-L", arrows[(p.id, ends[0])].tip, tail, [p.id]))
                continue
        out.dimlines.append(DimLine(f"{p.id}-D", p.p0, p.p1, [arrows[(p.id, e)] for e in ends], [p.id]))

    rest = [p for p in lines if p.id not in used]
    ext_ids: set[str] = set()
    for d in out.dimlines:
        axis = d.p1 - d.p0
        for k, end in enumerate((d.p0, d.p1)):
            tip = next((a.tip for a in d.arrows if np.linalg.norm(a.tip - end) < 6 * s), end)
            best = None
            for q in rest:
                if _angle(q.direction(), axis) < np.radians(70):
                    continue
                dist = _point_line_dist(tip, q.p0, q.p1)
                t = float(np.dot(tip - q.p0, _unit(q.p1 - q.p0)))
                if dist <= 2.5 * s and -3 * s <= t <= q.length + 3 * s:
                    if best is None or dist < best[0]:
                        best = (dist, q)
            if best:
                out.extensions[f"{d.id}:{k}"] = best[1]
                ext_ids.add(best[1].id)

    # stubs: skeleton spurs of arrowheads, and the bit of an extension line that pokes past its
    # dimension line; short lines ending near an arrow tip or a dimension-line end are not edges
    anchors = [a.tip for d in out.dimlines for a in d.arrows] + [lead.tip for lead in out.leaders]
    anchors += [e for d in out.dimlines for e in (d.p0, d.p1)]

    def is_stub(p: Prim) -> bool:
        return p.kind == "line" and p.length <= 4 * s and any(
            min(np.linalg.norm(p.p0 - q), np.linalg.norm(p.p1 - q)) <= 4 * s for q in anchors)

    for p in rest + others:
        if p.id in ext_ids or p.length < 2 * s or is_stub(p):
            continue
        out.visible.append(p)
    return out
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/sketch/test_classify.py -v` then `uv run ruff check .`
Expected: 6 passed. Tuning notes: arrowheads in the synthetic sheet are 16 px long and 10 px wide on 1 px lines; if the opening kernel removes them, lower the `1.8 * stroke` factor; if outline corners survive the opening and pose as arrowheads, raise it. The `stroke_px` passed in tests is 3.0, the value `capture` measures on the synthetic sheet.

- [ ] **Step 5: Commit**

```bash
git add s2c/sketch/classify.py tests/sketch/test_classify.py
git commit -m "Classify strokes into visible, hidden, centre, dimension, extension and leader lines"
```

---

### Task 9: Reference positions and linking values to geometry

**Files:**
- Create: `s2c/sketch/link.py`
- Test: `tests/sketch/test_link.py`

**Interfaces:**
- Consumes: `Classified`, `DimLine`, `Arrow`, `Leader` (Task 8); `Prim` (Task 7); `TextItem` (Task 5); `Issue` (Task 1).
- Produces: dataclass `Ref(id: str, view: ViewName, axis: "a" | "b", px: float, sources: list[str])` where `px` is the position along the view axis in sheet pixels with `a = x` and `b = -y` (up is positive); dataclass `Link(text: TextItem, view: ViewName, kind: str, axis: "a" | "b" | None, refs: tuple[str, str] | None, circles: list[str], measures: list[str], how: "dimension_line" | "leader" | "across" | "position" | "unplaced", confidence: float)`; `build_refs(view, classified) -> dict[str, list[Ref]]` (each list sorted by `px`); `link_view(view, classified, texts, stroke_px) -> tuple[dict[str, list[Ref]], list[Link], list[Issue]]`; constants `TOL_FRAC = 0.015`, `TOL_MIN = 4.0`; helper `ref_tolerance(refs_on_axis) -> float`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/sketch/test_link.py
import cv2
import numpy as np

from s2c.sketch.classify import Arrow, Classified, DimLine, Leader, classify
from s2c.sketch.grammar import parse_text
from s2c.sketch.link import build_refs, link_view
from s2c.sketch.models import Reading
from s2c.sketch.text import TextItem, erase_mask, find_text_boxes, read_texts
from s2c.sketch.vectorize import Prim, vectorize
from s2c.sketch.views import split_views
from tests.sketch.synth import FRONT_OUTLINE, F, Sheet, TruthReader, bridge_block


def line(pid, p, q, width=3.0):
    return Prim(pid, "line", np.array([p, q], float), width)


def text(tid, s, centre, w=50, h=24):
    box = (int(centre[0] - w / 2), int(centre[1] - h / 2), w, h)
    return TextItem(tid, box, [Reading(reader="a", text=s, confidence=0.95)], parse_text(s),
                    None, "dimension", "written", 0.95, [parse_text(s).value])


def hdim(did, x1, x2, y_obj, y_line):
    d = DimLine(did, np.array([x1, y_line], float), np.array([x2, y_line], float),
                [Arrow(np.array([x1, y_line], float), np.array([-1.0, 0]), did),
                 Arrow(np.array([x2, y_line], float), np.array([1.0, 0]), did)], [did])
    ext = {f"{did}:0": line(did + "e0", (x1, y_obj + 6), (x1, y_line + 8), 1),
           f"{did}:1": line(did + "e1", (x2, y_obj + 6), (x2, y_line + 8), 1)}
    return d, ext


def front_outline():
    pts = [F(*p) for p in FRONT_OUTLINE]
    return [line(f"v{k}", p, q) for k, (p, q) in enumerate(zip(pts, pts[1:] + pts[:1]))]


def test_refs_merge_positions_closer_than_the_tolerance():
    c = Classified(visible=[line("a", (300, 100), (300, 400)), line("b", (302, 100), (302, 400)),
                            line("c", (300, 400), (600, 400))])
    refs = build_refs("front", c)
    assert [round(r.px) for r in refs["a"]] == [301, 600]
    assert set(refs["a"][0].sources) == {"a", "b", "c"}
    assert [round(r.px) for r in refs["b"]] == [-400, -100]


def test_overall_and_chain_values_follow_their_extension_lines():
    d1, e1 = hdim("d1", 200, 600, 900, 990)
    d2, e2 = hdim("d2", 300, 350, 700, 660)
    d3, e3 = hdim("d3", 350, 450, 700, 660)
    c = Classified(visible=front_outline(), dimlines=[d1, d2, d3], extensions={**e1, **e2, **e3})
    texts = [text("t1", "100", (400, 974)), text("t2", "12.5", (325, 644)), text("t3", "25", (400, 644))]
    refs, links, issues = link_view("front", c, texts, 3.0)
    px = {r.id: r.px for r in refs["a"]}
    by = {l.text.id: l for l in links}
    assert [round(px[r]) for r in by["t1"].refs] == [200, 600]
    assert [round(px[r]) for r in by["t2"].refs] == [300, 350]
    assert [round(px[r]) for r in by["t3"].refs] == [350, 450]
    assert by["t2"].refs[1] == by["t3"].refs[0]          # the chain shares a position
    assert all(l.how == "dimension_line" and l.axis == "a" for l in links) and issues == []


def test_vertical_value_links_on_axis_b():
    d = DimLine("dv", np.array([150, 700.0]), np.array([150, 850.0]),
                [Arrow(np.array([150, 700.0]), np.array([0, -1.0]), "dv"),
                 Arrow(np.array([150, 850.0]), np.array([0, 1.0]), "dv")], ["dv"])
    ext = {"dv:0": line("x0", (194, 700), (142, 700), 1), "dv:1": line("x1", (194, 850), (142, 850), 1)}
    c = Classified(visible=front_outline(), dimlines=[d], extensions=ext)
    refs, links, _ = link_view("front", c, [text("t", "37.5", (114, 775))], 3.0)
    px = {r.id: r.px for r in refs["b"]}
    assert links[0].axis == "b" and [round(px[r]) for r in links[0].refs] == [-850, -700]


def test_leader_links_a_diameter_and_its_count_to_same_size_circles():
    circles = [Prim("c1", "circle", np.zeros((3, 2)), 3, (250, 470), 25),
               Prim("c2", "circle", np.zeros((3, 2)), 3, (550, 470), 25),
               Prim("c3", "circle", np.zeros((3, 2)), 3, (400, 470), 10)]
    tip = np.array([550, 470]) + 25 * np.array([0.707, -0.707])
    lead = Leader("L", tip, tip + np.array([45, -45]), ["L"])
    c = Classified(visible=front_outline() + circles, leaders=[lead])
    t = text("t", "2xØ12.5", (tip[0] + 45, tip[1] - 61), w=90)
    _, links, _ = link_view("top", c, [t], 3.0)
    assert links[0].how == "leader" and links[0].kind == "diameter"
    assert sorted(links[0].circles) == ["c1", "c2"]


def test_value_without_a_dimension_line_links_by_position_with_an_issue():
    c = Classified(visible=front_outline())
    _, links, issues = link_view("front", c, [text("t", "100", (400, 960))], 3.0)
    assert links[0].how == "position" and links[0].axis == "a" and links[0].confidence < 0.95
    assert [i.kind for i in issues] == ["unplaced"]


def test_a_value_far_from_everything_is_kept_unplaced():
    c = Classified(visible=front_outline())
    _, links, issues = link_view("front", c, [text("t", "7", (1500, 100))], 3.0)
    assert links[0].how == "unplaced" and links[0].refs is None
    assert issues[0].kind == "unplaced"


def test_bridge_front_values_all_link_through_real_strokes():
    sh = bridge_block(Sheet())
    ink = sh.ink()
    texts = read_texts(sh.bgr(), find_text_boxes(ink, 3.0),
                       [TruthReader(sh.texts, "a"), TruthReader(sh.texts, "b")])
    views, _ = split_views(ink, texts, 3.0)
    front = next(v for v in views if v.name == "front")
    c = classify(vectorize(front.ink, 3.0, "front"), front.ink, 3.0, front.texts)
    refs, links, issues = link_view("front", c, front.texts, 3.0)
    assert len(links) == 10 and all(l.how == "dimension_line" for l in links), issues
    px = {r.id: r.px for rs in refs.values() for r in rs}
    spans = sorted(round(abs(px[l.refs[1]] - px[l.refs[0]])) for l in links)
    assert spans == [50, 50, 50, 50, 100, 100, 150, 150, 150, 400]
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/sketch/test_link.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# s2c/sketch/link.py
"""Stage 6: reference positions per view axis, and which positions or circles each value measures."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.optimize import linear_sum_assignment

from s2c.sketch.classify import Classified, DimLine
from s2c.sketch.models import Issue, ViewName
from s2c.sketch.text import TextItem

TOL_FRAC, TOL_MIN = 0.015, 4.0
INVALID = 1e6


@dataclass
class Ref:
    id: str
    view: ViewName
    axis: Literal["a", "b"]
    px: float
    sources: list[str]


@dataclass
class Link:
    text: TextItem
    view: ViewName
    kind: str
    axis: Literal["a", "b"] | None
    refs: tuple[str, str] | None
    circles: list[str] = field(default_factory=list)
    measures: list[str] = field(default_factory=list)
    how: Literal["dimension_line", "leader", "across", "position", "unplaced"] = "unplaced"
    confidence: float = 0.0


def _pos(p, axis: str) -> float:
    return float(p[0]) if axis == "a" else -float(p[1])


def ref_tolerance(refs: list[Ref]) -> float:
    extent = refs[-1].px - refs[0].px if refs else 0.0
    return max(TOL_MIN, TOL_FRAC * extent)


def build_refs(view: ViewName, c: Classified) -> dict[str, list[Ref]]:
    raw: dict[str, list[tuple[float, str]]] = {"a": [], "b": []}
    for p in c.visible + c.hidden + c.centre:
        if p.kind == "line":
            for e in (p.p0, p.p1):
                raw["a"].append((_pos(e, "a"), p.id))
                raw["b"].append((_pos(e, "b"), p.id))
        elif p.kind in ("circle", "arc"):
            cx, cy, r = p.center[0], p.center[1], p.radius
            raw["a"] += [(cx - r, p.id), (cx, p.id), (cx + r, p.id)]
            raw["b"] += [(-cy - r, p.id), (-cy, p.id), (-cy + r, p.id)]
        else:
            xs, ys = p.pts[:, 0], p.pts[:, 1]
            raw["a"] += [(float(xs.min()), p.id), (float(xs.max()), p.id)]
            raw["b"] += [(-float(ys.max()), p.id), (-float(ys.min()), p.id)]
    refs: dict[str, list[Ref]] = {}
    for axis, vals in raw.items():
        vals.sort()
        if not vals:
            refs[axis] = []
            continue
        tol = max(TOL_MIN, TOL_FRAC * (vals[-1][0] - vals[0][0]))
        clusters = [[vals[0]]]
        for v in vals[1:]:
            if v[0] - clusters[-1][0][0] <= tol:
                clusters[-1].append(v)
            else:
                clusters.append([v])
        refs[axis] = [Ref(f"{view}.{axis}{k}", view, axis, float(np.mean([v[0] for v in cl])),
                          sorted({v[1] for v in cl})) for k, cl in enumerate(clusters)]
    return refs


def _nearest(refs: list[Ref], px: float, tol: float) -> Ref | None:
    best = min(refs, key=lambda r: abs(r.px - px), default=None)
    return best if best is not None and abs(best.px - px) <= tol else None


def _centre(t: TextItem) -> np.ndarray:
    x, y, w, h = t.box
    return np.array([x + w / 2, y + h / 2], float)


def _cost(t: TextItem, d: DimLine) -> float:
    c = _centre(t)
    axis = d.p1 - d.p0
    length = float(np.linalg.norm(axis))
    u = axis / (length + 1e-9)
    s = float(np.dot(c - d.p0, u))
    perp = float(abs((c - d.p0)[0] * u[1] - (c - d.p0)[1] * u[0]))
    size = max(t.box[2], t.box[3])
    margin = 0.25 * length + size
    if s < -margin or s > length + margin or perp > 2 * size + 10:
        return INVALID
    return perp + 0.1 * abs(s - length / 2)


def _assign(texts: list[TextItem], dimlines: list[DimLine]) -> dict[str, DimLine]:
    if not texts or not dimlines:
        return {}
    cost = np.array([[_cost(t, d) for d in dimlines] for t in texts])
    rows, cols = linear_sum_assignment(cost)
    return {texts[r].id: dimlines[c] for r, c in zip(rows, cols) if cost[r, c] < INVALID}


def _linear(view, t, d: DimLine, c: Classified, refs, tol) -> Link | None:
    axis = "a" if d.horizontal else "b"
    found = []
    for k, end in enumerate((d.p0, d.p1)):
        ext = c.extensions.get(f"{d.id}:{k}")
        if ext is not None:
            px = (_pos(ext.p0, axis) + _pos(ext.p1, axis)) / 2
        else:
            tip = min((a.tip for a in d.arrows), key=lambda q: np.linalg.norm(q - end), default=end)
            px = _pos(tip if np.linalg.norm(tip - end) < 20 else end, axis)
        found.append(_nearest(refs[axis], px, 2 * tol[axis]))
    if None in found or found[0].id == found[1].id:
        return None
    a, b = sorted(found, key=lambda r: r.px)
    return Link(t, view, t.parsed.kind, axis, (a.id, b.id), [], [a.id, b.id], "dimension_line",
                t.confidence)


def _by_position(view, t, c: Classified, refs, tol) -> Link | None:
    """No dimension line: the nearest parallel visible edge on the side where the text sits."""
    if not refs["a"] or not refs["b"]:
        return None
    ca, cb = _pos(_centre(t), "a"), _pos(_centre(t), "b")
    a0, a1, b0, b1 = refs["a"][0].px, refs["a"][-1].px, refs["b"][0].px, refs["b"][-1].px
    reach = 0.25 * max(a1 - a0, b1 - b0)
    lines = [p for p in c.visible if p.kind == "line"]
    if a0 <= ca <= a1 and (b1 < cb <= b1 + reach or b0 - reach <= cb < b0):
        axis, along, across = "a", ca, cb
        cands = [p for p in lines if abs(p.p1[1] - p.p0[1]) < 0.2 * abs(p.p1[0] - p.p0[0])
                 and min(p.p0[0], p.p1[0]) <= along <= max(p.p0[0], p.p1[0])]
        key = (lambda p: abs(-p.p0[1] - across))
    elif b0 <= cb <= b1 and (a1 < ca <= a1 + reach or a0 - reach <= ca < a0):
        axis, along, across = "b", cb, ca
        cands = [p for p in lines if abs(p.p1[0] - p.p0[0]) < 0.2 * abs(p.p1[1] - p.p0[1])
                 and min(-p.p0[1], -p.p1[1]) <= along <= max(-p.p0[1], -p.p1[1])]
        key = (lambda p: abs(p.p0[0] - across))
    else:
        return None
    if not cands:
        return None
    edge = min(cands, key=key)
    r0 = _nearest(refs[axis], _pos(edge.p0, axis), 2 * tol[axis])
    r1 = _nearest(refs[axis], _pos(edge.p1, axis), 2 * tol[axis])
    if r0 is None or r1 is None or r0.id == r1.id:
        return None
    a, b = sorted((r0, r1), key=lambda r: r.px)
    return Link(t, view, t.parsed.kind, axis, (a.id, b.id), [], [a.id, b.id], "position",
                0.7 * t.confidence)


def _round(view, t, c: Classified, used: set[str], stroke) -> Link | None:
    circles = [p for p in c.visible + c.hidden if p.kind in ("circle", "arc")]
    if not circles:
        return None
    centre = _centre(t)
    reach = 1.5 * max(t.box[2], t.box[3]) + 20
    lead = min(c.leaders, key=lambda L: np.linalg.norm(L.tail - centre), default=None)
    how, target = None, None
    if lead is not None and np.linalg.norm(lead.tail - centre) <= reach:
        target = min(circles, key=lambda p: abs(np.linalg.norm(lead.tip - np.array(p.center)) - p.radius))
        if abs(np.linalg.norm(lead.tip - np.array(target.center)) - target.radius) > 2 * stroke + 3 + 0.1 * target.radius:
            target = None
        how = "leader"
    if target is None:
        for d in c.dimlines:
            if d.id in used:
                continue
            mid, length = (d.p0 + d.p1) / 2, float(np.linalg.norm(d.p1 - d.p0))
            for p in circles:
                if np.linalg.norm(mid - np.array(p.center)) <= 0.3 * p.radius \
                        and abs(length - 2 * p.radius) <= 0.15 * 2 * p.radius and _cost(t, d) < INVALID:
                    target, how = p, "across"
                    used.add(d.id)
                    break
            if target is not None:
                break
    if target is None:
        return None
    group = [target.id]
    if t.parsed.count > 1:
        group = [p.id for p in circles if p.kind == "circle"
                 and abs(p.radius - target.radius) <= 0.08 * target.radius]
    return Link(t, view, t.parsed.kind, None, None, group, list(group), how, t.confidence)


def link_view(view: ViewName, c: Classified, texts: list[TextItem], stroke_px: float):
    refs = build_refs(view, c)
    tol = {axis: ref_tolerance(rs) for axis, rs in refs.items()}
    dims = [t for t in texts if t.role == "dimension" and t.parsed is not None]
    linear = [t for t in dims if t.parsed.kind == "linear"]
    assigned = _assign(linear, c.dimlines)
    used = {d.id for d in assigned.values()}
    links: list[Link] = []
    issues: list[Issue] = []
    for t in dims:
        link = None
        if t.parsed.kind == "linear":
            d = assigned.get(t.id)
            link = _linear(view, t, d, c, refs, tol) if d else None
            if link is None:
                link = _by_position(view, t, c, refs, tol)
                if link is not None:
                    issues.append(Issue(severity="amber", kind="unplaced", targets=[t.id],
                                        message=f"{t.readings[0].text} linked by its position only; check it."))
        elif t.parsed.kind in ("diameter", "radius", "thread"):
            link = _round(view, t, c, used, stroke_px)
        if link is None:
            link = Link(t, view, t.parsed.kind, None, None, how="unplaced", confidence=t.confidence)
            issues.append(Issue(severity="amber", kind="unplaced", targets=[t.id],
                                message=f"Could not tell what {t.readings[0].text} measures."))
        links.append(link)
    return refs, links, issues
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/sketch/test_link.py -v` then `uv run ruff check .`
Expected: 7 passed. If the integration test fails, print `issues` and the `how` of each link first: a `position` or `unplaced` link means the classifier missed that dimension line or its extension lines (fix in Task 8's thresholds), not the linker. Angle and chamfer values stay `unplaced` in this plan; the second plan links them.

- [ ] **Step 5: Commit**

```bash
git add s2c/sketch/link.py tests/sketch/test_link.py
git commit -m "Link written values to reference positions and circles"
```

---

### Task 10: Solve: one least-squares system across all views, with badges

**Files:**
- Create: `s2c/sketch/solve.py`
- Test: `tests/sketch/test_solve.py`

**Interfaces:**
- Consumes: `Ref`, `Link` (Task 9); `Prim` (Task 7); `Issue`, `Size`, `Badge`, `Evidence`, `ViewName` (Task 1).
- Produces:
  - `GROUPS`: global axis to `(view, local axis, orientation)`; orientation `-1` marks a mirrored view axis (spec 7.2): `x: front.a, top.a, bottom.a, back.a(-1)`; `y: front.b, right.b, left.b, back.b`; `z` (distance from the front face): `right.a, top.b, left.a(-1), bottom.b(-1)`.
  - dataclasses `DimResult(value: float | None, badge: Badge, evidence: Evidence, implied: float | None, candidates: list[float])`, `PredGap(view, axis, refs: tuple[str, str], value: float)`, `Solved(mm: dict[str, float], scale: dict[tuple[str, str], float], dims: dict[str, DimResult], circle_d: dict[str, tuple[float, Badge]], envelope: dict[str, Size], gaps: list[PredGap], issues: list[Issue])`. `mm` holds every ref's position in millimetres, measured from the lowest ref of its view axis. `dims` is keyed by `TextItem.id`.
  - `solve(views: dict[ViewName, dict[str, list[Ref]]], links: list[Link], circles: dict[str, Prim]) -> Solved`.
  - `px_to_mm(solved, refs_on_axis, view, axis, px) -> float` (piecewise-linear through the refs, extrapolated with the view scale).

Method (spec 7.3 to 7.5): unknowns are the refs' positions in mm. Hard rows: each view axis's lowest ref is 0; matched refs across views measure the same distance; every written value; every circle diameter. Soft rows: consecutive refs keep the drawing's proportions at the view axis's scale. Hard rows get weight 1000, soft rows 1, solved together with `numpy.linalg.lstsq`. Conflicts are found on the hard rows alone by leave-one-out; ties go to the value that disagrees most with the drawing's proportions. A size is `derived` when its row vector lies in the row space of the consistent hard rows (`arithmetic` without cross-view rows, `cross_view` with them), otherwise `predicted` from proportions.

- [ ] **Step 1: Write the failing tests**

```python
# tests/sketch/test_solve.py
import numpy as np
import pytest

from s2c.sketch.grammar import parse_text
from s2c.sketch.link import Link, Ref
from s2c.sketch.models import Reading
from s2c.sketch.solve import px_to_mm, solve
from s2c.sketch.text import TextItem
from s2c.sketch.vectorize import Prim


def refs_of(view, axis, pxs, sources=None):
    return [Ref(f"{view}.{axis}{k}", view, axis, float(p), (sources or {}).get(k, []))
            for k, p in enumerate(pxs)]


def lin(view, axis, refs, i, j, s, badge="written", candidates=None):
    p = parse_text(s)
    t = TextItem(f"{view}.{axis}.{i}-{j}", (0, 0, 10, 10),
                 [Reading(reader="a", text=s, confidence=0.95)], p, None, "dimension", badge, 0.95,
                 candidates or [p.value])
    return Link(t, view, "linear", axis, (refs[i].id, refs[j].id), [], [], "dimension_line", 0.95)


FRONT_A = [200, 300, 350, 450, 500, 600]      # 4 px per mm: 0 25 37.5 62.5 75 100
FRONT_B = [-900, -850, -800, -700]            # 0 12.5 25 50


def front(values=None, badges=None, candidates=None):
    a, b = refs_of("front", "a", FRONT_A), refs_of("front", "b", FRONT_B)
    spec = [(0, 5, "100"), (0, 2, "37.5"), (3, 5, "37.5"), (1, 2, "12.5"), (2, 3, "25"), (3, 4, "12.5")]
    values, badges, candidates = values or {}, badges or {}, candidates or {}
    links = [lin("front", "a", a, i, j, values.get(k, s), badges.get(k, "written"), candidates.get(k))
             for k, (i, j, s) in enumerate(spec)]
    links += [lin("front", "b", b, 1, 3, "37.5"), lin("front", "b", b, 2, 3, "25"),
              lin("front", "b", b, 1, 2, "12.5"), lin("front", "b", b, 0, 1, "12.5")]
    return {"front": {"a": a, "b": b}}, links


def test_front_solves_exactly_and_the_upright_start_is_derived():
    views, links = front()
    s = solve(views, links, {})
    got = [s.mm[f"front.a{k}"] for k in range(6)]
    assert got == pytest.approx([0, 25, 37.5, 62.5, 75, 100], abs=1e-3)
    assert all(d.badge == "written" for d in s.dims.values())
    assert s.gaps == []                                  # every gap follows from written values
    assert s.envelope["x"].value == pytest.approx(100) and s.envelope["x"].badge == "written"
    assert s.envelope["y"].value == pytest.approx(50) and s.envelope["y"].badge == "derived"


def test_a_conflict_marks_the_value_that_disagrees_with_the_drawing():
    views, links = front(values={4: "35"})              # 37.5 + 35 + 37.5 != 100
    s = solve(views, links, {})
    bad = [k for k, d in s.dims.items() if d.badge == "conflict"]
    assert bad == ["front.a.2-3"]
    assert any(i.kind == "conflict" and i.severity == "red" for i in s.issues)


def test_an_uncertain_value_gets_the_candidate_the_geometry_implies():
    views, links = front(badges={4: "uncertain"}, candidates={4: [85.0, 25.0]}, values={4: "85"})
    s = solve(views, links, {})
    d = s.dims["front.a.2-3"]
    assert d.badge == "uncertain" and d.value == pytest.approx(25)
    assert d.implied == pytest.approx(25, abs=0.5)


def test_the_decimal_trap_suggests_the_scaled_value():
    a = refs_of("front", "a", [0, 100, 200, 300])
    links = [lin("front", "a", a, 0, 1, "50"), lin("front", "a", a, 1, 2, "50"),
             lin("front", "a", a, 2, 3, "5")]
    s = solve({"front": {"a": a, "b": refs_of("front", "b", [-100, 0])}}, links, {})
    issue = next(i for i in s.issues if i.kind == "decimal")
    assert "50" in issue.message and issue.targets == ["front.a.2-3"]


def test_the_top_width_is_derived_across_views():
    views, links = front()
    views["top"] = {"a": refs_of("top", "a", [200, 250, 300, 350, 450, 500, 550, 600]),
                    "b": refs_of("top", "b", [-520, -470, -420])}
    s = solve(views, links, {})
    assert s.mm["top.a7"] == pytest.approx(100, abs=1e-3)
    assert s.mm["top.a1"] == pytest.approx(12.5, abs=0.5)     # not written: proportion only
    assert any(g.refs == ("top.a0", "top.a1") for g in s.gaps)


def test_an_unwritten_axis_is_predicted_from_proportions():
    a = refs_of("front", "a", [200, 600])
    b = refs_of("front", "b", [-900, -700])
    s = solve({"front": {"a": a, "b": b}}, [lin("front", "a", a, 0, 1, "100")], {})
    assert s.envelope["y"].badge == "predicted" and s.envelope["y"].evidence == "proportion"
    assert s.envelope["y"].value == pytest.approx(50)
    assert "z" not in s.envelope and any(i.kind == "predicted" for i in s.issues)


def test_a_circle_diameter_fixes_its_width():
    c = Prim("top-p9", "circle", np.zeros((3, 2)), 3, (250.0, 470.0), 25.0)
    a = refs_of("top", "a", [200, 225, 250, 275, 600], {1: ["top-p9"], 2: ["top-p9"], 3: ["top-p9"]})
    b = refs_of("top", "b", [-520, -495, -470, -445, -420], {1: ["top-p9"], 2: ["top-p9"], 3: ["top-p9"]})
    t = TextItem("t", (0, 0, 9, 9), [Reading(reader="a", text="Ø12.5", confidence=0.9)],
                 parse_text("Ø12.5"), None, "dimension", "written", 0.9, [12.5])
    links = [Link(t, "top", "diameter", None, None, ["top-p9"], ["top-p9"], "leader", 0.9),
             lin("top", "a", a, 0, 4, "100")]
    s = solve({"top": {"a": a, "b": b}}, links, {"top-p9": c})
    assert s.mm["top.a3"] - s.mm["top.a1"] == pytest.approx(12.5, abs=1e-3)
    assert s.circle_d["top-p9"] == (pytest.approx(12.5), "written")


def test_a_mirrored_left_view_matches_the_right_view():
    right = refs_of("right", "a", [800, 850, 900])     # distance from the front face grows rightward
    left = refs_of("left", "a", [0, 50, 100])          # mirrored: the front face is on its right
    y = refs_of("right", "b", [-900, -700])
    ly = refs_of("left", "b", [-900, -700])
    links = [lin("right", "a", right, 0, 2, "25"), lin("left", "a", left, 1, 2, "15")]
    s = solve({"right": {"a": right, "b": y}, "left": {"a": left, "b": ly}}, links, {})
    # left a1..a2 (near the front face) mirrors right a0..a1
    assert s.mm["right.a1"] == pytest.approx(15, abs=1e-3)


def test_px_to_mm_interpolates_between_refs():
    views, links = front()
    s = solve(views, links, {})
    assert px_to_mm(s, views["front"]["a"], "front", "a", 400) == pytest.approx(50, abs=1e-3)
    assert px_to_mm(s, views["front"]["a"], "front", "a", 640) == pytest.approx(110, abs=0.5)
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/sketch/test_solve.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# s2c/sketch/solve.py
"""Stages 7 and 8 (ranks 1, 2 and 5): one least-squares system over every view's reference positions.
Written values are exact, shared views agree, the drawing's proportions fill the rest."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from s2c.sketch.link import Link, Ref
from s2c.sketch.models import Badge, Evidence, Issue, Size, ViewName
from s2c.sketch.vectorize import Prim

GROUPS: dict[str, list[tuple[str, str, int]]] = {
    "x": [("front", "a", 1), ("top", "a", 1), ("bottom", "a", 1), ("back", "a", -1)],
    "y": [("front", "b", 1), ("right", "b", 1), ("left", "b", 1), ("back", "b", 1)],
    "z": [("right", "a", 1), ("top", "b", 1), ("left", "a", -1), ("bottom", "b", -1)],
}
HARD, SOFT = 1e3, 1.0
MATCH_TOL = 0.03            # of the view extent, for cross-view matching
DECIMAL_RATIO = 3.0
FALLBACK_SCALE = 0.25       # mm per px when the sheet has no value at all


@dataclass
class DimResult:
    value: float | None
    badge: Badge
    evidence: Evidence
    implied: float | None = None
    candidates: list[float] = field(default_factory=list)


@dataclass
class PredGap:
    view: ViewName
    axis: str
    refs: tuple[str, str]
    value: float


@dataclass
class Solved:
    mm: dict[str, float]
    scale: dict[tuple[str, str], float]
    dims: dict[str, DimResult]
    circle_d: dict[str, tuple[float, Badge]]
    envelope: dict[str, Size]
    gaps: list[PredGap]
    issues: list[Issue]


@dataclass
class _Row:
    coef: dict[str, float]
    rhs: float
    kind: str          # anchor | match | dim | circle | soft
    tag: str = ""      # text id for dim and circle rows


class _System:
    def __init__(self, ids: list[str]):
        self.ix = {r: k for k, r in enumerate(ids)}
        self.rows: list[_Row] = []

    def add(self, coef, rhs, kind, tag=""):
        self.rows.append(_Row(coef, rhs, kind, tag))

    def matrix(self, rows: list[_Row]) -> tuple[np.ndarray, np.ndarray]:
        A = np.zeros((len(rows), len(self.ix)))
        b = np.zeros(len(rows))
        for i, row in enumerate(rows):
            for rid, c in row.coef.items():
                A[i, self.ix[rid]] += c
            b[i] = row.rhs
        return A, b

    def vector(self, coef: dict[str, float]) -> np.ndarray:
        v = np.zeros(len(self.ix))
        for rid, c in coef.items():
            v[self.ix[rid]] += c
        return v


def _lstsq(A, b):
    if A.size == 0:
        return np.zeros(A.shape[1])
    return np.linalg.lstsq(A, b, rcond=None)[0]


def _in_rowspace(A: np.ndarray, v: np.ndarray) -> bool:
    if A.size == 0:
        return False
    x = np.linalg.lstsq(A.T, v, rcond=None)[0]
    return float(np.linalg.norm(A.T @ x - v)) <= 1e-6 * max(1.0, float(np.linalg.norm(v)))


def _norm(rs: list[Ref], orient: int) -> list[float]:
    lo, hi = rs[0].px, rs[-1].px
    span = max(hi - lo, 1e-9)
    return [(r.px - lo) / span if orient > 0 else (hi - r.px) / span for r in rs]


def _match(rs1, o1, rs2, o2) -> list[tuple[Ref, Ref]]:
    n1 = sorted(zip(_norm(rs1, o1), rs1), key=lambda t: t[0])
    n2 = sorted(zip(_norm(rs2, o2), rs2), key=lambda t: t[0])
    i = j = 0
    out = []
    while i < len(n1) and j < len(n2):
        if abs(n1[i][0] - n2[j][0]) <= MATCH_TOL:
            out.append((n1[i][1], n2[j][1]))
            i, j = i + 1, j + 1
        elif n1[i][0] < n2[j][0]:
            i += 1
        else:
            j += 1
    return out


def _distance(rs: list[Ref], r: Ref, orient: int, sign: float) -> dict[str, float]:
    """Coefficients of the distance from the axis origin face: u, or u_max - u when mirrored."""
    if orient > 0:
        return {r.id: sign}
    coef = {rs[-1].id: sign}
    coef[r.id] = coef.get(r.id, 0.0) - sign
    return coef


def _merge(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0.0) + v
    return out


def _circle_refs(rs: list[Ref], circle_id: str, lo_px: float, hi_px: float):
    mine = [r for r in rs if circle_id in r.sources]
    if len(mine) < 2:
        return None
    lo = min(mine, key=lambda r: abs(r.px - lo_px))
    hi = min(mine, key=lambda r: abs(r.px - hi_px))
    return (lo, hi) if lo.id != hi.id else None


def _circle_rows(link: Link, value: float, views, circles) -> list[tuple[dict[str, float], float]]:
    rows = []
    for cid in link.circles:
        c = circles.get(cid)
        if c is None or link.view not in views:
            continue
        cx, cy, r = c.center[0], c.center[1], c.radius
        span = value if link.kind != "radius" else 2 * value
        for axis, lo, hi in (("a", cx - r, cx + r), ("b", -cy - r, -cy + r)):
            pair = _circle_refs(views[link.view].get(axis, []), cid, lo, hi)
            if pair:
                rows.append(({pair[1].id: 1.0, pair[0].id: -1.0}, span))
    return rows


def _scales(views, links, issues) -> dict[tuple[str, str], float]:
    ref = {r.id: r for axes in views.values() for rs in axes.values() for r in rs}
    ratios: dict[tuple[str, str], list[tuple[float, Link]]] = {}
    for L in links:
        if L.refs and L.kind == "linear" and L.text.badge == "written":
            dpx = abs(ref[L.refs[1]].px - ref[L.refs[0]].px)
            if dpx > 0:
                ratios.setdefault((L.view, L.axis), []).append((L.text.parsed.value / dpx, L))
    scale = {k: float(np.median([r for r, _ in v])) for k, v in ratios.items()}
    for key, vals in ratios.items():  # decimal trap (spec 7.5)
        for k, (r, L) in enumerate(vals):
            others = [o for m, (o, _) in enumerate(vals) if m != k]
            if not others:
                continue
            med = float(np.median(others))
            if r / med > DECIMAL_RATIO or med / r > DECIMAL_RATIO:
                dpx = L.text.parsed.value / r
                alts = [L.text.parsed.value * f for f in (10, 0.1, 0.01, 100)]
                best = min(alts, key=lambda a: abs(a - med * dpx))
                issues.append(Issue(severity="amber", kind="decimal", targets=[L.text.id],
                                    message=f"{L.text.readings[0].text} looks off scale; did you mean {best:g}?"))
    everything = [r for v in ratios.values() for r, _ in v]
    overall = float(np.median(everything)) if everything else FALLBACK_SCALE
    if not everything:
        issues.append(Issue(severity="red", kind="predicted", targets=[],
                            message="No values found. Write at least the overall width, height and depth."))
    for v, axes in views.items():
        for ax in axes:
            if (v, ax) in scale:
                continue
            partner = [scale[(pv, pa)] for g in GROUPS.values() if (v, ax) in [(m[0], m[1]) for m in g]
                       for pv, pa, _ in g if (pv, pa) in scale]
            scale[(v, ax)] = partner[0] if partner else overall
    return scale


def solve(views: dict[ViewName, dict[str, list[Ref]]], links: list[Link],
          circles: dict[str, Prim]) -> Solved:
    issues: list[Issue] = []
    ids = [r.id for axes in views.values() for rs in axes.values() for r in rs]
    system = _System(ids)
    scale = _scales(views, links, issues)

    for v, axes in views.items():
        for ax, rs in axes.items():
            if rs:
                system.add({rs[0].id: 1.0}, 0.0, "anchor")
            for r0, r1 in zip(rs, rs[1:]):
                system.add({r1.id: 1.0, r0.id: -1.0}, scale[(v, ax)] * (r1.px - r0.px), "soft")
    for members in GROUPS.values():
        present = [(v, ax, o) for v, ax, o in members if views.get(v, {}).get(ax)]
        for v2, ax2, o2 in present[1:]:
            v1, ax1, o1 = present[0]
            rs1, rs2 = views[v1][ax1], views[v2][ax2]
            for r1, r2 in _match(rs1, o1, rs2, o2):
                system.add(_merge(_distance(rs1, r1, o1, 1.0), _distance(rs2, r2, o2, -1.0)), 0.0, "match")

    dim_rows: dict[str, list[tuple[dict[str, float], float]]] = {}
    for L in links:
        if L.text.parsed is None:
            continue
        value = L.text.parsed.value
        if L.refs and L.kind == "linear":
            dim_rows[L.text.id] = [({L.refs[1]: 1.0, L.refs[0]: -1.0}, value)]
        elif L.circles:
            dim_rows[L.text.id] = _circle_rows(L, value, views, circles)
    by_id = {L.text.id: L for L in links}
    written = [tid for tid in dim_rows if by_id[tid].text.badge == "written" and dim_rows[tid]]
    uncertain = [tid for tid in dim_rows if by_id[tid].text.badge == "uncertain" and dim_rows[tid]]

    def rows_for(tids, extra=None):
        rows = [r for r in system.rows if r.kind in ("anchor", "match")]
        for tid in tids:
            rows += [_Row(c, (extra or {}).get(tid, rhs), "dim", tid) for c, rhs in dim_rows[tid]]
        return rows

    # conflicts on the hard rows alone (leave-one-out, ties to the value most off the drawing)
    def worst(tids) -> float:
        A, b = system.matrix(rows_for(tids))
        x = _lstsq(A, b)
        res = np.abs(A @ x - b)
        tol = np.array([max(0.05, 0.005 * abs(rhs)) for rhs in b])
        return float(np.max(res - tol)) if len(b) else -1.0

    consistent = list(written)
    conflicts: list[str] = []
    ref = {r.id: r for axes in views.values() for rs in axes.values() for r in rs}

    def off_drawing(tid) -> float:
        L = by_id[tid]
        if not L.refs:
            return 0.0
        dpx = abs(ref[L.refs[1]].px - ref[L.refs[0]].px)
        drawn = scale[(L.view, L.axis)] * dpx
        return abs(L.text.parsed.value - drawn) / max(L.text.parsed.value, 1e-9)

    while consistent and worst(consistent) > 0:
        # every removal that leaves a consistent set scores 0, so the tie-break decides among them
        trials = [(max(0.0, round(worst([t for t in consistent if t != tid]), 6)), -off_drawing(tid), tid)
                  for tid in consistent]
        _, _, drop = min(trials)
        consistent.remove(drop)
        conflicts.append(drop)

    # implied sizes for uncertain values (spec 8.2), then the final solve
    def full(tids, extra=None):
        hard = rows_for(tids, extra)
        soft = [r for r in system.rows if r.kind == "soft"]
        A_h, b_h = system.matrix(hard)
        A_s, b_s = system.matrix(soft)
        return _lstsq(np.vstack([HARD * A_h, SOFT * A_s]), np.concatenate([HARD * b_h, SOFT * b_s]))

    first = full(consistent)
    proposals: dict[str, float] = {}
    implied: dict[str, float] = {}
    for tid in uncertain:
        coef, _ = dim_rows[tid][0]
        imp = float(system.vector(coef) @ first)
        implied[tid] = imp
        cands = list(by_id[tid].text.candidates) or [by_id[tid].text.parsed.value]
        cands += [c * f for c in list(cands) for f in (10, 0.1, 0.01)]
        proposals[tid] = min(cands, key=lambda c: abs(c - imp))
    sol = full(consistent + list(proposals), proposals)
    mm = {rid: float(sol[k]) for rid, k in system.ix.items()}

    A_all, _ = system.matrix(rows_for(consistent))
    A_own, _ = system.matrix([r for r in rows_for(consistent) if r.kind != "match"])

    def badge_of(coef) -> tuple[Badge, Evidence]:
        vec = system.vector(coef)
        if _in_rowspace(A_own, vec):
            return "derived", "arithmetic"
        if _in_rowspace(A_all, vec):
            return "derived", "cross_view"
        return "predicted", "proportion"

    dims: dict[str, DimResult] = {}
    for L in links:
        tid = L.text.id
        value = L.text.parsed.value if L.text.parsed else None
        if tid in conflicts:
            dims[tid] = DimResult(value, "conflict", "reader", candidates=list(L.text.candidates))
            issues.append(Issue(severity="red", kind="conflict", targets=[tid],
                                message=f"{L.text.readings[0].text} contradicts other written values."))
        elif tid in proposals:
            dims[tid] = DimResult(proposals[tid], "uncertain", "reader", implied[tid],
                                  list(L.text.candidates))
            issues.append(Issue(severity="amber", kind="uncertain", targets=[tid],
                                message=f"Readers disagree on {L.text.readings[0].text}; "
                                        f"the drawing suggests {proposals[tid]:g}."))
        else:
            dims[tid] = DimResult(value, L.text.badge or "uncertain", "reader",
                                  candidates=list(L.text.candidates))

    circle_d: dict[str, tuple[float, Badge]] = {}
    for L in links:
        for cid in L.circles:
            if L.text.id in consistent:
                d = L.text.parsed.value * (2 if L.kind == "radius" else 1)
                circle_d[cid] = (d, "written")
    for cid, c in circles.items():
        if cid in circle_d:
            continue
        view = next((v for v in views if cid.startswith(f"{v}-")), None)
        pair = _circle_refs(views.get(view, {}).get("a", []), cid, c.center[0] - c.radius,
                            c.center[0] + c.radius) if view else None
        if pair:
            b, _ = badge_of({pair[1].id: 1.0, pair[0].id: -1.0})
            circle_d[cid] = (mm[pair[1].id] - mm[pair[0].id], b)

    gaps: list[PredGap] = []
    for v, axes in views.items():
        for ax, rs in axes.items():
            for r0, r1 in zip(rs, rs[1:]):
                b, _ = badge_of({r1.id: 1.0, r0.id: -1.0})
                if b == "predicted":
                    gaps.append(PredGap(v, ax, (r0.id, r1.id), mm[r1.id] - mm[r0.id]))

    envelope: dict[str, Size] = {}
    direct = {tuple(sorted(by_id[t].refs)) for t in consistent if by_id[t].refs}
    for g, members in GROUPS.items():
        present = [(v, ax) for v, ax, _ in members if views.get(v, {}).get(ax)]
        if not present:
            issues.append(Issue(severity="red", kind="predicted", targets=[],
                                message=f"No view shows the {g.upper()} size; draw it or type it."))
            continue
        v, ax = present[0]
        lo, hi = views[v][ax][0], views[v][ax][-1]
        value = mm[hi.id] - mm[lo.id]
        if tuple(sorted((lo.id, hi.id))) in direct:
            envelope[g] = Size(value=value, badge="written", evidence="reader")
        else:
            b, e = badge_of({hi.id: 1.0, lo.id: -1.0})
            envelope[g] = Size(value=value, badge=b, evidence=e)
            if b == "predicted":
                issues.append(Issue(severity="amber", kind="predicted", targets=[g],
                                    message=f"The overall {g.upper()} size is predicted from the drawing; confirm it."))
    return Solved(mm, scale, dims, circle_d, envelope, gaps, issues)


def px_to_mm(solved: Solved, refs: list[Ref], view: str, axis: str, px: float) -> float:
    if not refs:
        return 0.0
    xs = np.array([r.px for r in refs])
    ys = np.array([solved.mm[r.id] for r in refs])
    s = solved.scale.get((view, axis), FALLBACK_SCALE)
    if px < xs[0]:
        return float(ys[0] - s * (xs[0] - px))
    if px > xs[-1]:
        return float(ys[-1] + s * (px - xs[-1]))
    return float(np.interp(px, xs, ys))
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/sketch/test_solve.py -v` then `uv run ruff check .`
Expected: 9 passed. Notes: `A_own` treats all non-match rows as "own view" evidence, so a size fixed by a value in the same view is `arithmetic` and one needing another view is `cross_view`. If the conflict test drops the wrong value, print `trials`: every candidate removal makes the rest consistent, so the tie-break (`off_drawing`) decides; `35` is the one that disagrees with its 100 px span at 0.25 mm per px.

- [ ] **Step 5: Commit**

```bash
git add s2c/sketch/solve.py tests/sketch/test_solve.py
git commit -m "Solve exact sizes across views with badges, conflicts and proposals"
```

---

### Task 11: Holes from circles and hidden lines

**Files:**
- Create: `s2c/sketch/features.py`
- Test: `tests/sketch/test_features.py`

**Interfaces:**
- Consumes: `Classified` (Task 8); `Ref`, `build_refs`, `Link` (Task 9); `Solved`, `GROUPS`, `px_to_mm`, `solve` (Task 10); `Feature`, `Issue` (Task 1).
- Produces: `LOS` (view to the global axis it looks along), `AXES` (view to `{local axis: global axis}`), `find_holes(classified: dict[ViewName, Classified], refs: dict[ViewName, dict[str, list[Ref]]], solved: Solved) -> tuple[list[Feature], list[Issue]]`; helper `to_distance(solved, refs, view, axis, px) -> float` (distance along the global axis, mirrored views handled). `Feature.position_mm` is `(X, Y, Z)` in the team's frame: X from the left, Y from the bottom, Z with the front face at `Z = depth` (team multi-view spec section 3); along the hole axis it is the end nearest the viewer of the view showing the circle.

Rule (spec 9): a circle in view A; in another view B that looks across the hole axis and shares one axis with A, two lines running along the hole axis at the circle's centre ± radius. Their common extent is the hole's extent. Both ends on a visible face: through; otherwise blind with that depth. No pair in any view: a predicted through hole with an amber issue (spec 8.3).

- [ ] **Step 1: Write the failing tests**

```python
# tests/sketch/test_features.py
import numpy as np
import pytest

from s2c.sketch.classify import Classified
from s2c.sketch.features import find_holes
from s2c.sketch.grammar import parse_text
from s2c.sketch.link import Link, build_refs
from s2c.sketch.models import Reading
from s2c.sketch.solve import solve
from s2c.sketch.text import TextItem
from s2c.sketch.vectorize import Prim


def seg(pid, p, q):
    return Prim(pid, "line", np.array([p, q], float), 3.0)


def rect(prefix, x0, y0, x1, y1):
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return [seg(f"{prefix}{k}", p, q) for k, (p, q) in enumerate(zip(corners, corners[1:] + corners[:1]))]


def item(tid, s):
    p = parse_text(s)
    return TextItem(tid, (0, 0, 9, 9), [Reading(reader="a", text=s, confidence=0.95)], p, None,
                    "dimension", "written", 0.95, [p.value])


def lin(refs, view, axis, px0, px1, s):
    rs = refs[view][axis]
    r0 = min(rs, key=lambda r: abs(r.px - px0))
    r1 = min(rs, key=lambda r: abs(r.px - px1))
    return Link(item(f"{view}{axis}{s}", s), view, "linear", axis, (r0.id, r1.id), [], [],
                "dimension_line", 0.95)


def setup(pair_y=(820, 900), with_pair=True):
    """FRONT 100 x 20 mm, TOP 100 x 40 mm, 4 px per mm; a Ø10 hole at X 50, 20 mm from the front face."""
    front = Classified(visible=rect("front-v", 200, 820, 600, 900))
    if with_pair:
        front.hidden = [seg("front-h0", (380, pair_y[0]), (380, pair_y[1])),
                        seg("front-h1", (420, pair_y[0]), (420, pair_y[1]))]
    circle = Prim("top-c", "circle", np.zeros((3, 2)), 3.0, (400.0, 440.0), 20.0)
    top = Classified(visible=rect("top-v", 200, 360, 600, 520) + [circle])
    cls = {"front": front, "top": top}
    refs = {v: build_refs(v, c) for v, c in cls.items()}
    links = [lin(refs, "front", "a", 200, 600, "100"), lin(refs, "front", "b", -900, -820, "20"),
             lin(refs, "top", "b", -520, -360, "40"),
             Link(item("dia", "Ø10"), "top", "diameter", None, None, ["top-c"], ["top-c"], "leader", 0.95)]
    solved = solve(refs, links, {"top-c": circle})
    return cls, refs, solved


def test_through_hole_from_a_circle_and_a_full_hidden_pair():
    cls, refs, solved = setup()
    feats, issues = find_holes(cls, refs, solved)
    assert len(feats) == 1 and issues == []
    h = feats[0]
    assert h.type == "hole" and h.axis == "y" and h.through is True and h.depth is None
    assert h.diameter == pytest.approx(10) and h.badge == "written"
    assert h.position_mm == pytest.approx((50, 20, 20), abs=0.3)
    assert set(h.evidence) == {"top-c", "front-h0", "front-h1"}


def test_blind_hole_depth_from_the_hidden_pair():
    cls, refs, solved = setup(pair_y=(820, 860))
    feats, _ = find_holes(cls, refs, solved)
    assert feats[0].through is False and feats[0].depth == pytest.approx(10, abs=0.3)


def test_circle_without_hidden_lines_is_a_predicted_through_hole():
    cls, refs, solved = setup(with_pair=False)
    feats, issues = find_holes(cls, refs, solved)
    assert feats[0].badge == "predicted" and feats[0].through is True
    assert [i.kind for i in issues] == ["predicted"]
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/sketch/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# s2c/sketch/features.py
"""Stage 9: holes from a circle in one view and a pair of lines along its axis in another."""
from __future__ import annotations

import numpy as np

from s2c.sketch.classify import Classified
from s2c.sketch.link import Ref
from s2c.sketch.models import Feature, Issue, ViewName
from s2c.sketch.solve import GROUPS, Solved, px_to_mm

LOS = {"front": "z", "back": "z", "top": "y", "bottom": "y", "right": "x", "left": "x"}
AXES = {v: {ax: g for g, members in GROUPS.items() for mv, ax, _ in members if mv == v} for v in LOS}
NEAR_VIEWER = {"front": min, "back": max, "top": max, "bottom": min, "right": max, "left": min}
ORDER = ["front", "top", "right", "left", "bottom", "back"]


def _orient(view: str, axis: str) -> int:
    return next((o for members in GROUPS.values() for v, ax, o in members if v == view and ax == axis), 1)


def _axis_px(point, axis: str) -> float:
    return float(point[0]) if axis == "a" else -float(point[1])


def to_distance(solved: Solved, refs, view: str, axis: str, px: float) -> float:
    rs = refs[view][axis]
    u = px_to_mm(solved, rs, view, axis, px)
    return u if _orient(view, axis) > 0 else solved.mm[rs[-1].id] - u


def _team_frame(d: dict[str, float], solved: Solved) -> tuple[float, float, float]:
    depth = solved.envelope["z"].value if "z" in solved.envelope else 0.0
    return d.get("x", 0.0), d.get("y", 0.0), depth - d.get("z", 0.0)


def _lines_along(c: Classified, view, axis_along, refs, solved, axis_pos, kinds=("hidden", "visible")):
    out = []
    for p in [q for k in kinds for q in getattr(c, k)]:
        if p.kind != "line":
            continue
        d = p.p1 - p.p0
        if np.linalg.norm(d) < 1 or (abs(d[0]) > abs(d[1])) != (axis_along == "a"):
            continue
        pos = to_distance(solved, refs, view, axis_pos, (_axis_px(p.p0, axis_pos) + _axis_px(p.p1, axis_pos)) / 2)
        span = sorted(to_distance(solved, refs, view, axis_along, _axis_px(e, axis_along)) for e in (p.p0, p.p1))
        out.append((pos, span, p))
    return out


def _faces(c: Classified, view, ax_pos, ax_along, refs, solved, at: float, tol: float) -> list[float]:
    """Positions along the hole axis of visible lines that cross the hole's position `at`."""
    out = []
    for p in c.visible:
        if p.kind != "line":
            continue
        d = p.p1 - p.p0
        if (abs(d[0]) > abs(d[1])) != (ax_pos == "a"):
            continue
        span = sorted(to_distance(solved, refs, view, ax_pos, _axis_px(e, ax_pos)) for e in (p.p0, p.p1))
        if span[0] - tol <= at <= span[1] + tol:
            out.append(to_distance(solved, refs, view, ax_along,
                                   (_axis_px(p.p0, ax_along) + _axis_px(p.p1, ax_along)) / 2))
    return out


def find_holes(classified: dict[ViewName, Classified], refs, solved: Solved) -> tuple[list[Feature], list[Issue]]:
    feats: list[Feature] = []
    issues: list[Issue] = []
    for va in ORDER:
        if va not in classified:
            continue
        hole_axis = LOS[va]
        for circ in [p for p in classified[va].visible + classified[va].hidden if p.kind == "circle"]:
            centre = {AXES[va][ax]: to_distance(solved, refs, va, ax,
                                                circ.center[0] if ax == "a" else -circ.center[1])
                      for ax in ("a", "b")}
            dia, badge = solved.circle_d.get(circ.id, (2 * circ.radius * solved.scale[(va, "a")], "predicted"))
            r, tol = dia / 2, max(0.5, 0.1 * dia / 2)
            pairs = []
            for vb in ORDER:
                if vb == va or vb not in classified or hole_axis not in AXES[vb].values():
                    continue
                shared = [g for g in centre if g in AXES[vb].values()]
                if not shared:
                    continue
                g = shared[0]
                ax_pos = next(a for a, gg in AXES[vb].items() if gg == g)
                ax_along = next(a for a, gg in AXES[vb].items() if gg == hole_axis)
                cands = _lines_along(classified[vb], vb, ax_along, refs, solved, ax_pos)
                lows = [c for c in cands if abs(c[0] - (centre[g] - r)) <= tol]
                highs = [c for c in cands if abs(c[0] - (centre[g] + r)) <= tol]
                faces = _faces(classified[vb], vb, ax_pos, ax_along, refs, solved, centre[g], tol)
                for lo in lows:
                    for hi in highs:
                        s0, s1 = max(lo[1][0], hi[1][0]), min(lo[1][1], hi[1][1])
                        if s1 - s0 > tol:
                            through = any(abs(f - s0) <= tol for f in faces) and any(abs(f - s1) <= tol for f in faces)
                            pairs.append((s0, s1, through, [lo[2].id, hi[2].id]))
                if pairs:
                    break
            if not pairs:
                ends = {"x": solved.envelope.get("x"), "y": solved.envelope.get("y"), "z": solved.envelope.get("z")}
                full = ends[hole_axis].value if ends[hole_axis] else 0.0
                along = NEAR_VIEWER[va](0.0, full)
                pos = _team_frame({**centre, hole_axis: along}, solved)
                feats.append(Feature(id=f"hole-{len(feats)}", type="hole", axis=hole_axis, position_mm=pos,
                                     diameter=dia, through=True, depth=None, evidence=[circ.id],
                                     badge="predicted"))
                issues.append(Issue(severity="amber", kind="predicted", targets=[f"hole-{len(feats) - 1}"],
                                    message=f"Ø{dia:g} hole has no hidden lines in another view; assumed through."))
                continue
            for s0, s1, through, ids in pairs:
                along = NEAR_VIEWER[va](s0, s1)
                pos = _team_frame({**centre, hole_axis: along}, solved)
                feats.append(Feature(id=f"hole-{len(feats)}", type="hole", axis=hole_axis, position_mm=pos,
                                     diameter=dia, through=through, depth=None if through else s1 - s0,
                                     evidence=[circ.id, *ids], badge=badge))
    return feats, issues
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/sketch/test_features.py -v` then `uv run ruff check .`
Expected: 3 passed. If the position test fails on Z, check `_team_frame`: the TOP view's `b` is the distance from the front face (20 mm here), and the team frame puts the front face at `Z = depth` (40), so `Z = 40 - 20 = 20`.

- [ ] **Step 5: Commit**

```bash
git add s2c/sketch/features.py tests/sketch/test_features.py
git commit -m "Infer through and blind holes from circles and hidden lines"
```

---

### Task 12: The pipeline, the JSON assembly and debug overlays

**Files:**
- Create: `s2c/sketch/pipeline.py`, `s2c/sketch/debug.py`
- Modify: `s2c/sketch/__init__.py`
- Test: `tests/sketch/test_pipeline.py`

**Interfaces:**
- Consumes: every earlier task.
- Produces: `read_sketch(image_bytes: bytes, readers: list[Reader] | None = None) -> SketchReading`; `analyse(image_bytes, readers=None) -> tuple[SketchReading, Trace]` where dataclass `Trace(captured, texts, views, classified, refs, links, solved)` holds the internals (every field may be `None` after an abstain) for the accuracy script and debugging; `debug.write_overlays(trace, out_dir: Path) -> list[Path]`. `SketchReading.image_size_px` is `(width, height)` of the original photo. `readers=None` means `readers_from_env()`.

Assembly rules:
- `View.bbox_px`: the view's sheet bbox mapped to the photo; `size_mm`: the extents of its `a` and `b` refs.
- `Entity`: one per visible, hidden and centre primitive. `mm` keys: line `p0`, `p1`; circle `centre`, `radius`; arc `centre`, `radius`, `start_deg`, `end_deg` (counter-clockwise, y up); curve `points`. `px` uses the same keys in photo pixels. Confidence 0.9 for visible, 0.8 for hidden and centre in this version.
- `Dimension`: one per link, id = the text id; `view` is `None` when the link is `unplaced`; `measures` holds ref ids for linear values and entity ids for circles. One more `Dimension` per predicted gap, id `pred-{k}`, with empty `text_raw` and `readings`, badge `predicted`, evidence `proportion`.
- After a text or view abstain, the dimensions read so far are returned as `unplaced`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/sketch/test_pipeline.py
import cv2
import numpy as np
import pytest

from s2c.sketch import read_sketch
from s2c.sketch.models import SketchReading
from s2c.sketch.pipeline import analyse
from tests.sketch.synth import Sheet, TruthReader, bridge_block


def png(img):
    return cv2.imencode(".png", img)[1].tobytes()


def readers(sh):
    return [TruthReader(sh.texts, "a"), TruthReader(sh.texts, "b")]


@pytest.fixture(scope="module")
def bridge():
    sh = bridge_block(Sheet())
    return sh, read_sketch(png(sh.bgr()), readers(sh))


def test_bridge_block_views_and_envelope(bridge):
    _, r = bridge
    assert r.abstain is None
    assert {v.name for v in r.views} == {"top", "front", "right"}
    assert r.envelope["x"].value == pytest.approx(100, abs=0.5) and r.envelope["x"].badge == "written"
    assert r.envelope["y"].value == pytest.approx(50, abs=0.5)
    assert r.envelope["z"].value == pytest.approx(25, abs=0.5)
    assert r.envelope["y"].badge in ("written", "derived") and r.envelope["z"].badge in ("written", "derived")


def test_bridge_block_holes(bridge):
    _, r = bridge
    holes = [f for f in r.features if f.type == "hole"]
    assert sorted(h.axis for h in holes) == ["x", "x", "y", "y"]
    assert all(h.diameter == pytest.approx(12.5, abs=0.3) and h.through for h in holes)
    base = sorted(h.position_mm[0] for h in holes if h.axis == "y")
    assert base == pytest.approx([12.5, 87.5], abs=0.5)


def test_bridge_block_values_are_all_trusted_and_no_red_issues(bridge):
    _, r = bridge
    written = [d for d in r.dimensions if d.readings]
    assert len(written) == 18 and all(d.badge == "written" for d in written)
    assert not [i for i in r.issues if i.severity == "red"]


def test_json_contract_round_trip(bridge):
    _, r = bridge
    assert SketchReading.model_validate_json(r.model_dump_json()) == r


def test_px_are_in_photo_pixels():
    sh = bridge_block(Sheet())
    big = cv2.resize(sh.bgr(), (3200, 2262), interpolation=cv2.INTER_CUBIC)
    r = read_sketch(png(big), readers(sh))
    assert r.image_size_px == (3200, 2262)
    front = next(v for v in r.views if v.name == "front")
    assert front.bbox_px[0] == pytest.approx(2 * 150, abs=30)   # the left dimension line sits at x 150


def test_blank_page_abstains_at_the_views_stage():
    r = read_sketch(png(np.full((1131, 1600, 3), 250, np.uint8)), [])
    assert r.abstain is not None and r.abstain.reason == "no_views_found"


def test_unreadable_bytes_abstain():
    r = read_sketch(b"not an image", [])
    assert r.abstain.stage == "capture" and r.abstain.reason == "unreadable_image"


def test_inch_looking_values_raise_a_unit_question():
    sh = Sheet()
    for p, q in [((300, 300), (700, 300)), ((700, 300), (700, 500)), ((700, 500), (300, 500)),
                 ((300, 500), (300, 300))]:
        sh.line(p, q)
    sh.hdim(300, 400, 500, 545, ".50")
    sh.hdim(400, 700, 500, 545, "1.50")
    sh.hdim(300, 700, 500, 590, "2.00")
    sh.vdim(300, 500, 700, 750, ".75")
    sh.text("FRONT", (500, 680), scale=1.0)
    r = read_sketch(png(sh.bgr()), readers(sh))
    assert any(i.kind == "unit" and i.severity == "red" for i in r.issues)


def test_debug_overlays_are_written(tmp_path):
    from s2c.sketch.debug import write_overlays
    sh = bridge_block(Sheet())
    _, trace = analyse(png(sh.bgr()), readers(sh))
    paths = write_overlays(trace, tmp_path)
    assert paths and all(p.exists() for p in paths)
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/sketch/test_pipeline.py -v`
Expected: FAIL with `ImportError: cannot import name 'read_sketch'`.

- [ ] **Step 3: Implement the pipeline**

```python
# s2c/sketch/pipeline.py
"""read_sketch: one photo of a hand-drawn sheet -> SketchReading. Stages 1 to 9 of the spec."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from s2c.sketch.capture import Captured, capture, sheet_to_original
from s2c.sketch.classify import Classified, classify
from s2c.sketch.features import find_holes
from s2c.sketch.link import Link, Ref, link_view
from s2c.sketch.models import (Dimension, Entity, Issue, SketchAbstain, SketchReading, View,
                               ViewName)
from s2c.sketch.readers import Reader, readers_from_env
from s2c.sketch.solve import Solved, px_to_mm, solve
from s2c.sketch.text import TextItem, find_text_boxes, read_texts
from s2c.sketch.vectorize import Prim, vectorize
from s2c.sketch.views import ViewRegion, split_views


@dataclass
class Trace:
    captured: Captured | None = None
    texts: list[TextItem] | None = None
    views: list[ViewRegion] | None = None
    classified: dict[ViewName, Classified] | None = None
    refs: dict[ViewName, dict[str, list[Ref]]] | None = None
    links: list[Link] | None = None
    solved: Solved | None = None


def read_sketch(image_bytes: bytes, readers: list[Reader] | None = None) -> SketchReading:
    return analyse(image_bytes, readers)[0]


def _box_to_photo(box, H) -> tuple[float, float, float, float]:
    x, y, w, h = box
    pts = sheet_to_original(np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]), H)
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    return float(x0), float(y0), float(x1 - x0), float(y1 - y0)


def _pt_to_photo(p, H) -> list[float]:
    return [float(v) for v in sheet_to_original(np.float32([p]), H)[0]]


def _empty(size, abstain: SketchAbstain, timings, dims=None) -> SketchReading:
    return SketchReading(image_size_px=size, views=[], entities=[], dimensions=dims or [], features=[],
                         envelope={}, issues=[], abstain=abstain, timings_ms=timings)


def _unplaced(texts: list[TextItem], H) -> list[Dimension]:
    return [Dimension(id=t.id, view=None, kind=t.parsed.kind, value=t.parsed.value,
                      text_raw=t.readings[0].text if t.readings else "", readings=t.readings,
                      candidates=t.candidates, measures=[], badge=t.badge or "uncertain",
                      evidence="reader", bbox_px=_box_to_photo(t.box, H))
            for t in texts if t.role == "dimension"]


def _unit_issue(texts: list[TextItem]) -> list[Issue]:
    """Spec 3: values that look like inches (all under 10, mostly written with a leading dot)."""
    dims = [t for t in texts if t.role == "dimension" and t.parsed and t.parsed.kind == "linear"]
    if len(dims) < 3 or not all(t.parsed.value < 10 for t in dims):
        return []
    dotted = sum(any(r.text.strip().startswith(".") for r in t.readings) for t in dims)
    if dotted < len(dims) / 2:
        return []
    return [Issue(severity="red", kind="unit", targets=[t.id for t in dims],
                  message="The values look like inches. Are they mm or inches?")]


def _entity(p: Prim, view: str, line_type: str, refs, solved: Solved, H) -> Entity:
    def mm(pt):
        return [px_to_mm(solved, refs[view]["a"], view, "a", pt[0]),
                px_to_mm(solved, refs[view]["b"], view, "b", -pt[1])]

    s = solved.scale.get((view, "a"), 0.25)
    if p.kind == "line":
        mmd = {"p0": mm(p.p0), "p1": mm(p.p1)}
        pxd = {"p0": _pt_to_photo(p.p0, H), "p1": _pt_to_photo(p.p1, H)}
    elif p.kind in ("circle", "arc"):
        radius = solved.circle_d.get(p.id, (2 * p.radius * s, "predicted"))[0] / 2
        mmd = {"centre": mm(p.center), "radius": radius}
        edge = _pt_to_photo((p.center[0] + p.radius, p.center[1]), H)
        centre_px = _pt_to_photo(p.center, H)
        pxd = {"centre": centre_px, "radius": float(np.hypot(edge[0] - centre_px[0], edge[1] - centre_px[1]))}
        if p.kind == "arc":
            ang = np.degrees(np.unwrap(np.arctan2(-(p.pts[:, 1] - p.center[1]), p.pts[:, 0] - p.center[0])))
            mmd["start_deg"], mmd["end_deg"] = float(ang.min()), float(ang.max())
            pxd["start_deg"], pxd["end_deg"] = mmd["start_deg"], mmd["end_deg"]
    else:
        mmd = {"points": [mm(q) for q in p.pts]}
        pxd = {"points": [_pt_to_photo(q, H) for q in p.pts]}
    return Entity(id=p.id, view=view, type=p.kind, line_type=line_type, mm=mmd, px=pxd,
                  confidence=0.9 if line_type == "visible" else 0.8)


def analyse(image_bytes: bytes, readers: list[Reader] | None = None) -> tuple[SketchReading, Trace]:
    t0 = time.perf_counter()
    timings: dict[str, float] = {}
    trace = Trace()

    def lap(name):
        timings[name] = round((time.perf_counter() - t0) * 1000 - sum(timings.values()), 1)

    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return _empty((0, 0), SketchAbstain(stage="capture", reason="unreadable_image",
                                            remedy="Send a JPEG or PNG photo of the sheet."), timings), trace
    size = (int(img.shape[1]), int(img.shape[0]))
    cap = capture(img)
    lap("capture")
    if isinstance(cap, SketchAbstain):
        return _empty(size, cap, timings), trace
    trace.captured = cap
    H = cap.to_original

    readers = readers_from_env() if readers is None else readers
    texts = read_texts(cap.sheet, find_text_boxes(cap.ink, cap.stroke_px), readers)
    lap("text")
    if isinstance(texts, SketchAbstain):
        return _empty(size, texts, timings), trace
    trace.texts = texts

    split = split_views(cap.ink, texts, cap.stroke_px)
    lap("views")
    if isinstance(split, SketchAbstain):
        return _empty(size, split, timings, _unplaced(texts, H)), trace
    views, issues = split
    issues += _unit_issue(texts)
    trace.views = views

    classified, refs, links = {}, {}, []
    for v in views:
        c = classify(vectorize(v.ink, cap.stroke_px, v.name), v.ink, cap.stroke_px, v.texts)
        r, lk, iss = link_view(v.name, c, v.texts, cap.stroke_px)
        classified[v.name], refs[v.name] = c, r
        links += lk
        issues += iss
    trace.classified, trace.refs, trace.links = classified, refs, links
    lap("geometry")

    circles = {p.id: p for c in classified.values() for p in c.visible + c.hidden if p.kind == "circle"}
    solved = solve(refs, links, circles)
    trace.solved = solved
    issues += solved.issues
    features, fi = find_holes(classified, refs, solved)
    issues += fi
    lap("solve")

    out_views = []
    for v in views:
        ext = [solved.mm[refs[v.name][ax][-1].id] if refs[v.name][ax] else 0.0 for ax in ("a", "b")]
        out_views.append(View(name=v.name, label_text=v.label_text, bbox_px=_box_to_photo(v.bbox, H),
                              size_mm=(ext[0], ext[1])))
    entities = [_entity(p, v, lt, refs, solved, H) for v, c in classified.items()
                for lt, group in (("visible", c.visible), ("hidden", c.hidden), ("centre", c.centre))
                for p in group]
    dimensions = []
    for L in links:
        d = solved.dims.get(L.text.id)
        dimensions.append(Dimension(
            id=L.text.id, view=None if L.how == "unplaced" else L.view, kind=L.kind,
            value=d.value if d else L.text.parsed.value,
            text_raw=L.text.readings[0].text if L.text.readings else "", readings=L.text.readings,
            candidates=d.candidates if d else L.text.candidates, implied=d.implied if d else None,
            tolerance=L.text.parsed.tolerance, measures=L.measures, axis=L.axis,
            badge=d.badge if d else (L.text.badge or "uncertain"), evidence=d.evidence if d else "reader",
            bbox_px=_box_to_photo(L.text.box, H)))
    ref_px = {r.id: r for axes in refs.values() for rs in axes.values() for r in rs}
    for k, g in enumerate(solved.gaps):
        a, b = ref_px[g.refs[0]], ref_px[g.refs[1]]
        vb = next(v.bbox for v in views if v.name == g.view)
        if g.axis == "a":
            box = (a.px, vb[1] + vb[3], b.px - a.px, 1)
        else:
            box = (vb[0], -b.px, 1, b.px - a.px)
        dimensions.append(Dimension(id=f"pred-{k}", view=g.view, kind="linear", value=g.value, text_raw="",
                                    readings=[], measures=list(g.refs), axis=g.axis, badge="predicted",
                                    evidence="proportion", bbox_px=_box_to_photo(box, H)))
    lap("assemble")
    timings["total"] = round((time.perf_counter() - t0) * 1000, 1)
    reading = SketchReading(image_size_px=size, views=out_views, entities=entities, dimensions=dimensions,
                            features=features, envelope=solved.envelope, issues=issues, timings_ms=timings)
    debug_dir = os.environ.get("SKETCH_DEBUG_DIR")
    if debug_dir:
        from s2c.sketch.debug import write_overlays
        write_overlays(trace, Path(debug_dir))
    return reading, trace
```

```python
# s2c/sketch/__init__.py
"""Hand-sketch recognition: one photo of a multi-view sheet -> SketchReading JSON."""
from s2c.sketch.pipeline import read_sketch

__all__ = ["read_sketch"]
```

- [ ] **Step 4: Implement the debug overlays**

```python
# s2c/sketch/debug.py
"""Debug overlays for the lab screen: what each stage saw. Off unless SKETCH_DEBUG_DIR is set."""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

COLOURS = {"visible": (0, 160, 0), "hidden": (200, 80, 0), "centre": (180, 0, 180),
           "dimension": (0, 140, 255), "extension": (0, 200, 200), "leader": (200, 200, 0)}
BADGE = {"written": (0, 160, 0), "uncertain": (0, 165, 255), "predicted": (0, 165, 255),
         "conflict": (0, 0, 230), "derived": (0, 160, 0), "edited": (160, 0, 0)}


def _pt(p):
    return int(round(p[0])), int(round(p[1]))


def write_overlays(trace, out_dir: Path) -> list[Path]:
    if trace.captured is None:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    img = trace.captured.sheet.copy()
    for c in (trace.classified or {}).values():
        for kind in ("visible", "hidden", "centre"):
            for p in getattr(c, kind):
                cv2.polylines(img, [np.int32(p.pts)], p.kind == "circle", COLOURS[kind], 2)
        for d in c.dimlines:
            cv2.line(img, _pt(d.p0), _pt(d.p1), COLOURS["dimension"], 2)
            for a in d.arrows:
                cv2.circle(img, _pt(a.tip), 5, (0, 0, 255), -1)
        for e in c.extensions.values():
            cv2.line(img, _pt(e.p0), _pt(e.p1), COLOURS["extension"], 2)
        for lead in c.leaders:
            cv2.line(img, _pt(lead.tail), _pt(lead.tip), COLOURS["leader"], 2)
    records = []
    dims = trace.solved.dims if trace.solved else {}
    for t in trace.texts or []:
        x, y, w, h = t.box
        badge = dims[t.id].badge if t.id in dims else (t.badge or "uncertain")
        cv2.rectangle(img, (x, y), (x + w, y + h), BADGE.get(badge, (128, 128, 128)), 2)
        crop_path = out_dir / f"{stamp}_{t.id}.png"
        cv2.imwrite(str(crop_path), trace.captured.sheet[y:y + h, x:x + w])
        records.append({"id": t.id, "role": t.role, "badge": badge, "crop": crop_path.name,
                        "readings": [r.model_dump() for r in t.readings]})
    for v in trace.views or []:
        x, y, w, h = v.bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), (90, 90, 90), 1)
        cv2.putText(img, v.name, (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 90), 2)
    overlay = out_dir / f"{stamp}_overlay.png"
    cv2.imwrite(str(overlay), img)
    log = out_dir / f"{stamp}_readings.jsonl"
    log.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")
    return [overlay, log]
```

- [ ] **Step 5: Run tests and lint**

Run: `uv run pytest tests/sketch/test_pipeline.py -v` then `uv run pytest -q` then `uv run ruff check .`
Expected: all pass, and the full suite stays green. This is the first end-to-end test: when it fails, set `SKETCH_DEBUG_DIR=tmp/debug`, run once, open the overlay, and fix the earliest stage that looks wrong (views, then line types, then links), in that stage's own tests first.

- [ ] **Step 6: Commit**

```bash
git add s2c/sketch/pipeline.py s2c/sketch/debug.py s2c/sketch/__init__.py tests/sketch/test_pipeline.py
git commit -m "Assemble the SketchReading end to end with debug overlays"
```

---

### Task 13: Golden sheets and the accuracy report

**Files:**
- Create: `tests/golden_sketch/README.md`, `tests/golden_sketch/bridge_block_synthetic/expected.json`, `tests/sketch/test_golden.py`, `scripts/sketch_accuracy.py`
- Test: `tests/sketch/test_golden.py`

**Interfaces:**
- Consumes: `analyse` (Task 12).
- Produces: the golden format; `scripts/sketch_accuracy.py` printing a markdown table; `score(reading, trace, expected) -> dict` in `scripts/sketch_accuracy.py`, reused by the test.

- [ ] **Step 1: Write the golden format**

~~~markdown
<!-- tests/golden_sketch/README.md -->
# Golden hand-drawn sheets

One folder per sheet: `image.jpg` (or `.png`) and `expected.json`. No people, no personal data.

```json
{
  "views": ["top", "front", "right"],
  "envelope": {"x": 100, "y": 50, "z": 25},
  "dimensions": [
    {"view": "front", "value": 100, "kind": "linear", "axis": "a", "span_mm": [0, 100]},
    {"view": "top", "value": 12.5, "kind": "diameter"}
  ],
  "holes": [{"axis": "y", "diameter": 12.5, "through": true, "at": [12.5, 12.5]}]
}
```

- `span_mm`: the two positions the value measures, in mm from the lowest position of that view axis.
- `holes[].at`: the hole centre on the two axes across the hole, in the team frame (x, y, z order, hole axis left out; Z has the front face at Z = depth).
- Real sheets need real readers: run with `SKETCH_GOLDEN=1` and `SKETCH_READERS` set.
~~~

Write `tests/golden_sketch/bridge_block_synthetic/expected.json` with the bridge block values (the synthetic sheet is generated in the test, not stored):

```json
{
  "views": ["top", "front", "right"],
  "envelope": {"x": 100, "y": 50, "z": 25},
  "dimensions": [
    {"view": "front", "value": 100, "kind": "linear", "axis": "a", "span_mm": [0, 100]},
    {"view": "front", "value": 37.5, "kind": "linear", "axis": "a", "span_mm": [0, 37.5]},
    {"view": "front", "value": 37.5, "kind": "linear", "axis": "a", "span_mm": [62.5, 100]},
    {"view": "front", "value": 12.5, "kind": "linear", "axis": "a", "span_mm": [25, 37.5]},
    {"view": "front", "value": 25, "kind": "linear", "axis": "a", "span_mm": [37.5, 62.5]},
    {"view": "front", "value": 12.5, "kind": "linear", "axis": "a", "span_mm": [62.5, 75]},
    {"view": "front", "value": 37.5, "kind": "linear", "axis": "b", "span_mm": [12.5, 50]},
    {"view": "front", "value": 25, "kind": "linear", "axis": "b", "span_mm": [25, 50]},
    {"view": "front", "value": 12.5, "kind": "linear", "axis": "b", "span_mm": [12.5, 25]},
    {"view": "front", "value": 12.5, "kind": "linear", "axis": "b", "span_mm": [0, 12.5]},
    {"view": "top", "value": 12.5, "kind": "diameter"},
    {"view": "top", "value": 12.5, "kind": "linear", "axis": "a", "span_mm": [0, 12.5]},
    {"view": "top", "value": 12.5, "kind": "linear", "axis": "b", "span_mm": [12.5, 25]},
    {"view": "right", "value": 12.5, "kind": "diameter"},
    {"view": "right", "value": 12.5, "kind": "linear", "axis": "a", "span_mm": [0, 12.5]},
    {"view": "right", "value": 12.5, "kind": "linear", "axis": "a", "span_mm": [12.5, 25]},
    {"view": "right", "value": 12.5, "kind": "linear", "axis": "b", "span_mm": [37.5, 50]},
    {"view": "right", "value": 50, "kind": "linear", "axis": "b", "span_mm": [0, 50]}
  ],
  "holes": [
    {"axis": "y", "diameter": 12.5, "through": true, "at": [12.5, 12.5]},
    {"axis": "y", "diameter": 12.5, "through": true, "at": [87.5, 12.5]},
    {"axis": "x", "diameter": 12.5, "through": true, "at": [37.5, 12.5]},
    {"axis": "x", "diameter": 12.5, "through": true, "at": [37.5, 12.5]}
  ]
}
```

- [ ] **Step 2: Write the scorer and the report script**

```python
# scripts/sketch_accuracy.py
"""Accuracy of the sketch reader on the golden sheets. Prints a markdown table for the README.
Usage: uv run python scripts/sketch_accuracy.py [folder ...]   (default: every tests/golden_sketch/*)"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from s2c.sketch.pipeline import analyse

TOL = lambda v: max(1.0, 0.05 * abs(v))  # noqa: E731  (team golden tolerance: 5 percent or 1 mm)
TRUSTED = ("written", "derived", "edited")


def score(reading, trace, expected: dict) -> dict:
    s = {"values": 0, "read_ok": 0, "link_ok": 0, "silent_errors": 0, "edits": 0,
         "envelope_ok": 0, "holes_ok": 0, "holes": len(expected.get("holes", []))}
    got = [d for d in reading.dimensions if d.readings]
    pool = list(got)
    link_by_id = {L.text.id: L for L in (trace.links or [])}
    for e in expected.get("dimensions", []):
        s["values"] += 1
        match = next((d for d in pool if d.view == e["view"] and d.kind == e["kind"]
                      and d.value is not None and abs(d.value - e["value"]) < 1e-6), None)
        if match is None:
            s["edits"] += 1
            continue
        pool.remove(match)
        s["read_ok"] += 1
        L = link_by_id.get(match.id)
        if "span_mm" in e and L is not None and L.refs and trace.solved:
            span = sorted(trace.solved.mm[r] for r in L.refs)
            if all(abs(a - b) <= 0.5 for a, b in zip(span, e["span_mm"])):
                s["link_ok"] += 1
        elif "span_mm" not in e and match.measures:
            s["link_ok"] += 1
        if match.badge not in TRUSTED:
            s["edits"] += 1
    for d in pool:  # values we produced that match nothing expected
        if d.badge in TRUSTED:
            s["silent_errors"] += 1
        s["edits"] += 1
    env = expected.get("envelope", {})
    ok = 0
    for axis, value in env.items():
        size = reading.envelope.get(axis)
        if size is None:
            continue
        right = abs(size.value - value) <= TOL(value)
        ok += right
        if not right and size.badge in TRUSTED:
            s["silent_errors"] += 1
    s["envelope_ok"] = int(ok == len(env))
    holes = [f for f in reading.features if f.type == "hole"]
    for h in expected.get("holes", []):
        idx = {"x": 0, "y": 1, "z": 2}
        across = [k for k in ("x", "y", "z") if k != h["axis"]]
        hit = next((f for f in holes if f.axis == h["axis"] and f.through == h["through"]
                    and abs((f.diameter or 0) - h["diameter"]) <= TOL(h["diameter"])
                    and all(abs(f.position_mm[idx[a]] - v) <= TOL(v) for a, v in zip(across, h["at"]))),
                   None)
        if hit:
            holes.remove(hit)
            s["holes_ok"] += 1
    return s


def main(folders: list[Path]) -> None:
    load_dotenv()
    rows, total = [], {}
    for folder in folders:
        image = next((p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")), None)
        if image is None:
            continue
        expected = json.loads((folder / "expected.json").read_text(encoding="utf-8"))
        t0 = time.perf_counter()
        reading, trace = analyse(image.read_bytes())
        s = score(reading, trace, expected)
        s["ms"] = (time.perf_counter() - t0) * 1000
        rows.append((folder.name, s))
        for k, v in s.items():
            total[k] = total.get(k, 0) + v
    print("| Sheet | Values read | Linked | Envelope | Holes | Edits | Silent errors | ms |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, s in rows:
        print(f"| {name} | {s['read_ok']}/{s['values']} | {s['link_ok']}/{s['values']} | "
              f"{'ok' if s['envelope_ok'] else 'no'} | {s['holes_ok']}/{s['holes']} | {s['edits']} | "
              f"{s['silent_errors']} | {s['ms']:.0f} |")
    if rows:
        n = len(rows)
        print(f"| **all** | {total['read_ok']}/{total['values']} | {total['link_ok']}/{total['values']} | "
              f"{total['envelope_ok']}/{n} | {total['holes_ok']}/{total['holes']} | {total['edits'] / n:.1f} per sheet | "
              f"{total['silent_errors']} | {total['ms'] / n:.0f} |")


if __name__ == "__main__":
    args = [Path(a) for a in sys.argv[1:]]
    main(args or sorted(p for p in Path("tests/golden_sketch").iterdir() if (p / "expected.json").exists()))
```

In the bridge block, holes along `y` list `at` as (x, z) and holes along `x` as (y, z); `z = 12.5` for all four, the middle of the 25 mm depth.

- [ ] **Step 3: Write the tests**

```python
# tests/sketch/test_golden.py
import json
import os
import sys
from pathlib import Path

import cv2
import pytest

from s2c.sketch.pipeline import analyse
from tests.sketch.synth import Sheet, TruthReader, bridge_block

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from sketch_accuracy import score  # noqa: E402

GOLDEN = Path(__file__).resolve().parents[1] / "golden_sketch"


def test_synthetic_bridge_block_scores_perfectly():
    sh = bridge_block(Sheet())
    expected = json.loads((GOLDEN / "bridge_block_synthetic" / "expected.json").read_text(encoding="utf-8"))
    reading, trace = analyse(cv2.imencode(".png", sh.bgr())[1].tobytes(),
                             [TruthReader(sh.texts, "a"), TruthReader(sh.texts, "b")])
    s = score(reading, trace, expected)
    assert s["read_ok"] == s["values"] == 18
    assert s["link_ok"] == 18
    assert s["envelope_ok"] == 1 and s["holes_ok"] == 4
    assert s["silent_errors"] == 0 and s["edits"] == 0


def test_misread_values_are_never_silent():
    sh = bridge_block(Sheet())
    expected = json.loads((GOLDEN / "bridge_block_synthetic" / "expected.json").read_text(encoding="utf-8"))
    readers = [TruthReader(sh.texts, "a"), TruthReader(sh.texts, "b", swap={"25": "35", "50": "5O"})]
    reading, trace = analyse(cv2.imencode(".png", sh.bgr())[1].tobytes(), readers)
    assert score(reading, trace, expected)["silent_errors"] == 0


REAL = [p for p in GOLDEN.iterdir() if (p / "expected.json").exists()
        and any(q.suffix.lower() in (".jpg", ".jpeg", ".png") for q in p.iterdir())]


@pytest.mark.skipif(os.environ.get("SKETCH_GOLDEN") != "1", reason="needs real readers")
@pytest.mark.parametrize("folder", REAL, ids=[p.name for p in REAL])
def test_real_golden_sheet_has_no_silent_errors(folder):
    image = next(q for q in folder.iterdir() if q.suffix.lower() in (".jpg", ".jpeg", ".png"))
    expected = json.loads((folder / "expected.json").read_text(encoding="utf-8"))
    reading, trace = analyse(image.read_bytes())
    assert score(reading, trace, expected)["silent_errors"] == 0
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/sketch/test_golden.py -v` then `uv run pytest -q` then `uv run ruff check .`
Expected: the two synthetic tests pass, the real-sheet test is skipped. Then `uv run python scripts/sketch_accuracy.py` prints the table (empty until real sheets with images exist).

- [ ] **Step 5: Commit**

```bash
git add tests/golden_sketch scripts/sketch_accuracy.py tests/sketch/test_golden.py
git commit -m "Add golden sheet format, scorer and accuracy report"
```

---

## After this plan

1. Draw the 15 to 20 real golden sheets (spec 14.1), photograph them, write their `expected.json`, run `SKETCH_GOLDEN=1 uv run python scripts/sketch_accuracy.py` with the chosen readers, and paste the table into the README.
2. Second plan (spec "Should" row): angles and chamfers in the solve, prediction ranks 3 and 4 (symmetry, standard sizes), the meaning check, `resolve(reading, edits)` and correction records, `scripts/bakeoff_readers.py`, the Brev setup script. Also the two issue kinds this plan does not raise yet: `gap_closed` (spec 8.3) and the amber issue for views that share an axis but whose inner positions do not match (spec 11).
