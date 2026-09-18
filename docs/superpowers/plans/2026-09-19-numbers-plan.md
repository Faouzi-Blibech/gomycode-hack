# Numbers Implementation Plan (coin metrology and dimension OCR)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce every millimetre value in the system: coin-scaled measurements from photos, and dimension annotations read from sketches and drawings, each linked to the field it describes.

**Architecture:** `s2c/metrology.py` is pure OpenCV: coin, scale, part contour, hole circles. `s2c/ocr.py` is a three-stage pipeline: find text regions with classical CV, read each region with an injectable reader (VLM-assisted by default, a trained recogniser if ready), then link values to fields using the Topology. Both modules replace their fakes automatically once they exist.

**Tech Stack:** Python 3.11, OpenCV headless, NumPy, pytest. The VLM-assisted reader uses `s2c/vision/client.py`.

**Spec:** `docs/superpowers/specs/2026-09-19-sketch-to-cad-design.md` (Rule 2, Rule 4, sections 4.1, 4.2, 4.7). Contracts are in `s2c/partspec/models.py`; read that file before starting.

## Global Constraints

- Numbers are measured or user-written, never model-estimated. The model may read a number the user wrote; it may never estimate a size. (spec Rule 2)
- Coin eccentricity above 0.30, or no coin, abstains with `coin_tilted` or `coin_not_found` and remedy "Lay the coin flat, shoot top-down, retake." (spec Rule 4)
- mm-per-pixel error under 2 percent on three coin photos at three distances. (spec 4.7)
- Every value read is returned; unlinked values carry `linked_to: "unknown"`. Never drop a value. (role brief)
- Work at a fixed longest side of 1600 px; scale boxes back to the original. (role brief)
- Commit messages: plain, no AI attribution, no co-author trailers. (spec 5)
- Write the failing test first. (CLAUDE.md)

Prerequisite: the integrator's Tasks 1 to 7 are merged (package, contracts, silhouette, VLM client, fakes).

---

## File structure

| Path | Responsibility |
| --- | --- |
| `s2c/metrology.py` | `find_coin`, `measure` |
| `s2c/coins.py` | Coin diameter table |
| `s2c/ocr.py` | `find_text_regions`, `read_annotations`, `link` |
| `s2c/ocr_readers.py` | `vlm_reader` (default) and the `Reader` protocol a trained model implements |
| `tests/test_metrology.py` | Synthetic coin and part images |
| `tests/test_ocr.py` | Synthetic sketches with printed digits, fake reader |
| `tests/synth.py` | Helpers that draw synthetic sketches and photos |
| `scripts/accuracy_report.py` | Runs OCR and metrology on the golden set and prints the numbers for the README |

---

### Task 1: Coin table and coin detection

**Files:**
- Create: `s2c/coins.py`, `s2c/metrology.py`, `tests/synth.py`
- Test: `tests/test_metrology.py`

**Interfaces:**
- Produces: `COINS: dict[str, float]` (name to diameter in mm); `find_coin(image_bgr) -> CoinFit | None` where `CoinFit` is a dataclass `(cx: float, cy: float, diameter_px: float, eccentricity: float, confidence: float)`; `resize_longest(image, longest=1600) -> tuple[np.ndarray, float]` returning the resized image and the scale factor applied.

- [ ] **Step 1: Write the synthetic helpers**

```python
# tests/synth.py
"""Draw synthetic photos and sketches so tests need no real images."""
import cv2
import numpy as np


def photo(coin_px=250, coin_axes=None, part=(400, 260), holes=((60, 60, 40), (340, 200, 40)), size=(1200, 1600)):
    """Grey table, dark part on the left, bright coin on the right. coin_axes=(a, b) draws a tilted coin."""
    img = np.full((size[0], size[1], 3), 170, np.uint8)
    x0, y0 = 200, 300
    cv2.rectangle(img, (x0, y0), (x0 + part[0], y0 + part[1]), (40, 40, 40), -1)
    for hx, hy, hd in holes:
        cv2.circle(img, (x0 + hx, y0 + hy), hd // 2, (170, 170, 170), -1)
    axes = coin_axes or (coin_px // 2, coin_px // 2)
    cv2.ellipse(img, (1200, 500), axes, 0, 0, 360, (230, 210, 120), -1)
    return img, (x0, y0)


def sketch(labels, part=(600, 400), size=(1200, 1600), holes=()):
    """White paper, pen rectangle, printed dimension text. labels: list of (text, x, y)."""
    img = np.full((size[0], size[1], 3), 255, np.uint8)
    x0, y0 = 300, 300
    cv2.rectangle(img, (x0, y0), (x0 + part[0], y0 + part[1]), (0, 0, 0), 3)
    for hx, hy, hd in holes:
        cv2.circle(img, (x0 + hx, y0 + hy), hd // 2, (0, 0, 0), 3)
    for text, x, y in labels:
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    return img, (x0, y0)
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_metrology.py
import numpy as np
from s2c.coins import COINS
from s2c.metrology import find_coin, resize_longest
from tests.synth import photo


def test_coin_table_has_tunisian_dinar():
    assert COINS["1 TND"] == 25.0


def test_resize_longest_scales_and_reports_factor():
    img = np.zeros((600, 3200, 3), np.uint8)
    out, factor = resize_longest(img, 1600)
    assert out.shape[1] == 1600 and abs(factor - 0.5) < 1e-6


def test_find_flat_coin():
    img, _ = photo(coin_px=250)
    fit = find_coin(img)
    assert fit is not None
    assert abs(fit.cx - 1200) < 4 and abs(fit.cy - 500) < 4
    assert abs(fit.diameter_px - 250) < 6
    assert fit.eccentricity < 0.1


def test_tilted_coin_has_high_eccentricity():
    img, _ = photo(coin_axes=(125, 70))
    fit = find_coin(img)
    assert fit is not None and fit.eccentricity > 0.3


def test_no_coin_returns_none():
    img = np.full((1200, 1600, 3), 170, np.uint8)
    assert find_coin(img) is None
```

- [ ] **Step 3: Run to confirm failure**

Run: `uv run pytest tests/test_metrology.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Implement**

```python
# s2c/coins.py
"""Reference coin diameters in millimetres. Add the coins the team actually owns."""
COINS: dict[str, float] = {
    "1 TND": 25.0,
    "2 TND": 29.0,
    "500 millimes": 24.0,
    "1 EUR": 23.25,
    "2 EUR": 25.75,
    "50 cent EUR": 24.25,
}
DEFAULT_COIN = "1 TND"
```

```python
# s2c/metrology.py
"""Coin -> scale -> part measurements. Pure OpenCV. No model anywhere in this file."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from s2c.coins import COINS, DEFAULT_COIN
from s2c.partspec.models import Abstain, Bbox, Circle, Coin, Measurements

ECCENTRICITY_MAX = 0.30


@dataclass
class CoinFit:
    cx: float
    cy: float
    diameter_px: float
    eccentricity: float
    confidence: float


def resize_longest(image: np.ndarray, longest: int = 1600) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    factor = longest / max(h, w)
    if factor >= 1.0:
        return image, 1.0
    return cv2.resize(image, (round(w * factor), round(h * factor)), interpolation=cv2.INTER_AREA), factor


def _eccentricity(a: float, b: float) -> float:
    a, b = max(a, b), min(a, b)
    return float(np.sqrt(1 - (b / a) ** 2)) if a > 0 else 1.0


def find_coin(image_bgr: np.ndarray) -> CoinFit | None:
    """Coins are the roundest, most saturated blob. Returns the fit in the input image's pixels."""
    small, f = resize_longest(image_bgr)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    sat = cv2.GaussianBlur(hsv[:, :, 1], (7, 7), 0)
    _, th = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best: CoinFit | None = None
    for c in contours:
        if len(c) < 20 or cv2.contourArea(c) < 500:
            continue
        (cx, cy), (w, h), _ = cv2.fitEllipse(c)
        area, hull_area = cv2.contourArea(c), cv2.contourArea(cv2.convexHull(c))
        solidity = area / hull_area if hull_area else 0
        if solidity < 0.9:
            continue
        ecc = _eccentricity(w, h)
        fit = CoinFit(cx / f, cy / f, max(w, h) / f, ecc, confidence=float(solidity * (1 - ecc)))
        if best is None or fit.confidence > best.confidence:
            best = fit
    return best
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_metrology.py -v`
Expected: PASS. If the synthetic coin colour is not saturated enough on your threshold, raise the saturation of the coin colour in `tests/synth.py` (it is a yellow-gold `(230, 210, 120)` in BGR); real coins are less saturated than paper but more than a grey table, which is what the threshold relies on. Check on a real photo in Task 2.

- [ ] **Step 6: Commit**

```bash
git add s2c/coins.py s2c/metrology.py tests/synth.py tests/test_metrology.py
git commit -m "Detect the reference coin and its eccentricity"
```

---

### Task 2: Scale, part contour, holes and abstention gates

**Files:**
- Modify: `s2c/metrology.py`
- Test: `tests/test_metrology.py` (append)

**Interfaces:**
- Produces: `measure(image_bgr: np.ndarray, coin_name: str = DEFAULT_COIN) -> Measurements | Abstain`. Contour and circles are in millimetres relative to the bottom-left of the part's bounding box, y upward, matching the builder convention.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_metrology.py
from s2c.metrology import measure
from s2c.partspec.models import Measurements, Abstain


def test_measure_plate_with_two_holes():
    img, _ = photo(coin_px=250, part=(400, 260), holes=((60, 60, 40), (340, 200, 40)))
    m = measure(img, "1 TND")  # 25 mm over 250 px = 0.1 mm/px
    assert isinstance(m, Measurements), m
    assert abs(m.mm_per_px - 0.1) < 0.002
    assert abs(m.bbox_mm.width - 40) < 0.8 and abs(m.bbox_mm.height - 26) < 0.8
    assert len(m.circles_mm) == 2
    holes = sorted(m.circles_mm, key=lambda c: c.x)
    assert abs(holes[0].x - 6) < 0.6 and abs(holes[0].y - 20) < 0.6 and abs(holes[0].diameter - 4) < 0.5
    assert abs(holes[1].x - 34) < 0.6 and abs(holes[1].y - 6) < 0.6


def test_tilted_coin_abstains():
    img, _ = photo(coin_axes=(125, 70))
    out = measure(img)
    assert isinstance(out, Abstain) and out.reason == "coin_tilted"
    assert "flat" in out.remedy


def test_missing_coin_abstains():
    img = np.full((1200, 1600, 3), 170, np.uint8)
    cv2.rectangle(img, (200, 300), (600, 560), (40, 40, 40), -1)
    out = measure(img)
    assert isinstance(out, Abstain) and out.reason == "coin_not_found"


def test_no_part_abstains():
    img, _ = photo(part=(1, 1), holes=())
    out = measure(img)
    assert isinstance(out, Abstain) and out.reason == "part_not_found"
```

Add `import cv2` at the top of the test file.

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_metrology.py -v`
Expected: the new tests FAIL with `ImportError: cannot import name 'measure'`.

- [ ] **Step 3: Implement**

```python
# append to s2c/metrology.py

def _part_mask(image_bgr: np.ndarray, coin: CoinFit) -> np.ndarray:
    """Foreground that is not the coin. Otsu on grey; the part is darker or lighter than the table."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.circle(th, (int(coin.cx), int(coin.cy)), int(coin.diameter_px * 0.6), 0, -1)
    return cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def measure(image_bgr: np.ndarray, coin_name: str = DEFAULT_COIN) -> Measurements | Abstain:
    remedy_coin = "Lay the coin flat, shoot top-down, retake."
    coin = find_coin(image_bgr)
    if coin is None:
        return Abstain(stage="metrology", reason="coin_not_found", remedy="No coin found. " + remedy_coin)
    if coin.eccentricity > ECCENTRICITY_MAX:
        return Abstain(stage="metrology", reason="coin_tilted", remedy=remedy_coin)
    mm_per_px = COINS[coin_name] / coin.diameter_px

    mask = _part_mask(image_bgr, coin)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return Abstain(stage="metrology", reason="part_not_found",
                       remedy="No part found. Use a plain background that contrasts with the part.")
    outer_ids = [i for i in range(len(contours)) if hierarchy[0][i][3] == -1]
    best = max(outer_ids, key=lambda i: cv2.contourArea(contours[i]))
    outer = contours[best]
    if cv2.contourArea(outer) < 0.002 * mask.size:
        return Abstain(stage="metrology", reason="part_not_found",
                       remedy="The part is too small in the frame. Move the phone closer.")

    x, y, w, h = cv2.boundingRect(outer)
    to_mm = lambda px, py: (float((px - x) * mm_per_px), float((y + h - py) * mm_per_px))  # noqa: E731
    contour_mm = [to_mm(px, py) for px, py in outer.reshape(-1, 2)]

    circles: list[Circle] = []
    child = hierarchy[0][best][2]
    while child != -1:
        c = contours[child]
        if cv2.contourArea(c) > 30:
            (cx, cy), r = cv2.minEnclosingCircle(c)
            circularity = cv2.contourArea(c) / (np.pi * r * r) if r > 0 else 0
            if circularity > 0.75:
                mx, my = to_mm(cx, cy)
                circles.append(Circle(x=mx, y=my, diameter=float(2 * r * mm_per_px)))
        child = hierarchy[0][child][0]

    return Measurements(
        mm_per_px=float(mm_per_px),
        coin=Coin(name=coin_name, pixel_diameter=float(coin.diameter_px), eccentricity=float(coin.eccentricity),
                  confidence=float(coin.confidence)),
        outer_contour_mm=contour_mm,
        bbox_mm=Bbox(width=float(w * mm_per_px), height=float(h * mm_per_px)),
        circles_mm=circles,
        confidence=float(min(coin.confidence, 0.95)),
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_metrology.py -v`
Expected: PASS. The full suite `uv run pytest -q` must stay green; the pipeline now uses the real `measure`.

- [ ] **Step 5: Real photos at three distances**

Photograph the same coin and a ruler at roughly 20, 30 and 40 cm, top-down. For each, run:

```bash
uv run python -c "
import cv2; from s2c.metrology import find_coin
f = find_coin(cv2.imread('tmp/coin_20cm.jpg')); print(f)
"
```

Measure a known 50 mm ruler span in pixels in an image editor and check that `50 / (span_px * 25.0 / f.diameter_px)` is within 2 percent of 1. Record the three errors in `README.md`.

- [ ] **Step 6: Commit**

```bash
git add s2c/metrology.py tests/test_metrology.py README.md
git commit -m "Measure part bounding box and holes in millimetres from a coin-scaled photo"
```

---

### Task 3: Text region detection on sketches

**Files:**
- Create: `s2c/ocr.py`
- Test: `tests/test_ocr.py`

**Interfaces:**
- Produces: `find_text_regions(image_bgr) -> list[Region]` where `Region` is a dataclass `(x: int, y: int, w: int, h: int)` in original-image pixels; `strip_lines(binary) -> np.ndarray` (helper, removes long straight strokes).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ocr.py
from s2c.ocr import find_text_regions
from tests.synth import sketch


def contains(region, px, py):
    return region.x <= px <= region.x + region.w and region.y <= py <= region.y + region.h


def test_finds_each_printed_label_and_not_the_outline():
    img, (x0, y0) = sketch([("60", 560, 260), ("40", 940, 520), ("5", 560, 780)])
    regions = find_text_regions(img)
    assert 3 <= len(regions) <= 5
    for cx, cy in [(590, 240), (970, 500), (575, 760)]:
        assert any(contains(r, cx, cy) for r in regions), (cx, cy)
    # no region should be as wide as the part outline
    assert all(r.w < 300 for r in regions)
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_ocr.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# s2c/ocr.py
"""Dimension annotations on sketches and drawings: find, read, link."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from s2c.metrology import resize_longest


@dataclass
class Region:
    x: int
    y: int
    w: int
    h: int


def _binarize(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)


def strip_lines(binary: np.ndarray, min_len: int = 80) -> np.ndarray:
    """Remove long horizontal and vertical strokes (outline, dimension lines), keep the text."""
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_len))
    lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel) | cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    lines = cv2.dilate(lines, np.ones((5, 5), np.uint8))
    return cv2.bitwise_and(binary, cv2.bitwise_not(lines))


def find_text_regions(image_bgr: np.ndarray) -> list[Region]:
    small, f = resize_longest(image_bgr)
    text = strip_lines(_binarize(small))
    # glue characters of one number together
    glued = cv2.morphologyEx(text, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(glued, connectivity=8)
    regions = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 120 or h < 12 or h > 200 or w > 400 or w / max(h, 1) > 8:
            continue
        pad = 6
        regions.append(Region(int((x - pad) / f), int((y - pad) / f), int((w + 2 * pad) / f), int((h + 2 * pad) / f)))
    return regions
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_ocr.py -v`
Expected: PASS. If the outline survives as a region, raise `min_len` or the dilate kernel; if digits split, widen the close kernel.

- [ ] **Step 5: Try it on a golden sketch**

```bash
uv run python -c "
import cv2; from s2c.ocr import find_text_regions
img = cv2.imread('tests/golden/plate_60x40x5_2holes/image.jpg')
for r in find_text_regions(img): cv2.rectangle(img, (r.x, r.y), (r.x+r.w, r.y+r.h), (0,0,255), 3)
cv2.imwrite('tmp/regions.jpg', img)
"
```

Open `tmp/regions.jpg`. Every written number should have a box.

- [ ] **Step 6: Commit**

```bash
git add s2c/ocr.py tests/test_ocr.py
git commit -m "Find handwritten dimension regions on a sketch"
```

---

### Task 4: Readers, VLM-assisted by default, trained model optional

**Files:**
- Create: `s2c/ocr_readers.py`
- Test: `tests/test_ocr_readers.py`

**Interfaces:**
- Produces: `Reading` dataclass `(value_mm: float | None, kind: Literal["linear","diameter","radius"], confidence: float)`; `Reader = Callable[[np.ndarray], Reading]`; `vlm_reader(client: VLMClient) -> Reader`; `parse_reading(text: str) -> Reading` (pure, tested). A trained model is any function with the `Reader` signature.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ocr_readers.py
import numpy as np
from s2c.ocr_readers import parse_reading, vlm_reader
from s2c.vision.client import VLMClient


def test_parse_reading_handles_plain_and_diameter_and_radius():
    assert parse_reading('{"text": "60", "confidence": 0.9}').value_mm == 60.0
    r = parse_reading('{"text": "Ø6", "confidence": 0.8}')
    assert r.value_mm == 6.0 and r.kind == "diameter"
    r = parse_reading('{"text": "D 12.5", "confidence": 0.8}')
    assert r.value_mm == 12.5 and r.kind == "diameter"
    r = parse_reading('{"text": "R3", "confidence": 0.8}')
    assert r.value_mm == 3.0 and r.kind == "radius"
    assert parse_reading('{"text": "hello", "confidence": 0.8}').value_mm is None
    assert parse_reading("garbage").value_mm is None


def test_vlm_reader_sends_crop_and_never_asks_for_estimates(tmp_path):
    seen = {}
    def chat(messages):
        seen["system"] = messages[0]["content"]
        return '{"text": "40", "confidence": 0.95}'
    client = VLMClient(chat=chat, model="fake", log_path=tmp_path / "l")
    reading = vlm_reader(client)(np.full((40, 80, 3), 255, np.uint8))
    assert reading.value_mm == 40.0
    assert "transcribe" in seen["system"].lower() and "estimate" in seen["system"].lower()
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_ocr_readers.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# s2c/ocr_readers.py
"""Read one cropped annotation. The model transcribes what the user wrote; it never estimates."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Literal

import cv2
import numpy as np

from s2c.vision.client import VLMClient

Kind = Literal["linear", "diameter", "radius"]


@dataclass
class Reading:
    value_mm: float | None
    kind: Kind
    confidence: float


Reader = Callable[[np.ndarray], Reading]

SYSTEM = (
    "You transcribe a single handwritten or printed dimension label cropped from an engineering sketch. "
    "Return ONLY JSON: {\"text\": \"<exactly the characters written>\", \"confidence\": 0..1}. "
    "Transcribe, do not estimate, do not guess a size from the drawing, do not convert units. "
    "Diameter symbols may appear as Ø, ⌀, D or phi. Radius as R. If unreadable, return {\"text\": \"\", \"confidence\": 0}."
)

_NUM = re.compile(r"(?P<prefix>[øØ⌀]|phi|dia\.?|d|r)?\s*(?P<num>\d+(?:[.,]\d+)?)", re.I)


def parse_reading(text: str) -> Reading:
    try:
        data = json.loads(text[text.find("{"): text.rfind("}") + 1])
        raw, conf = str(data.get("text", "")), float(data.get("confidence", 0))
    except (ValueError, AttributeError):
        return Reading(None, "linear", 0.0)
    m = _NUM.search(raw)
    if not m:
        return Reading(None, "linear", 0.0)
    value = float(m.group("num").replace(",", "."))
    prefix = (m.group("prefix") or "").lower()
    kind: Kind = "radius" if prefix == "r" else "diameter" if prefix else "linear"
    return Reading(value, kind, max(0.0, min(1.0, conf)))


def vlm_reader(client: VLMClient) -> Reader:
    def read(crop: np.ndarray) -> Reading:
        crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC) if crop.shape[0] < 60 else crop
        ok, buf = cv2.imencode(".png", crop)
        return parse_reading(client.complete_json(SYSTEM, "Transcribe this label.", buf.tobytes(), mime="image/png"))
    return read
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_ocr_readers.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add s2c/ocr_readers.py tests/test_ocr_readers.py
git commit -m "Add annotation readers with a VLM transcriber by default"
```

Optional trained-model path, only if you already have one working: add `trained_reader(model_path) -> Reader` in the same file that runs your recogniser on the crop and returns a `Reading`. Select it with `OCR_READER=trained` in `read_annotations` (Task 5). Report its accuracy next to the VLM reader's in the README.

---

### Task 5: Linking values to fields and the `read_annotations` entry point

**Files:**
- Modify: `s2c/ocr.py`
- Test: `tests/test_ocr.py` (append)

**Interfaces:**
- Consumes: `Topology` (hole positions as fractions), `Reader`.
- Produces: `link(regions, readings, part_bbox, topology) -> list[Annotation]`; `part_outline_bbox(image_bgr) -> tuple[int,int,int,int]`; `read_annotations(image_bgr, topology, reader: Reader | None = None) -> Annotations`.

Linking rules for version 1:

1. `diameter` readings link to `hole_diameter`; `hole_index` is the nearest topology hole in pixels if one is within 15 percent of the part width, else `None` (shared).
2. `radius` readings link to `corner_radius`.
3. `linear` readings inside or touching the main outline bounding box, horizontally elongated placement (region centre above or below the box), link to `width`. Region centre left or right of the box links to `height`.
4. `linear` readings far outside the box (more than 25 percent of the box width away) link to `thickness` when the topology says the part is a plate or bracket. Otherwise `unknown`.
5. Any remaining `linear` reading is `unknown`, and `merge.py` assigns it largest-first.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_ocr.py
import numpy as np
from s2c.ocr import read_annotations, link, Region, part_outline_bbox
from s2c.ocr_readers import Reading
from s2c.partspec.models import Topology


def test_part_outline_bbox_finds_the_rectangle():
    img, (x0, y0) = sketch([])
    x, y, w, h = part_outline_bbox(img)
    assert abs(x - x0) < 8 and abs(y - y0) < 8 and abs(w - 600) < 12 and abs(h - 400) < 12


def test_link_by_position():
    bbox = (300, 300, 600, 400)
    topo = Topology.model_validate({"part_type": "plate", "holes": [{"u": 0.1, "v": 0.85}], "confidence": 0.9})
    regions = [Region(560, 230, 60, 40),   # above the box -> width
               Region(920, 480, 60, 40),   # right of the box -> height
               Region(560, 900, 40, 40),   # far below (more than 25% of the width away) -> thickness
               Region(340, 330, 60, 40)]   # near the top-left hole -> hole diameter, index 0
    readings = [Reading(60.0, "linear", 0.9), Reading(40.0, "linear", 0.9),
                Reading(5.0, "linear", 0.9), Reading(6.0, "diameter", 0.9)]
    ann = link(regions, readings, bbox, topo)
    by_value = {a.value_mm: a for a in ann}
    assert by_value[60.0].linked_to == "width"
    assert by_value[40.0].linked_to == "height"
    assert by_value[5.0].linked_to == "thickness"
    assert by_value[6.0].linked_to == "hole_diameter" and by_value[6.0].hole_index == 0


def test_read_annotations_end_to_end_with_injected_reader():
    img, _ = sketch([("60", 560, 260), ("40", 940, 520), ("5", 560, 780)])
    topo = Topology.model_validate({"part_type": "plate", "holes": [], "annotation_count": 3, "confidence": 0.9})
    answers = iter([Reading(60.0, "linear", 0.9), Reading(40.0, "linear", 0.9), Reading(5.0, "linear", 0.9)])
    ann = read_annotations(img, topo, reader=lambda crop: next(answers))
    assert sorted(a.value_mm for a in ann.items) == [5.0, 40.0, 60.0]
    assert {a.linked_to for a in ann.items} <= {"width", "height", "thickness", "unknown"}
    assert 0 < ann.confidence <= 1
```

Note: the region order given to the reader is left-to-right, top-to-bottom by region centre; the test relies on that order for the fake reader. Sort regions by `(y // 100, x)` before reading.

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_ocr.py -v`
Expected: the new tests FAIL with `ImportError`.

- [ ] **Step 3: Implement**

```python
# append to s2c/ocr.py
import os

from s2c.ocr_readers import Reader, Reading, vlm_reader
from s2c.partspec.models import Annotation, Annotations, Topology


def part_outline_bbox(image_bgr: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box of the largest closed outline, in original pixels."""
    small, f = resize_longest(image_bgr)
    binary = _binarize(small)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = image_bgr.shape[:2]
        return 0, 0, w, h
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return int(x / f), int(y / f), int(w / f), int(h / f)


def _centre(r: Region) -> tuple[float, float]:
    return r.x + r.w / 2, r.y + r.h / 2


def link(regions: list[Region], readings: list[Reading], part_bbox: tuple[int, int, int, int],
         topology: Topology) -> list[Annotation]:
    bx, by, bw, bh = part_bbox
    far = 0.25 * bw
    hole_px = [(bx + h.u * bw, by + (1 - h.v) * bh) for h in topology.holes]
    out: list[Annotation] = []
    for r, rd in zip(regions, readings):
        if rd.value_mm is None:
            continue
        cx, cy = _centre(r)
        linked, hole_index = "unknown", None
        if rd.kind == "diameter":
            linked = "hole_diameter"
            if hole_px:
                d, idx = min((((cx - hx) ** 2 + (cy - hy) ** 2) ** 0.5, i) for i, (hx, hy) in enumerate(hole_px))
                hole_index = idx if d < 0.15 * bw + r.w else None
        elif rd.kind == "radius":
            linked = "corner_radius"
        else:
            inside_x = bx - far <= cx <= bx + bw + far
            inside_y = by - far <= cy <= by + bh + far
            above_or_below = inside_x and (cy < by or cy > by + bh) and abs(cy - (by if cy < by else by + bh)) <= far
            left_or_right = inside_y and (cx < bx or cx > bx + bw) and abs(cx - (bx if cx < bx else bx + bw)) <= far
            if above_or_below:
                linked = "width"
            elif left_or_right:
                linked = "height"
            elif not (inside_x and inside_y) and topology.part_type in ("plate", "l_bracket"):
                linked = "thickness"
        out.append(Annotation(value_mm=float(rd.value_mm), kind=rd.kind,
                              bbox_px=(float(r.x), float(r.y), float(r.w), float(r.h)),
                              linked_to=linked, hole_index=hole_index, confidence=float(rd.confidence)))
    return out


def read_annotations(image_bgr: np.ndarray, topology: Topology, reader: Reader | None = None) -> Annotations:
    if reader is None:
        from s2c.vision.client import VLMClient
        reader = vlm_reader(VLMClient.from_env())
    regions = sorted(find_text_regions(image_bgr), key=lambda r: (_centre(r)[1] // 100, _centre(r)[0]))
    readings = [reader(image_bgr[max(r.y, 0): r.y + r.h, max(r.x, 0): r.x + r.w]) for r in regions]
    items = link(regions, readings, part_outline_bbox(image_bgr), topology)
    if not items:
        return Annotations(items=[], confidence=0.0)
    expected = topology.annotation_count or len(items)
    coverage = min(1.0, len(items) / max(expected, 1))
    conf = float(np.mean([a.confidence for a in items]) * coverage)
    return Annotations(items=items, confidence=conf)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_ocr.py tests/test_ocr_readers.py -v`
Expected: PASS. Then `uv run pytest -q` for the full suite; the pipeline now uses the real OCR.

- [ ] **Step 5: Run on the golden sketches with a real key**

```bash
uv run pytest tests/test_golden.py -v
```

Expected: at least 7 of the 10 sketch cases pass by 24 September. For each failure, open the lab UI (`uv run python app_gradio.py`) and look at the Annotations JSON to see whether the value was misread (reader problem) or mislinked (linking problem).

- [ ] **Step 6: Commit**

```bash
git add s2c/ocr.py tests/test_ocr.py
git commit -m "Link read dimensions to part fields using position and topology"
```

---

### Task 6: Accuracy report for the README

**Files:**
- Create: `scripts/accuracy_report.py`

- [ ] **Step 1: Write the script**

```python
# scripts/accuracy_report.py
"""Value accuracy, linking accuracy and coin scale error on the golden set. Prints a markdown table."""
import json
from pathlib import Path

import cv2
from dotenv import load_dotenv

from s2c.metrology import measure
from s2c.ocr import read_annotations
from s2c.partspec.models import Measurements
from s2c.vision.client import VLMClient
from s2c.vision.topology import topology_from_image

load_dotenv()
client = VLMClient.from_env()
cases = sorted(p for p in Path("tests/golden").iterdir() if (p / "expected.json").exists())

ocr_hits = ocr_total = link_hits = 0
scale_errors = []
for case in cases:
    e = json.loads((case / "expected.json").read_text())
    img_path = next(p for p in case.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    img = cv2.imread(str(img_path))
    truth = {k.removesuffix("_mm"): v for k, v in e["part"].items() if isinstance(v, (int, float)) and k != "type"}
    if e["input_kind"] == "photo":
        m = measure(img)
        if isinstance(m, Measurements) and "width" in truth:
            scale_errors.append(abs(m.bbox_mm.width - truth["width"]) / truth["width"])
        continue
    topo = topology_from_image(img_path.read_bytes(), client, e["input_kind"])
    ann = read_annotations(img, topo)
    for a in ann.items:
        ocr_total += 1
        if any(abs(a.value_mm - v) < 0.01 for v in truth.values()):
            ocr_hits += 1
            key = a.linked_to if a.linked_to != "inner_diameter" else "bore_diameter"
            if key in truth and abs(truth[key] - a.value_mm) < 0.01:
                link_hits += 1

print("| Metric | Value |\n| --- | --- |")
print(f"| OCR values read correctly | {ocr_hits}/{ocr_total} |")
print(f"| Values linked to the right field | {link_hits}/{ocr_total} |")
if scale_errors:
    print(f"| Photo width error (mean) | {100 * sum(scale_errors) / len(scale_errors):.1f}% |")
```

- [ ] **Step 2: Run it and paste the table into the README**

Run: `uv run python scripts/accuracy_report.py`

- [ ] **Step 3: Commit**

```bash
git add scripts/accuracy_report.py README.md
git commit -m "Report OCR and metrology accuracy on the golden set"
```

---

## Self-review against the spec

- Rule 2: `metrology.py` has no model call. `ocr_readers.py` asks the model to transcribe only, tested in `test_vlm_reader_sends_crop_and_never_asks_for_estimates`.
- Rule 4 coin gates: `coin_tilted`, `coin_not_found`, `part_not_found` with remedies, tested in Task 2.
- Contracts: `read_annotations(image_bgr, topology) -> Annotations` and `measure(image_bgr) -> Measurements | Abstain` match the fakes and the `Pipeline` fields in the integrator plan.
- Coordinate convention: `measure` reports mm from the bottom-left of the part bbox, y up, tested with the two-hole plate.
- Accuracy numbers for the README (spec 4.7): Task 2 step 5 and Task 6.
- Known simplification: linking uses position only. Arrow detection is a stretch goal if 7 of 10 golden sketches do not pass by 24 September.
