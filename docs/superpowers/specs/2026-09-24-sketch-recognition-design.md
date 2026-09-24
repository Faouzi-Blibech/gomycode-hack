# Hand-sketch recognition: design spec

Date: 2026-09-24
Author: numbers owner (hand-sketch recognition)
Status: design approved section by section in review; written spec pending review
Related: `docs/superpowers/specs/2026-09-19-sketch-to-cad-design.md` (team rules), the multi-view specs on the `geometry/*` branches (the 3D builder that consumes our output)
Research behind the choices: the numbers owner's local research report "Sketch to technical drawing tools" (not in the repo; summary of its conclusions in sections 4 and 5)

## 1. What this module does

One phone photo of a hand-drawn sheet goes in: several labelled orthographic views of one object (TOP, FRONT, SIDE, ...) with handwritten dimensions, dashed hidden lines and centre lines, drawn like a standard engineering drawing. One JSON document comes out, the `SketchReading`: every view, every shape, every line type, every dimension and angle with the geometry it measures, the features the hidden lines imply, and a prediction for every size that is missing or unclear, each with a badge saying where it came from.

### Ownership

| In this module | Owned by the team |
| --- | --- |
| Reading the sheet: views, shapes, line types, text | 3D construction from the JSON |
| Linking every value to the geometry it measures | The clean output sheet (DXF, SVG, PDF) |
| Solving exact sizes, predicting missing or unclear ones | The review and edit screen (we supply its data and `resolve()`) |
| Interpreting hidden lines as holes and slots | Converting the JSON to `MultiViewSpec` if they want it |
| The JSON contract, `resolve(reading, edits)`, the correction record | |
| Accuracy numbers for the pitch | |

### Success criteria

- A `SketchReading` for every sheet in the golden set, or an abstain with the right reason.
- No silent errors on the golden set: every wrong value carries an `uncertain`, `predicted` or `conflict` badge.
- Reported numbers: reader accuracy, line-type precision and recall, linking accuracy, sheets fully correct without edits, edits per sheet, latency per sheet (section 14).

## 2. Rules for this module

The team's four rules, as they apply here:

1. **No model writes code.** Models return text or JSON that we validate with Pydantic. Geometry comes from our own code.
2. **Every number has a source and a badge.** A value is either written by the user and read by two readers, derived exactly from written values, or predicted and marked as such. A model never produces a coordinate or a size; it may transcribe what the user wrote and give a semantic check.
3. **Faithful first.** The JSON contains everything the user drew, including what no builder can use yet. Nothing is dropped silently; anything uninterpreted becomes an issue.
4. **Abstention is a feature.** A stage that cannot work returns an abstain with a reason slug and a one-sentence remedy.

Privacy: the image lives only for the request. Keeping crops for training is opt-in, off by default, crops only (section 10.4).

## 3. Input assumptions

- One sheet per photo, roughly A4, pen or dark pencil on light paper, the whole sheet in frame.
- Views laid out in third-angle projection, as in a standard drawing: TOP above FRONT, the right SIDE to the right of FRONT. Optional BOTTOM, LEFT, BACK in their third-angle places.
- Each view carries a written label: TOP, FRONT, SIDE, RIGHT, LEFT, BOTTOM, BACK, or the French DESSUS, FACE, CÔTÉ, DROITE, GAUCHE, DESSOUS, ARRIÈRE. A missing label falls back to the layout position and becomes an issue.
- Units are millimetres. A sheet whose values look like inches (leading-dot decimals such as `.50` throughout, all values under 10) raises an issue asking "mm or inches?".
- Drawing conventions: thick continuous lines for visible edges, dashed lines for hidden edges, dash-dot lines for centre lines and axes, thin dimension lines with arrowheads between extension lines, leaders for `Ø` and `R`, chained and overall dimensions, angles with `°`, chamfers as `2×45°` or `C2`, hole callouts such as `Ø6 THRU` or `Ø6 ↧10`.
- Any object, not only mechanical parts: outlines may contain free curves.

## 4. Pipeline

```text
photo
  1. capture    find the sheet, rectify perspective, remove shadows, binarise
  2. text       find text, read it with two readers, parse, classify: view label | dimension | other
  3. views      group the ink into views, name each view from its label, fall back to layout
  4. vectorize  skeleton graph, then lines, circles and arcs, free curves
  5. classify   visible | hidden | centre | dimension line | extension line | leader | arrowhead
  6. link       every value to the geometry it measures
  7. solve      one least-squares solve per global axis across all views, badges
  8. predict    fill missing and unclear values from the best available evidence
  9. features   hidden lines plus circles become holes and slots with depth
     output     SketchReading JSON, debug overlays
```

| Stage | Tool | Why |
| --- | --- | --- |
| 1 capture | OpenCV: largest quadrilateral, perspective warp, background division, Sauvola threshold (scikit-image) | exact, fast, no training |
| 2 text | two readers from a pluggable set (section 5), plus parsing | handwriting is where models add real value |
| 3 views | connected ink clusters, label text, third-angle layout rules | deterministic |
| 4 vectorize | skeleton (scikit-image) to a graph (skan, or a small in-house tracer), least-squares lines, RANSAC circles and arcs, smoothed splines | exact; the research found no model that adds value here |
| 5 classify | geometric rules (section 6.1) | no public code or weights exist for hand-drawn line types; rules are explainable |
| 6 link | geometric tracing (section 6) | no existing code anywhere; this is our core contribution |
| 7-8 solve, predict | NumPy and SciPy least squares | deterministic, handles chains, totals, conflicts |
| 9 features | cross-view matching rules | deterministic |

A vision model is used in two places only: as one of the readers in stage 2, and for an optional meaning check (section 5.4). It never overrides a geometric result.

Expected latency: every stage except the vision-model call runs in milliseconds to under a second; a sheet takes a few seconds.

## 5. Reading text

### 5.1 Readers are pluggable

```python
class ReaderResult(BaseModel):
    text: str
    confidence: float  # 0..1

class Reader(Protocol):
    name: str
    def read(self, crops: list[np.ndarray]) -> list[ReaderResult]: ...
```

Adapters in the first version:

- `paddle`: PaddleOCR PP-OCRv5 recognition, local. Also provides text detection for stage 2.
- `trocr`: `microsoft/trocr-base-handwritten`, local.
- `vlm`: any vision model behind the team's OpenAI-compatible `s2c/vision/client.py` (`VLMClient.complete_json`). All crops of one sheet are tiled into one numbered grid image, so one call reads the whole sheet. The reply is validated with Pydantic; on a validation error, one retry with the error appended; on a second failure the reader returns nothing and the pipeline continues with the other reader.

`SKETCH_READERS` selects the pair, for example `paddle,vlm`. No model name is hard-coded; the vision model is set by `VLM_BASE_URL`, `VLM_MODEL`, `VLM_API_KEY` as for the rest of the team. The final pair is chosen by the head-to-head in section 14.3.

The `vlm` prompt asks the model to transcribe exactly what is written in each numbered tile, to use `Ø` for a diameter sign and `R` for a radius, to return `""` when a tile is unreadable, and never to guess a value that is not written.

### 5.2 Two readers, one badge

- Both readers parse to the same value and kind: the value is `written`, confidence is the higher of the two.
- They disagree, or only one reader is available, or a confidence is below 0.6: the value is `uncertain` and the JSON keeps both readings. Section 8.2 proposes the most plausible one.
- Neither parses: the text is kept as `other` with an issue if it sits where a dimension is expected.

### 5.3 Text grammar

```text
text      := [count ("×" | "x" | "X")] [prefix] number [unit] [tolerance] [callout]
prefix    := "Ø" | "⌀" | "∅" | "D" | "R" | "C" | "M"
number    := digits ["." digits] | "." digits          (comma accepted as decimal point)
unit      := "mm" | "°"
tolerance := "±" number | "+" number "/" "-" number
callout   := "THRU" | "TYP" | ("↧" | "DP" | "DEEP") number
```

Kinds: no prefix is `linear`; `Ø`, `⌀`, `∅`, `D` are `diameter`; `R` is `radius`; a trailing `°` is `angle`; `C` or `n×45°` is `chamfer`; `M` is a thread size, kept as text with its nominal diameter. `O` or `o` in digit positions reads as `0`; a leading `o` before digits reads as `Ø`. Tolerances are kept as text next to the nominal value. Labels are matched case-insensitively against the vocabulary in section 3.

### 5.4 Meaning check (optional, should-have)

One vision-model call on the whole rectified sheet returns JSON: the view labels and their rough positions, the number of holes per view, and the list of values it sees. Each disagreement with the geometric result becomes an amber issue. It is never used to change a value.

## 6. Linking

### 6.1 Line types from geometry (stage 5)

- **Arrowhead**: a small filled or open wedge at the end of a skeleton branch, its axis along the branch.
- **Dimension line**: a thin segment with an arrowhead at one or both ends, or a segment between two arrowheads in a chain.
- **Extension line**: a short segment perpendicular to a dimension line at an arrow tip, with a small gap before it reaches the object.
- **Leader**: a segment, possibly bent once, from a text box to an arrowhead that touches a circle or an arc.
- **Hidden**: a chain of three or more short pieces with gaps, each piece continuing the direction of the previous one; the chain may be straight or curved. Pieces and gaps are measured relative to the median stroke length on the sheet.
- **Centre**: the same chaining with alternating long and short pieces, or crossing marks at a circle centre.
- **Visible**: every other continuous stroke.

### 6.2 From a value to what it measures

| Case | Rule |
| --- | --- |
| Value on a dimension line | Match texts to dimension lines one to one by distance to the line's midpoint and span (Hungarian assignment). Each arrow tip leads to its extension line, which leads to the object; the landing point is a reference position (section 7.1). |
| Chained values | Each value sits between its own pair of arrow tips; consecutive values share a tip, hence a reference position. |
| Diameter or radius with a leader | The leader's arrow tip touches a circle or arc: that circle's diameter or that arc's radius. `n×Ø6` applies to that circle and the other circles of the same drawn size in the view. |
| Diameter across a circle | A dimension line through the centre with arrows on the circle: that circle. |
| Angle | A dimension arc between two lines with the value inside or beside it: the angle between those two lines. |
| Chamfer | `C2` or `2×45°` with a leader to a corner: that corner. |
| Hole callout | `Ø6 THRU` or `Ø6 ↧10` with a leader: the hole's diameter and its explicit depth, which overrides depth from hidden lines. |
| Value with no dimension line | Link to the nearest edge parallel to the text's placement: below or above a view is a horizontal extent, left or right is a vertical extent. Confidence is lowered and an amber issue is raised. |
| Nothing fits | Kept as an `unplaced` dimension with an amber issue. Never dropped. |

## 7. Solving

### 7.1 Reference positions

In an orthographic view almost every size is a distance between two positions along one axis. Each view is reduced to two lists of reference positions:

- along its horizontal axis: vertical edges, vertices, circle centres, circle extremes (centre minus radius, centre plus radius), arc extremes;
- along its vertical axis: the same for horizontal edges.

Positions closer than a tolerance (1.5 percent of the view's extent, at least 4 px) merge into one. A horizontal dimension becomes the equation `position_B - position_A = value`.

### 7.2 Global axes and views

Global frame, the same as the team's multi-view spec: X is width, Y is height, Z is depth. Each view has a local frame `(a, b)` in millimetres, `a` to the right and `b` up, origin at the bottom-left of that view's bounding rectangle. The face table is the team's: front `a -> X`, `b -> Y`; top `a -> X`, `b -> Z = z - b`; right `a -> Z = z - a`, `b -> Y`; back, bottom and left are mirrored as in their spec.

Views share axes: FRONT and TOP share X, FRONT and SIDE share Y, TOP and SIDE share Z. For each shared axis, the two views' reference positions are matched: fit a 1D affine map from one view's pixels to the other's using their outer extents, then match inner positions in order within the tolerance. Matched positions become one unknown.

### 7.3 One solve per global axis

Unknowns: every merged reference position on that axis, with the minimum fixed at 0.

- **Hard equations**: every `written` and `edited` value that measures along this axis. An `uncertain` value is left out of a first solve, which gives the size the geometry implies without it (section 8.2); its proposed candidate then enters the final solve as a hard equation, and its badge stays `uncertain`.
- **Angles**: a written angle θ between an edge and the axis gives `Δb - tan(θ)·Δa = 0` between the edge's end positions, linear because θ is known. With both lengths written, the angle is a check.
- **Soft equations**: the drawing's proportions, `p_i - p_j ≈ s·(px_i - px_j)`, with a small weight. The scale `s` is estimated per view and axis as the median of written value over pixel length in that view and axis; with no written value there, from the matched view on the same axis; else from the sheet median. Each view may be drawn at its own scale.

Method: solve the hard equations by least squares. Any hard equation with a residual above tolerance (0.5 percent of the value, at least 0.05 mm) is a conflict candidate; a leave-one-out test names the value whose removal makes the rest consistent. Conflicting values are excluded from the final solve and marked `conflict`. The final solve minimises the soft equations subject to the consistent hard equations exactly (equality-constrained least squares).

### 7.4 Badges

| Badge | Meaning |
| --- | --- |
| `written` | both readers agree on what the user wrote |
| `edited` | the user changed or confirmed it through `resolve()` |
| `derived` | exactly determined by written and edited values (for example 4.00 - 1.50 - 1.50 = 1.00) |
| `uncertain` | the readers disagree or are unsure; a proposal is given (section 8.2) |
| `predicted` | not determined by any written value; filled by the prediction layer, with its evidence |
| `conflict` | contradicts other written values; both values are shown |

`derived` versus `predicted` is decided by linear algebra: a size `p_i - p_j` is derived when the vector `e_i - e_j` lies in the row space of the hard-equation matrix.

### 7.5 Safety nets

- **Decimal trap**: after the solve, each written value implies a pixel-per-mm scale. A value whose scale is more than three times off the median of its view and axis is flagged, with the alternatives ×10, ×0.1, ×0.01 ("did you mean .50?").
- **Conflicts** carry both values and the leave-one-out result.
- **Unused geometry**: an edge that no value touches and whose size is only predicted appears in the issue list with its predicted size.

### 7.6 Free curves

Curves are stored as control points. They are mapped from pixels to millimetres piecewise-linearly between the neighbouring reference positions on each axis, so the drawn shape is kept and every written value is honoured.

## 8. Prediction

### 8.1 Missing values, in order of evidence

| Rank | Evidence | Example | Badge and `evidence` |
| --- | --- | --- | --- |
| 1 | exact from written values | 4.00 - 1.50 - 1.50 = 1.00 | `derived`, `arithmetic` |
| 2 | the same size in another view | TOP width not written, FRONT gives 4.00 | `derived`, `cross_view` |
| 3 | symmetry or repetition | a mirror-symmetric view repeats a written size; identical circles share a written diameter | `predicted`, `symmetry` or `pattern` |
| 4 | standard sizes | a hole drawn at about Ø6.4 snaps to 6.5, an M6 clearance hole | `predicted`, `standard` |
| 5 | drawing proportions | scaled from the sketch | `predicted`, `proportion` |

Symmetry is accepted when the view's reference positions mirror about the centre within the tolerance and a centre line or matching features support it. Standard sizes come from a small table: preferred numbers (R10 series), metric drill and clearance sizes, common sheet and plate thicknesses. A value snaps only when it is within 5 percent of a table entry.

### 8.2 Unclear readings

For an `uncertain` value, each candidate reading (both readers' parses, and the decimal alternatives from section 7.5) is compared with the size the geometry implies without that value. The candidate closest to the implied size is proposed; the JSON keeps all candidates and the implied size. The badge stays `uncertain` until the user confirms.

### 8.3 Missing geometry

- A small gap in an outline (under 3 percent of the view's extent) is closed and noted.
- A circle in one view with no matching hidden lines in the view that should show them becomes a predicted through hole, with an amber issue.
- A missing view is not predicted here; it stays the team's job.

## 9. Features from hidden lines

| Pattern | Feature |
| --- | --- |
| A circle in view A, and in an orthogonal view B a pair of hidden (or visible) parallel lines whose positions equal the circle's centre ± radius on the shared axis | hole along view A's line of sight |
| The pair runs across the material from boundary to boundary | through hole |
| The pair stops inside the material | blind hole, depth from the solved length of the pair |
| An explicit callout (`THRU`, `↧10`) | overrides the depth from hidden lines |
| Two parallel lines closed by two arcs, plus a matching hidden pair | slot |
| A centre line through the pair | extra evidence, raises confidence |

Patterns the team's builder cannot use yet (rectangular blind pockets, counterbores, countersinks, internal cavities) are still reported as entities and appear as `features` with `type: "other"` and an issue, so nothing is lost.

## 10. Output contract

### 10.1 `SketchReading` (Pydantic, `extra="forbid"`)

```python
Badge = Literal["written", "edited", "derived", "uncertain", "predicted", "conflict"]
Evidence = Literal["reader", "user", "arithmetic", "cross_view", "symmetry", "pattern",
                   "standard", "proportion", "geometry"]
ViewName = Literal["front", "top", "right", "left", "bottom", "back"]

class View(_Strict):
    name: ViewName
    label_text: str | None           # None when named from the layout
    bbox_px: tuple[float, float, float, float]
    size_mm: tuple[float, float]      # width and height in the view's (a, b) frame

class Entity(_Strict):
    id: str
    view: ViewName
    type: Literal["line", "arc", "circle", "curve"]
    line_type: Literal["visible", "hidden", "centre"]
    mm: dict                          # line: p0, p1; arc: centre, radius, start_deg, end_deg; circle: centre, radius; curve: points
    px: dict                          # the same in original image pixels
    confidence: float

class Reading(_Strict):
    reader: str                       # adapter name, the model name is in the log
    text: str
    confidence: float

class Dimension(_Strict):
    id: str
    view: ViewName | None             # None when unplaced
    kind: Literal["linear", "diameter", "radius", "angle", "chamfer", "thread"]
    value: float | None               # mm or degrees
    text_raw: str
    readings: list[Reading]
    candidates: list[float] = []      # for uncertain values (section 8.2)
    implied: float | None = None      # size implied by the geometry without this value
    tolerance: str | None = None
    measures: list[str]               # entity ids or reference ids
    axis: Literal["a", "b"] | None = None
    badge: Badge
    evidence: Evidence
    bbox_px: tuple[float, float, float, float]

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
    evidence: list[str]               # entity and dimension ids
    badge: Badge

class Size(_Strict):
    value: float
    badge: Badge
    evidence: Evidence

class Issue(_Strict):
    severity: Literal["red", "amber"]
    kind: Literal["conflict", "uncertain", "predicted", "unplaced", "decimal", "unit",
                  "label", "meaning_check", "not_in_3d", "gap_closed"]
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

Coordinates: `mm` values are in each view's `(a, b)` frame (section 7.2); `position_mm` of a feature is in the global frame; `px` values are in the original photo so a UI can highlight the spot. When `abstain` is set, the lists hold whatever was recovered before the failing stage.

### 10.2 Entry points

```python
def read_sketch(image_bytes: bytes, readers: list[Reader] | None = None) -> SketchReading: ...
def resolve(reading: SketchReading, edits: list[Edit]) -> SketchReading: ...
```

`Edit` is `{target_id, action: "set_value" | "confirm" | "not_a_dimension" | "rename_view", value}`. `resolve()` re-runs linking, solving, prediction and features with the edits applied (edited values become hard equations with the `edited` badge) and without re-reading the image. It runs in milliseconds, so the team's UI can call it after every edit.

### 10.3 Debug overlays

Every stage can write an overlay image to `SKETCH_DEBUG_DIR`: rectified sheet, text boxes with both readings, view boxes, line types in colour, link arrows from values to reference positions, solved sizes with badges. Every reading is logged with its crop. Off by default in production.

### 10.4 Correction records (learning loop)

Every edit received by `resolve()` can be appended to `corrections.jsonl`: reading id, target id, stage, the value before, the value after, both readings, the badge before, a timestamp. The crop is stored only when the user opted in. For the hackathon the loop collects data only; retraining comes later.

## 11. Failure handling

| Situation | Result |
| --- | --- |
| No sheet found | abstain `capture` / `sheet_not_found`: "Put the whole sheet in the frame on a darker surface and retake." |
| Too blurry or too dark | abstain `capture` / `image_quality`: "Retake in good light and hold the phone steady." |
| No ink clusters that look like views | abstain `views` / `no_views_found`: "Draw the views with a dark pen and retake." |
| A view label is missing | named from the layout, amber `label` issue |
| Views cannot be matched on a shared axis | amber issue, the views are solved alone, their shared sizes are `predicted` |
| No values on the sheet | every size `predicted`, red issue: "Write at least the overall width, height and depth." |
| One reader fails | continue with the other, every value `uncertain` |
| Both readers fail | abstain `text` / `readers_unavailable`: "Reading service unavailable. Retry in a minute." |
| Values look like inches | red `unit` issue asking "mm or inches?" |

## 12. Configuration

| Variable | Use | Default |
| --- | --- | --- |
| `SKETCH_READERS` | reader pair, comma-separated | `paddle,vlm` |
| `VLM_BASE_URL`, `VLM_MODEL`, `VLM_API_KEY` | the vision model, as for the team | from `.env` |
| `SKETCH_DEBUG_DIR` | debug overlays, off when unset | unset |
| `SKETCH_CORRECTIONS` | path of `corrections.jsonl`, off when unset | unset |

Event day: the vision model and PaddleOCR are served from an NVIDIA Brev GPU; only these variables change.

## 13. Code layout

```text
s2c/sketch/
  models.py        SketchReading and friends (section 10.1)
  capture.py       stage 1
  readers.py       Reader protocol and the paddle, trocr, vlm adapters
  text.py          stage 2: detection, grammar, classification
  views.py         stage 3
  vectorize.py     stage 4
  classify.py      stage 5
  link.py          stage 6
  solve.py         stage 7 and badges
  predict.py       stage 8, standard-size table
  features.py      stage 9
  corrections.py   correction records
  debug.py         overlays
  pipeline.py      read_sketch, resolve
tests/sketch/      unit tests per stage, vector fixtures, golden runner
tests/golden_sketch/<name>/image.jpg + expected.json
scripts/bakeoff_readers.py
scripts/sketch_accuracy.py
```

The package depends only on `s2c/vision/client.py` from the rest of the codebase. New dependencies: `scikit-image`, `scipy`, `networkx`, `skan`; PaddleOCR and `transformers` (for TrOCR) go into an optional dependency group so the core installs light. Each file stays small and single-purpose.

## 14. Testing and accuracy

### 14.1 Test sets

| Set | Content | Use |
| --- | --- | --- |
| Golden sheets | 15 to 20 real hand-drawn sheets by several people: the reference drawing (a bridge block with four holes) redrawn by hand, and a mug, a phone, a bottle, a bracket, a box with a pocket. Each with `expected.json`: every dimension and what it measures, features, envelope. | the reported numbers |
| Vector fixtures | reference positions and dimensions written in code, for example the FRONT view of the reference drawing (X positions 0, 1.00, 1.50, 2.50, 3.00, 4.00) | exact unit tests of linking, solving, badges, prediction |
| Crops | 200 to 400 labelled value crops cut from the golden sheets | the reader head-to-head |
| Synthetic sheets (optional) | random CadQuery parts projected with OCCT hidden-line removal and drawn in a hand-drawn style | a larger regression set if time allows |

The golden sheets contain no people and no personal data.

### 14.2 Metrics and first targets

| Metric | First target |
| --- | --- |
| Reader exact match on crops, overall and for `.50`-style decimals, `Ø`, `R`, multi-digit | 95 percent for the chosen pair |
| Hidden-line precision and recall | 90 percent |
| Arrowhead precision and recall | 90 percent |
| Linking accuracy | 90 percent of values linked to the right positions |
| Sheets fully correct without edits | reported, no target yet |
| Edits needed per sheet | reported |
| Silent error rate | 0 on the golden sheets |
| Latency per sheet on the event GPU | under 10 s |

Targets are first guesses; the numbers we report are the measured ones.

### 14.3 Reader head-to-head

`scripts/bakeoff_readers.py` runs every candidate reader on the crop set: PaddleOCR PP-OCRv5, TrOCR, Qwen3-VL-8B, Qwen3.5 and Qwen3.8 variants, Nemotron OCR v2, and optionally a hosted frontier model as a ceiling. It reports exact-match accuracy per category, latency and memory. The chosen pair is the one whose errors overlap least while each stays accurate, because agreement is only a safety signal when the two readers fail on different crops. A small CRNN trained on synthetic labels is built only if the best pair misses the target on symbol categories.

### 14.4 Unit tests

Every stage has tests on small synthetic inputs, written before the code. Every abstain reason has a test. The golden runner is a separate, slower test.

## 15. Scope for 27 September

| Priority | Content |
| --- | --- |
| Must | capture; two readers with agreement; grammar; views and labels; lines and circles; hidden, arrowhead, dimension, extension and leader classification; linking for linear, chained, diameter and radius values; solve with cross-view matching and badges; prediction ranks 1, 2 and 5; holes through and blind; JSON and abstains; debug overlays; golden runner and the accuracy numbers |
| Should | angles and chamfers; prediction ranks 3 and 4; unclear-reading proposals; meaning check; `resolve()` with edits; reader head-to-head; Brev setup script; correction records |
| Could | free-curve quality; slots; centre lines as evidence; synthetic sheets; opt-in crop storage |
| Later | CRNN; fine-tuning a vision model on corrections; the photo path |

## 16. Open items

- Units: mm assumed; inches support only if the team asks.
- Label vocabulary: English and French as listed; add Arabic if users need it.
- The team's UI and builder read `SketchReading`; the adapter to `MultiViewSpec` is theirs unless agreed otherwise.
- PaddleOCR's GPU build on Windows may be fragile; its CPU mode is the fallback on the laptop, and the event GPU runs Linux.
- `uv` is not installed on the numbers owner's laptop yet; the project needs Python 3.11 or 3.12.
