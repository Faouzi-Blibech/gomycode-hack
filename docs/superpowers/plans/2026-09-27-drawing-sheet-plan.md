# Drawing Sheets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One image with several orthographic views is split into views, each view gets its face name, and the part is built through the existing multi-view pipeline.

**Architecture:**
- A new `sheet.py` splits a sheet image into views and names them: labels, layout and the projection symbol, all checked against the shared scale.
- The outline stage learns line drawings: spurs, openings and hidden lines. Fuse decides which drawn circles are holes.
- A turned part with one side view gets the other one copied.
- The Studio expands a dropped sheet into tagged view items.

**Tech Stack:** Python 3.11, uv, pytest, OpenCV, numpy, CadQuery, Gradio 6.28.

**Spec:** `docs/superpowers/specs/2026-09-27-drawing-sheet-design.md`. Read the whole spec: its section numbers are cited below.

## Global Constraints

- `s2c/multiview/spec.py` is frozen. No new dependencies. ruff line length 120. TDD.
- Implementers do not commit, stage, stash or push, and never run `git checkout --` or `git reset`. There is no AI attribution anywhere.
- CLAUDE.md rule 2: a sheet gives proportions only, never millimetres. Sizes stay typed, measured or accepted by the user.
- Behaviour for sketches, photos and filled renders (the reverse-engineering benchmark feeds renders as kind "drawing") must not change. The line-drawing rules apply only when kind is "drawing" and the view is line art (spec 3.3).
- Test fixtures are synthetic, drawn by our own helper. Never commit the user's sample image or anything from the `Reverce engineering` dataset.

## Review Focus

1. A photo or sketch that happens to contain two separate blobs (a part and its shadow, two sketches side by side) must not be split into a sheet by accident. Only an item with face "auto" is split. The split then needs 2 or more views that are aligned in a row or column (centres within 10 %) and pass the scale check.
2. A filled render sent as kind "drawing" (the benchmark) must keep its current outline, openings and holes exactly.
3. A wrong or unreadable label never wins over a consistent layout, and a view with no consistent name stays "auto" and never gets a guessed face.
4. The projection symbol is never built as the part when another drawing is present.
5. Changing the projection switch after upload renames sheet views but never overwrites a face the user picked by hand.

---

### Task 1: `split_sheet` (spec 3.1) and the synthetic sheet test helper

**Files:**
- Create: `s2c/multiview/sheet.py` (split part only), `tests/sheet_helpers.py`, `tests/test_mv_sheet_split.py`

**Interfaces (produced):**
- `@dataclass View: box: tuple[int, int, int, int]` (x, y, w, h in image px); `label_box: tuple[int, int, int, int] | None`
- `@dataclass Drawing: views: list[View]`
- `@dataclass Sheet: drawings: list[Drawing]; shape: tuple[int, int]; warnings: list[str]`
- `split_sheet(image_bgr: np.ndarray) -> Sheet`
- `is_sheet(sheet: Sheet) -> bool`: True when some drawing has 2 or more views that are aligned (Review Focus 1).
- `tests/sheet_helpers.py`:
  - `draw_sheet(views: dict[str, np.ndarray], layout: str = "first", labels: bool = True, gap: int = 60, line: int = 2, symbol: str | None = None, border: bool = False, centre_lines: bool = False, hidden: dict[str, list] | None = None) -> np.ndarray`.
  - `views` maps a face to a binary silhouette mask; each is drawn as its contours (visible lines) on white.
  - The views are placed per the projection (spec section 2), with a text label ("FRONT VIEW", "TOP VIEW", …) under each when `labels`.
  - `symbol` in {"first", "third"} draws the projection symbol as a separate drawing on the right.
  - `hidden` maps a face to dashed line segments to draw.
  - It also returns the ground-truth boxes: `draw_sheet(...) -> tuple[np.ndarray, dict[str, tuple]]`.
  - `part_views(solid_or_mesh, env) -> dict[str, np.ndarray]`: face masks from `raster.face_mask` for front, top, right, left, back and bottom.

- [ ] **Step 1: Failing tests** (`tests/test_mv_sheet_split.py`, fixtures from `sheet_helpers` with an L-bracket box solid):
  - `test_a_first_angle_sheet_splits_into_its_views`: 5 views give 5 `View`s whose boxes match the truth within 3 px, each with its `label_box`.
  - `test_centre_lines_and_inner_circles_stay_inside_their_view`: `centre_lines=True`, and a front view with a hole circle, still give 5 views.
  - `test_the_border_frame_is_removed`: `border=True` gives the same 5 views.
  - `test_two_drawings_are_two_drawings`: sheet plus `symbol="first"` gives 2 drawings, 5 views and 2 views.
  - `test_a_single_part_photo_is_not_a_sheet`: one filled blob, and also two blobs that do not align, give `is_sheet == False`.
  - `test_a_black_separator_band_splits_drawings`: a 25 px full-width black band between two drawings gives 2 drawings.
- [ ] **Step 2: Run** `uv run pytest tests/test_mv_sheet_split.py -v`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** spec 3.1 exactly (the thresholds are in the spec).
- [ ] **Step 4: Run** the tests, then ruff on the touched files. Expected: PASS.

### Task 2: Line drawings in the outline stage (spec 3.3)

**Files:**
- Modify: `s2c/multiview/outline.py` (new `line_art` path in `extract`), `s2c/multiview/fuse.py` (`Observation` gains `line_art: bool = False`, `hidden: list = field(default_factory=list)`), `s2c/multiview/pipeline.py` (`observe` passes `drawing=kind == "drawing"` and copies `line_art` / `hidden` into the Observation)
- Test: `tests/test_mv_line_art.py`

**Interfaces (produced):**
- `PixelOutline` gains `line_art: bool = False` and `hidden: list[tuple[str, float, float, float]]`. Each entry is ("h" | "v", pos, start, end), normalised 0..1 to the outline bbox; "h" means a horizontal line at y = pos from x = start to x = end.
- `extract(image_bgr, mask_out=(), band=EDGE_BAND_PX, drawing: bool = False) -> PixelOutline | MvAbstain`. The line-art rules run only when `drawing` is set and ink fills under 35 % of the filled outer region.
- `find_hidden_lines(ink: np.ndarray, bbox) -> list[tuple[str, float, float, float]]`

- [ ] **Step 1: Failing tests:**
  - `test_a_centre_line_does_not_stretch_the_outline`: a drawn 400 × 200 rectangle with a horizontal chain line extending 80 px beyond both ends gives a bbox of 400 ± 6 by 200 ± 6.
  - `test_inner_visible_edges_are_not_openings`: a drawn rectangle with an inner line dividing it, plus an inner drawn rectangle, gives `inner == []`.
  - `test_a_drawn_circle_is_still_a_circle`: a drawn circle inside a drawn rectangle gives 1 circle.
  - `test_dashed_lines_are_found_as_hidden_lines`: two dashed vertical lines inside a drawn rectangle give 2 "v" entries at the right positions ± 0.03.
  - `test_a_filled_render_is_unchanged_in_drawing_mode`: a filled grey part with a see-through slot and a hole gives the same `PixelOutline` (outer, inner, circles, bbox) with `drawing=True` as with `drawing=False`, and `line_art` is False.
  - `test_sketches_are_unchanged`: the existing sketch fixtures from `tests/test_mv_outline.py` give identical results (kind sketch does not pass `drawing`).
- [ ] **Step 2: Run** `uv run pytest tests/test_mv_line_art.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest tests/test_mv_line_art.py tests/test_mv_outline.py tests/test_mv_pipeline.py tests/test_mv_rescue.py tests/test_mv_merge_views.py -q` and ruff. Expected: PASS.

### Task 3: `name_views` and `crop_views` (spec 3.2)

**Files:**
- Modify: `s2c/multiview/sheet.py` (add naming; do not change Task 1's split behaviour)
- Test: `tests/test_mv_sheet_name.py`

**Interfaces:**
- Consumes: Task 1's `Sheet`, `Drawing`, `View`, `split_sheet`, and `tests/sheet_helpers.draw_sheet`.
- Produces:
  - `@dataclass Naming: drawing: int` (index into `sheet.drawings`); `faces: list[str]` (per view, a face or "auto"); `projection: str` ("first" | "third"); `projection_source: str` ("symbol" | "setting"); `warnings: list[str]`.
  - `name_views(sheet: Sheet, image_bgr: np.ndarray, projection: str = "first", reader: Reader | None = None) -> Naming`
  - `crop_views(sheet: Sheet, image_bgr: np.ndarray, naming: Naming) -> list[tuple[bytes, str]]`: PNG bytes and face, per view of the part drawing, with a 4 % white margin.
  - `find_symbol(sheet, image_bgr) -> tuple[int, str] | None`: the drawing index and "first" | "third".
- Label matching is pure: `label_face(text: str) -> str | None`. It is tested directly with "Right Side View" → right, "REAR VIEW" → back, "Plan" → top, "Front Elevation" → front, "Frnt view" → front, "Section A-A" → None.

- [ ] **Step 1: Failing tests** (synthetic sheets; the reader is a fake that returns the drawn label text for a label crop):
  - `test_layout_names_a_first_angle_sheet_without_labels`: `labels=False, layout="first"` names all 5 views correctly.
  - `test_layout_names_a_third_angle_sheet_with_the_switch`: `layout="third"`, `projection="third"`: all correct.
  - `test_labels_name_the_views_when_read`: `labels=True` with the fake reader, and the sheet drawn in third-angle but `projection="first"`: the labels win, all correct.
  - `test_a_label_that_does_not_match_its_size_is_overruled`: the bottom view is labelled "REAR VIEW" (the user's case). It is named bottom, and the warning mentions "does not match its size".
  - `test_the_projection_symbol_sets_the_projection_and_is_not_the_part`: sheet drawn third-angle, `symbol="third"`, `projection="first"`: projection "third", source "symbol", the part drawing is the 5-view one, all faces correct.
  - `test_a_lone_symbol_is_the_part`: only the truncated-cone symbol (first-angle): 2 views, the trapezoid named front, the circles named per first-angle (left).
  - `test_an_unnamable_view_stays_auto`: an extra view not aligned with anything is "auto".
  - `test_crops_keep_the_view_and_its_scale`: each crop's non-white bbox equals the view box size within 2 px.
  - `test_label_face_keywords`: as listed above.
- [ ] **Step 2: Run** `uv run pytest tests/test_mv_sheet_name.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** spec 3.2.
- [ ] **Step 4: Run** `uv run pytest tests/test_mv_sheet_split.py tests/test_mv_sheet_name.py -q` and ruff. Expected: PASS.

### Task 4: Which drawn circles are holes (spec 3.4), and the missing side view of a turned part (spec 3.5)

**Files:**
- Modify:
  - `s2c/multiview/fuse.py`: new `classify_drawn_circles`; `features_from(observations, env, edges=None)` skips the circles in `edges`.
  - `s2c/multiview/turned.py`: new `complete_turned`.
  - `s2c/multiview/pipeline.py`: `fuse` calls both, and adds their warnings.
- Test: `tests/test_mv_drawn_circles.py`

**Interfaces:**
- Consumes: Task 2's `Observation.line_art` and `Observation.hidden`, and the `PixelOutline.hidden` format.
- Produces:
  - `classify_drawn_circles(observations: list[Observation], env: Envelope) -> tuple[dict[int, set[int]], list[str]]`: observation index → the circle indices that are edges, plus the warnings.
  - `complete_turned(outlines: dict[str, Outline], env: Envelope) -> tuple[dict[str, Outline], list[str]]`: adds the missing side view, with source "inferred", when spec 3.5 applies. In `fuse`, the added face's provenance is "inferred".

- [ ] **Step 1: Failing tests** (Observations built from drawn line-art images through `outline.extract(..., drawing=True)`):
  - `test_the_small_end_of_a_cone_is_an_edge`: side view is a trapezoid, end view has two concentric circles whose inner diameter equals the trapezoid's short side. The inner circle is an edge, the outer circle is the outline, and there is no hole feature.
  - `test_a_washer_bore_is_a_hole`: rectangle side view, concentric circles: the bore is a hole.
  - `test_hidden_lines_make_a_hole`: front circle Ø8 with a pair of dashed lines at ±4 mm in the top view: a hole.
  - `test_a_lug_hole_without_hidden_lines_is_a_hole`: bracket front view with a circle in a rounded lug, top view plain: a hole, with the warning "no other view explains it".
  - `test_a_two_view_cone_builds_as_a_cone`:
    - Given only front (the trapezoid) and left (the circles), built through `MvPipeline().observe(...)` + `fuse` + `build`, with the envelope typed.
    - Expected: the volume is within 3 % of the frustum `πL(R² + Rr + r²)/3`, and the spec has no hole.
  - `test_filled_renders_keep_every_circle`: kind "drawing" but filled (`line_art` False): `classify_drawn_circles` returns no edges.
- [ ] **Step 2: Run** `uv run pytest tests/test_mv_drawn_circles.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** spec 3.4 and 3.5.
- [ ] **Step 4: Run** `uv run pytest tests/test_mv_drawn_circles.py tests/test_mv_fuse.py tests/test_mv_turned.py tests/test_mv_pipeline.py tests/test_studio_pipeline.py -q` and ruff. Expected: PASS.

### Task 5: Studio: split on upload, projection switch, example sheet (spec 3.6)

**Files:**
- Modify:
  - `s2c/studio/handlers.py`: `add_images` splits sheets; new `set_projection`; `load_sheet_example`.
  - `s2c/studio/session.py`: `Item` gains `sheet_id: str | None = None`, `view: int | None = None`, `hand_face: bool = False`. `Session` gains `projection: str = "first"`, `sheets: dict[str, tuple]` (sheet_id → (image path, Sheet, Naming)) and `sheet_notes: list[str]`.
  - `s2c/studio/app.py`: projection radio in Capture, a "Try a drawing sheet" button, and the capture card with the sheet notes.
- Create: `examples/mv/sheet/sheet.png` (generated by a small script `scripts/make_example_sheet.py` using `tests/sheet_helpers`, from a flanged bracket built in CadQuery; labels on; first-angle symbol on; committed), `examples/mv/sheet/README.md` (the part's true envelope, so the demo can type it).
- Test: `tests/test_studio_sheet.py`

**Interfaces:**
- Consumes: Task 1 `split_sheet`, `is_sheet`; Task 3 `name_views`, `crop_views`.
- Produces:
  - `Studio.set_projection(sid: str, projection: str) -> None`
  - `Studio.load_sheet_example(sid: str) -> None`
  - Crops are saved under the Studio root, as `<root>/sheets/<sid>/<sheet_id>_<i>.png`. They are removed with the session's other files by the existing sweep.
  - `set_face` sets `hand_face = True`.

- [ ] **Step 1: Failing tests:**
  - `test_dropping_a_sheet_gives_one_item_per_view`: a synthetic 5-view sheet with face "auto" gives 5 items with the right faces, kind "drawing", `sheet_id` set.
  - `test_a_tagged_image_is_never_split`: the same sheet with face "front" stays 1 item.
  - `test_a_photo_is_not_split`: a single blob stays 1 item.
  - `test_the_projection_switch_renames_but_keeps_hand_picked_faces`: switch to "third" renames the views, except one face the user set by hand.
  - `test_the_symbol_overrides_the_switch_and_says_so`: the notes mention the symbol.
  - `test_the_sheet_example_builds`: load the example, analyze with the envelope from its README, build. The part builds, and every face match is at least 0.9.
- [ ] **Step 2: Run** `uv run pytest tests/test_studio_sheet.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement,** and generate the example with the script.
- [ ] **Step 4: Run** `uv run pytest tests/test_studio_sheet.py tests/test_studio_ui.py tests/test_studio_pipeline.py -q` and ruff. Expected: PASS.

### Task 6: Sheet benchmark and docs (spec section 6)

**Files:**
- Modify: `s2c/multiview/benchmark.py` (a `--sheet` mode: compose a first-angle line-art sheet from a part's front, top and right renders with Canny edges, labels on even-numbered parts, then split, name and score with the same metrics plus `named_ok`), `scripts/re_benchmark.py` (the `--sheet` flag), `tests/test_re_benchmark.py`, `README.md` (one accuracy row), the review doc
- The controller runs it on 40 parts (`--limit 40` spread over the categories) and records sheet vs per-face numbers in the ledger.

- [ ] **Step 1: Failing test:** `test_sheet_mode_scores_a_synthetic_part`: the synthetic dataset part in sheet mode gives `result == "built"`, `named_ok is True`, `voxel_iou > 0.85`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the tests, then the 40-part run (controller).

Execution order:
- Wave 1: Tasks 1 and 2 in parallel (disjoint files).
- Wave 2: Tasks 3 and 4 in parallel (sheet.py against fuse/turned/pipeline).
- Then Task 5.
- Then Task 6.
- Then the final review and the push.
