# Reverse-Engineering Benchmark and 3D Modeling Fixes Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the 3D modeling against about 400 real parts with known STLs, fix the largest error sources, and close the Track 2 must-fixes before the 2026-09-27 demo.

**Architecture:**
- A benchmark module turns each dataset part into a scored run through the real multi-view pipeline. Inputs are the six face renders, tagged by face, and the true envelope given as user values. Scoring is against the ground-truth STL.
- A CLI runs the benchmark in parallel and writes JSONL results plus a per-category summary.
- Fixes to the modeling come from the baseline data, in a second wave, each with a before/after number.

**Tech Stack:** Python 3.11, uv, pytest, trimesh (voxelize/fill; `rtree` is not installed, so no `contains`), CadQuery, OpenCV, Pillow.

**Spec:** the design was approved in chat on 2026-09-26 (Track 2 of `docs/superpowers/reviews/2026-09-24-project-review.md`, "Benchmark + improve"). The multi-view contracts are in `docs/superpowers/specs/2026-09-22-multiview-gcode-design.md`.

## Global Constraints

- The dataset (`C:\Users\moham\Desktop\GoMyCode\Reverce engineering`, about 400 parts, 336 MB, mostly CC-BY-NC / NC-ND / NC-SA) never enters git. Code reads it from `--dataset` or the `RE_DATASET` environment variable. Nothing derived from it is committed except aggregate numbers.
- Dataset layout: `<category>/<category>_NNN/{front,back,left,right,top,bottom,iso}.png` (800×800 RGBA, white background, shaded orthographic, one shared scale per part), `model.stl`, `metadata.json` (`bounds_mm`, `watertight`, `license`); `index.csv` at the root.
- Frame: the dataset is Z-up. Ours is Y-up with the face frames of `s2c/multiview/raster.py:face_coords`. Map with the proper rotation `ours = (x, z_ds, -y_ds)`, then shift so the minimum corner is at 0. With it, the dataset's `front.png` matches our `face_mask(mesh, "front", env)` (probe: per-view IoU 0.99). Never use the reflection `(x, z, y)`: it flips the volume sign.
- `spec.py` is frozen. No new dependencies. ruff line length 120. TDD. Implementers do not commit, stage, stash or push. There is no AI attribution anywhere.
- CLAUDE.md rule 2: the benchmark passes the true envelope as user values. The report must say so, and label the results "clean renders, true size given", never "phone photos".

## Review Focus

1. A part whose STL is not watertight, or is huge (over 200 k triangles), must not crash or stall the whole run. It is scored as skipped, with a reason, within a per-part time limit.
2. A part where the pipeline abstains counts as "abstain <stage>:<reason>" in the summary. It must not be silently dropped from the denominators.
3. Two parts with the same name in different categories must not overwrite each other's results.
4. A failed fillet in the Studio must not blank the viewer. The suggested size must actually build.
5. After P0-3, no millimetre value produced by the vision model may reach `MultiViewSpec` geometry: through holes stay through unless Solaria or the user makes them blind.

---

### Task 1: Benchmark module, CLI and tests

**Files:**
- Create: `s2c/multiview/benchmark.py`, `scripts/re_benchmark.py`, `tests/test_re_benchmark.py`
- Modify: `pyproject.toml` (register a `dataset` pytest marker only)

**Interfaces (produced):**
- `load_part(part_dir: Path) -> RefPart`. `RefPart` is a dataclass with `name` (`"<category>/<dir>"`), `category`, `mesh` (`raster.Mesh` in our frame, min corner at 0), `envelope` (`spec.Envelope`), `renders` (face → Path), `watertight: bool` and `triangles: int`.
- `score_part(ref: RefPart, pipe: MvPipeline, root: Path, timeout_s: float = 120) -> dict`. It returns one JSON-able row: `part`, `category`, `result` (`"built"`, `"abstain <stage>:<reason>"`, `"skipped <reason>"` or `"error <ExcName>"`), `frame_iou` (mean over six faces of `iou(normalize_mask(face_mask(ref.mesh, f, env)), normalize_mask(render_mask(png)))`), `vol_true`, `vol_built`, `vol_err` (`vol_built / vol_true - 1`), `voxel_iou`, `view_iou` (face → IoU of the rebuilt vs the true silhouette, both from `face_mask` in our frame), `features`, `secs`.
- `render_mask(png: Path) -> np.ndarray`: 255 where the pixel differs from white by more than 30 in the sum of |RGB − 255|.
- `voxel_iou(a: trimesh.Trimesh, b: trimesh.Trimesh, n: int = 64) -> float`. It uses `voxelized(pitch).fill()` with `pitch = max(b.extents) / n`, compares sets of rounded cell indices, and never uses `contains`.
- `summarize(rows: list[dict]) -> dict`. Per category and overall, it reports `n`, `built`, `abstained` (reason → count), `skipped`, `errors`, and the medians of `abs(vol_err)`, `voxel_iou` and the mean view IoU (over built rows only), plus the share of built parts with `abs(vol_err) <= 0.05`.
- `summary_markdown(summary: dict) -> str`: a table plus a header line stating "clean renders, true envelope given as user values".
- `scripts/re_benchmark.py main(argv=None)`:
  - Arguments: `--dataset` (defaults to `RE_DATASET`), `--out` (default `tmp/re_bench/<UTC timestamp>`), `--category` (repeatable), `--parts` (comma list of `<category>/<dir>`), `--limit N`, `--jobs 4`, `--timeout 120`.
  - It runs `score_part` in a `ProcessPoolExecutor`, writes `results.jsonl` (one row per part, flushed as they finish) and `summary.md` / `summary.json`, and prints the overall line.
  - Pipeline for scoring: `MvPipeline()` with no chat, no reader, no mesh provider, no image generator and no depth. Every render is `ImageInput(png_bytes, face, "drawing")`. User values: `{"envelope.x_mm": .., "envelope.y_mm": .., "envelope.z_mm": ..}`. Build with `artifacts.build_part(spec, root=<out>/parts)`.
  - Skip a part (`"skipped not_watertight"` or `"skipped too_large"`) before running it when the STL is not watertight or has more than 200 000 triangles.
  - Enforce the per-part time limit in the worker; on timeout the row is `"skipped timeout"`.

- [ ] **Step 1: Write the failing tests** (`tests/test_re_benchmark.py`). Use a synthetic dataset built in `tmp_path`, so no real data is needed:
  - A helper `make_part(tmp, name, solid)` exports a CadQuery solid (for example `box(40, 20, 10)` with one Ø6 through hole along the depth). It converts the solid to the dataset frame (the inverse of the mapping: `ds = (x, -z_ours, y_ours)`), writes `model.stl` and `metadata.json` (`bounds_mm`, `watertight`) and six 800×800 PNGs. Each PNG is the part's `face_mask` in our frame, drawn grey (128) on white, with openings left white. The masks are drawn from the ours-frame mesh, so names line up.
  - `test_load_part_maps_the_frame`: the envelope equals the solid's size in our frame within 0.01 mm, and the mesh volume is positive.
  - `test_score_part_builds_a_box_with_a_hole`: `result == "built"`, `abs(vol_err) < 0.06`, `voxel_iou > 0.9`, `frame_iou > 0.95`.
  - `test_not_watertight_is_skipped`: a part whose metadata says `watertight: false` gets `result == "skipped not_watertight"`.
  - `test_summary_counts_abstains_and_skips`: `summarize` over hand-made rows keeps abstained and skipped parts in `n`, and computes medians over built rows only.
  - `test_cli_writes_results_and_summary`: `main(["--dataset", str(tmp), "--out", str(out), "--jobs", "1"])` on a two-part synthetic dataset writes two JSONL lines and a `summary.md` that contains "clean renders".
  - `test_real_dataset_smoke` (marked `@pytest.mark.dataset`, skipped unless `RE_DATASET` is set): score `bracket/bracket_001` and `washer_spacer/washer_spacer_001`. Both build, with `voxel_iou > 0.85`.
- [ ] **Step 2: Run** `uv run pytest tests/test_re_benchmark.py -v`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `benchmark.py` and `scripts/re_benchmark.py` as specified. Register the `dataset` marker in `pyproject.toml`.
- [ ] **Step 4: Run** `uv run pytest tests/test_re_benchmark.py -v` and `RE_DATASET="C:/Users/moham/Desktop/GoMyCode/Reverce engineering" uv run pytest tests/test_re_benchmark.py -m dataset -v`, plus ruff on the touched files. Then run the CLI on 10 parts (`--limit 10 --jobs 4`) and paste its summary into the report.

### Task 2: P0-3 — no model millimetres or model blind flags in geometry

**Files:**
- Modify: `s2c/multiview/label.py`, `s2c/multiview/fuse.py`, `s2c/multiview/merge_views.py`
- Test: `tests/test_mv_label.py`, `tests/test_mv_fuse.py`, `tests/test_mv_merge_views.py`

**Findings:**
- The label prompt asks the vision model for hole-depth guesses in millimetres (`label.py:48`), and `label_image` keeps them (`label.py:84-86`).
- `attach_label` copies them into `Observation.depth_estimates` and copies the model's `blind` flags into `Observation.blind` (`fuse.py:49-52`).
- `features_from` turns them into `depth_mm` with provenance "estimated" (`fuse.py:224-231`).
- `merge_views` votes the model's blind flags (`merge_views.py:184-190`).
- `tests/test_mv_label.py:22` asserts that the depths are kept.

- [ ] **Step 1: Write the failing tests.**
  - `test_mv_label.py`: flip the test at line 22. Model estimates, including `holes[0].depth_mm`, are dropped (`label.estimates == {}`), and the system prompt no longer mentions "estimates" or millimetre guesses.
  - `test_mv_fuse.py`: a label with `blind=True` and a depth estimate gives a feature that is a through hole, with no `depth_mm` from the model. A hole with a Solaria `depth_ratio` still becomes blind with provenance "estimated".
  - `test_mv_merge_views.py`: blind votes that come only from label flags no longer make a merged hole blind.
- [ ] **Step 2: Run** the three test files. Expected: the new assertions FAIL.
- [ ] **Step 3: Fix.**
  - Remove the estimates line from the prompt, and drop every estimate in `label_image`. Keep the field in `MvLabel` so old replies still validate; ignore it.
  - `attach_label` no longer sets `blind` or `depth_estimates` from the label. Only Solaria (`depth_from_image` / `depth_ratio`) may make a hole blind.
  - Delete the `depth_estimates` branch in `features_from`, and the label-vote branch in `merge_views` (Solaria-marked holes keep their current path).
  - If `Observation.depth_estimates` becomes unused, delete it and every reference to it.
- [ ] **Step 4: Run** `uv run pytest tests/test_mv_label.py tests/test_mv_fuse.py tests/test_mv_merge_views.py tests/test_mv_pipeline.py tests/test_mv_depth.py -q` and ruff on the touched files. Grep `s2c/` for `estimates`: only the ignored `MvLabel` field may remain.

### Task 3: P0-11 — a failed finish keeps the last good part and suggests a size that builds

**Files:**
- Modify: `s2c/studio/handlers.py`, `s2c/studio/app.py`, `s2c/multiview/finish.py`
- Test: `tests/test_studio_ui.py`, `tests/test_studio_finish.py`

**Findings:**
- `Studio._model` sets `session.part = None` when the build fails (`handlers.py:341-348`), so a fillet that does not fit blanks the viewer.
- The message does not say which size would work.

- [ ] **Step 1: Write the failing tests.**
  - `test_studio_finish.py::test_largest_finish_that_builds`: `largest_finish(spec, geometry, lo=0.2)` returns a size `s` such that `build(apply_geometry(spec, geometry with finish_mm=s)[0])` succeeds. It returns `None` when even 0.2 fails. It uses at most 7 builds; count them with a wrapper around `build`.
  - `test_studio_ui.py::test_failed_finish_keeps_the_last_part`:
    1. Build a part.
    2. Rebuild it with a fillet too large to fit.
    3. The returned Model has `ok False`, `preview` equal to the previous part's preview, views and stats kept, and a message containing "largest that builds" and the suggested number.
    4. `session.part` is still the previous part.
  - Keep the existing `open_step=2` behaviour of `test_the_build_button_shows_a_failed_finish_in_step_3`.
- [ ] **Step 2: Run** both files. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - `finish.largest_finish(spec, geometry, lo=0.2) -> float | None`: bisect between `lo` and `min(geometry.finish_mm, max_finish_mm(spec))` with at most 7 `build` calls. Call `build.build` directly, not `build_part`, so no files or cache entries are written. Return the largest size that built, rounded down to 0.1 mm.
  - In `_model`, on a finish failure:
    - keep `session.part` as it was;
    - return a Model with `ok False`, the previous part's preview, views and stats, `open_step=2`, and the card "Fillet X mm does not fit. The largest that builds is Y mm." (or "No fillet fits this part." when the result is `None`).
  - A failure that is not a finish failure keeps today's behaviour.
  - In `app.py`, make sure a failed Model does not clear the 3D viewer: map `preview` onto the viewer whenever it is set.
- [ ] **Step 4: Run** `uv run pytest tests/test_studio_ui.py tests/test_studio_finish.py tests/test_studio_api.py -q` and ruff on the touched files.

### Task 4: Baseline run and error analysis (controller)

- [ ] Run `uv run python scripts/re_benchmark.py --dataset "<dataset>" --jobs 4` over the whole dataset and keep `summary.md`.
- [ ] Rank the error sources by their total contribution to `abs(vol_err)` and to `1 - voxel_iou`, per category. Look at the renders and rebuilt views of the ten worst built parts.
- [ ] Append the fix tasks (Task 5 onwards) to this plan, each with its finding, the parts that show it, the change, and the metric it must move. Commit the addendum. Rulings go in the ledger.

### Task 5+: fixes from the baseline

Written after Task 4, from its data. Each fix is TDD, plus a benchmark rerun on the affected category, with before/after numbers in the commit message and the ledger. Inherent visual-hull limits (hidden pockets and cavities) are handled by a warning or an abstain, never by guessing.

### Task 6: README accuracy line

- [ ] After the fixes, rerun the full benchmark. In `README.md`'s accuracy table, add one row: "Reference parts (≈400, clean renders, true size given): built X %, median volume error Y %, median 3D IoU Z", with the date and the command. State that phone photos are still unmeasured.

---

## Addendum (2026-09-26, from the baseline): Tasks 7 and 8

Baseline findings (clean renders, true size given, snapping on, no providers; all 400 parts, `tmp/re_bench/baseline`, HEAD b5b7555):
- Overall: 356 of 400 built (26 abstained, 18 skipped), median |volume error| 15.6 %, median 3D IoU 0.873, 20.5 % of built parts within 5 % volume.
- Almost every error is over-volume (314 of 356 built parts more than 2 % over, 1 under). The rebuilt silhouettes match the true ones (view IoU ≈ 1.0) on most bad parts, so the error is material that no view shows.
- **Finding A, outline snapping.** `fuse.snap` moves every vertex of a scaled outline: points within 0.5 mm of an edge go to the edge, thin walls go to standard thicknesses, and the rest go to the 0.5 mm grid.
  - This squares off curved outlines. On `pulley_001` the round top view gets flats, 0.46 mm beyond the circle.
  - It also collapses thin features. On `bolt_nut_031` the 0.3 mm flange snaps to 0, the flange vanishes, and the hex is stretched to the full width: view IoU 0.35–0.67.
  - Turning outline snapping off on 12 probe parts raised the voxel IoU on 7 of them (for example `gear_002` 0.959 → 0.998, `gear_001` 0.858 → 0.901) and lowered none by more than 0.003.
- **Finding B, turned parts.** For a part that is round along one axis, the hull of three extrusions has square cross-sections wherever the diameter steps down (hubs, collars, chamfered rims).
  - The error source is the knob, pulley, gear, bolt_nut, washer_spacer, shaft_collar and shaft_coupler categories, about half the dataset.
  - A spike that intersects the hull with the solid of revolution of the side profile moved the volume ratio as follows: `knob_001` 1.215 → 1.168, `shaft_coupler_001` 1.10 → 1.076, `washer_spacer_001` 1.022 → 0.992, `shaft_collar_001` 1.087 → 1.067.
  - `pulley_001` and `gear_001` were not detected until Finding A is fixed, because their round views had been squared off.
- **Inherent limits (not fixed, documented).**
  - Open boxes and trays (`bearing_holder`), three-plate corner brackets (`bracket_012/013/014/041`) and blind pockets. Every silhouette is right, yet the hull fills them in.
  - Nothing in the three outlines can tell these parts apart from solid ones, so an honest per-part warning is not possible from silhouettes alone.
  - The README states the limit and reports the error per category.

Task order: 7 and 8 in parallel (disjoint files), then a full benchmark rerun (before/after in the ledger and the commit message), then Task 6.

### Task 7: Outline snapping keeps curves and thin features

**Files:**
- Modify: `s2c/multiview/fuse.py` (the outline part of `snap`; `snap_coord` and the feature snapping stay as they are)
- Test: `tests/test_mv_fuse.py`

**Rule.** Snapping an outline is for cleaning hand-drawn straight lines, so it only moves straight, axis-parallel lines, and only when the move is small for the part:
- An edge counts as horizontal when `|Δb| <= 0.035 * |Δa|` (2°) and it is at least 1 mm long; vertical likewise with a and b swapped. Only the b of a vertex on a horizontal edge, or the a of a vertex on a vertical edge, may be snapped. Curves (polylines with short or sloped edges) are left alone.
- Work per level: the distinct b values of horizontal edges (and the a values of vertical edges), rounded to 1e-6. Each level snaps with `snap_coord(v, length)`.
- A level is not snapped when:
  - the snapped value equals another level's snapped or original value (this is what collapses the 0.3 mm flange onto 0);
  - or the move exceeds 2 % of that axis's length (a 0.25 mm grid move on a 6 mm part is 4 %).
- Every vertex on a snapped level takes the new value. Other vertices keep theirs. The path is recorded in `snapped` only when something moved.

- [ ] **Step 1: Write the failing tests** in `tests/test_mv_fuse.py`, calling `assemble` with provenance `"scaled"` for the views (reuse the existing `test_assemble_snaps_only_untrusted_values_and_applies_edits` setup):
  - `test_snapping_leaves_a_round_outline_round`: the front view is a 64-point circle of radius 6.5 centred in a 13 × 13 face (envelope 13 × 13 × 5). After `assemble`, every outer point is within 0.01 mm of where it was.
  - `test_snapping_never_collapses_a_thin_flange`: the front outline is a hex-on-flange profile in a 17.5 × 6 face: `[(0,0), (17.5,0), (17.5,0.3), (14.5,0.3), (14.5,6), (3,6), (3,0.3), (0,0.3)]`. After `assemble`, the points at b = 0.3 are still at 0.3.
  - `test_snapping_still_squares_a_sketched_l_bracket`: an L outline in a 50 × 30 face with the inner corner at a = 4.8 and b = 4.8 (edges axis-parallel, all longer than 1 mm) snaps to 5.0 (thin-wall table), and `"views.front.outer"` is in `snapped`.
  - `test_snapping_does_not_move_a_small_part_by_more_than_two_percent`: in a 6 × 6 face, a horizontal level at b = 2.3 stays at 2.3. Snapping would move it by 0.3 mm, which is 5 %.
- [ ] **Step 2: Run** `uv run pytest tests/test_mv_fuse.py -v`. Expected: the round, flange and small-part tests FAIL; the L-bracket test passes (existing behaviour).
- [ ] **Step 3: Implement** a helper `_snap_outline(points, a_len, b_len) -> list[Point]` in `fuse.py` and use it in `snap` for the three canonical faces.
- [ ] **Step 4: Run** `uv run pytest tests/test_mv_fuse.py tests/test_studio_pipeline.py tests/test_studio_ui.py -q` and ruff on the touched files. Expected: PASS.

### Task 8: Turned parts are built as solids of revolution

**Files:**
- Create: `s2c/multiview/turned.py`, `tests/test_mv_turned.py`
- Modify: `s2c/multiview/build.py` (intersect with the revolve before features and finishes), `s2c/multiview/pipeline.py` (`fuse` adds one warning when the part reads as turned)

**Interfaces (produced):**
- `turned_axis(spec: MultiViewSpec) -> str | None`: `"x"`, `"y"` or `"z"` when the part reads as turned about that axis, else `None`. Deterministic, from the spec alone. Candidate axes in the order y, z, x. The axis view and the side views are:
  - y: axis view top; side views front (radial x) and right (radial z);
  - z: axis view front; side views top (radial x) and right (radial y);
  - x: axis view right; side views front (radial y) and top (radial z).
- The part reads as turned about axis k when all three hold:
  1. The two envelope lengths across k are equal within 2 %.
  2. Every outer point of the axis view (in global coordinates via `to_global`) lies within `1.03 * R + 0.2` mm of the centre, where R is half the larger of the two lengths.
  3. The four half-profiles agree:
     - A half-profile is the extent from the centre line, on each side of each side view, at 256 heights spread over the open interval along k.
     - The extent at a height is the max (right half) or min (left half) radial coordinate of the outline's edge crossings, measured from the centre line.
     - Every pair's mean absolute difference, divided by R, is at most 0.03.
- `WARNING = "Built as a turned part around the {axis} axis: its view along that axis is round and both side views match"`.
- `revolve(spec: MultiViewSpec, axis: str) -> cq.Workplane`: the solid of revolution about the centre line of the envelope along `axis`.
  - Its profile is `r(h)` = the maximum of the four half-extents at height h.
  - It is evaluated at each outline vertex height h ± 1e-4 mm, so steps stay sharp, and closed along the axis from h = 0 to h = L.
  - The maximum keeps the true part inside the revolve.
- `build.build`: after the hull passes `_check`, if `turned_axis(spec)` is not None, `solid = solid.intersect(revolve(spec, axis))`, then `_check` again. Features and finishes follow as before.
- `MvPipeline.fuse`: when `turned_axis(spec)` is not None, it appends `WARNING.format(axis=axis)` to the spec's warnings.

- [ ] **Step 1: Write the failing tests** (`tests/test_mv_turned.py`, using `tests/mv_helpers.make_spec`, `circle`, `outline`, `rect`):
  - `test_a_stepped_round_part_is_built_round`: envelope 20 × 15 × 20.
    - The top view is `circle(10, 10, 10, 180)`.
    - The front and right views are the same stepped profile: `[(0,0), (20,0), (20,5), (15,5), (15,15), (5,15), (5,5), (0,5)]`, a flange of radius 10 and height 5 under a hub of radius 5 and height 10.
    - `turned_axis == "y"`, and the built volume is `π(10²·5 + 5²·10) ≈ 2356.2` within 1.5 %. Without the fix it is about 2570.
  - `test_a_square_plate_is_not_turned`: the envelope 20 × 5 × 20 with rectangle views gives `turned_axis is None`, and the volume is 2000.
  - `test_different_side_views_are_not_turned`: the same round top view, but the right view is the rectangle `rect(20, 15)`, gives `None`.
  - `test_a_round_disk_with_a_through_hole_keeps_its_hole`:
    - The front view is a circle, with envelope 30 × 30 × 6.
    - It has a centred Ø8 through hole on the front.
    - Expected: `turned_axis == "z"`, and the volume is `π(15² − 4²)·6` within 1.5 %.
  - `test_fuse_warns_when_the_part_is_turned`: run `MvPipeline().fuse` on an `Observed` for a round part, built the way `tests/test_studio_pipeline.py` builds its observations (or monkeypatch `turned_axis` to return `"y"` on any spec). Assert that the warning `WARNING.format(axis="y")` is in `spec.warnings`.
- [ ] **Step 2: Run** `uv run pytest tests/test_mv_turned.py -v`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `turned.py`, then the two calls in `build.py` and `pipeline.py`.
- [ ] **Step 4: Run** `uv run pytest tests/test_mv_turned.py tests/test_mv_build.py tests/test_mv_pipeline.py tests/test_studio_pipeline.py -q` and ruff on the touched files. Expected: PASS, with no change to any existing build test's volume.
