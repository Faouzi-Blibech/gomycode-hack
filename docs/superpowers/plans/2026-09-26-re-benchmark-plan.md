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
