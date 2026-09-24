# Deferred Minors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every minor finding the reviews of the Qwen/Solaria and Studio branches deferred, without changing any feature's scope.

**Architecture:** Nine small, file-disjoint tasks. Each fixes the listed findings in its own files, with a test that fails before the fix. No new modules, no new dependencies, no contract (`spec.py`) changes.

**Tech Stack:** Python 3.11, uv, pytest, ruff, CadQuery 2.8 / OCP, trimesh, ezdxf, Gradio 6.28, FastAPI.

**Spec:** `docs/superpowers/specs/2026-09-23-studio-design.md` and `docs/superpowers/specs/2026-09-23-qwen-solaria-design.md`. The findings come from the task and final reviews of those two plans.

## Global Constraints

- Run everything from the repo root with `uv run ...`. Windows 11; paths may contain spaces.
- ruff line length 120; `uv run ruff check <your files>` must report nothing new.
- Write the failing test first, watch it fail, then fix (CLAUDE.md "Testing").
- Touch only the files your task lists. `spec.py` (the contract) is frozen.
- Implementers do not commit and do not push. The controller commits with plain messages and no AI attribution (CLAUDE.md "Git rules").
- Never read or write `.env`.
- Keep the surrounding style: short docstrings that say why, no comment on obvious lines.

## Review Focus

1. A file that exists in the cache folder is trusted as finished (mesh, drawing, G-code, .blend): after a crash or timeout mid-write, the next request must rebuild it, not serve a truncated file.
2. Two browser tabs exporting the same part at the same time: no half-written file, no OCCT global-state race.
3. A user who clears a size box expects the measured or suggested value back, not their old typed number.
4. A user who uploads a HEIC or a PDF expects to be told which file is wrong in plain words, not `outline: bad_image`.
5. A slow or broken DashScope or Solaria call must not hang a request past its stated budget or crash the whole analysis.

---

### Task 1: Subprocess log, settings defaults, dead constant

**Files:**
- Modify: `s2c/multiview/proc.py`, `s2c/multiview/fuse.py:20`
- Test: `tests/test_studio_settings.py`

**Findings:**
1. `proc.run` opens the log with `encoding`/`errors` kwargs that never apply: the child writes raw bytes to the file descriptor.
2. `proc.tail` reads the whole log to return the last lines.
3. No test pins the defaults of `AiSettings` booleans and `GeometrySettings` snap / finish / edges / clearance.
4. `fuse.CLEARANCE_MM` is an unused alias.

- [ ] **Step 1: Write the failing tests** in `tests/test_studio_settings.py`:
  - `test_tail_reads_only_the_end`: write a log of 200 000 lines `line <i>` (well over 64 KiB), monkeypatch `pathlib.Path.read_text` to raise `AssertionError("whole file read")`, and assert `tail(path, 3) == "line 199997\nline 199998\nline 199999"`.
  - `test_tail_short_and_missing`: a 2-line log with `n=20` returns both lines; a missing path returns `""`; a log whose bytes are not UTF-8 (`b"\xff\xfeok\n"`) returns a string ending in `ok`.
  - `test_run_log_is_binary`: `run([sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'\\xff\\xfeok\\n')"], 30, log)` returns 0, and `log.read_bytes()` ends with `b"\xff\xfeok\n"`.
  - `test_settings_defaults`: assert every boolean default of `AiSettings()` and the `snap`, `finish`, `edges` and `clearance` defaults of `GeometrySettings()`. Copy the expected values from the Studio spec's parameter tables; if code and spec disagree, stop and report instead of changing either.
- [ ] **Step 2: Run** `uv run pytest tests/test_studio_settings.py -v`. Expected: `test_tail_reads_only_the_end` FAILs with "whole file read"; the others may pass already (they pin behaviour).
- [ ] **Step 3: Fix.** `run`: open the log with `log_path.open("wb")`. `tail`: open in binary, seek to `max(0, size - TAIL_BYTES)` with `TAIL_BYTES = 65536`, read to the end, decode as UTF-8 with `errors="replace"`, drop the first (possibly partial) line when the read did not start at 0, return the last `n` lines joined by `"\n"`; `""` on `OSError`. Delete `CLEARANCE_MM` from `fuse.py` after `grep -rn CLEARANCE_MM s2c tests scripts app_*.py` shows no other user.
- [ ] **Step 4: Run** `uv run pytest tests/test_studio_settings.py tests/test_mv_fuse.py tests/test_studio_blend.py tests/test_studio_print.py -q` and `uv run ruff check s2c/multiview/proc.py s2c/multiview/fuse.py tests/test_studio_settings.py`. Expected: all pass, no lint.

### Task 2: Atomic, serialised file writers (mesh, G-code, Blender)

**Files:**
- Modify: `s2c/multiview/exporters.py`, `s2c/multiview/slice.py`, `s2c/multiview/blend.py`
- Test: `tests/test_studio_exporters.py`, `tests/test_studio_blend.py`, `tests/test_studio_print.py`

**Findings:**
1. `_STEP_LOCK` guards STEP only; the other OCCT writers (STL triangulation, 3MF, BREP) are not serialised.
2. Mesh files, G-code and `.blend` treat "exists" as "finished" but are not written atomically, so a crash or timeout mid-write leaves a truncated file that later requests serve.
3. `write_step` / `write_brep` take a `quality` parameter they ignore.
4. `write_blend` trusts a cached `part.blend` without a sanity check; a stale `part_blender_kit.zip` can sit next to a later real `.blend`; `SCRIPT`'s usage line omits `--factory-startup`.

- [ ] **Step 1: Write the failing tests.**
  - `tests/test_studio_exporters.py::test_all_occt_writes_hold_the_lock`: monkeypatch `cadquery.exporters.export` and `cadquery.Shape.exportStl` with wrappers that record `exporters._OCCT_LOCK.locked()` and then call the original; export `stl, step, 3mf, brep` and assert every recorded value is `True`.
  - `test_failed_write_leaves_no_file`: monkeypatch `exporters.WRITERS["obj"]` with a writer that writes 10 bytes to the path it is given and raises `RuntimeError`; `export_mesh_formats(box, d, ["obj"])` raises, and `d` then holds neither `part.obj` nor any `*.tmp*` file.
  - `test_concurrent_exports`: four threads export `["stl", "step", "3mf", "glb"]` of the same box into four folders; every STL loads with trimesh with volume within 1 % of the box, every STEP starts with `ISO-10303-21`.
  - `tests/test_studio_print.py::test_timed_out_slice_leaves_no_gcode`: monkeypatch the `run` that `slice.py` calls with a fake that writes `b"; partial"` to the path after `--output` in the command and returns `None`; `slice_solid(...)` returns an `MvAbstain` and the output folder holds no `part.gcode`.
  - `tests/test_studio_blend.py::test_cached_blend_is_checked`: with `blender_runner` monkeypatched to `None`, a 0-byte `part.blend` and a `part.blend` holding `b"garbage"` are both deleted and the kit is returned; a `part.blend` starting with `b"BLENDER"` (and one starting with the zstd magic `b"\x28\xb5\x2f\xfd"`) is returned as is.
  - `test_real_blend_removes_stale_kit`: create `part_blender_kit.zip`, monkeypatch `blender_runner` to return `["python"]` and the module's `run` to write `b"BLENDER-v402"` to the target and return 0; after `write_blend`, the kit zip is gone.
  - `test_script_usage_line`: `"--factory-startup" in SCRIPT.splitlines()[1]`.
- [ ] **Step 2: Run** `uv run pytest tests/test_studio_exporters.py tests/test_studio_blend.py tests/test_studio_print.py -v`. Expected: the new tests FAIL (lock not held for STL/3MF/BREP; partial files remain; garbage blend returned; kit remains; usage line).
- [ ] **Step 3: Fix.**
  - `exporters.py`: rename `_STEP_LOCK` to `_OCCT_LOCK` (update its comment: OCCT writer and mesher state is process-wide) and hold it around every OCCT call in `write_stl`, `write_step`, `write_3mf`, `write_brep`. `mesh_of` goes through `write_stl`, so it is covered. `export_mesh_formats` writes each file to `path.with_name(f"{path.stem}.{uuid4().hex[:8]}.tmp{path.suffix}")`, then `os.replace`s it onto the final name; on any exception it unlinks the temp file and re-raises. Keep the `quality` parameter on `write_step` / `write_brep` (one signature for `WRITERS`) and say so in one docstring line: B-rep formats are exact.
  - `slice.py`: PrusaSlicer writes to a temp name in the same folder; rename onto `part.gcode` only when the run returned 0 and the file is non-empty; unlink the temp otherwise.
  - `blend.py`: `_valid_blend(path)` is true when the file is non-empty and starts with `b"BLENDER"`, the zstd magic or the gzip magic `b"\x1f\x8b"`; an invalid cached `part.blend` is unlinked and rebuilt. After a successful Blender run, unlink `out_dir / KIT_NAME`. `SCRIPT` usage line: `Run: blender -b --factory-startup --python open_in_blender.py -- part.obj part.blend`.
- [ ] **Step 4: Run** `uv run pytest tests/test_studio_exporters.py tests/test_studio_blend.py tests/test_studio_print.py tests/test_mv_slice.py tests/test_studio_artifacts.py -q` and ruff on the touched files. Expected: all pass (the real-bpy test may skip when `vendor/bpy-env` is absent), no lint.

### Task 3: Drawing callouts for far-side holes and finishes; atomic drawing files

**Files:**
- Modify: `s2c/multiview/drawing.py`
- Test: `tests/test_studio_drawing.py`

**Findings:**
1. Holes on the back, left and bottom faces get only a text note; their diameter is not dimensioned.
2. Fillets and chamfers get no callout (the geometry shows them, the text does not say their size).
3. `write_drawings` writes in place; `artifacts.export_part` treats an existing drawing file as finished.

- [ ] **Step 1: Write the failing tests** in `tests/test_studio_drawing.py`, using a spec with one through hole on `back` (and one on `left`) plus a fillet finish (build the spec the way the existing tests do):
  - `test_far_side_hole_is_dimensioned`: the DXF has a DIAMETER dimension (`msp.query("DIMENSION")`, `dimtype & 7 == 3`) whose text contains `Ø` and `(back)` (resp. `(left)`), centred where the hole appears in the opposite view (`back` → front view, `left` → right view, `bottom` → top view), within 0.01 mm. Derive the mirrored position from the face frames in `s2c/multiview/spec.py` / `raster.py`, not from a guess; the test computes the expected centre from the built solid's hole axis.
  - `test_finish_callout`: the TEXT layer holds a line like `Fillet R2 on <edge group>` (use the finish's own fields and `settings.EDGE_LABELS` for the wording).
  - `test_drawing_write_is_atomic`: monkeypatch `drawing._pdf` to write 5 bytes and raise; `write_drawings(..., ["dxf", "pdf"])` raises and the folder holds no `drawing.pdf` and no temp file (`drawing.dxf` may exist, it finished).
- [ ] **Step 2: Run** `uv run pytest tests/test_studio_drawing.py -v`. Expected: the three new tests FAIL.
- [ ] **Step 3: Fix.** Dimension a hole on a far face in the view of its opposite face (`back`→`front`, `left`→`right`, `bottom`→`top`) at the mirrored position, with `text="Ø<> (<face>)"` so the measured diameter stays live. Add one TEXT line per finish. Each drawing file goes to a temp name in the same folder and is `os.replace`d onto its final name; unlink the temp on failure.
- [ ] **Step 4: Run** `uv run pytest tests/test_studio_drawing.py tests/test_studio_artifacts.py -q` and ruff on the touched files. Render the SVG of the test spec once (`uv run python -c ...` writing to your scratch directory) and read it to confirm the callouts sit inside the page and do not overlap the views; say so in your report.

### Task 4: Build cache: per-key lock and swept-folder test

**Files:**
- Modify: `s2c/multiview/artifacts.py`
- Test: `tests/test_studio_artifacts.py`

**Findings:**
1. `build_part` is not atomic across concurrent calls for the same new key: both calls rebuild.
2. No test covers "cached part, folder swept, rebuild" (the `preview.exists()` guard).

- [ ] **Step 1: Write the failing tests.**
  - `test_concurrent_builds_share_one_build`: monkeypatch `artifacts.build` with a wrapper that counts calls and sleeps 0.3 s before calling the real one; two threads call `build_part(spec, geometry, tmp_path)` for the same spec; assert the count is 1 and both results have the same `key` and existing `preview`.
  - `test_swept_folder_rebuilds`: build once, `shutil.rmtree` the part's folder, build again: the preview exists again and `build` was called twice.
- [ ] **Step 2: Run** `uv run pytest tests/test_studio_artifacts.py -v`. Expected: the concurrency test FAILs with a count of 2; the swept-folder test may pass (it pins behaviour).
- [ ] **Step 3: Fix.** A module-level `_key_locks: dict[tuple[str, str], threading.Lock]` guarded by one `threading.Lock`; `build_part` takes the key's lock, re-checks the LRU inside it, builds only on a miss. Locks for keys that leave the LRU may stay (bounded by distinct keys per process); say so in a comment.
- [ ] **Step 4: Run** `uv run pytest tests/test_studio_artifacts.py tests/test_studio_ui.py -q` and ruff on the touched files.

### Task 5: Studio UI and session fixes

**Files:**
- Modify: `s2c/studio/handlers.py`, `s2c/studio/app.py`, `s2c/studio/session.py`, `s2c/studio/theme.py`
- Test: `tests/test_studio_ui.py`

**Findings:**
1. `on_analyze` (app.py) builds `AiSettings` without the `ValueError` guard the other handlers have.
2. `_review` has a redundant nested conditional for `known` (handlers.py:215).
3. Clearing a previously typed size box keeps the old edit instead of reverting to the measured or suggested value (`build`, handlers.py:281).
4. At phone width the drop-zone label is clipped (theme.py CSS).
5. Gradio never fires the session `delete_callback`; sessions are swept only when a new visitor arrives.
6. The bad-image card does not name the file and shows raw `stage: reason` slugs (`_abstain_card`).
7. After a clearance change in step 3 (`rebuild_geometry`), the Review table keeps the old diameters until the next Build.
8. `SessionStore.get(None)` uses two different UUIDs for the store key and `Session.id`.

- [ ] **Step 1: Write the failing tests** in `tests/test_studio_ui.py` (use the existing fakes and fixtures there):
  - `test_cleared_size_reverts`: analyze, build with `{"x": "60"}`, then build with `{"x": ""}`; `session.edits` no longer has `envelope.x_mm` and the review shows the measured value again.
  - `test_bad_image_names_file`: add a file `notes.pdf` holding `b"%PDF-1.4"`; `analyze` returns a Review whose `message_html` contains `notes.pdf` and does not contain `bad_image`.
  - `test_abstain_titles_are_words`: `_abstain_card` for `MvAbstain(stage="outline", reason="no_outline", remedy="...")` contains no underscore-joined slug.
  - `test_clearance_change_refreshes_review`: with a spec that has a hole, `rebuild_geometry` with a different `clearance` returns a result whose review rows show the new snapped diameter, and `session.shown` holds it (so the next Build does not read the table as an edit).
  - `test_get_none_uses_one_id`: `store.get(None).id` is a key of `store._items`.
  - `test_idle_sessions_swept_without_new_visitor`: a store with `ttl_s=0.05`; create two sessions, sleep 0.1 s, call `get` on one: the other is gone. Throttle the sweep (`SWEEP_EVERY_S`) and monkeypatch it to 0 in the test.
  - `test_analyze_guard`: the app's analyze handler, fed an out-of-range attempts value, returns the capture card with a message instead of raising (call through `build_app` the way existing tests reach handlers, or extract the settings builder so it is testable).
- [ ] **Step 2: Run** `uv run pytest tests/test_studio_ui.py -v`. Expected: the new tests FAIL.
- [ ] **Step 3: Fix.**
  - `build`: an empty box removes that axis from the edits.
  - `_review`: `known = (abstain.partial or {}).get("known", {}) if abstain else {}`.
  - `analyze`: before `observe`, decode each image (`cv2.imdecode`); on failure return a capture card naming the file: `"<name> is not an image we can read. Upload a JPEG or PNG."`. `_abstain_card` titles come from a small reason → words map with a fallback of the reason with underscores turned into spaces, capitalised; the remedy stays as is.
  - `rebuild_geometry` refreshes the review from the new spec (same `_review` call as `build`) and returns it with the model; `app.py` maps the rows onto the values table and the sizes.
  - `SessionStore.get`: one id (`sid = sid or uuid4().hex`); sweep idle sessions from `get` at most every `SWEEP_EVERY_S = 60` seconds. Check the installed Gradio's `State` source for when `delete_callback` runs; if it cannot run for a state that is never an output, remove `delete_callback` and `time_to_live` from `app.py` and say so in the session docstring; otherwise keep them.
  - `on_analyze`: wrap the settings construction in the same `ValueError` guard the other handlers use.
  - `theme.py`: let the drop-zone label wrap at narrow widths (`white-space: normal`, `overflow-wrap: anywhere` under a `@media (max-width: 640px)` rule, targeting the same selector the upload label uses).
- [ ] **Step 4: Run** `uv run pytest tests/test_studio_ui.py tests/test_studio_api.py -q` and ruff on the touched files.

### Task 6: API route, CLI and small docs

**Files:**
- Modify: `s2c/multiview/routes.py`, `scripts/mv_export.py`, `scripts/make_examples.py`, `README.md`
- Test: `tests/test_studio_api.py`

**Findings:**
1. `routes._NAME` accepts `..` as a path segment (the containment check is what protects).
2. No test covers the export route's abstain branch.
3. `mv_export` shows a pydantic traceback for a bad `--material` / `--layer`, and sweeps before argparse (even `--help` deletes old builds).
4. `make_examples.py` comment overstates the shading range (6 levels, not 12).
5. README no longer mentions `scripts/setup_triposr.ps1`.

- [ ] **Step 1: Write the failing tests** in `tests/test_studio_api.py`:
  - `test_artifact_route_rejects_dotdot`: `GET /mv/artifacts/<key>/..%2Fx` and `/mv/artifacts/<key>/a/../b` return 400 or 404 and never 200; `_NAME.fullmatch("..")` is `None`.
  - `test_export_abstain`: an export request whose images abstain (a non-image upload) returns the route's abstain status and a JSON body with `stage`, `reason`, `remedy`.
  - `test_cli_bad_material`: `main(["--material", "WOOD", ...])` exits with code 2 (argparse error) and stderr names the allowed materials; no traceback.
  - `test_cli_help_does_not_sweep`: monkeypatch the CLI's `sweep` to raise; `main(["--help"])` exits 0.
- [ ] **Step 2: Run** `uv run pytest tests/test_studio_api.py -v`. Expected: the new tests FAIL.
- [ ] **Step 3: Fix.** `_NAME` rejects segments made only of dots. CLI: `choices=` from `settings.MATERIALS` (and the layer bounds from `PrintSettings` via a `ValidationError` → `parser.error(...)`), and `sweep` after `parse_args`. Correct the `make_examples.py` comment to the real number of shading levels (read the code). README: one line under the TripoSR / optional-models part: `powershell scripts/setup_triposr.ps1` installs the local TripoSR fallback.
- [ ] **Step 4: Run** `uv run pytest tests/test_studio_api.py tests/test_mv_routes.py -q` and ruff on the touched files.

### Task 7: Model-provider robustness (DashScope, Solaria, face completion)

**Files:**
- Modify: `s2c/multiview/qwen_image.py`, `s2c/multiview/depth.py`, `s2c/multiview/pipeline.py`, `s2c/multiview/complete.py`, `s2c/multiview/qwen_faces.py`
- Test: `tests/test_mv_qwen_image.py`, `tests/test_mv_depth.py`, `tests/test_mv_pipeline.py`, `tests/test_mv_complete.py`, `tests/test_mv_qwen_faces.py`

**Findings:**
1. DashScope: the 90 s budget is not enforced across the POST, the polls and the download; a SUCCEEDED task with no image URL polls until the timeout.
2. `depth.py` allocates one float64 H × W array per circle (~15 MB each); `pipeline._depths` does not guard `apply_depth`.
3. `default_pipeline` logs nothing when Qwen-Image or Solaria is not configured (the Qwen/Solaria spec §10 asks for one warning each).
4. `complete.py`'s "Qwen-Image view rejected" warning also appears when Qwen was never called (no refs) or after the failure breaker tripped.
5. `qwen_faces`: non-circular openings in a drawn face become through-cuts the consistency gate cannot see, while the module docstring says holes come only from photographed faces.

- [ ] **Step 1: Write the failing tests.**
  - `test_mv_qwen_image.py::test_dashscope_total_budget`: a fake transport whose POST takes 50 s and each poll 30 s (use a fake clock, not real sleeps) makes `dashscope_gen` raise `ImageGenError` once 90 s of the fake clock have passed, and the download is never attempted after that.
  - `test_dashscope_succeeded_without_image`: a task that reports SUCCEEDED with no image URL raises `ImageGenError` after one poll.
  - `test_mv_depth.py::test_depth_masks_are_small`: for a 1600 × 1200 depth map with 10 circles, peak extra memory stays under 20 MB (`tracemalloc`), or assert the helper works on per-circle bounding-box crops; pick the assertion that fails today.
  - `test_mv_pipeline.py::test_depth_failure_in_apply_is_a_warning`: a depth provider that returns an array of the wrong shape makes `observe` return an `Observed` with a `"<face>: depth unavailable"` warning, not raise.
  - `test_default_pipeline_warns_when_unconfigured`: with `SOLARIA_SPACE`, `DASHSCOPE_API_KEY` and `QWEN_IMAGE_SPACE` (use the variables `default_gen` and `default_pipeline` actually read) unset, `caplog` holds one warning naming Qwen-Image and one naming Solaria.
  - `test_mv_complete.py::test_no_rejected_warning_without_a_call`: `complete(...)` with `gen=None` or empty refs emits no warning containing "rejected".
  - `test_mv_qwen_faces.py::test_drawn_face_keeps_outer_only`: a drawn image with a square opening yields an outline whose `inner` is empty.
- [ ] **Step 2: Run** the five test files with `-v`. Expected: the new tests FAIL.
- [ ] **Step 3: Fix.** `dashscope_gen`: one `deadline = clock() + 90` shared by POST, polls and download (each request's timeout is the time left; raise `ImageGenError("timed out")` at zero); SUCCEEDED without an image URL raises at once. `depth.py`: compute each circle's hole and ring statistics on the circle's bounding-box crop with a boolean mask. `pipeline._depths`: wrap `apply_depth` in the same try/except as the provider call. `default_pipeline`: `log.warning` once each when Qwen-Image or Solaria is not configured. `complete.py`: warn only when Qwen actually returned an image that the gate rejected. `qwen_faces.outline_from_image`: keep the outer loop only (`inner=[]`), matching the docstring.
- [ ] **Step 4: Run** `uv run pytest tests/test_mv_qwen_image.py tests/test_mv_depth.py tests/test_mv_pipeline.py tests/test_mv_complete.py tests/test_mv_qwen_faces.py tests/test_mv_rescue.py -q` and ruff on the touched files.

### Task 8: Same-face merge fidelity

**Files:**
- Modify: `s2c/multiview/merge_views.py`
- Test: `tests/test_mv_merge_views.py`

**Findings:**
1. The recursive re-merge numbers "photo n" within the subgroup, so the warning names the wrong photo.
2. Dropped hole clusters and their linked values vanish without a warning.
3. `extract` on the 512 grid uses `EDGE_BAND_PX = 31`, tuned for 1600 px images, so non-circular openings near the edge can be lost.
4. A voted opening that turns out circular is dropped instead of kept as a hole.

- [ ] **Step 1: Write the failing tests** in `tests/test_mv_merge_views.py` (reuse the synthetic-image helpers there):
  - `test_disagreeing_photo_named_by_input_index`: three photos of a face where the third disagrees and the recursion path is taken; the warning says `photo 3`.
  - `test_dropped_hole_is_reported`: a hole present in only one of three photos (so the vote drops it) produces a warning naming the face and the hole's diameter, and any value linked to it is reported as dropped.
  - `test_opening_near_edge_survives_merge`: a square opening 20 px (at 512) from the outline edge in all photos survives the merge.
  - `test_voted_circular_opening_is_kept`: an opening that is circular after the vote appears in the merged outline's circles.
- [ ] **Step 2: Run** `uv run pytest tests/test_mv_merge_views.py -v`. Expected: the new tests FAIL.
- [ ] **Step 3: Fix.** Carry the original photo indices through the recursion; add the warnings; scale the edge band to the grid (`round(EDGE_BAND_PX * grid / 1600)`, never below 3 px) or pass a band parameter to `extract`; keep a voted opening that becomes circular as a circle.
- [ ] **Step 4: Run** `uv run pytest tests/test_mv_merge_views.py tests/test_mv_pipeline.py -q` and ruff on the touched file.

### Task 9: Lab app fixes (`app_mv_gradio.py`)

Held until the project debate rules on whether the lab app stays. If it stays:

**Files:**
- Modify: `app_mv_gradio.py`
- Test: `tests/test_mv_gradio.py`

**Findings:**
1. The "merged ×N" badge is missing for merged opposite-face photos; the reject list also offers observed faces (no effect).
2. A bad number wipes the gallery, reads and warnings (`_pack` resets outputs); unchanged rows are validated too; a cleared edit cannot be undone.
3. A pre-filled envelope suggestion counts as confirmed on Rebuild without an explicit touch (CLAUDE.md rule 2: a suggestion must not become a user value silently).

- [ ] **Step 1: Write the failing tests** for each finding in `tests/test_mv_gradio.py`.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Fix** in the same way as the Studio (Task 5): suggestions are placeholders, only changed rows are validated, an empty box reverts, errors keep the other outputs.
- [ ] **Step 4: Run** `uv run pytest tests/test_mv_gradio.py -q` and ruff.
