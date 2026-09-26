# Project review, 2026-09-24: what to enhance, change, rewrite, remove and add

Four reviews of the `geometry/studio-ui` branch, each from one angle: architecture, vision/AI accuracy, product/demo, and reliability/security. In a second round each reviewer read the other three reports, checked the claims they doubted against the code, and ranked everything together. Where they disagreed, the ruling is noted. The demo is on 2026-09-27, three days from now.

## Bottom line

- The deterministic core is sound: visual hull, CadQuery build, snapping, provenance, exporters, slicer, cache. The code is small, typed and tested (291 tests pass).
- The risk is at the edges. The network models have never completed a live call. One vision-API error crashes Analyze. Real phone photos and real sketches have never been measured.
- Two claims we plan to make on stage are false in code today: "the model never produces a millimetre" and "images live only for the request".
- The fix is mostly subtraction. Turn the hosted drawing/depth/3D models off for the demo, degrade instead of crashing, stop model millimetres reaching geometry, and gate the sketch front end so it abstains instead of guessing.
- `main` holds only docs, and no single-view module (`partspec/`, `builder.py`, `api.py`, `web/`) exists on any branch. The team has to decide today what the demo is.

## Decisions the team must take today

1. **What is the demo, and what merges to `main` by 09-26?** All four reviews recommend the Studio (`app_mv_studio.py`). CLAUDE.md rules 2 and 3 ("no general multi-view reconstruction") contradict this branch, and sign-off on the multi-view spec (§10) is still pending. Add a short multi-view section to CLAUDE.md with the decision.
2. **Model depths.** Remove the hole depths the vision model guesses in mm (see P0-3). The multi-view spec allowed them with an amber "estimated" badge, but CLAUDE.md rule 2 and the pitch say no model millimetres ever.
3. **Which AI stays on stage.** Recommended: Qwen-VL reads the handwriting and names the face (the visible AI). Qwen-Image face drawing, TripoSR, Solaria and sketch rescue are off. For an extruded part the missing face is a rectangle, and the Qwen gate accepts even a 2 mm strip when only one face is observed (verified).
4. **Phone on stage or not.** If yes, bind the Studio to the LAN only with a password, over the presenter's own hotspot (P0-12).

## P0: before the demo (consensus ranking)

| # | What | Action | Where | Effort |
|---|------|--------|-------|--------|
| 1 | A vision-API error, a 429, the `replace-me` placeholder key, or a malformed reply (`label_invalid`) aborts Analyze even when the user tagged every image. Skip the label call when face and kind are tagged; otherwise fall back to `hint_label` on an exception **or** `label_invalid`; treat placeholder keys as unset; add a catch-all stop card in the handlers. | CHANGE | `pipeline.py:104-109`, `label.py:71,79-81`, `studio/handlers.py:165-186`, `studio/app.py:196` | S |
| 2 | Hosted Qwen-Image, TripoSR, Solaria and rescue off for the demo, through Studio defaults read from the environment (keep `AiSettings` code defaults, which tests pin). TripoSR never runs on a sketch. The coverage chip says "extruded (rectangle)" instead of "AI will draw it". | CHANGE | `studio/app.py:86-102`, `studio/handlers.py:159`, `complete.py:19,107`, `pipeline.py:198-200` | S |
| 3 | Model millimetres reach geometry: the prompt asks for hole depths in mm, `fuse` turns them into `depth_mm`, and a model `blind` flag turns a through hole into a blind one. Delete both; holes are through unless the user says blind (per-hole control is P1). Flip the test that asserts the depths are kept. | REMOVE | `label.py:48,84-86`, `fuse.py:50-52,224-231`, `merge_views.py:184-186`, `tests/test_mv_label.py:22` | S |
| 4 | A failed TripoSR is called again on every Build, finish-slider move and Redraw (verified: one call per move, each blocking the shared `cad` lane). Remember the failure, enforce the Space timeout (a 1 s limit took 6.1 s), close each gradio client, and default rembg to `u2netp` (the default `bria-rmbg` is about 1 GB and CC BY-NC). | CHANGE | `complete.py:105-110`, `hf3d.py:50-53,111-126`, `depth.py:51`, `qwen_image.py:115` | S |
| 5 | The paper is taken as the part: a sheet on a desk, a shadow gradient or ruled paper gives a full-frame slab without abstaining. In one PR, behind the harness of #7: (a) abstain when the outline touches the border, covers >85 % of the frame, or is a 4-corner quad that contains other ink; (b) find the sheet and warp it with `reference._quads`/`_rectify`, which also fixes keystone; (c) flat-field inside the sheet. | REWRITE | `outline.py:48-90` (feeds ocr, reference, qwen_faces, merge_views) | M |
| 6 | Trusted junk numbers: `parse_value` removes spaces, so "12 48" becomes 1248 mm and a phantom "0 1" becomes 1 mm, badged green `user_written` and never snapped. Split on whitespace and never join digits; link ⌀ and R only to a nearby hole; flag a value off by >3× from the pixel measurement; gate TrOCR at 0.7; show Qwen reads amber unless TrOCR agrees; send at most 16 crops. | CHANGE | `ocr.py:43,62-81,88-100,112-124`, `qwen_reader.py:16`, `fuse.py:79-83` | S |
| 7 | No accuracy evidence: `tests/golden_mv/` is empty, the README accuracy table says "pending", and the round-trip IoU is circular (0.95 on a part with 18 % extra volume). Add a degradation test file (shadow, ruled paper, sheet on desk, tilt, noise) and 3-5 real phone captures of the demo parts measured with calipers; rehearse only on inputs that pass; put one measured number in the README. | ADD | `tests/golden_mv/`, `tests/test_mv_golden.py`, `README.md` | M |
| 8 | Pen-stroke width inflates walls and shrinks holes: the bundled example builds a 3.5 mm wall for 3.0 (+18 % volume) and a 5.0 mm hole for 5.5 (an M5 bolt no longer fits). Move the outer contour in by w/2 and grow holes by w. A scratch test brought holes within ±3 %. | CHANGE | `outline.py:61-90`, `fuse.py:249-266` | S |
| 9 | Privacy lines are not true: the vision model gets the original upload with its EXIF (GPS), the Studio never says images go to DashScope and Hugging Face, and Gradio keeps uploads up to about 2 h. Send a re-encoded 1600 px JPEG; add an upload note; set `delete_cache=(600, 3600)`; turn analytics off; correct `CLAUDE.md:69`, `status.py:25`, `app.py:23` and `disclosure.md` (rembg licence); list the models in the manifest. | CHANGE | `pipeline.py:106`, `label.py:62,67`, `studio/app.py:23,41`, `status.py:25`, `docs/disclosure.md`, `artifacts.py:186-194` | S |
| 10 | Stale results: after Build or a finish change, the visible "Download all (.zip)" still serves the previous part; after a new Analyze, step 3 still shows the old model, views and zip. Clear them in `on_build`, `on_geometry` and `on_analyze`. | CHANGE | `studio/app.py:250,266-272` | S |
| 11 | A failed fillet clears the model (`session.part = None`). Keep the last good part on screen and suggest the largest size that builds (bisection, at most 7 builds, only after a failure). | CHANGE | `studio/handlers.py:318-325`, `studio/app.py:145,223-228` | S |
| 12 | Preflight and warm-up: a script that checks each key and Space (`view_api`), runs the two example sketches offline and asserts 50/30/20 and one Ø5.5, and warms TrOCR (10.8 s cold) in a thread at launch, with a lock so it loads once. Add a timestamp and the error to every log line, and take timeouts from the environment. | ADD | new `scripts/preflight.py`, `studio/app.py:276`, `ocr.py:49`, `qwen_image.py:33-38`, `label.py:108-118` | S-M |
| 13 | README quick start for this branch (`uv sync --extra ai`), plus `docs/demo.md`: a 3-minute beat sheet with a fallback for each step. Use the ISO card, not a coin, as the photo reference. Move the lab app under a "Fallback" heading. | REWRITE | `README.md:48-65,110-125,138-140`, new `docs/demo.md` | S |
| 14 | Only if the phone is demoed: `STUDIO_HOST` (default 127.0.0.1) plus `STUDIO_PASSWORD` as Gradio `auth`; never `share=True` with keys loaded; the presenter's hotspot. | ADD | `studio/app.py:276-279` | S |

## Code blocks, by action

**REMOVE**
- Model depth estimates and model blind flags in geometry: `label.py:48` (prompt line), `label.py:84-86`, `fuse.py:50-52` and `:224-231`, `merge_views.py:184-186`. P0.
- After the demo: `app_mv_gradio.py` and `tests/test_mv_gradio.py`; the old build path (`MvPipeline.build`, `BuildResult`, `build.export`, `scripts/mv_build.py`) once `/mv/build` moves onto `artifacts`; the copied helpers (`REFERENCES`, `_value`, face gallery, `TRUSTED`) the two UIs duplicate. P1.

**REWRITE**
- `outline.extract` front end: gate → sheet warp → flat-field → stroke-width compensation, one author, behind the degradation harness (`outline.py:48-90`). P0.
- `ocr.parse_value` and `ocr.link` (`ocr.py:43,112-124`): no digit joining, distance-limited links, ratio check, crop cap. P0.
- `pipeline._label` (`pipeline.py:104-109`): model only when needed, fallback to tags. P0.
- `/mv/build` onto `build_part` + `export_part`, keeping its response shape (`routes.py:105-122`). P1.
- `complete()` (14 parameters, mutates a cache bag without a lock) into a small face-filler object (`complete.py:82-131`). P2.

**ADD**
- Degradation tests plus real captures (`tests/golden_mv/`). P0.
- `scripts/preflight.py` and `GET /mv/health`. P0.
- `docs/demo.md` and the README quick start. P0.
- Three failure-path tests next to `test_configured_switches_are_per_request`: a raising `chat` with a tagged face still gives a Review; a raising `mesh_provider` is called at most once across analyze, build and 3× rebuild; a failed finish keeps `session.part`. P0.
- A lazy "Drawing" tab in step 3 that renders SVG only when opened (not after each build), with "≈" on dimensions until the stroke-width fix lands. P1.
- A per-hole through/blind control (`user_edited`). P1.
- Hole positions the user wrote: `link` never ties a linear value to a feature, so a written "12 from the edge" is read and ignored (multi-view spec §4.4 step 1 was never implemented). Match each non-largest linear reading to the hole coordinate on the same axis. P1 (P0 if the demo sketch has position dimensions).
- A per-request call recorder (a `contextvars.ContextVar` that `log_call` and `env_chat` append to) for a "What the AI did" panel and the manifest. P1. Do not read log-file offsets: two concurrent sessions would see each other's calls.
- CI (`.github/workflows/ci.yml`: ruff + pytest without network markers). P1.
- The PartSpec → MultiViewSpec adapter (`s2c/builder.py`, `s2c/views.py`), so there is one geometry kernel, only after decision 1. P1.

**CHANGE**
- Studio AI defaults from the environment; honest status chips (`status.py:12-25`, `handlers.py:159`). P0.
- `Suggest` fills a missing axis from sketch proportions and then counts as `user_edited` after one Build. Keep a "from sketch proportions, check" note until the user types in the box (`fuse.py:127-140`, `studio/app.py:206`). P1.
- Hole edits are keyed by list index, and the list can change when an envelope edit flips the duplicate-through test. Key edits by face and position, or drop them when the feature count changes (`fuse.py:197-202`, `handlers.py:277-295`). P2.
- `OPENCV_IO_MAX_IMAGE_PIXELS` pixel cap (one line). P0. The rest of the API limits and CORS come with the web client. P1.
- Accessible names for the size boxes (they are announced as "mm") and named steps. P1.
- Pin `gradio` and `gradio_client` to the tested versions; make output and log paths independent of the working directory (`artifacts.py:34`, `routes.py:26`, `label.py:98`). P1.
- Face-frame table written out six times: one `FACE_FRAMES` (`spec.py:151-168`, `raster.py:20-29`, `build.py:28-44,81-94`, `qwen_faces.py:196-208`, `drawing.py:26`). P2.

## Cut list: do not do before 2026-09-27

- The PartSpec adapter: no `partspec/` exists on any branch, so the work would be redone.
- `StudioService` extraction, the `complete()` rewrite, face-frame unification, the `/mv/build` rewrite: refactors with no stage value that touch 291 green tests.
- A per-request deadline and per-stage progress plumbing: environment timeouts are enough once hosted AI is off.
- `spec.py` size bounds: contract churn; put request limits in `routes.py` after the demo.
- Size boxes on step 3: they duplicate `on_build` wiring. A PDF render lane: just leave PDF unticked live.
- ArUco markers, EXIF parallax correction, confidence calibration, ring-based hole detection: correct direction, multi-day work.
- Parallel label and read calls: they double the 429 risk before any budget exists.
- Any generic retry/backoff layer: it multiplies the timeout budget. Degrade instead.
- Tuning the Qwen-Image gate or swapping Solaria for another depth model: make them opt-in instead.

## Disagreements, and how they were settled

- **AI off, by code default or by environment.** The environment decides the Studio's initial checkboxes, and `AiSettings` code defaults stay as tested, so the API is unchanged.
- **Outline fix order.** A border gate alone misses a sheet lying inside the frame (68 % of the frame in the product test), and flat-field alone broke the front view in the vision test. All three parts land together behind the harness, and nothing merges into `extract` without it.
- **OCR gate vs parsing.** Parsing and linking come first because they work for both readers. A confidence gate does nothing for Qwen reads, which carry a constant 0.9.
- **Show AI on stage vs speed.** Skip the label call for tagged images, but leave one image on "auto" in the demo so the audience sees the model name a face.
- **Lab app.** Keep it as a fallback until the demo, then delete it.
- **Drawing tab.** Render it lazily, and only after the stroke-width fix (or with "≈"), so the stage does not show a wrong Ø.
- **Phone access.** Reliability wins: no LAN binding without a password.

## Measured facts behind the ranking

- Example flow: analyze 0.14 s without OCR; TrOCR 10.8 s cold, 2.0 s warm; build 0.18 s; finish rebuild 0.09 s; export of 4 formats 0.65 s, of all 12 formats 3.41 s. PDF takes 1.1–5 s depending on fillets.
- Noisy phone photo: about 40 text crops, 47.6 s of reading, phantom reads at confidence 0.31–0.51 (real reads 0.99+).
- Bundled example: 5286 mm³ built against 4477 mm³ intended (+18 %); hole Ø5.0 against 5.5; IoU still 0.95 / 0.99.
- `logs/vlm.jsonl`: two live calls ever, both failed (433 ms and 111 ms, no reason recorded).
- Worst-case Analyze with every hosted model on and stalling: more than 25 minutes.

## Already fixed (2026-09-24 and 09-25)

All 35 minor findings the two earlier reviews had deferred are fixed, plus the issues the final review of that work found. Details are in `docs/superpowers/plans/2026-09-24-deferred-minors-plan.md`. The suite now stands at 291 passed and 5 skipped, up from 235.

- **Files.** Every cached file is written atomically: mesh, drawing, G-code, preview and `.blend`. A crash can no longer leave a half-written file that is served later. Every OCCT writer is serialised. Two exports of the same part wait for each other. A cached `.blend` is checked before it is reused.
- **Builds.** Two requests for the same new part build it once.
- **Drawing.** Holes on the back, left and bottom faces are dimensioned in the opposite view. Fillets are called out as R and chamfers as C. The text block never overlaps a view.
- **Studio.**
  - A cleared size box or table cell goes back to the measured value.
  - An unreadable image is named in plain words.
  - The review refreshes after a clearance change.
  - Rejected AI faces stay rejected across builds.
  - Idle sessions are swept.
- **Model providers.**
  - DashScope has one time budget for the whole call.
  - Depth masks are cropped to each hole, and a depth error no longer stops the analysis.
  - A warning is logged when a provider is not configured.
  - Faces drawn by Qwen-Image keep only their outline.
- **Merging.** Dropped photos are named by their real position. Dropped holes and their values are reported. Openings near the edge survive. An opening that becomes round only after merging is flagged.
- **API, CLI and lab app.**
  - Artifact paths reject dot segments.
  - The CLI checks its arguments before deleting any old builds.
  - The lab app never counts a suggested size as the user's. It keeps its outputs when a value is bad, and remembers rejected faces.
- **Checked in the running Studio.** A cleared size reverts, the review refreshes, export leaves no temp files, and the upload label wraps on a phone.

## Task checklist: four tracks (updated 2026-09-26)

The remaining work is split into four tracks, one per person or agent. ★ marks what is needed for the 2026-09-27 demo. Tick items as they merge. P0 numbers refer to the table above.

**Before anything else, whole team:** decisions 1–4 above.

### Track 1: Recognition (images, OCR, outlines)
Code: `outline.py`, `ocr.py`, `qwen_reader.py`, `label.py` (prompt), `reference.py`.
- [ ] ★ P0-5: the paper sheet is taken as the part. Add a border/coverage/quad abstain, rectify the sheet with `reference._quads`/`_rectify`, and even out the lighting (`outline.py:48-90`). Land it together with the degradation tests.
- [ ] ★ P0-6: OCR gives trusted junk numbers.
  - Never join digits across spaces (`ocr.py:43`).
  - Link ⌀ and R values only to a nearby hole.
  - Flag a value more than 3× off the pixel measurement.
  - Gate TrOCR reads at 0.7 confidence.
  - Show Qwen reads amber unless TrOCR agrees.
  - Send at most 16 crops.
- [ ] P0-8: correct for pen-stroke width, which makes walls too thick and holes too small (`outline.py:61-90`). The 3D modeling track checks it on the golden parts.
- [ ] P0-3 (prompt half): stop asking the vision model for hole depths and blind flags (`label.py:48`).
- [ ] P0-7 (tests half): degradation tests for shadow, ruled paper, sheet on a desk, tilt and noise.
- [ ] After the demo: hole positions written by the user (multi-view spec §4.4); report the dropped-hole warning in mm, not grid pixels; drafting conventions (centre lines, pen gaps).

### Track 2: 3D modeling (geometry, build, drawing)
Code: `fuse.py`, `build.py`, `finish.py`, `drawing.py`, `artifacts.py`.
- [ ] ★ P0-3 (geometry half): remove the model's millimetre depths and blind flags from geometry (`fuse.py:50-52,224-231`, `merge_views.py:184-186`). Holes are through unless the user says otherwise. Flip `tests/test_mv_label.py:22`.
- [ ] ★ P0-11: a failed fillet or chamfer keeps the last good part and suggests the largest size that builds, by bisection with at most 7 builds (`studio/handlers.py:318-325`).
- [ ] P0-7 (parts half): 3–5 real demo parts measured with calipers for the golden set (`tests/golden_mv/`), and one measured accuracy number in the README.
- [ ] After the demo:
  - a per-hole through/blind control;
  - a lazy Drawing tab in step 3;
  - the PartSpec → MultiViewSpec adapter, once decision 1 is taken;
  - one `FACE_FRAMES` table;
  - edits keyed by face and position.

### Track 3: Security and privacy
Code: `pipeline.py`, `label.py`, `studio/app.py`, `status.py`, `routes.py`, `docs/disclosure.md`, `CLAUDE.md`.
- [ ] ★ P0-9: re-encode images as a 1600 px JPEG before the vision model, so no EXIF or GPS data leaves the machine (`pipeline.py:106`).
- [ ] ★ P0-9: make the privacy lines true.
  - Add an upload notice naming DashScope and Hugging Face.
  - Set `delete_cache=(600, 3600)`.
  - Turn analytics off.
  - Correct `CLAUDE.md:69`, `status.py:25`, `studio/app.py:23` and `docs/disclosure.md`, including the rembg `bria-rmbg` CC BY-NC licence.
  - List the models in the manifest.
- [ ] ★ Treat placeholder API keys (`replace-me`) as unset.
- [ ] ★ Add the `OPENCV_IO_MAX_IMAGE_PIXELS` pixel cap.
- [ ] P0-14, only if the phone is demoed: LAN access only through `STUDIO_HOST` plus `STUDIO_PASSWORD` (Gradio auth), over the presenter's hotspot, never `share=True`.
- [ ] After the demo:
  - `/mv` API limits: upload size, spec size, concurrency;
  - CORS;
  - token scope;
  - pinned `gradio` and `gradio_client` versions;
  - paths that do not depend on the working directory.

### Track 4: Backend and orchestration (pipeline, providers, UI flow, release)
Code: `pipeline.py`, `complete.py`, `hf3d.py`, `depth.py`, `qwen_image.py`, `studio/handlers.py`, `studio/app.py`, `scripts/`, `README.md`.
- [ ] ★ P0-1: a vision-API error, a 429 or a `label_invalid` reply falls back to the user's face tags instead of crashing Analyze. Skip the label call when face and kind are tagged, and add a catch-all card (`pipeline.py:104-109`).
- [ ] ★ P0-2: hosted Qwen-Image, TripoSR, Solaria and rescue are off for the demo, through environment-driven Studio defaults. Never run TripoSR on a sketch. The coverage chip says "extruded (rectangle)".
- [ ] ★ P0-4: remember a TripoSR failure, enforce the Space timeout, close each Space client, and default rembg to `u2netp` (`complete.py:105-110`, `hf3d.py`).
- [ ] ★ P0-10: clear the stale zip and step 3 after Build, a finish change or a new Analyze (`studio/app.py:250,266-272`).
- [ ] ★ P0-12: add `scripts/preflight.py` (keys, Spaces, and the offline example asserting 50/30/20 and Ø5.5), a TrOCR warm-up at launch, timestamped error logs, and timeouts from the environment.
- [ ] ★ P0-13: a README quick start and a `docs/demo.md` beat sheet with a fallback for each step.
- [ ] ★ Merge the demo branch to `main`, then two full rehearsals on the demo laptop with the hotspot and the real parts. Whole team.
- [ ] After the demo:
  - delete the lab app and the old build path;
  - move `/mv/build` onto `artifacts`;
  - CI;
  - a per-request model-call recorder and "What the AI did" panel;
  - a shared `atomic_write` helper;
  - the "merged" badge for opposite-face photos in the Studio.
