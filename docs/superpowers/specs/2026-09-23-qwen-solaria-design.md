# Qwen-Image faces, Qwen-VL reading, photo merging and Solaria depth: design spec

Date: 2026-09-23
Author: multi-view path owner
Status: approved in design review, pending written-spec review and team sign-off
Extends: `docs/superpowers/specs/2026-09-22-multiview-gcode-design.md` (the multi-view spec)

## 1. What changes

The multi-view path keeps its geometry: the deterministic visual hull in `build.py`, sliced by `slice.py`. This spec improves what goes into it:

1. **Reading.** Handwritten values are read by a Qwen vision-language model (Qwen-VL) on DashScope, one call per image. TrOCR stays as the fallback.
2. **Merging.** Several photos of the same face are merged into one observation: aligned masks, a per-pixel vote, median values, median scale. Today only the most confident photo is kept.
3. **Missing faces.** Qwen-Image-2.1 draws each missing canonical face, conditioned on every observed image, and a consistency gate accepts or rejects it. TripoSR and the assumed rectangle stay as fallbacks.
4. **Sketch rescue.** When a sketch has no closed outline, Qwen-Image redraws it as a clean silhouette, accepted only if it matches the raw strokes.
5. **Depth.** For photos with holes or slots, the Solaria Space (Marigold V2 depth) decides through versus blind and estimates blind depth.
6. **UI.** A Gradio app to upload, review, edit numbers, rebuild and download.

Final output is unchanged: an editable `MultiViewSpec`, then STEP, STL and G-code. Solaria's point cloud and Qwen's images are inputs to our own code, never exported.

Not in this spec: editing faces by text or drawing, a mesh output, the React web app, any change to `spec.py`, `build.py`, `slice.py`, `partspec/` or the envelope gate.

## 2. Rules

The multi-view spec rules hold unchanged:

1. **The model never writes code.** Qwen-VL returns JSON validated by pydantic. Qwen-Image returns images. Solaria returns a point cloud. Our code turns each into outlines and numbers.
2. **Numbers have a source and a badge.** Qwen-VL only transcribes what the user wrote, so its reads are `user_written`. Outlines drawn by Qwen-Image are `inferred`. Solaria depths are `estimated`. The envelope still needs `user_written`, `measured` or `user_edited`; nothing in this spec can satisfy the gate.
3. **Six views, three outlines.** Unchanged.
4. **Abstention is a feature.** Every new stage falls back to the existing behaviour with a warning. No new abstention reasons.

## 3. Configuration

All providers come from `.env`. No model name or Space name is hard-coded in source; defaults below live in `.env.example`.

| Variable | Use | Example |
| --- | --- | --- |
| `VLM_BASE_URL` | Qwen-VL, OpenAI-compatible | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` |
| `VLM_MODEL` | Qwen-VL model from the Model Studio catalogue | a current Qwen-VL entry, recorded in `docs/models.md` with the date |
| `VLM_API_KEY` | DashScope key | |
| `QWEN_IMAGE_BACKEND` | `dashscope` or `space` | `space` |
| `QWEN_IMAGE_BASE_URL` | DashScope native API root | `https://dashscope-intl.aliyuncs.com/api/v1` |
| `QWEN_IMAGE_MODEL` | DashScope image-edit model name | a Qwen-Image edit entry from the catalogue |
| `QWEN_IMAGE_SPACE` | Hugging Face Space | `Qwen/Qwen-Image-2.1` |
| `SOLARIA_SPACE` | Hugging Face Space | `CronosSa/Solaria1.0` |
| `HF_TOKEN` | Space quota (ZeroGPU) | |

`QWEN_IMAGE_BACKEND` defaults to `space`. `dashscope` needs `QWEN_IMAGE_MODEL` and `VLM_API_KEY`; without them it falls back to `space`.

## 4. Reading handwriting: `qwen_reader.py`

- `BatchReader = Callable[[list[np.ndarray]], list[tuple[str, float]]]`: crops in, one `(text, confidence)` per crop out, same order.
- `ocr.read_values` accepts either the existing `Reader` or a `BatchReader`. With a batch reader it collects every crop from `text_regions` and makes one call. Parsing (`parse_value`) and linking (`link`) are unchanged.
- `qwen_batch_reader(chat: Chat) -> BatchReader`. One chat message holds the prompt and the crops as numbered images:

  > Each image is a crop of handwriting from a mechanical sketch. Return exactly what is written in each crop, as JSON only: `{"reads": [{"i": 1, "text": "⌀6"}, ...]}`. Use `⌀` for a diameter sign and `R` for a radius. Return `""` when a crop is unreadable. Never guess a value that is not written.

- The reply is validated with a pydantic model (`reads`, each `i` in 1..n, `text` a string). On a validation error, one retry with the error appended. On a second failure or any exception, it returns `None` and `read_values` uses TrOCR if available, else no values and the existing warning "OCR unavailable: enter the dimensions by hand".
- A read that parses gets confidence 0.9; a read that does not parse is dropped with a log line, as today.
- `label.py` is unchanged; it simply runs on Qwen-VL through `env_chat`. The user's face tag still overrides the model.
- Every call is logged by `env_chat` to `logs/vlm.jsonl` with `stage: "mv_read"`.

## 5. Merging photos of the same face: `merge_views.py`

`merge_same_face(observations: list[Observation], images: list[np.ndarray]) -> tuple[list[Observation], list[np.ndarray], list[str]]`: merged observations, one image per merged observation (the reference photo), and warnings.

Runs in `MvPipeline.observe` after every image has an `Observation`. Groups are by exact face tag; opposite faces keep the existing mirror check in `fuse.canonical_outlines`. A group of one passes through unchanged.

For a group of two or more:

1. **Normalise.** Each outline mask is cropped to its bounding box and resized to a shared grid: long side 512 px, aspect ratio the median of the group's bounding-box ratios.
2. **Align.** The reference is the mask of the most confident observation. Every other mask is aligned to it with `cv2.findTransformECC`, `MOTION_AFFINE` (a tilted photo also changes the bounding-box scale, which a rotation alone cannot undo), 100 iterations, epsilon 1e-5, on float masks blurred with sigma 8. If ECC fails to converge, the unaligned mask is used.
3. **Outliers.** A first vote is taken. A mask whose IoU against it is below 0.7, or whose bounding-box aspect ratio differs from the group median by more than 10 percent (the grid stretch would otherwise hide a wrong shape), is dropped with the warning "`<face>`: photo `<n>` disagrees with the others, ignored". The vote is retaken without it. If fewer than two masks remain, the most confident observation passes through alone.
4. **Vote.** A pixel is part when at least half of the masks agree. The merged `PixelOutline` is extracted from the voted mask with the same contour rules as `outline.extract` (largest external contour, children, circularity 0.85).
5. **Circles.** Circles from all photos are mapped into the grid and clustered by centre distance under 5 percent of the grid diagonal. A cluster present in at least half of the photos becomes one circle, with the median centre and median diameter. These replace the circles found on the voted mask.
6. **Scale.** Each photo with `mm_per_px` gives a grid scale: `mm_per_px` times its crop-to-grid resize factor. The merged scale is the median. If the spread (max minus min over median) exceeds 3 percent, warning "`<face>`: scale varies between photos, check the reference object".
7. **Values.** Linked readings keep their axis. Readings on the same axis (or the same merged hole) are clustered at 5 percent. A cluster read on at least half of the photos that have a value there is kept, as its median value with the confidence of its best reading; one face may carry several values on an axis (a total and a partial length). A cluster read on fewer photos is dropped with a warning naming the value. Hole links are remapped to the merged circle clusters by nearest centre.
8. **Labels and depth.** `blind` and `depth_estimates` are remapped the same way. A Solaria result (section 9) for a circle wins over every vision-model label; otherwise a hole is blind when most photos that saw it say so.
9. **Agreement.** The mean IoU of the kept masks against the vote is recorded. The merged observation's confidence is the group's max confidence times the agreement. An info line "`<face>`: merged `<k>` photos, agreement `<a>`" is added to the warnings list so the UI shows it.

The merged observation lives in grid pixels. `observed.images` keeps the reference photo for that face, and `observed.masks[face]` is the normalised voted mask.

## 6. Qwen-Image client: `qwen_image.py`

`ImageGen = Callable[[list[np.ndarray], str, int, str], np.ndarray]`: BGR reference images, a prompt, a seed and a log stage in, one BGR image out. Raises `ImageGenError` on any failure.

- `dashscope_gen()`: POST `{QWEN_IMAGE_BASE_URL}/services/aigc/multimodal-generation/generation` with `{"model": QWEN_IMAGE_MODEL, "input": {"messages": [{"role": "user", "content": [{"text": prompt}, {"image": "data:image/png;base64,..."}, ...]}]}, "parameters": {"prompt_extend": false, "watermark": false, "seed": 7}}` and `Authorization: Bearer <VLM_API_KEY>`. If the response holds `output.task_id`, poll `{QWEN_IMAGE_BASE_URL}/tasks/<id>` every 2 s. The image URL is read from `output.choices[0].message.content[*].image` or `output.results[0].url` and downloaded. Total timeout 90 s.
- `space_gen()`: `gradio_client.Client(QWEN_IMAGE_SPACE, hf_token=HF_TOKEN).predict(input_images=<files>, original_prompt=prompt, enable_extend=False, seed=7, randomize_seed=False, api_name="/generate_with_enhance")`. Remaining parameters take the Space defaults. Total timeout 120 s. The first returned image is used.
- `default_gen() -> ImageGen | None` picks the backend from section 3, or `None` when neither is configured.
- Up to 10 reference images per call (the model limit). References are resized to a long side of 1024 px.
- Each call appends to `logs/vlm.jsonl`: backend, model or Space, stage (`mv_face` or `mv_rescue`), latency.
- Every call passes an explicit `seed`; a retry uses `seed + 1`.

## 7. Predicting a missing face

Lives in `qwen_faces.py`; `complete.py` calls it first, ahead of TripoSR.

`predict_face(gen: ImageGen, refs: list[tuple[str, np.ndarray]], face: str, seed: int) -> np.ndarray | None` returns a binary mask or `None`.

Prompt, with the face table from the multi-view spec section 3:

> Images 1..n show one mechanical part: image 1 is the `<face>` view, image 2 is the `<face>` view, ... Draw the orthographic `<target>` view of the same part, as seen from `<viewer description>`, with `<image-up description>` at the top of the image. Solid black silhouette on a pure white background. Through-holes are white. No text, no dimension lines, no shading, no perspective, no background objects.

Viewer descriptions: front "the front, looking along -Z"; top "above, looking down along -Y, with the front edge at the bottom"; right "the right side, looking along -X". References are the input photos of every observed face (after merging, one per face), most confident first.

The generated image goes through `outline.extract`. An abstention means `None`. Otherwise the outline is stretched to `face_size(face, env)` and becomes `Outline(source="inferred", confidence=0.6)`, provenance `inferred`.

**Consistency gate.** Qwen may draw a plausible but wrong face. The gate checks that the predicted face never removes material the user photographed:

1. Build a 128 x 128 x 128 boolean voxel grid over the envelope. Carve it with the observed and mirrored canonical outline masks and the candidate, each extruded along its axis. A canonical face that is still missing counts as its full rectangle.
2. Project the carved grid back onto every observed canonical face.
3. Every observed face must reach IoU 0.95 against its own outline mask at the same resolution.

Failing the gate, or `None` from extraction, triggers one retry with `seed + 1`. A second failure falls through to TripoSR, then to the assumed rectangle, with the warning "`<face>`: Qwen-Image view rejected, `<fallback>` used".

Order in `complete()`: observed, mirrored, Qwen-Image, TripoSR, assumed. `rejected` from `/mv/merge` or the UI skips Qwen-Image and TripoSR for that face, as it does today. Qwen images are cached per request with the mesh, so `fuse` after an edit never calls Qwen again.

## 8. Sketch rescue

In `MvPipeline.observe`, when `outline.extract` returns `MvAbstain(reason="no_outline")` for a `sketch` or `drawing` and an `ImageGen` is configured:

1. Call `gen([image], prompt)`:

   > Redraw this hand sketch of a mechanical part face as a clean solid black silhouette on a pure white background. Keep the proportions and position exactly. Remove all text, numbers, arrows and dimension lines. Holes are white.

2. Resize the result to the input size and extract its outline.
3. Raw region: foreground strokes of the input, closed with an elliptical kernel 3 percent of the image's long side (enough to bridge a pen gap), then the largest region filled.
4. Accept when the IoU of the cleaned mask against the raw region is at least 0.85 and the bounding-box aspect ratios differ by at most 5 percent. Otherwise return the original abstention.

An accepted rescue keeps `source = observed`, multiplies the observation's confidence by 0.8, and adds the warning "`<face>`: sketch cleaned by Qwen-Image, check it". OCR still runs on the original image; the rescued outline, already at the input size, is what `text_regions` removes before finding text.

## 9. Solaria depth: `depth.py`

Runs in `MvPipeline.observe` before merging, once per face: on the most confident `photo` observation of that face with at least one circle, in that photo's own pixels. Merging then carries its result to the merged circles (section 5, step 8). It never runs on sketches or on Qwen images, so each face costs at most one Solaria call.

**Call.** `gradio_client.Client(SOLARIA_SPACE, hf_token=HF_TOKEN).predict(image, full_white_mask, api_name="/gerar_3d")`, timeout 180 s. The second input is the mask; passing an all-white mask the size of the image makes Solaria return depth for the whole frame, table included. The call returns `(depth_image, ply_path, status)`.

**Depth map from the PLY.** Solaria writes points `x = (col - W/2) / W`, `y = -(row - H/2) / H`, `z = 2 * depth`, on a stride-4 grid. `read_depth(ply_path, W, H) -> np.ndarray` inverts that into an `(H/4, W/4)` depth array with NaN where no point exists. Solaria rescales depth to 0..1 over the mask, and which way depth grows does not matter: the ratio below is unchanged by scale, shift and sign.

**Calibration by ratios.** Marigold depth is affine-invariant: correct up to an unknown scale and shift. Ratios of depth differences are invariant to both. The part lies flat, so the table is exactly `L` below the photographed face, where `L` is the envelope length along that face's axis.

- `d_face`: median depth in a ring from r to 1.5 r around each circle, inside the part mask.
- `d_hole`: median depth inside the circle, radius 0.7 r.
- `d_table`: median depth in a band 5 to 15 percent of the bounding-box diagonal outside the part outline, excluding the reference object box.
- `ratio = (d_hole - d_face) / (d_table - d_face)`.

| Condition | Result |
| --- | --- |
| `abs(d_table - d_face)` below 5 percent of the frame's depth range | uncalibrated: no change, warning "depth: part too flat to measure, check hole depths" |
| `ratio >= 0.9` | through: `blind[i] = False` |
| `0.1 <= ratio < 0.9` | blind: `blind[i] = True`, `depth_ratio[i] = ratio`; `fuse` turns it into `ratio * L` once the envelope is known |
| `ratio < 0.1` | not an opening: no change, warning "`<face>`: hole `<i>` looks like a mark, not a hole" |

Solaria overrides the label's `blind` and `depth_estimates` for that circle. `fuse.features_from` gives these depths provenance `estimated`, so they snap to 0.5 mm and show amber. Any failure (timeout, HTTP, parse) leaves the observation unchanged with the warning "depth unavailable". The depth array is cached per request.

Stated assumption, shown in the app next to the photo upload: photos of real parts are taken top-down, with the part lying flat on a plain surface.

## 10. Pipeline wiring

- `MvPipeline.__init__` gains `image_gen: ImageGen | None`, `batch_reader: BatchReader | None`, `depth: DepthProvider | None`. `DepthProvider = Callable[[np.ndarray], np.ndarray]` (BGR image to depth array).
- `observe`: per image, label, reference, outline (with rescue if needed), OCR (batch reader, else TrOCR); then depth per face; then `merge_same_face`.
- `fuse`: passes `image_gen` and the reference images to `complete`.
- `build`: unchanged.
- `default_pipeline()` wires Qwen-VL chat and batch reader from `VLM_*`, `default_gen()`, the Solaria provider when `SOLARIA_SPACE` is set, TrOCR and TripoSR as today. A missing piece logs one warning and is skipped.
- `routes.py` keeps its endpoints; `/mv/analyze` returns the new warnings and, per canonical face, the provider that filled it.

## 11. Gradio app: `app_mv_gradio.py`

Repository root, `uv run python app_mv_gradio.py`, port 7860. Calls `default_pipeline()` in-process. Per-session state in `gr.State`: the `Observed` object and the latest spec.

1. **Upload.** Multiple image files. A table with one row per file: face (auto, front, back, left, right, top, bottom) and kind (auto, sketch, photo, drawing). Reference object (none, the coins, card, A4). The top-down assumption from section 9. Button Analyze; Gradio's progress indicator shows while it runs.
2. **Review.**
   - One thumbnail per canonical face with its badge: observed, merged xN, mirrored, Qwen-Image, TripoSR, assumed. Inferred faces have a reject checkbox.
   - OCR reads: image, text, value, linked axis or hole.
   - Values table: field path, value, provenance. The value column is editable. Envelope rows first; while missing they are empty, labelled red, with the suggestion from `partial.suggested`.
   - Warnings list.
   - Button Rebuild: `fuse` with the edited values as `user_values` (provenance `user_edited`) and the rejected faces, then `build`.
3. **Result.** `gr.Model3D` of the STL, the six rendered views with IoU per observed face, downloads for STL, STEP and G-code, print time and filament grams.

An `MvAbstain` renders as a card with its stage, reason and remedy.

## 12. Errors and logging

- Every external call has a timeout and catches transport, HTTP and parse errors. A failure falls back as described and adds a warning. Nothing crashes the pipeline; nothing is guessed silently.
- `logs/vlm.jsonl` records every Qwen-VL, Qwen-Image and Solaria call for the disclosure.
- Images sent to DashScope or Hugging Face Spaces leave the machine. The app says so next to the upload, and `docs/disclosure.md` lists both services.

## 13. Testing

Unit tests never touch the network. Fakes stand in for `Chat`, `ImageGen`, `DepthProvider` and `gradio_client`.

- `test_mv_qwen_reader.py`: valid JSON parsed in order; malformed JSON retried once; second failure returns `None` and TrOCR fake is used; unparseable text dropped; the prompt contains every crop.
- `test_mv_merge_views.py`: three synthetic photos of a plate with two holes, with noise, a 20 px shift and a 3 degree rotation, plus one outlier photo. Voted outline IoU against the truth at least 0.97; outlier dropped with its warning; circle centres within 1 percent of the grid diagonal and diameters within 3 percent; scale median and spread warning; value clustering picks the majority.
- `test_mv_qwen_image.py`: DashScope payload shape; synchronous and task-polling responses from recorded JSON fixtures; Space backend through a fake client; backend selection from the environment; timeout raises `ImageGenError`.
- `test_mv_complete.py` additions: fake gen returning the true top silhouette of an L-bracket is accepted; a fake gen returning a silhouette that cuts observed material fails the gate, retries, then falls to the TripoSR fake, then to assumed; rejected faces skip Qwen.
- `test_mv_rescue.py`: a sketch with a broken outline abstains; a fake gen returning the clean silhouette is accepted with the warning; a fake gen returning a distorted shape is rejected and the abstention stands.
- `test_mv_depth.py`: PLY written in Solaria's exact format round-trips to a depth array; synthetic depth maps with a face plane, table plane, a blind hole at 40 percent and a through hole give the right classes and a depth within 5 percent; a flat case is uncalibrated; a failing provider leaves the observation unchanged.
- `test_mv_pipeline.py` addition: one front sketch plus typed envelope, all providers faked; top and right come from the fake Qwen; the part builds.
- `test_mv_gradio.py`: the Blocks build; analyze and rebuild handlers run on fakes.
- Live tests marked `network` (new pytest marker, skipped unless `-m network`): one real call each to Qwen-VL, Qwen-Image and Solaria.
- The existing suite stays green. The failing `test_trocr_reads_printed_digits` is fixed first by adding `sentencepiece` to the `ai` extra.

## 14. Order of work

A working demo exists after every step. If time runs short, cut from the bottom.

1. Setup: git repository, `sentencepiece`, `.env.example`, the `network` marker.
2. `qwen_reader.py` and the batch path in `ocr.read_values`.
3. `merge_views.py` and its wiring in `observe`.
4. `qwen_image.py`, `predict_face` and the gate in `complete.py`.
5. `app_mv_gradio.py`.
6. Sketch rescue.
7. `depth.py` and Solaria.
8. Golden captures, live tests, `README.md`, `docs/models.md`, `docs/disclosure.md`.

## 15. Risks

- **Licence.** Qwen-Image-2.1 is under the Qwen Research License. Confirm that a hackathon demo is allowed before event day, and disclose it.
- **Availability.** Qwen-Image-2.1 may not be on public DashScope yet. The Space backend covers it; the Space is rate-limited.
- **Solaria.** A third-party Space created on 21 September 2026. It can change or disappear, and ZeroGPU quotas need `HF_TOKEN`. Every result it feeds has a fallback. If it breaks, record the Space commit that worked in `docs/models.md`.
- **Latency.** Qwen-Image 20 to 40 s per face, Solaria 60 to 180 s per photo in the ZeroGPU queue. The app shows progress; caching means edits never repeat a model call.
- **Team sign-off.** The multi-view spec is still pending sign-off; this spec depends on it.

## 16. Reference: Hunyuan3D-2.1

`tencent/Hunyuan3D-2.1` (Hugging Face Space, ZeroGPU A10G, Tencent Hunyuan Community License) is recorded as a reference, not wired in. Its `/shape_generation` endpoint takes one image or up to four views (`mv_image_front`, `mv_image_back`, `mv_image_left`, `mv_image_right`) and returns a mesh. It is the strongest candidate to replace TripoSR as the mesh fallback in `complete.py` later, because it can use several of the user's faces at once. Only its silhouettes would be used, exactly like TripoSR today. Check the licence's territory terms before using it.
