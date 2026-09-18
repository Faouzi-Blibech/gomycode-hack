# Sketch-to-CAD: design spec

Date: 2026-09-19
Event: GOMYCODE "Come Build with AI" hackathon, 27 September 2026, 09:00 to 20:00, submission 17:30 Tunis time
Team: 3 people (integrator, geometry owner, numbers owner)
Status: approved design, pre-implementation

## 1. What we are building

A phone camera that turns a real object or a hand-drawn sketch into an editable, parametric CAD file (STEP + STL) you can print or machine.

One-line pitch: point your phone at a broken part or a sketch of one, get a parametric 3D model back.

Three inputs converge on one intermediate representation, the PartSpec, before any 3D is generated.

| Input | How it arrives | Numbers come from | Priority |
| --- | --- | --- | --- |
| Hand sketch | Drawn on paper, dimensions written on it, photographed | Handwritten annotations (OCR) | Primary. Build first. |
| Photo of a real part | Object photographed top-down next to a coin | Coin metrology (OpenCV) | Second |
| Existing 2D drawing | Clean orthographic drawing, printed or exported | Printed annotations (OCR) | Third, cheapest |

## 2. The four rules

### Rule 1: the model never writes code

The vision-language model (VLM) emits a Topology JSON that conforms to a Pydantic schema. `builder.py` is our own deterministic Python that turns a validated PartSpec into geometry.

We never ask an LLM to emit CadQuery, OpenSCAD, Python, or any executable output. Reasons, in the order we give them to a judge:

1. Code generation has unbounded failure modes and cannot be validated field by field.
2. A schema can be checked before anything is built or rendered.
3. No arbitrary code execution path exists in the product at all.
4. Sliders become real parameter edits that re-derive geometry, not mesh scaling.

If a task seems to need generated code, stop and extend the schema instead.

### Rule 2: numbers are measured or user-written, never model-estimated

There are exactly two legitimate sources for a millimetre value:

- Measured: OpenCV coin detection gives mm-per-pixel, and contours are measured in mm.
- User-written: a dimension the user wrote on the sketch or drawing, read by OCR.

The VLM owns topology only: what kind of part, how many holes, roughly where (normalised 0 to 1 positions inside the bounding box are allowed, absolute millimetres are not), which view is which. If the model returns a number where the schema expects a measured or user-written value, that is a bug and the value is discarded.

VLMs are bad at metric estimation and return confident wrong numbers. A number that has no measured or written source is a missing field, and a missing field is handled by Rule 4, never by guessing.

### Rule 3: the 6-face view is the mental model and the validation surface

Internally, geometry is an extruded profile plus features:

- front view: the 2D profile we extrude
- side or top view: extrusion depth
- any view: hole positions, slots, fillets

We do not attempt general multi-view solid reconstruction. When a part cannot be expressed as profile + features, we abstain and say why.

### Rule 4: abstention is a feature

Every stage has a confidence gate. Low confidence means we tell the user what went wrong and what to do about it. We never silently guess. Every abstention carries a machine-readable `reason` and a human-readable `remedy`.

| Gate | Condition | Remedy shown to the user |
| --- | --- | --- |
| Coin geometry | Coin ellipse eccentricity above 0.30, or no coin found | "Lay the coin flat, shoot top-down, retake." |
| Grammar | Topology part_type not in the frozen grammar | "Unsupported geometry: <reason>." |
| Missing dimension | A required PartSpec field has no measured or written source | "We could not read <field>. Enter it below." The UI shows an input, the field is marked `user_edited`. |
| Schema | Model output fails Pydantic validation twice | "The model could not describe this part. Try a cleaner photo." |
| Round-trip | IoU between the input silhouette and the reprojected solid below 0.85 | "Low confidence, check the dimensions." Result shown in amber, not blocked. |

## 3. The part grammar (frozen)

Supported part types. Anything else abstains. Do not extend this list on build day.

| Type | Parameters |
| --- | --- |
| `plate` | width_mm, height_mm, thickness_mm, corner_radius_mm (default 0) |
| `l_bracket` | leg_a_mm, leg_b_mm, width_mm, thickness_mm, angle_deg (default 90) |
| `flange` | outer_diameter_mm, bore_diameter_mm, thickness_mm, bolt_circle_diameter_mm, bolt_count, bolt_hole_diameter_mm |
| `spacer` | outer_diameter_mm, inner_diameter_mm (0 means solid), length_mm |
| `profile_extrusion` | points_mm (closed polyline, at least 3 points), depth_mm |

Features that may attach to any type:

| Feature | Parameters |
| --- | --- |
| `hole` | x_mm, y_mm, diameter_mm, depth_mm (null means through), leg (`a` or `b`, l_bracket only) |
| `slot` | x_mm, y_mm, width_mm, length_mm, angle_deg, depth_mm (null means through), leg |
| `fillet` / `chamfer` | edges (`all`, `all_vertical`, `top`, `bottom`), radius_mm |

All dimensions are millimetres, floats. No inches anywhere. No unit strings. Positions are measured from the bottom-left corner of the front-view bounding box of the part, x to the right, y upward.

## 4. Architecture

```text
image ──┬─ metrology.py   OpenCV: coin -> mm/px, contours in mm       (photo path)
        ├─ ocr.py         handwritten/printed dims -> Annotations     (sketch, drawing)
        └─ vision.py      VLM -> Topology JSON, never numbers          (all paths)
                 │
             merge.py     fuse + confidence gates -> PartSpec | Abstain
                 │
        PartSpec (Pydantic, single source of truth) <── slider/field edits from the UI
                 │
             builder.py   CadQuery -> STEP + STL
                 │
             views.py     6 orthographic silhouettes -> IoU vs input silhouette
```

### 4.1 Modules and owners

All Python modules live in the `s2c` package (`s2c/partspec/`, `s2c/vision/`, `s2c/merge.py` and so on). `s2c/fakes/` holds stand-ins for every module so the pipeline runs before the real modules land.

| Module | Responsibility | Owner |
| --- | --- | --- |
| `partspec/` | Pydantic models for PartSpec, Topology, Annotations, Measurements, Abstain. JSON schema export. | Integrator |
| `vision/` | OpenAI-compatible client, prompt templates, schema-in-prompt, validate, retry once with the error, abstain on second failure. | Integrator |
| `merge.py` | Fuse Topology + Annotations + Measurements into a PartSpec. Confidence gates. Provenance per field. | Integrator |
| `api.py` | FastAPI: `/analyze`, `/build`, `/files/{id}`. Stateless, files in a temp dir with TTL. | Integrator |
| `app_gradio.py` | Lab UI: upload, run pipeline, show PartSpec JSON, silhouettes, IoU, download. | Integrator |
| `web/` | React + Three.js mobile web app: camera capture, sliders, STL viewer, download. | Integrator, with geometry owner on the viewer |
| `builder.py` | PartSpec to CadQuery solid. STEP + STL export. | Geometry owner |
| `views.py` | Solid to 6 silhouettes as binary images. | Geometry owner |
| `silhouette.py` | Input image to a normalised binary silhouette. `normalize_mask` and `iou`, shared by views and the API. | Integrator |
| `tests/golden/` ground-truth parts | Hand-modelled reference parts with known dimensions, sketches and photos of them. | Geometry owner |
| `ocr.py` | Find dimension annotations on a sketch or drawing, read the value, link it to an edge or a hole. | Numbers owner |
| `metrology.py` | Coin detection, eccentricity gate, mm-per-pixel, outer contour and hole circles in mm. | Numbers owner |
| `tests/` for OCR and metrology accuracy | Labelled set, accuracy report. | Numbers owner |

### 4.2 Contracts (frozen on day one)

These are Pydantic models in `partspec/`. Everyone codes against them and against fakes of the other modules.

Topology, produced by `vision.py`. No absolute numbers.

```text
Topology
  part_type: plate | l_bracket | flange | spacer | profile_extrusion | unsupported
  unsupported_reason: str | null
  view: front | top | side | isometric | unknown
  holes: list of { u: 0..1, v: 0..1, kind: through | blind | unknown }     # normalised position in bbox
  slots: list of { u, v, orientation: horizontal | vertical | diagonal }
  rounded_corners: bool
  symmetric: bool
  bolt_count: int | null           # flange only
  annotation_count: int | null     # how many written dimensions the model sees, not their values
  confidence: 0..1
  notes: str
```

Annotations, produced by `ocr.py`.

```text
Annotations
  items: list of Annotation
  confidence: 0..1
Annotation
  value_mm: float
  kind: linear | diameter | radius
  bbox_px: [x, y, w, h]
  linked_to: width | height | thickness | depth | length | leg_a | leg_b | corner_radius | hole_diameter | hole_x | hole_y | slot_length | slot_width | outer_diameter | inner_diameter | bolt_circle_diameter | bolt_hole_diameter | unknown
  hole_index: int | null
  confidence: 0..1
```

Measurements, produced by `metrology.py`.

```text
Measurements
  mm_per_px: float
  coin: { name: str, pixel_diameter: float, eccentricity: float, confidence: 0..1 }
  outer_contour_mm: list of [x, y]
  bbox_mm: { width, height }
  circles_mm: list of { x, y, diameter }
  confidence: 0..1
```

Abstain, produced by any stage.

```text
Abstain
  stage: metrology | ocr | vision | merge | build | verify
  reason: str          # machine-readable slug, e.g. coin_tilted, unsupported_geometry, missing_thickness
  remedy: str          # one sentence for the user
  partial: dict | null # values already recovered, so the UI can show inputs for the missing ones
```

PartSpec, produced by `merge.py`, edited by the UI, consumed by `builder.py`.

```text
PartSpec
  version: "1"
  source_input: sketch | photo | drawing
  part: Plate | LBracket | Flange | Spacer | ProfileExtrusion   (discriminated on "type")
  features: list of Hole | Slot
  finishes: list of Fillet | Chamfer
  provenance: { "<field path>": measured | user_written | user_edited | default }
  confidence: 0..1
  warnings: list of str
```

Every numeric field in `part` and `features` must have a provenance entry. `merge.py` raises if one is missing. `default` is allowed only for fields the grammar marks with a default.

### 4.3 Merge rules

Sketch and drawing path:
1. Topology gives part_type and feature count and rough positions.
2. Annotations give values. Each is matched to a PartSpec field through `linked_to`. Unlinked annotations are matched by heuristics: the largest linear value goes to the largest bounding-box side, a diameter annotation nearest a hole goes to that hole.
3. Hole positions: if hole_x and hole_y annotations exist, use them. Otherwise, use the normalised topology position times the measured or written width and height, and mark provenance `default` with a warning, since the user will confirm it on the sliders.
4. Any required field with no source triggers a missing-dimension abstention with a remedy that names the field and a `partial` dict of what was recovered. The UI collects the value and calls `/merge` again with `user_values`, which are recorded as `user_edited`.
5. `profile_extrusion` is not supported from a sketch in version 1. Its outline needs a measured contour, so a sketch of one abstains with reason `profile_needs_photo`.

Photo path:
1. Metrology gives scale, the outer bounding box, and circles.
2. Topology gives part_type and which circles are holes versus the bore.
3. Thickness cannot be measured from a top-down photo. It is always a missing-dimension prompt unless a second side photo is provided. Version 1 asks the user.

### 4.4 Model provider layer

One client built on the `openai` Python package with a configurable base URL. Three environment variables: `VLM_BASE_URL`, `VLM_MODEL`, `VLM_API_KEY`. Presets documented in `docs/models.md`:

| Provider | Base URL | Use |
| --- | --- | --- |
| Google Gemini free tier | `https://generativelanguage.googleapis.com/v1beta/openai/` | Pre-event testing |
| Groq free tier | `https://api.groq.com/openai/v1` | Pre-event testing, fast |
| Ollama local | `http://localhost:11434/v1` | Offline fallback |
| NVIDIA Build | `https://integrate.api.nvidia.com/v1` | Event day, primary |

Model names are chosen from each provider's catalogue at the time and recorded in `docs/models.md` with the date checked. The call sends the image plus the Topology JSON schema in the prompt, asks for JSON only, validates with Pydantic, retries once with the validation error appended, and returns an Abstain on the second failure. Every response is logged with provider, model, latency and token counts for the tool disclosure.

### 4.5 API

| Endpoint | Input | Output |
| --- | --- | --- |
| `POST /analyze` | image file, `input_kind` (sketch, photo, drawing) | `{ partspec, topology, annotations?, measurements?, silhouette_id }` or `{ abstain }` |
| `POST /merge` | topology, annotations, measurements, `source_input`, `user_values` | `{ partspec }` or `{ abstain }`. Re-fuses without a new model call. |
| `POST /build` | PartSpec JSON, `silhouette_id` | `{ stl_url, step_url, iou, views: [6 image urls], warnings }` or `{ abstain }` |
| `GET /files/{id}` | file id | the file, deleted after 1 hour |

Images are never written to disk beyond the request lifetime except the silhouette needed for `/build`, which lives in the temp directory with the same TTL.

### 4.6 User interface

Gradio lab view: single page, upload, choose input kind, run, see Topology, Annotations, Measurements and PartSpec JSON side by side, the six silhouettes, IoU, and download buttons. This is the developer harness and the "lab" screen for technical judges.

Mobile web app: three screens.
1. Capture: camera input, pick sketch or photo, a coin hint overlay for photos.
2. Review: PartSpec as sliders and number fields, each labelled with its provenance badge (measured, written, edited, default), 3D preview re-derived on every change, IoU shown with a green or amber badge, abstention card when needed.
3. Export: download STL and STEP, share link.

### 4.7 Verification

- Round-trip: `views.py` renders the built solid to the same view as the input, `silhouette.iou` compares it with the input silhouette after both are cropped to their bounding box and resized, so the check is scale-invariant. Threshold 0.85 for green.
- Golden set: at least 10 sketches and 5 photos with known dimensions. A test asserts every PartSpec numeric field is within 5 percent or 1 mm of ground truth, whichever is larger.
- Builder: known-volume tests for every part type, STEP and STL files open in FreeCAD.
- Views: self round-trip IoU of a built solid against its own silhouette is above 0.98.
- OCR: digit and value accuracy on the labelled set, reported as a number in the README.
- Metrology: mm-per-pixel error under 2 percent on three coin photos at three distances.
- Abstention tests: tilted coin, unsupported part, missing thickness, garbage image each produce the right reason.

## 5. Team workflow

- `main` is always demoable. Feature branches per person, PRs reviewed by one other person, squash merge.
- Commit messages are plain, in the team's voice. No AI attribution lines and no co-author trailers, in commits or PR bodies.
- Contracts in `partspec/` change only through a PR that all three approve, and never on event day.
- Each person has a role brief in `docs/roles/` written to be pasted into their AI assistant.
- `CLAUDE.md` at the repo root carries the rules, grammar and contracts so any AI tool in the repo respects them.

## 6. Timeline

| Milestone | Date | Done means |
| --- | --- | --- |
| M0 Contracts | 20 Sep | `partspec/` models merged, JSON schema exported, fakes for every module, repo scaffold, CI runs tests. |
| M1 Geometry + model | 22 Sep | Builder produces STEP and STL for all five types from hand-written PartSpecs. Views produce 6 silhouettes. Provider layer returns valid Topology on a free model. |
| M2 Sketch path | 24 Sep | Sketch to STL end to end in Gradio. OCR reads the golden sketches. Merge and gates work. Golden test passes on at least 7 of 10 sketches. |
| M3 Photo path + app | 26 Sep | Coin metrology on 5 photos. Mobile web app runs on a phone against the API. Video script drafted, demo parts sourced, one part printed if a printer is found. |
| Event | 27 Sep | See day plan. |

Event day plan:
- 09:00 to 10:00: get NVIDIA key, set env vars, run the golden set on the NVIDIA model. If accuracy drops, keep the fallback provider and disclose both.
- 10:00 to 14:00: fixes, polish, mentors.
- 14:00 to 15:30: record the 90-second video. Not later.
- 15:30 to 17:00: project card, tool disclosure, README, final deploy.
- 17:00 to 17:30: submit. Nothing new after 17:00.

## 7. Judging map

| Criterion | Points | What we show |
| --- | --- | --- |
| Problem + user value | 20 | A broken bracket with no replacement available. Sketch it, print it. |
| Functional execution | 20 | Live sketch to printable STL on a phone, sliders re-derive geometry. |
| Quality of AI use | 20 | The model does topology only. Why we never generate code. Provider swap and why NVIDIA Build. |
| Testing + reliability | 15 | Abstention gates, round-trip IoU, golden-set accuracy numbers, latency and cost per request. |
| Experience + demo | 15 | 90 seconds: sketch, phone, sliders, print. |
| Responsible AI + data | 10 | No image storage beyond the request, human edits before export, no code execution, full model and tool disclosure. |

## 8. Out of scope for version 1

General multi-view reconstruction, parts outside the grammar, inch units, assemblies, texture or colour, thickness from a single top-down photo, account systems, persistent storage.

## 9. Open items

- Confirm the tools of the geometry owner and the OCR approach of the numbers owner when they are present. The role briefs cover both a trained-model and a VLM-assisted OCR path.
- Source demo parts and a 3D printer before 26 September.
- Pick model names per provider and record them in `docs/models.md`.
- Pre-event build is disclosed honestly in the project card.
