# Multi-view reconstruction and G-code: design spec

Date: 2026-09-22
Author: geometry owner
Status: approved by the geometry owner, pending team sign-off (see section 10)
Extends: `docs/superpowers/specs/2026-09-19-sketch-to-cad-design.md`

## 1. What changes

The first spec builds one extruded profile from one image. This spec adds a second path, the multi-view path, next to it:

- The user gives 1 to 6 images of a part, one per face (front, back, left, right, top, bottom). Each image is a hand sketch, a photo of the real part, or a clean drawing.
- Faces the user did not give are completed: first by the mirror rule, then by a pretrained image-to-3D model (TripoSR) whose output is used only to predict the missing outlines.
- Dimensions come from handwritten values (OCR), reference objects in the frame, one known dimension, standard-size snapping, and, last, a model estimate from context. A minimum set is mandatory; the user is asked when it is missing.
- The solid is the intersection of the three extruded outlines (the visual hull), built by our own deterministic CadQuery code.
- Outputs: STL, STEP, and G-code for an FDM printer, sliced by PrusaSlicer CLI with one fixed profile.

The existing PartSpec path, `builder.py`, `views.py`, `merge.py` and the `partspec/` contracts are not modified by this work.

## 2. Rules for this path

The four rules of the first spec are restated for the multi-view path:

1. **The model never writes code.** Unchanged. Models return JSON or meshes. A mesh is only rendered to silhouettes; it is never exported or executed.
2. **Numbers have a source and a badge.** Model estimates are now allowed, but only for values that are not part of the minimum set (section 4.1), always with provenance `estimated` or `inferred`, always shown in amber until the user confirms them.
3. **Six views, three outlines.** A silhouette seen from the back is the mirror of the silhouette seen from the front, and the same holds for left/right and top/bottom. Six views carry three independent outlines. Opposite faces add depth features (blind holes, slots) and a consistency check.
4. **Abstention is a feature.** Unchanged. Every gate returns a reason slug and a one-sentence remedy.

## 3. Coordinates

Global frame: X is width, Y is height, Z is depth. The part fills the envelope box from `(0, 0, 0)` to `(x_mm, y_mm, z_mm)`.

Each face has a local 2D frame `(a, b)` in millimetres. `a` points right in the image and `b` points up. The origin is the bottom-left of the envelope rectangle as seen from that face. Viewers look at the part from outside, with the image up direction given below.

| Face | Viewer at | Image up | `a` maps to | `b` maps to | Outline axis |
| --- | --- | --- | --- | --- | --- |
| front | +Z | +Y | `X = a` | `Y = b` | Z |
| back | -Z | +Y | `X = x_mm - a` | `Y = b` | Z |
| top | +Y | -Z | `X = a` | `Z = z_mm - b` | Y |
| bottom | -Y | +Z | `X = a` | `Z = b` | Y |
| right | +X | +Y | `Z = z_mm - a` | `Y = b` | X |
| left | -X | +Y | `Z = a` | `Y = b` | X |

The front-view convention matches the first spec: front view in the XY plane, depth along +Z, origin at the bottom-left.

Mirror rule, used to turn an opposite face into the canonical one:

- back to front: `(a, b) -> (x_mm - a, b)`
- bottom to top: `(a, b) -> (a, z_mm - b)`
- left to right: `(a, b) -> (z_mm - a, b)`

The canonical faces are `front`, `top` and `right`. The builder consumes only those three outlines.

## 4. Dimensions

### 4.1 Minimum gate

The envelope `x_mm`, `y_mm`, `z_mm` must each have a trusted source before anything is built. Trusted sources are:

- `user_written`: a value the user wrote on a sketch or drawing, read by OCR
- `measured`: a length measured on a photo of the real part with a reference object in the frame
- `user_edited`: a value the user typed in the app

If any axis has no trusted source, the path stops with `MvAbstain(stage="dimensions", reason="missing_x" | "missing_y" | "missing_z", remedy="Enter the width | height | depth in mm.", partial=<values recovered so far>)`. The app shows an input for exactly that axis and calls merge again. When another view shows the missing axis next to a known one, `partial.suggested` carries a value scaled from the pixel proportions, so the input can be pre-filled; the user still has to confirm it, and it is then `user_edited`. A model estimate never satisfies the gate.

Axis names shown to the user: X is "width", Y is "height", Z is "depth".

### 4.2 OCR of handwritten values

Every sketch or drawing image goes through OCR.

- Backend: a reader function `crop -> (text, confidence)`. The numbers owner's reader plugs into this hook when `ocr.py` lands; until then the fallback reader below is used.
- Fallback reader: text regions are the connected components left after removing the largest outline; each region is read by `microsoft/trocr-base-handwritten` (Hugging Face, runs on the RTX 3050 or on CPU). The result is parsed with the pattern `^(⌀|D|R)?\s*\d+(\.\d+)?$`; anything else is dropped with a log line. `⌀` or `D` means diameter, `R` means radius, no prefix means linear.
- Linking, per image, relative to the outline bounding box: a linear value whose text sits below or above the outline links to that face's `a` axis; left or right links to its `b` axis; a diameter or radius value links to the nearest circular opening. On each axis, the largest linear value is the envelope length on that axis.
- The face table in section 3 turns `a` and `b` into X, Y or Z. When two images give different values for the same axis, and they differ by more than 5 percent, the value with the higher OCR confidence wins and a warning names both.

### 4.3 Reference objects

Photos of a real part may include one reference object. Its known size gives millimetres per pixel for that image.

| Object | Size (mm) | Detected by |
| --- | --- | --- |
| 1 TND coin | 25.0 diameter | numbers owner's `metrology.measure` |
| 1 EUR coin | 23.25 diameter | numbers owner's `metrology.measure` |
| 2 EUR coin | 25.75 diameter | numbers owner's `metrology.measure` |
| Credit card (ISO ID-1) | 85.60 x 53.98 | largest quadrilateral with aspect 1.586 ± 3 percent |
| A4 sheet | 210 x 297 | largest quadrilateral with aspect 1.414 ± 3 percent; its four corners also rectify perspective |

A hand sketch is not drawn to scale. Pixel lengths on a sketch are never `measured`, even if the sketch sits on an A4 sheet. For sketches, pixel ratios only give proportions (`scaled`).

### 4.4 Fusion order

Every number that is not in the envelope (hole diameters, hole positions, slot sizes, fillet radii) takes the first available source in this order:

1. `user_written`: OCR value linked to that feature
2. `measured`: pixels times the reference-object scale, photos only
3. `scaled`: pixels times the scale fixed by the trusted envelope lengths of that view
4. `inferred`: taken from a view predicted by TripoSR
5. `estimated`: a value the vision model returns from context
6. `default`: a documented default, for example "blind hole depth defaults to half the depth along its axis"

`user_edited` replaces any of these when the user changes a value in the app.

### 4.5 Standard-size snapping

Snapping applies to values whose provenance is `scaled`, `inferred` or `estimated`. It never changes `user_written`, `user_edited` or `measured`.

- Hole diameters snap to the nearest metric clearance hole when within 0.4 mm: 2.7 (M2.5), 3.4 (M3), 4.5 (M4), 5.5 (M5), 6.6 (M6), 9.0 (M8).
- Derived wall thicknesses snap to 1, 1.5, 2, 3, 4, 5, 6, 8 or 10 mm when within 0.3 mm. The envelope itself is trusted and never snaps.
- Every other length snaps to a 0.5 mm grid.

Each snapped field path is listed in `snapped`, so the UI can show "snapped from 5.37".

### 4.6 Consistency checks

- If a view's pixel aspect ratio differs from the ratio of its two trusted envelope lengths by more than 5 percent, the view gets the warning "photo is not square-on, retake it facing the part".
- If a face and its opposite face are both given, their outlines after the mirror rule must reach IoU 0.9. Below that, a warning is added and the outline with the higher confidence is kept.
- If a written value and a reference-object measurement of the same axis differ by more than 5 percent, the written value wins and a warning names both.

## 5. Contract

New models live in `s2c/multiview/spec.py`, owned by the geometry owner. They do not change `partspec/`. When the integrator's `partspec.Abstain` lands, `MvAbstain` can become an alias of it.

```text
Face          = front | back | left | right | top | bottom
CanonicalFace = front | top | right
MvProvenance  = user_written | measured | user_edited | scaled | inferred | estimated | default
ViewSource    = observed | mirrored | inferred | assumed

Envelope
  x_mm: float > 0
  y_mm: float > 0
  z_mm: float > 0

Outline                                  # in the face's (a, b) frame, millimetres
  outer: list of (a, b), at least 3 points, closed implicitly, inside the envelope rectangle
  inner: list of list of (a, b)          # non-circular through-openings along the face axis
  source: ViewSource
  confidence: 0..1

FaceFeature = FaceHole | FaceSlot      # discriminated on "type"
FaceHole
  type: "hole"
  face: Face
  a_mm, b_mm: float                      # centre, in that face's frame
  diameter_mm: float > 0
  depth_mm: float > 0 | null             # null means through along the face axis
FaceSlot
  type: "slot"
  face: Face
  a_mm, b_mm: float
  width_mm, length_mm: float > 0         # length is end to end
  angle_deg: float = 0                   # in the face frame, from +a toward +b
  depth_mm: float > 0 | null

Finish = Fillet | Chamfer                # same fields and edge selectors as the frozen grammar

MultiViewSpec
  version: "mv1"
  envelope: Envelope
  views: { front: Outline, top: Outline, right: Outline }
  features: list of FaceFeature
  finishes: list of Finish
  provenance: { "<field path>": MvProvenance }
  snapped: list of field paths
  warnings: list of str
  confidence: 0..1

MvAbstain
  stage: label | outline | dimensions | complete | build | slice | verify
  reason: str
  remedy: str
  partial: dict | null
```

Validation rules:

- Every numeric field of `envelope`, `features` and `finishes` has a provenance entry. Outline points carry one entry per outline (`views.front.outer`, and so on).
- `envelope.x_mm`, `y_mm`, `z_mm` provenance must be `user_written`, `measured` or `user_edited`.
- Outline points lie inside the envelope rectangle of their face, with 0.5 mm tolerance.
- Circular through-openings found on an image become `FaceFeature` holes with `depth_mm = null`, so their diameters are editable and snap. Only non-circular openings (circularity below 0.85) stay in `inner`.

Edge selectors keep the frozen grammar meaning: `all`, `all_vertical` (edges parallel to Z), `top` (edges on the `>Z` face, the one the front viewer sees), `bottom` (edges on the `<Z` face).

## 6. Stages

All code lives in `s2c/multiview/`. One file per stage.

| File | Input | Output |
| --- | --- | --- |
| `spec.py` | | Contract models, face frame conversions, mirror rule |
| `raster.py` | mesh or solid triangles | binary masks per face; mask normalisation and IoU |
| `outline.py` | image | outer contour, inner contours and circles, in pixels |
| `ocr.py` | image, outline | linked handwritten values (numbers owner's reader or the TrOCR fallback) |
| `reference.py` | photo | mm per pixel from a coin, card or A4 sheet, or none |
| `label.py` | images | face per image, features seen, estimates, through the existing VLM client |
| `fuse.py` | all of the above, user values | `MultiViewSpec` or `MvAbstain` |
| `hf3d.py` | one image | mesh, from local TripoSR or the Hugging Face Space |
| `complete.py` | spec with missing views, mesh | spec with all three canonical outlines |
| `build.py` | `MultiViewSpec` | CadQuery solid |
| `slice.py` | solid | oriented STL, G-code, print stats |
| `pipeline.py` | images, user values | the whole path, with fakes for anything not installed |
| `routes.py` | HTTP | FastAPI `APIRouter` the integrator mounts |

### 6.1 Label

The vision model receives each image and returns JSON only:

```text
MvLabel (per image)
  face: Face | unknown
  input_kind: sketch | photo | drawing
  holes: list of { u: 0..1, v: 0..1, blind: bool }
  description: str                       # "shelf bracket", "motor mount plate"
  estimates: { "<field path>": float }   # optional, context guesses, only for non-envelope fields
  confidence: 0..1
```

If the user tagged the face in the app, the tag wins over the model. If the face stays `unknown`, the path stops with `MvAbstain(stage="label", reason="face_unknown", remedy="Tell us which face this photo shows.")`. The call reuses the integrator's `VLMClient` (validate, retry once with the error, log provider, model, latency, tokens).

### 6.2 Outline

The image is resized so the longest side is 1600 px. Then: grayscale, Gaussian blur, Otsu threshold, morphological close. For photos, the reference object is masked out before this step. The largest external contour is the outer outline. Its children are inner contours; a child with circularity at least 0.85 is a circle. No contour larger than 2 percent of the image means `MvAbstain(stage="outline", reason="no_outline", remedy="Retake on a plain background with the whole part in frame.")`.

Pixels become millimetres with the view scale: the reference scale for photos, otherwise the scale that maps the outline bounding box to the trusted envelope lengths of that face.

### 6.3 Complete

For each canonical face (front, top, right):

1. `observed`: an image of that face exists.
2. `mirrored`: only the opposite face exists; apply the mirror rule.
3. `inferred`: neither exists.
   - Take the observed image with the highest label confidence, remove its background with `rembg`, and run TripoSR (`stabilityai/TripoSR`, MIT licence) at marching-cubes resolution 256.
   - Local first: on CUDA, with the renderer chunk size reduced to fit 6 GB. On a CUDA out-of-memory error, no CUDA device, or 60 seconds without a result, call the Hugging Face Space named by `TRIPOSR_SPACE` (default `stabilityai/TripoSR`) through `gradio_client`, with a 90 second timeout.
   - Orientation: try the 24 axis-aligned rotations of the mesh. For each, render the silhouette along the axis of the observed face, and score its IoU against the observed outline. Keep the best. A best IoU below 0.6 means the mesh is unreliable; fall through to step 4 with the warning "predicted view unreliable".
   - Scale the rotated mesh on each axis so its bounding box equals the envelope, then translate it to the origin.
   - Render the missing face, take its largest contour, simplify it with Douglas-Peucker at 0.5 mm, and drop openings smaller than 3 mm across. The outline gets `source = inferred` and its provenance is `inferred`.
4. `assumed`: TripoSR and the Space both failed, or the mesh was unreliable. The outline is the full envelope rectangle, with the warning "assumed rectangular <face>, check it".

The TripoSR mesh is cached in the temp store under the request id for one hour, so `merge` never reruns it.

### 6.4 Build

Deterministic CadQuery. No model output is ever executed.

1. Front outline in the XY plane at Z = 0, inner openings cut, extruded to Z = `z_mm`.
2. Top outline in the XZ plane (converted with section 3), extruded across Y from 0 to `y_mm`.
3. Right outline in the ZY plane (converted with section 3), extruded across X from 0 to `x_mm`.
4. `solid = A ∩ B ∩ C`.
5. Each `FaceFeature` is cut from its face inward along the face axis: through when `depth_mm` is null, otherwise to `depth_mm`.
6. Finishes are applied last with the frozen edge selectors.

Workplanes are built from explicit `cq.Plane(origin, xDir, normal)` values, and a bounding-box test covers each one.

Errors are `BuildError(reason, remedy)`, as in `builder.py`:

- `intersection_empty`: the intersection has no volume. Remedy: "The views do not describe one part. Check which face each photo shows."
- `invalid_solid`: the result is not one valid solid. Remedy: "Simplify the outline or retake the photo."
- `feature_outside_part`: a feature centre lies outside its face rectangle. Remedy: "A hole or slot lies outside the part. Check its position."
- `fillet_failed` / `chamfer_failed`: remedy "Reduce the fillet radius." or "Reduce the chamfer size."

Exports: `part.step` and `part.stl` (tolerance 0.01, angular tolerance 0.1) in the design frame.

### 6.5 Slice

- Slicer: PrusaSlicer CLI, found through `SLICER_PATH`, or `prusa-slicer-console` on the path.
- Profile: `profiles/fdm_default.ini` in the repo, overridable with `SLICER_PROFILE`. Generic Marlin printer, 220 x 220 x 250 mm bed, 0.4 mm nozzle, PLA, 0.2 mm layers, 3 perimeters, 20 percent infill, supports on build plate only, 45 degree support threshold.
- Orientation, chosen by us, not by the slicer: for each of the six axis-aligned "down" directions, sum the area of the planar faces whose outward normal points down and that lie on the minimum plane in that direction. Keep the direction with the largest area; on a tie, keep the lowest resulting height. The solid is rotated with CadQuery and written as `part_print.stl`.
- Bed check: the oriented bounding box must fit the bed in the profile. Otherwise `MvAbstain(stage="slice", reason="too_big_for_bed", remedy="The part is larger than the printer bed. Scale it down or split it.")`.
- Command: `prusa-slicer-console --export-gcode --load <profile> --center <bed centre> --output part.gcode part_print.stl`, with a 120 second timeout. The bed centre is read from `bed_shape` in the profile (110,110 for the default). A non-zero exit is `MvAbstain(stage="slice", reason="slicer_failed", remedy="The slicer could not process this part. Check the model in the viewer.")` with the last 20 lines of stderr logged.
- No slicer installed: STL and STEP are still returned, with the warning "G-code unavailable: slicer not installed". This is not an abstention.
- Print stats: print time from `; estimated printing time (normal mode) = ...` and filament weight from `; filament used [g] = ...` in the G-code.

### 6.6 Verify

`raster.py` renders the built solid from all six faces. Each face with an input image is compared with that image's outline mask after both are cropped to their bounding box, padded to a square and resized to 512 px. IoU below 0.85 turns that face amber with the warning "Low confidence on <face>, check the dimensions."

Rasterising fills each triangle with its own `cv2.fillPoly` call. One call with every triangle would fill overlapping triangles with the even-odd rule and cancel the front face against the back face.

## 7. API and UI

### 7.1 API

`s2c/multiview/routes.py` exposes an `APIRouter` with the prefix `/mv`. The integrator mounts it in `api.py`. Files live in the temp store for one hour, like the rest of the API.

| Endpoint | Input | Output |
| --- | --- | --- |
| `POST /mv/analyze` | 1 to 6 image files; per image an optional face tag and input kind; optional reference object name | `{ request_id, spec }` or `{ request_id, abstain }`, plus labels, outlines and OCR reads for the lab view |
| `POST /mv/merge` | `request_id`, `user_values` (envelope and field paths), accepted or rejected inferred faces | `{ spec }` or `{ abstain }`. No new model calls. |
| `POST /mv/build` | `MultiViewSpec` | `{ stl_url, step_url, gcode_url | null, print_time_s, filament_g, views: 6 image urls, iou: { face: float }, warnings }` or `{ abstain }` |

A rejected inferred face is replaced by the `assumed` rectangle.

### 7.2 Web app

Changes to the three screens in `web/`, built after the integrator's web scaffold exists:

- Capture: add up to six photos; each gets a face picker (front, back, left, right, top, bottom, auto) and a kind picker (sketch, photo, drawing).
- Review: three required envelope fields at the top, red and blocking the build while empty; one thumbnail per canonical face with its badge (observed, mirrored, inferred, assumed) and accept or replace buttons; every value with its provenance badge, amber for `scaled`, `inferred`, `estimated` and `default`; the 3D viewer rebuilds on each change after 400 ms.
- Export: STL, STEP and G-code downloads, print time, filament grams.

## 8. Testing

- `spec.py`: every face frame conversion and mirror rule; round-trip conversions; validation rejects an envelope with an untrusted source and outline points outside the envelope.
- `fuse.py`: the minimum gate returns `missing_x`, `missing_y`, `missing_z` with the right `partial`; fusion order picks the right source; snapping tables; snapping never touches `user_written`.
- `outline.py`: synthetic images with a rectangle, a rectangle with two circles, a slot, and a blank page.
- `ocr.py` linking: synthetic annotation boxes around a known outline link to the right axis.
- `build.py`: volume within 0.5 percent for a box, a box with two through holes, a box with a blind hole, a cylinder (front circle, top and right rectangles), an L-shape, and a flange; a bounding-box test for each workplane; `intersection_empty` for disjoint views.
- `complete.py`: TripoSR replaced by a fake that returns a trimesh box or cylinder in one of the 24 orientations; the search recovers the orientation and the inferred outline matches the true one at IoU 0.95 or more; the fallback to `assumed` when the fake raises.
- `raster.py`: a solid rendered from all six faces against analytic masks, IoU 0.98 or more.
- `slice.py`: orientation choice on an L-shape and a plate; bed check; G-code stats parsing on a stored sample header. Slicer tests are marked `slicer` and skip when no slicer is installed.
- One real TripoSR test marked `gpu`, skipped in CI.
- Golden set: five parts, each captured once as a single view and once as three views, under `tests/golden_mv/<name>/` with `expected.json` holding the true envelope and features.

## 9. Order of work

Five days remain before the event. The order makes sure a demo exists after step 2, even if TripoSR never installs.

1. `spec.py`, `raster.py`, `build.py` with a hand-written `MultiViewSpec`.
2. `slice.py` and the profile. Demo: hand-written spec to G-code.
3. `outline.py`, `reference.py`, `ocr.py`, `fuse.py`, `label.py`; the mirror rule in `complete.py`.
4. `hf3d.py` and the TripoSR branch of `complete.py`.
5. `pipeline.py`, `routes.py`, a command-line script `scripts/mv.py` for the lab demo.
6. Web changes, once the integrator's web scaffold is on `main`.

Main risk: TripoSR's `torchmcubes` dependency compiles native code and can fail on Windows. Mitigations: the Hugging Face Space fallback, and the `assumed` rectangle fallback. Neither stops the build.

New dependencies, in an optional `ai` group so the base install stays light: `torch` (CUDA build), `transformers` (TrOCR), `rembg`, `trimesh`, `gradio_client`, and TripoSR from its GitHub repository. `slice.py` needs only CadQuery, so the base install can slice without the `ai` group.

## 10. Team sign-off

This spec changes three positions of the first spec. Each needs the other two owners' agreement before it is demoed as the team's approach:

- Rule 2: model estimates are allowed for non-envelope values, badged `estimated` or `inferred`.
- Rule 3: multi-view reconstruction is in scope, as a visual hull of three outlines.
- Output: G-code is added next to STEP and STL.

The multi-view path does not change `partspec/`, so it can be built and merged without the three-approval contract rule. The disclosure must list TripoSR, TrOCR, rembg and PrusaSlicer.

## 11. Out of scope

Hidden cavities and undercuts; curved 3D surfaces (a sphere becomes the intersection of three discs); assemblies; CNC or laser output; printer profiles beyond the one default; colour or texture; exporting the TripoSR mesh itself.
