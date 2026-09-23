# Sketch-to-CAD Studio: UI/UX, parameters and exports — design spec

Date: 2026-09-23
Author: multi-view path owner
Status: design approved under the owner's standing instruction ("pick the recommended option, never stop"); pending the owner's review on waking
Extends: `2026-09-22-multiview-gcode-design.md` and `2026-09-23-qwen-solaria-design.md`
Inputs: three advisor reports (UX, CAD/export, performance), each verified against the installed libraries (gradio 6.28, CadQuery 2.8, trimesh 5.1, ezdxf 1.4.4, PrusaSlicer 2.9.6)

## 1. Goal

A professional, demo-ready app for the multi-view path:

- A guided flow, **Capture → Review → Model & Export**, that a first-time user completes without help.
- Hugging Face–style advanced parameters: seed and AI toggles, geometry finish, mesh quality and 3D-print settings.
- Downloads in every useful format: STL, STEP, 3MF, OBJ, GLB, PLY, BREP, Blender, DXF/SVG/PDF drawings, G-code, and a zip of everything.
- Fast. Nothing the user did not change is recomputed.

Out of scope: the React `web/` app (the integrator's), user accounts, cloud storage, CNC/laser output, and any change to `spec.py`, `build.py` or `partspec/`.

## 2. Decisions

| Topic | Decision | Why |
| --- | --- | --- |
| Surface | Gradio 6.28 "Studio" app, in-process pipeline, new launcher `app_mv_studio.py`; the lab app `app_mv_gradio.py` stays as is | Ships in the time left, Hugging Face look, reuses the pipeline; the existing lab app and its tests keep working |
| Layout | One `gr.Walkthrough` with three `gr.Step`s. No `gr.Sidebar` | Locked steps guide first-time users; a sidebar covers the page on phones; each setting group sits where it takes effect |
| Settings | New pydantic models in `settings.py`, shared by the UI, API and CLI | One source of defaults, ranges and validation |
| Caching | Content-addressed LRU keyed on geometry, mesh quality and print settings | Geometry takes about 0.1 s and slicing 0.5–3 s; format or print changes never rebuild geometry |
| Mesh presets | Absolute tolerance (`relative=False`) on a `shape.copy()`: draft 0.1 mm / 0.5 rad, normal 0.02 / 0.2, fine 0.005 / 0.1 | Measured volume error 0.013 / 0.002 / 0.000 %; the copy stops OCCT reusing a finer mesh |
| Blender | Real `.blend` when `BLENDER_PATH` (a blender executable) or `BLENDER_PYTHON` (a Python with bpy 4.2) is set; otherwise a "Blender kit" of OBJ plus an `open_in_blender.py` script | bpy is about 300 MB, Python 3.11 only and may clash on numpy, so it never runs inside the app process |
| Drawings | OCCT hidden-line projection of the final solid into a third-angle, three-view DXF with dimensions; SVG and PDF rendered from the DXF | Shows features and finishes, which the spec outlines do not |
| Print scale | Applies to the print STL and G-code only; CAD exports stay true size | The CAD file must stay the measured part |
| Fillet "vertical" edges | Unchanged: edges along the depth axis (Z), as the multi-view spec §5 defines them. The UI names them "outline corners" | The CAD advisor read Y-up as vertical; the spec's grammar is extrusion-based |

## 3. Modules

| File | Responsibility |
| --- | --- |
| `s2c/multiview/settings.py` | `AiSettings`, `GeometrySettings`, `MeshSettings`, `PrintSettings`, `ExportSettings`, `StudioSettings`; material presets; the format registry (names, labels, extensions) |
| `s2c/multiview/finish.py` | Applies `GeometrySettings` to a `MultiViewSpec`: fillet or chamfer finish, clamped radius, hole-clearance class |
| `s2c/multiview/exporters.py` | STL, STEP, 3MF, OBJ, GLB, PLY, BREP writers with the quality presets; reload validation |
| `s2c/multiview/drawing.py` | Three-view hidden-line drawing: DXF with dimensions, SVG, PDF |
| `s2c/multiview/blend.py` | Blender `.blend` through a subprocess, or the Blender kit |
| `s2c/multiview/print_settings.py` | `PrintSettings` → PrusaSlicer command-line flags, from a whitelist |
| `s2c/multiview/slice.py` | Gains `overrides` and `scale`, UTF-8 decoding, killing the process tree on timeout, `--threads` and a private `--datadir` |
| `s2c/multiview/artifacts.py` | Geometry key, the LRU of built parts, the preview GLB, exports on demand, print on demand, the zip with a manifest, the one-hour sweep |
| `s2c/multiview/pipeline.py`, `complete.py`, `qwen_faces.py` | Seed, attempt count and per-request AI toggles |
| `s2c/multiview/fuse.py` | The snap table takes a clearance class |
| `s2c/studio/` (`theme.py`, `session.py`, `status.py`, `handlers.py`, `app.py`) | The Studio app |
| `app_mv_studio.py` | Launcher |
| `scripts/make_examples.py`, `examples/mv/sketches/` | Generated demo sketches for "Try an example" |
| `s2c/multiview/routes.py`, `scripts/mv_export.py` | API and CLI parity: formats, quality and print settings |

## 4. Settings

All ranges are validated by pydantic. Every model has defaults, so `StudioSettings()` is valid.

**AiSettings** (take effect at Analyze, or at Redraw)

| Field | Type, default, range | Help |
| --- | --- | --- |
| `use_reader` | bool, True | Qwen-VL reads your handwriting. Off: TrOCR only |
| `use_qwen_image` | bool, True | Qwen-Image draws faces you did not photograph |
| `use_rescue` | bool, True | Qwen-Image redraws a sketch whose outline is open |
| `use_triposr` | bool, True | TripoSR as the fallback for missing faces |
| `use_solaria` | bool, True | Solaria depth for holes in photos; adds 60–180 s |
| `seed` | int, 7, 0..2³¹−1 | Same seed, same drawing |
| `randomize_seed` | bool, False | A new seed each time; the seed used is shown |
| `attempts` | int, 2, 1..4 | Drawings tried per face before falling back |

**GeometrySettings** (take effect at Build, about 0.1 s)

| Field | Default, range | Help |
| --- | --- | --- |
| `snap` | True | Snap amber values to standard sizes; your numbers are never snapped |
| `clearance` | `medium` of fine / medium / coarse | ISO 273 clearance class for holes that are not written |
| `finish` | `none` of none / fillet / chamfer | Edge finish |
| `finish_mm` | 1.0, 0.2..10 | Finish size, clamped to 0.45 × the smallest envelope side |
| `finish_edges` | `all_vertical` of all_vertical / top / bottom / all | Outline corners, front-face edges, back-face edges, all edges |

**MeshSettings**: `quality` is `normal` of draft / normal / fine (section 2 numbers).

**PrintSettings** (take effect at Export when G-code is selected)

| Field | Default, range | PrusaSlicer flag |
| --- | --- | --- |
| `material` | PLA of PLA / PETG / ABS / ASA / TPU | `--filament-type`, `--temperature`, `--first-layer-temperature`, `--bed-temperature`, `--first-layer-bed-temperature` from the preset (PLA 210/60, PETG 240/80, ABS 250/100, ASA 255/100, TPU 225/50 °C) |
| `nozzle_mm` | 0.4 of 0.2 / 0.4 / 0.6 / 0.8 | `--nozzle-diameter` |
| `layer_mm` | 0.2, 0.05..0.32, at most 0.75 × nozzle | `--layer-height`, `--first-layer-height` (the larger of the layer height and 0.2) |
| `infill_pct` | 20, 0..100 | `--fill-density N%` |
| `infill_pattern` | grid of grid / gyroid / rectilinear / honeycomb / cubic / lightning | `--fill-pattern` |
| `perimeters` | 3, 1..8 | `--perimeters` |
| `supports` | buildplate of off / buildplate / everywhere | `--no-support-material`, or `--support-material --support-material-auto` plus `--support-material-buildplate-only` |
| `brim_mm` | 0, 0..10 | `--brim-width` |
| `scale_pct` | 100, 50..200 | Scales the print solid before orientation; the bed check runs after scaling |

**ExportSettings**: `formats` is a set from the registry: `stl`, `step`, `3mf`, `obj`, `glb`, `ply`, `brep`, `blend`, `dxf`, `svg`, `pdf`, `gcode`. Default {stl, step, 3mf, gcode}.

## 5. Build and cache (`artifacts.py`)

- `geometry_key(spec, geometry)` is the sha256 of the canonical JSON of envelope, views, features and finishes after `finish.apply`, plus the geometry settings. Provenance, warnings and confidence are left out, so a changed warning never invalidates the cache.
- `build_part(spec, geometry) -> Part | MvAbstain`. `Part` holds key, solid, volume in mm³, bbox, the six view masks and the preview GLB (normal quality, part colour). It lives in a module LRU (`maxsize=16`, thread-safe) and on disk under `tmp/mv_gradio/<key>/`.
- `export_files(part, formats, mesh, print_settings) -> ExportResult`: files, sizes, print stats and warnings. Every file is written once per key (mesh quality and print settings are part of the file key) and reused. Folder mtimes are touched on each hit.
- `bundle(result) -> Path`: a zip of the files plus `manifest.json` (spec, settings, provenance, the tool and model list, print stats). Written to a temporary name, then `os.replace`d.
- `sweep(root, ttl=3600)` deletes stale folders.
- A `too_big_for_bed` or `slicer_failed` result drops only the G-code, with a warning; the other files are still returned.

## 6. Exports (`exporters.py`, `drawing.py`, `blend.py`)

| Format | How | Frame and units | Check (tests) |
| --- | --- | --- | --- |
| STL | CadQuery `exportStl(relative=False, ascii=False)` on a copy, preset tolerances | Design frame, mm | trimesh: watertight, volume within 0.5 % |
| STEP | `cq.exporters.export`, AP214, under a process-wide lock | Design frame, mm | `importStep` volume within 0.1 % |
| 3MF | CadQuery `THREEMF` with preset tolerances | mm | zip holds `3D/3dmodel.model` |
| OBJ | trimesh from the preset tessellation | Design frame (Y-up), mm | reload: watertight |
| GLB | trimesh, scaled to metres, part colour | glTF Y-up, m | extents equal envelope / 1000 |
| PLY | trimesh, binary | mm | reload: volume |
| BREP | `cq.exporters.export(.., "BREP")` | mm | `importBrep` volume |
| Blender | `.blend` via subprocess (§2), else a kit zip of `part.obj` and `open_in_blender.py` | mm scene units | `.blend` starts with `BLENDER`; the kit holds both files |
| DXF | Hidden-line removal on three views, third-angle layout; overall dimensions, hole diameters, title block | `$INSUNITS=4` (mm) | `doc.audit()` clean; three view groups |
| SVG / PDF | ezdxf `SVGBackend` / `MatplotlibBackend`, A4 landscape | | non-empty; the PDF starts `%PDF` |
| G-code | `slice.py` with overrides | print frame | the stats parse |

The ezdxf "EZDXF" dimension style needs `dimlfac=1`. Shapes go into DXF wrapped in a Workplane.

## 7. AI toggles and seed

- `MvPipeline.configured(ai: AiSettings) -> MvPipeline` returns a shallow copy with the switched-off providers set to None. The shared pipeline is never mutated.
- `qwen_faces.qwen_face(..., seed, attempts)` and `complete(..., seed=, attempts=)` replace the fixed `SEED` and `TRIES`. The cache stays keyed by (face, seed), so a new seed makes new calls only for missing faces.
- `randomize_seed` draws a seed at Analyze and at Redraw; the seed used is written back to the Seed field.
- "Redraw AI faces" adds `attempts` to the seed and runs `fuse` again.

## 8. The Studio UI

**Shell.** `gr.Blocks(title="Sketch-to-CAD Studio", delete_cache=(3600, 3600))`. The theme, CSS and fonts go to `launch()`: `gr.themes.Base(primary_hue="indigo", secondary_hue="amber", neutral_hue="slate")` with flat blocks, a 1 px border and large radius. `launch(max_file_size="20mb")`, `queue(max_size=20, default_concurrency_limit=1)`.

**Header** (`gr.HTML`): product name, one line of purpose, and status chips (● ready / ○ off) for Qwen-VL, Qwen-Image, TripoSR, Solaria, PrusaSlicer and Blender, computed once at start, plus "images deleted after 1 h".

**Step 1, Capture.**
- Left (`scale=3`): a drop zone (`gr.File(file_count="multiple", file_types=["image"])`) that appends to an items state and clears itself; a `@gr.render` card per image: thumbnail, face `gr.Dropdown` (auto + six faces), type `gr.Radio` (auto/sketch/photo/drawing), Remove. Stable `key=`s; face and type are kept in a separate tags state so choosing does not redraw.
- Right (`scale=2`): a coverage strip (six chips: front ✓2, top ✓1, right "AI will draw it"); reference object dropdown; the "shoot top-down, part flat" note; "Try an example" (loads the generated demo sketches); the **Reading & AI** accordion; Analyze (primary) and Cancel.
- Analyze is a generator with `gr.Progress` and stage text. It runs with `concurrency_id="models"` and `concurrency_limit=2`, and ends by unlocking and selecting Step 2.

**Step 2, Review.**
- A status card: green "ready to build", or the red stop card (stage, reason, remedy). `missing_x` names the Width box.
- Left: three size boxes (`gr.Textbox`, since an empty `gr.Number` shows 0), with the suggestion as a placeholder, where the value came from in `info`, and a red "Required" style while empty; "Use suggested sizes"; the values table (`gr.Dataframe(type="array", datatype=["str","number","html"], static_columns=[0,2])`) with an HTML chip per source (green trusted, amber "check", purple AI); a closed "Handwriting read" accordion.
- Right: three face cards (front, top, right): silhouette, a badge (observed, merged ×N with agreement, mirrored, Qwen-Image, TripoSR, assumed), a reject checkbox for AI-drawn faces, and "Redraw AI faces"; the warnings list grouped as "Check" and "Info"; "Build part →" (primary).

**Step 3, Model & Export.**
- Left (`scale=3`): `gr.Model3D(height=520)` showing the preview GLB, set in a `.then()` after the step switch; six rendered views with IoU in the captions.
- Right (`scale=2`): stat cards (size X × Y × Z mm, volume cm³, and after Export print time, filament g and m); the **Geometry** accordion (open; changing it rebuilds with `trigger_mode="always_last"`); **Mesh & export**; **3D print**; the format `gr.CheckboxGroup`; "Export selected" (primary), then a `gr.DownloadButton` for the zip and a `gr.File` list of the files with their sizes. If amber values were never touched, the Build button in Review says "Build anyway (N unchecked) →".

**States.** The empty state says what to drop. Loading shows stage text. Errors are cards with a remedy and never a traceback. `gr.Warning` is used only for non-blocking notices.

**Session.** A server-side dict keyed by a UUID holds the `Observed`, spec, settings and last part key. `gr.State` holds only the UUID (`time_to_live=3600`, with a `delete_callback` that drops the entry). No `.change` handler is attached to a State.

**Accessibility.** Every input has a label. Chip text says the meaning, so colour is never the only signal. Chip colours meet 4.5:1 contrast. Columns wrap at `min_width=320`.

## 9. API and CLI parity

- `POST /mv/export`: body `{spec, settings}`. Returns file URLs per format, the zip URL, print stats and warnings. `/mv/build` is unchanged.
- `scripts/mv_export.py` (new; a spec JSON in, files and a zip out): `--format` (repeatable), `--quality`, `--material`, `--layer`, `--infill`, `--supports`, `--scale`, `--out`. `scripts/mv.py` (images in) is unchanged.

## 10. Errors, logging, privacy

- Subprocesses (slicer, Blender): argument lists, `encoding="utf-8", errors="replace"`, output to a log file, and a timeout (slicer 120 s, Blender 90 s) that kills the process tree (`taskkill /T /F` on Windows).
- Every warning reaches the user. Abstentions stay as the multi-view spec defines them; no new reasons.
- Build files and Gradio uploads are deleted after one hour, as `docs/disclosure.md` states.

## 11. Testing

- Unit tests never touch the network, the real slicer or Blender. Real slicing is marked `slicer`; real Blender is marked `blender` and skipped unless configured.
- One module-scoped fixture builds the example L-bracket once. The export tests reload every file and compare volume and bbox within 0.5 %.
- UI handlers are plain methods tested without a browser; one test builds the Blocks.
- The whole suite stays under 90 s.

## 12. Work split (parallel agents, file ownership)

- **Wave 1:** `settings.py`, plus `pyproject.toml` (trimesh moves to the main dependencies; `blender` marker). One agent.
- **Wave 2**, in parallel with disjoint files, each agent running only its own tests:
  - A: `exporters.py`
  - B: `drawing.py`
  - C: `blend.py`
  - D: `print_settings.py` and `slice.py`
  - E: `finish.py` and the snap table in `fuse.py`
  - F: seed and toggles in `pipeline.py`, `complete.py`, `qwen_faces.py`
- **Wave 3:** `artifacts.py`.
- **Wave 4:** `s2c/studio/`, `app_mv_studio.py`, examples.
- **Wave 5:** API and CLI parity, docs, end-to-end run of the app.
- Agents do not commit. The controller reviews each task, runs the whole suite and commits. Nothing is pushed.
