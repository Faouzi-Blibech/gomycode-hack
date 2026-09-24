# Sketch-to-CAD

Point your phone at a broken part or a hand-drawn sketch of one. Get an editable, parametric CAD file back, ready to print or machine.

Built for the GOMYCODE "Come Build with AI" hackathon, 27 September 2026, on NVIDIA Build.

## The problem

A bracket snaps, a spacer goes missing, a plastic clip breaks. The replacement does not exist or costs more to ship than to print. Modelling it in CAD takes an hour you do not have, and mesh generators give you a blob you cannot edit or trust.

## What it does

Three inputs, one pipeline, one output.

| Input | How | Where the numbers come from |
| --- | --- | --- |
| Hand sketch | Draw it on paper, write the dimensions in mm, photograph it | Your handwriting, read by OCR |
| Real part | Photograph it top-down next to a coin | Coin scale, measured with OpenCV |
| 2D drawing | A clean orthographic view with dimensions | Printed dimensions, read by OCR |

Output: a STEP file for CAD tools and an STL file for slicers, plus a live 3D preview and sliders that re-derive the geometry when you change a number.

## How it works

```text
image ──┬─ metrology   OpenCV: coin -> mm per pixel, contours in mm
        ├─ ocr         written dimensions -> values linked to edges and holes
        └─ vision      vision model -> topology only: part type, hole count, rough positions
                │
             merge      fuse + confidence gates -> PartSpec, or a clear abstention
                │
        PartSpec        the single source of truth, edited by sliders
                │
             builder    our own deterministic CadQuery code -> STEP + STL
                │
             views      six orthographic silhouettes -> round-trip match against the input
```

Four rules shape every design decision:

1. **The model never writes code.** The vision model returns a small JSON describing topology. A schema validates it before anything is built. There is no code execution path from model output.
2. **Numbers are measured or written, never estimated.** Every millimetre comes from the coin scale or from what the user wrote. The model is not allowed to produce a dimension; if it does, the value is discarded.
3. **Profile plus features.** A part is a 2D outline extruded straight, with holes, slots, fillets and chamfers. That covers plates, brackets, flanges, spacers and free-form outlines, and it is honest about what it cannot do.
4. **Abstention is a feature.** Tilted coin, unsupported shape, missing thickness, poor round-trip match: each one stops with a reason and tells you what to do next.

Supported part types: `plate`, `l_bracket`, `flange`, `spacer`, `profile_extrusion`. Features: `hole`, `slot`, `fillet`, `chamfer`. All dimensions in millimetres.

## Run it

Requirements: Python 3.11, [uv](https://docs.astral.sh/uv/), Node 20.

```bash
cp .env.example .env            # add a vision model key, see docs/models.md
uv sync
uv run pytest                   # everything green before you start

uv run uvicorn s2c.api:app --host 0.0.0.0 --port 8000     # API
uv run python app_gradio.py                                # lab view on :7860

cd web && npm install && npm run dev -- --host             # mobile web app
```

Open the web app on your phone using the LAN URL Vite prints. The API and the app must be reachable from the phone.

The vision model is chosen by three environment variables: `VLM_BASE_URL`, `VLM_MODEL`, `VLM_API_KEY`. Any OpenAI-compatible endpoint works. The table in `docs/models.md` lists NVIDIA Build, Gemini, Groq and Ollama presets.

## Multi-view path (pending team sign-off)

Give one or more images per face, several of the same face if you have them: they are aligned and voted into one cleaner outline. Qwen-VL reads the numbers you wrote. Faces you did not give are drawn by Qwen-Image and kept only if they agree with the faces you did give; otherwise TripoSR, otherwise a rectangle. Solaria's depth map tells through holes from blind ones. The part is the intersection of the three extruded outlines, sliced to G-code. Designs: `docs/superpowers/specs/2026-09-22-multiview-gcode-design.md` and `docs/superpowers/specs/2026-09-23-qwen-solaria-design.md`.

    uv run python app_mv_studio.py                                                           # the Studio on :7860 (guided flow, parameters, every export)
    uv run python app_mv_gradio.py                                                           # the simple lab app
    uv run python scripts/mv_build.py examples/mv/l_bracket.json --out tmp/mv_demo          # spec -> STEP, STL, G-code
    uv run python scripts/mv_export.py examples/mv/l_bracket.json --format stl --format step --format pdf  # spec -> chosen formats + zip
    uv run python scripts/mv.py --image front.jpg@front@sketch --image top.jpg@top@sketch   # images -> the same
    uv run uvicorn s2c.multiview.app:app --port 8001                                        # /mv API
    NETWORK_TESTS=1 uv run pytest tests/test_mv_network.py -v                               # live check of the hosted models

Settings are in `.env.example`: Qwen-VL and Qwen-Image on DashScope or Hugging Face, Solaria on Hugging Face. Photos of real parts: shoot top-down with the part lying flat.

G-code needs PrusaSlicer: `winget install --id Prusa3D.PrusaSlicer -e` (needs admin), or unzip the portable zip from the PrusaSlicer GitHub release into `vendor/`. Without it you still get STL and STEP.

Export formats: STL, STEP, 3MF, OBJ, GLB, PLY, BREP, Blender, DXF/SVG/PDF drawing, G-code, and a zip with a manifest. Blender: set `BLENDER_PATH`, or run `scripts/setup_blender.ps1`; without it the download is a Blender kit.

TripoSR (the local fallback when Qwen-Image cannot complete a face): `powershell scripts/setup_triposr.ps1` installs it; without it, that face falls straight to a rectangle.

## Repository layout

```text
s2c/                Python package
  partspec/         frozen contracts: PartSpec, Topology, Annotations, Measurements, Abstain
  vision/           provider-agnostic client, prompts, topology extraction
  merge.py          fuse stage outputs into a PartSpec, confidence gates
  builder.py        PartSpec -> CadQuery solid -> STEP + STL
  views.py          six silhouettes for the round-trip check
  ocr.py            dimension reading and linking
  metrology.py      coin scale and measured contours
  silhouette.py     input silhouette, mask normalisation, IoU
  api.py            FastAPI surface
  fakes/            stand-ins for every stage so the pipeline runs before a module lands
app_gradio.py       lab UI showing every stage output
web/                React + Three.js mobile web app
tests/              pytest suite and the golden set of ground-truth parts
docs/
  superpowers/specs/   the design spec
  superpowers/plans/   one implementation plan per owner
  roles/               one brief per team member
  models.md            provider presets
  disclosure.md        tools, models and data used
```

## Testing and reliability

- **Golden set.** Ten hand-drawn sketches and five coin photos of parts with known dimensions. A test runs the full pipeline and checks every number within 5 percent or 1 mm.
- **Round trip.** The built solid is re-projected and compared with the input silhouette. Below 0.85 IoU the result is shown in amber with a warning.
- **Builder.** Volume tests for every part type and feature.
- **Abstention tests.** Each gate has a test that produces the right reason.
- **Call log.** Every model call records provider, model, latency and tokens for the disclosure.

Accuracy numbers, updated as tests land:

| Metric | Value |
| --- | --- |
| Golden sketches passing | pending |
| Coin scale error | pending |
| OCR value accuracy | pending |
| Median sketch-to-STL latency | pending |

## Responsible AI and data

- Images are processed in memory. One silhouette PNG and the exported files live in a temp folder for one hour, then they are deleted.
- The user sees and can edit every number before export. Each value carries a badge saying where it came from: measured, written, edited, or a default guess.
- No model output is ever executed.
- Models, providers and tools are listed in `docs/disclosure.md`.

## Team

Three people, three owners. The integrator owns the contracts, model layer, merge, API and UIs. The geometry owner owns the builder, views and the golden parts. The numbers owner owns coin metrology and dimension OCR. Briefs for each role are in `docs/roles/`.

## Status

Design and plans are complete. See `docs/superpowers/plans/` for the task lists and `docs/superpowers/specs/` for the full design.
