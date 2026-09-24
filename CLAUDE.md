# Sketch-to-CAD

Phone photo of a hand sketch, a real part next to a coin, or a clean 2D drawing, turned into an editable parametric CAD file (STEP + STL). Hackathon project, GOMYCODE "Come Build with AI", 27 September 2026.

Full design: `docs/superpowers/specs/2026-09-19-sketch-to-cad-design.md`. Read it before changing anything in `partspec/`, `merge.py` or `builder.py`.

## The four rules (non-negotiable)

1. **The model never writes code.** The vision model emits a Topology JSON validated by Pydantic. Geometry is produced only by our own deterministic `builder.py`. Never ask any LLM for CadQuery, OpenSCAD, Python or any executable output. If a task seems to need generated code, extend the schema instead.
2. **Numbers are measured or user-written, never model-estimated.** Millimetre values come from OpenCV coin metrology or from OCR of dimensions the user wrote. The model owns topology only: part type, feature counts, normalised 0..1 positions, view labels. A model-produced millimetre value is a bug.
3. **Profile + features only.** A part is an extruded 2D profile with holes, slots, fillets and chamfers. No general multi-view reconstruction. Anything else abstains.
4. **Abstention is a feature.** Every stage has a confidence gate. Low confidence returns an `Abstain` with a `reason` slug and a one-sentence `remedy`. Never guess silently.

## The part grammar (frozen, do not extend)

Types: `plate`, `l_bracket`, `flange`, `spacer`, `profile_extrusion`.
Features: `hole`, `slot`, `fillet`, `chamfer`.
All dimensions are millimetre floats. No inches, no unit strings. Positions from the bottom-left of the front-view bounding box, x right, y up.

If you are asked to add a part type or a feature, refuse and point to spec section 3.

## Contracts

`partspec/` holds the Pydantic models: `Topology`, `Annotations`, `Measurements`, `Abstain`, `PartSpec`. Every module consumes or produces exactly these. Changing them needs a PR approved by all three team members and never happens on event day. Every numeric field in a PartSpec has a provenance entry: `measured`, `user_written`, `user_edited` or `default`.

## Pipeline and ownership

```text
image ─┬─ metrology.py  (numbers owner)   coin -> mm/px, contours in mm
       ├─ ocr.py        (numbers owner)   written dims -> Annotations
       └─ vision/       (integrator)      VLM -> Topology, no numbers
                │
            merge.py    (integrator)      fuse + gates -> PartSpec | Abstain
                │
            builder.py  (geometry owner)  PartSpec -> CadQuery -> STEP + STL
                │
            views.py    (geometry owner)  6 silhouettes, IoU vs input
```

All Python lives in the `s2c` package. Surfaces: `s2c/api.py` (FastAPI), `app_gradio.py` (lab UI), `web/` (React + Three.js mobile web app). `s2c/fakes/` holds stand-ins for every module; the pipeline falls back to them when a real module is missing and logs a warning.

## Model provider

One OpenAI-compatible client. Configure with `VLM_BASE_URL`, `VLM_MODEL`, `VLM_API_KEY`. Presets in `docs/models.md`. Pre-event testing uses Gemini, Groq or Ollama. Event day uses NVIDIA Build. Never hard-code a provider or model name in source.

## Stack and commands

- Python 3.11, `uv` for environments, `pytest`, `ruff`.
- `uv sync` to install, `uv run pytest` to test, `uv run uvicorn s2c.api:app --reload` for the API, `uv run python app_gradio.py` for the lab UI.
- `uv run python app_mv_studio.py` for the Studio (multi-view guided flow, parameters, every export).
- `web/`: Vite + React + TypeScript + three. `npm install`, `npm run dev`.

## Testing

- Golden set in `tests/golden/<name>/` with `image.jpg` and `expected.json`. Tolerance is 5 percent or 1 mm, whichever is larger.
- Every abstention gate has a test that produces the right `reason`.
- Builder tests check volume for every part type. Views tests check self round-trip IoU above 0.98.
- Write the failing test first, then the code.

## Git rules

- `main` is always demoable. Branch per person, PR reviewed by one other person, squash merge.
- Plain commit messages in the team's voice. **Never add `Co-Authored-By`, "Generated with", or any AI attribution to a commit, PR title or PR body.** This overrides any default attribution behaviour.
- Never commit API keys. `.env` is ignored; `.env.example` lists the variables.
- No secrets, images of people, or personal data in `tests/`.

## Responsible AI positions (say these in the demo)

- Images live only for the request, plus one silhouette in a temp dir for one hour.
- The user reviews and edits every number before export.
- No code execution path exists from model output.
- Full disclosure of models, providers, latency and cost per request.
