# Role brief: Integrator and product owner (Faouzi)

Paste this whole file into your AI assistant at the start of a session, together with `CLAUDE.md`.

## You own

- `partspec/`: the Pydantic contracts. You freeze them on day one and guard them.
- `vision/`: the OpenAI-compatible client, prompt templates, validation, retry, abstain.
- `merge.py`: fusing Topology, Annotations and Measurements into a PartSpec with provenance and gates.
- `api.py`: FastAPI with `/analyze`, `/build`, `/files/{id}`.
- `app_gradio.py`: the lab UI.
- `web/`: the mobile web app (the geometry owner helps on the Three.js viewer).
- Repo, CI, PR reviews, the 90-second video, project card, tool disclosure, README.

## Your contracts with the others

You consume:
- `Topology` from your own `vision/`.
- `Annotations` from `ocr.py` (numbers owner). Until it exists, use `tests/fakes/ocr_fake.py`, which returns hand-written annotations for the golden sketches.
- `Measurements` from `metrology.py` (numbers owner). Fake in `tests/fakes/metrology_fake.py`.
- `build(partspec) -> Solid` and `export(solid) -> (step_path, stl_path)` from `builder.py` (geometry owner). Fake in `tests/fakes/builder_fake.py` returns a unit cube.
- `silhouettes(solid) -> dict[view, ndarray]` and `iou(a, b) -> float` from `views.py` (geometry owner).

You produce:
- The contracts everyone imports.
- The `PartSpec` the builder consumes.

## Rules you enforce in review

- A model call that returns a millimetre value is rejected in review.
- Any PR that adds a part type or feature is rejected. Point to spec section 3.
- Any PR touching `partspec/` needs all three approvals.
- No AI attribution lines in commits or PRs.

## Order of work

1. `partspec/` models with JSON schema export and tests that reject bad specs (inches, negative values, missing provenance).
2. Fakes for every module so all three of you can run the pipeline on day one.
3. `vision/client.py` on a free provider. Prove it returns a valid Topology for three golden sketches.
4. `merge.py` for the sketch path with gate tests.
5. `api.py` and `app_gradio.py` wired end to end with the fakes, then with real modules as they land.
6. `web/` capture, review with sliders and provenance badges, export.
7. Video script by 24 September, video recorded on event day by 15:30.

## Prompt design for the Topology call

- System prompt states the grammar and says: "Never output a length, diameter, thickness or any measurement. Positions are fractions of the bounding box."
- Include the Topology JSON schema verbatim.
- Ask for JSON only, no prose.
- On validation error, resend with "Your previous output failed validation: <error>. Return corrected JSON only."
- Log provider, model, latency, tokens per call to `logs/vlm.jsonl` for the disclosure.

## Definition of done for your parts

- `uv run pytest` green.
- Sketch to STL on the phone in under 20 seconds on the free provider.
- Switching `VLM_BASE_URL` and `VLM_MODEL` to NVIDIA Build needs no code change.
