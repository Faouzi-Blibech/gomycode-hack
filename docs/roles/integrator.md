# Role brief: Integrator and product owner (Faouzi)

Paste this whole file into your AI assistant at the start of a session, together with `CLAUDE.md`.

## You own

- `partspec/`: the Pydantic contracts. You freeze them on day one and guard them.
- `vision/`: the OpenAI-compatible client, prompt templates, validation, retry, abstain.
- `merge.py`: fusing Topology, Annotations and Measurements into a PartSpec with provenance and gates.
- `pipeline.py`: wiring every stage together, with per-module fallback to `s2c/fakes/`.
- `web/`: the mobile web app, built from your Claude design (the geometry owner helps on the Three.js viewer).
- Repo, the 90-second video, pitch deck, project card, tool disclosure, README, the event-day switch to NVIDIA Build.

Moved to the backend and security owner (`docs/roles/backend-security.md`): `api.py`, `app_gradio.py`, the golden-set harness, and all security hardening. They also review your PRs.

## Your contracts with the others

You consume:
- `Topology` from your own `vision/`.
- `Annotations` from `ocr.py` (numbers owner). Until it exists, the pipeline uses `s2c/fakes/ocr.py`.
- `Measurements` from `metrology.py` (numbers owner). Fake in `s2c/fakes/metrology.py`.
- `build(partspec) -> Solid` and `export(solid, out_dir) -> (step_path, stl_path)` from `builder.py` (geometry owner). Fake in `s2c/fakes/builder.py`.
- The HTTP API from `api.py` (backend and security owner). The web app is built against the response shapes in their plan's Task 1.
- `silhouettes(solid) -> dict[view, ndarray]` and `iou(a, b) -> float` from `views.py` (geometry owner).

You produce:
- The contracts everyone imports.
- The `PartSpec` the builder consumes.
- The `Pipeline` the API, the lab UI and the golden harness call.

## Rules you enforce in review

- A model call that returns a millimetre value is rejected in review.
- Any PR that adds a part type or feature is rejected. Point to spec section 3.
- Any PR touching `partspec/` needs all four approvals.
- No AI attribution lines in commits or PRs.

## Order of work

1. `partspec/` models with JSON schema export and tests that reject bad specs (inches, negative values, missing provenance).
2. Fakes for every module so everyone can run the pipeline on day one. Done.
3. `vision/client.py` on a free provider. Done, verified with gemma3:4b on Ollama.
4. `merge.py` for the sketch and photo paths with gate tests. Done.
5. `pipeline.py` wired end to end with the fakes. Done; real modules are picked up as they land.
6. `web/` capture, review with sliders and provenance badges, export, from your Claude design. On the review screen, send back as `user_values` only the numbers the user actually typed. Never echo `abstain.partial` back: that would re-stamp measured and written values as `user_edited`.
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
