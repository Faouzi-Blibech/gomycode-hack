# Tool disclosure

| Category | Used | Notes |
| --- | --- | --- |
| Vision model (event) | pending, NVIDIA Build | topology only, never dimensions; model name unknown until the event-day workshop issues vouchers |
| Vision model (pre-event testing) | Ollama local, `gemma3:4b` | verified working against the project's own committed code: returned a valid topology for a synthetic hand sketch in about 18 seconds, correctly identifying a plate with two holes and three written dimensions, and returning no measurement of any kind. See `docs/models.md` for the broken alternative we tried first. |
| OCR | pending | trained model or VLM-assisted, see numbers plan |
| Classical CV | OpenCV | coin scale, contours, silhouettes |
| CAD kernel | CadQuery (OpenCascade) | deterministic geometry |
| Coding assistants | pending | list what the team used |
| Datasets | our own golden set, no external data | golden set not yet built as of this writing |
| Generated assets | none | |

## Numbers behind the claims above

- Latency: one measured vision-model call against `gemma3:4b` took about 17.8 seconds (17785 ms), logged in `logs/vlm.jsonl`. This is a single sample, not a median across a golden set, and it is model latency only, not full sketch-to-STL pipeline latency.
- Tokens: that same call used about 1599 prompt tokens and 172 completion tokens.
- The vision model returned zero measurements in that call, consistent with rule 2 (numbers are measured or user-written, never model-estimated).

## Privacy

Images are processed in memory. One silhouette PNG and the exported STEP/STL files live in a temp folder for one hour, then they are deleted. The per-call log confirmed to back this up (`logs/vlm.jsonl`) contains only: timestamp, provider, model, latency, output length, prompt and completion token counts, and a status field. It contains no API key, no prompt text and no image content.

## Pre-event work

The pipeline was scaffolded and tested in the week before the event. Event day is integration against the sponsor's model (after the roster and voucher-activation workshop), polish, and the demo. See the day plan in `docs/superpowers/specs/2026-09-19-sketch-to-cad-design.md` section 6.
