# Tool and model disclosure

| Category | Used | Notes |
| --- | --- | --- |
| Vision model (event) | pending, NVIDIA Build | topology only, never dimensions; model name unknown until the event-day workshop issues vouchers |
| Vision model (pre-event testing) | Ollama local, `gemma3:4b` | verified working against the project's own committed code: returned a valid topology for a synthetic hand sketch in about 18 seconds, correctly identifying a plate with two holes and three written dimensions, and returning no measurement of any kind. See `docs/models.md` for the broken alternative we tried first. |
| Vision model (multi-view) | Qwen-VL on DashScope | face labels and written dimensions |
| Image completion (multi-view) | Qwen-Image-2.1 | draws missing silhouettes |
| Depth estimation (multi-view) | Marigold V2 via Solaria 1.0 | depth map of photo to detect through vs blind holes |
| 3D fallback (multi-view) | TripoSR | fallback face prediction |
| OCR | pending / TrOCR / Qwen-VL | dimension reading and linking |
| Classical CV | OpenCV | coin scale, contours, silhouettes |
| CAD kernel | CadQuery (OpenCascade) | deterministic geometry |
| Coding assistants | pending | list what the team used |
| Datasets | our own golden set, no external data | golden set not yet built as of this writing |
| Generated assets | none | |

## Models called at run time

| Model | Provider | What it does here | What it never does |
| --- | --- | --- | --- |
| Qwen-VL | Alibaba Cloud Model Studio (DashScope) | Says which face a photo shows; copies the handwritten numbers | Estimate a size; write code |
| Qwen-Image-2.1 | Hugging Face Space `Qwen/Qwen-Image-2.1` or DashScope | Draws the silhouette of a face nobody photographed; cleans up a sketch with an open outline | Produce a file we export; set a dimension |
| Marigold V2 via Solaria 1.0 | Hugging Face Space `CronosSa/Solaria1.0` | Depth map of a photo, to tell through from blind holes | Set a dimension without an amber "estimated" badge |
| TripoSR | Local GPU or Hugging Face Space | Fallback face prediction | Produce a file we export |
| TrOCR | Local | Fallback handwriting reader | |

Every call is logged to `logs/vlm.jsonl` with provider, model, stage and latency.

## Numbers behind the claims above

- Latency: one measured vision-model call against `gemma3:4b` took about 17.8 seconds (17785 ms), logged in `logs/vlm.jsonl`. This is a single sample, not a median across a golden set, and it is model latency only, not full sketch-to-STL pipeline latency.
- Tokens: that same call used about 1599 prompt tokens and 172 completion tokens.
- The vision model returned zero measurements in that call, consistent with rule 2 (numbers are measured or user-written, never model-estimated).

## Data and privacy

- Images are processed in memory. One silhouette PNG and the exported STEP/STL files live in a temp folder for one hour, then they are deleted.
- The per-call log confirmed to back this up (`logs/vlm.jsonl`) contains only: timestamp, provider, model, latency, output length, prompt and completion token counts, and a status field. It contains no API key, no prompt text and no image content.
- For multi-view hosted models, images are sent to DashScope and to Hugging Face Spaces for the calls above. The app displays an upload notice. Use DashScope directly (`QWEN_IMAGE_BACKEND=dashscope`) for anything confidential.
- API: images live only for the request; silhouettes and exported files stay in `tmp/mv/` for one hour.
- Gradio lab app: uploads stay in Gradio's cache and built files in `tmp/mv_gradio/`, both deleted after one hour. While a browser session is open, its photos stay in that session's memory.
- Studio: builds also live under `tmp/mv_gradio/` for one hour after their last use. Each export's zip carries a `manifest.json` listing the part's values, their sources, and the settings used to build and export it.
- No model output is executed. Geometry is built by our own CadQuery code from validated JSON.

## Licences

- Qwen-Image-2.1: Qwen Research License. Confirmed allowed for this hackathon demo on: pending.
- TripoSR: MIT. PrusaSlicer: AGPL, run as a separate program. rembg: MIT.

## Pre-event work

The pipeline was scaffolded and tested in the week before the event. Event day is integration against the sponsor's model (after the roster and voucher-activation workshop), polish, and the demo. See the day plan in `docs/superpowers/specs/2026-09-19-sketch-to-cad-design.md` section 6.
