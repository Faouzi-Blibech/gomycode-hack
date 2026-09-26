# Tools, models and data

## Models called at run time

| Model | Provider | What it does here | What it never does |
| --- | --- | --- | --- |
| Qwen-VL | Alibaba Cloud Model Studio (DashScope) | Says which face a photo shows; copies the handwritten numbers | Estimate a size; write code |
| Qwen-Image-2.1 | Hugging Face Space `Qwen/Qwen-Image-2.1` or DashScope | Draws the silhouette of a face nobody photographed; cleans up a sketch with an open outline | Produce a file we export; set a dimension |
| Marigold V2 via Solaria 1.0 | Hugging Face Space `CronosSa/Solaria1.0` | Depth map of a photo, to tell through from blind holes | Set a dimension without an amber "estimated" badge |
| TripoSR | Local GPU or Hugging Face Space | Fallback face prediction | Produce a file we export |
| TrOCR | Local | Fallback handwriting reader | |

Every call is logged to `logs/vlm.jsonl` with provider, model, stage and latency.

## Data

- Images are sent to DashScope and to Hugging Face Spaces for the calls above. The app says so next to the upload. Those services keep their own logs: the public Qwen-Image Space, for one, saves the images it receives. Use DashScope directly (`QWEN_IMAGE_BACKEND=dashscope`) for anything confidential.
- API: images live only for the request; silhouettes and exported files stay in `tmp/mv/` for one hour.
- Gradio lab app: uploads stay in Gradio's cache and built files in `tmp/mv_gradio/`, both deleted after one hour. While a browser session is open, its photos stay in that session's memory.
- Studio: builds also live under `tmp/mv_gradio/` for one hour after their last use. Each export's zip carries a `manifest.json` listing the part's values, their sources, and the settings used to build and export it.
- No model output is executed. Geometry is built by our own CadQuery code from validated JSON.

## Licences

- Qwen-Image-2.1: Qwen Research License. Confirmed allowed for this hackathon demo on: pending.
- TripoSR: MIT. PrusaSlicer: AGPL, run as a separate program. rembg: MIT.
