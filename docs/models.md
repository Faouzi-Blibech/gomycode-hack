# Model providers

One OpenAI-compatible client, configured by environment variables. Fill in the model name from the provider catalogue on the day you use it and record the date.

| Provider | `VLM_BASE_URL` | Suggested `VLM_MODEL` | Key from | Checked |
| --- | --- | --- | --- | --- |
| NVIDIA Build (event day) | `https://integrate.api.nvidia.com/v1` | pick a vision model from build.nvidia.com, for example a Llama 3.2 Vision or Nemotron VL entry | given at the event | pending, model name genuinely unknown until the event-day workshop |
| Google Gemini free tier | `https://generativelanguage.googleapis.com/v1beta/openai/` | a current Gemini Flash model | aistudio.google.com | pending |
| Groq free tier | `https://api.groq.com/openai/v1` | a current Llama 4 Scout vision entry | console.groq.com | pending |
| Ollama local | `http://localhost:11434/v1` | `gemma3:4b` | none, set `VLM_API_KEY=ollama` | verified working, see below |

### Ollama local: verified working provider

`http://localhost:11434/v1`, model `gemma3:4b`, `VLM_API_KEY=ollama`. Run against a synthetic hand sketch through the project's own committed code, it returned a valid topology in about 18 seconds (17785 ms, logged in `logs/vlm.jsonl`), correctly identifying a plate with two holes and three written dimensions, and returning no measurement of any kind. About 1599 prompt tokens and 172 completion tokens for that call. This is the fallback provider for pre-event testing and demo rehearsal.

### Ollama local: `qwen2.5vl:7b` is broken on this machine — do not use it

This model was tried first and emits incoherent output on both text and image prompts, on this same Ollama server, while a text-only model on that server works correctly. Do not spend time re-diagnosing this; use `gemma3:4b` instead. If a fresh machine or an updated model tag fixes it, update this note and re-verify before relying on it.

Rules:
- Never hard-code a provider or model in source.
- Every call is logged to `logs/vlm.jsonl` with provider, model, latency in ms, prompt and completion tokens.
- On event day, run `uv run pytest tests/test_golden.py` on the NVIDIA model before switching the demo to it. If the pass rate drops below the fallback provider, keep the fallback and disclose both.

## Multi-view path

| Model | Source | Use | Runs | Checked |
| --- | --- | --- | --- | --- |
| Qwen-VL | Alibaba Cloud Model Studio (DashScope), OpenAI-compatible | Face labels; transcribes handwritten values, one call per image | `VLM_BASE_URL`, `VLM_MODEL` | pending |
| Qwen-Image-2.1 | `Qwen/Qwen-Image-2.1` (Qwen Research License) | Draws faces nobody photographed; redraws sketches whose outline is open. Only read back as outlines | `QWEN_IMAGE_SPACE`, or DashScope with `QWEN_IMAGE_BACKEND=dashscope` | pending |
| Solaria 1.0 (Marigold V2 depth) | `CronosSa/Solaria1.0` Space, commit `f531e56` on 2026-09-23 | Depth map of a photo: through or blind holes, blind depth as a ratio | `SOLARIA_SPACE` (ZeroGPU, `HF_TOKEN` for quota) | pending |
| TripoSR | `stabilityai/TripoSR` (MIT) | Fallback when Qwen-Image is unavailable or rejected; only its silhouettes are used | Local CUDA, else `TRIPOSR_SPACE` | |
| Hunyuan3D-2.1 (reference, not wired in) | `tencent/Hunyuan3D-2.1` Space (Tencent Hunyuan Community License; check territory terms) | Candidate replacement for TripoSR: `/shape_generation` takes one image or front, back, left and right views and returns a mesh; only its silhouettes would be used | ZeroGPU Space | |
| TrOCR base handwritten | `microsoft/trocr-base-handwritten` | Fallback reader when Qwen-VL is unavailable | Local, CUDA or CPU | |
| rembg (u2net) | `rembg` | Removes the background before TripoSR | Local CPU | |
| PrusaSlicer | prusa3d.com (AGPL) | Slices the STL to G-code with `profiles/fdm_default.ini` | Local CLI | |
| ezdxf | `ezdxf` (MIT) | Writes the DXF, SVG and PDF drawings | Local | |
| trimesh | `trimesh` (MIT) | Writes the OBJ, GLB and PLY meshes | Local | |
| Blender / bpy 4.2 | blender.org (GPL) | Writes a native `.blend`; optional, runs as its own process | Local, `BLENDER_PATH` or `scripts/setup_blender.ps1` | |

Before the demo: `NETWORK_TESTS=1 uv run pytest tests/test_mv_network.py -v`, then fill in the Checked column with the date.
