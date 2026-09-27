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
| TrOCR base handwritten | `microsoft/trocr-base-handwritten` | Parallel co-reader whose agreement with Qwen-VL confirms a value | Local, CUDA or CPU | |
| rembg (u2net) | `rembg` | Removes the background before TripoSR | Local CPU | |
| PrusaSlicer | prusa3d.com (AGPL) | Slices the STL to G-code with `profiles/fdm_default.ini` | Local CLI | |
| ezdxf | `ezdxf` (MIT) | Writes the DXF, SVG and PDF drawings | Local | |
| trimesh | `trimesh` (MIT) | Writes the OBJ, GLB and PLY meshes | Local | |
| Blender / bpy 4.2 | blender.org (GPL) | Writes a native `.blend`; optional, runs as its own process | Local, `BLENDER_PATH` or `scripts/setup_blender.ps1` | |

Before the demo: `NETWORK_TESTS=1 uv run pytest tests/test_mv_network.py -v`, then fill in the Checked column with the date.

## Reading handwriting

The Studio reads written dimensions with two readers at once, through `s2c/reading/`:

- **Qwen-VL** (the `VLM_*` provider): reads every crop of an image in one call and keeps the ⌀ and R signs. Its confidence is a constant, so a Qwen read alone is never trusted.
- **TrOCR** (`microsoft/trocr-base-handwritten`, local, needs `uv sync --extra ai`): reads every crop in one batch and gives a real confidence. Reads below 0.7 are ignored.

A value is trusted (green, `user_written`) when both readers give the same number. When they disagree, or only Qwen answered, the Studio asks the user to confirm the size (pre-filled) or shows the hole diameter amber. With only TrOCR configured, its confident reads are trusted, as before. When Qwen is configured but fails or times out, TrOCR is again the only reader that answered: its reads are unconfirmed too, so every size becomes a suggestion during a VLM outage rather than staying silently trusted.

Both readers run in parallel, so a request waits for the slower one, not the sum. TrOCR loads in the background when the Studio starts, and Qwen is warmed right after with a tiny text-only request (same background thread, TrOCR first so Ollama sees TrOCR's GPU share already claimed when it decides how many layers to keep on the GPU), so the first real read does not pay Ollama's ~60 s cold-load cost. For Ollama, also set `OLLAMA_KEEP_ALIVE=30m` (or `-1`) in the environment of the Ollama server itself, so it does not unload the model after 5 idle minutes. Settings:

| Variable | Default | Meaning |
| --- | --- | --- |
| `READ_TIMEOUT_S` | 20 | time budget of the Qwen-VL read, in seconds |
| `TROCR_MODEL` | `microsoft/trocr-base-handwritten` | the local handwriting model |
| `GPU_BUDGET_TROCR_GB` | 1.0 | TrOCR's share of the GPU (`torch.cuda.set_per_process_memory_fraction`); an out-of-memory load or read moves TrOCR to CPU for the rest of the process instead of crashing the request |
| `READING_LOG` | `logs/reading.jsonl` | one line per reader call: reader, crops, cached, status, latency; no image, no text |

GPU budget on the demo laptop (RTX 4060, 8 GB): TrOCR is capped at `GPU_BUDGET_TROCR_GB` (1 GB by default), and Ollama's own server keeps `OLLAMA_GPU_OVERHEAD` bytes free (1 GiB = 1073741824 on that laptop), so the two stay at roughly 7 GB of 8 GB combined; any Qwen layers that do not fit run from system RAM on the CPU instead. `OLLAMA_GPU_OVERHEAD` is read by the Ollama server, so set it in its own environment, not this app's.

Measured latency: run `uv run python scripts/reading_latency.py <sketch image>` and paste the table here with the date and the machine.

Measured 2026-09-27 on the team laptop (CPU only, torch 2.14.0+cpu, transformers 5.17.0), TrOCR alone, 3 runs each; Qwen-VL not measured (no `VLM_*` key configured on that machine):

| sketch | TrOCR load | read, median | min | max | crops |
| --- | --- | --- | --- | --- | --- |
| `examples/mv/sketches/front.png` | 5.0 s | 2376 ms | 2325 ms | 2391 ms | 2 |
| `examples/mv/sketches/top.png` | 5.5 s | 1413 ms | 1304 ms | 1455 ms | 1 |

On CPU a read costs roughly 1-1.2 s per crop even in one batch, so a sheet with 16 crops can take well over 10 s.

Measured 2026-09-27 on the team laptop (RTX 4060 Laptop GPU, torch 2.14.0+cu126, transformers 5.17.0), TrOCR alone, 5 runs each:

| sketch | TrOCR load (first time) | read, median | min | max | crops |
| --- | --- | --- | --- | --- | --- |
| `examples/mv/sketches/front.png` | 20.1 s | 101 ms | 100 ms | 479 ms | 2 |
| `examples/mv/sketches/top.png` | 5.6 s | 63 ms | 60 ms | 337 ms | 1 |

The GPU cuts the read itself to roughly 1/23rd of the CPU time (2376 ms -> 101 ms on front.png).
