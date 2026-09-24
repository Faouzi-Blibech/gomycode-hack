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
