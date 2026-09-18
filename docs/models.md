# Model providers

One OpenAI-compatible client, configured by environment variables. Fill in the model name from the provider catalogue on the day you use it and record the date.

| Provider | `VLM_BASE_URL` | Suggested `VLM_MODEL` | Key from | Checked |
| --- | --- | --- | --- | --- |
| NVIDIA Build (event day) | `https://integrate.api.nvidia.com/v1` | pick a vision model from build.nvidia.com, for example a Llama 3.2 Vision or Nemotron VL entry | given at the event | pending |
| Google Gemini free tier | `https://generativelanguage.googleapis.com/v1beta/openai/` | a current Gemini Flash model | aistudio.google.com | pending |
| Groq free tier | `https://api.groq.com/openai/v1` | a current Llama 4 Scout vision entry | console.groq.com | pending |
| Ollama local | `http://localhost:11434/v1` | `qwen2.5vl:7b` or `llama3.2-vision` | none, set `VLM_API_KEY=ollama` | pending |

Rules:
- Never hard-code a provider or model in source.
- Every call is logged to `logs/vlm.jsonl` with provider, model, latency in ms, prompt and completion tokens.
- On event day, run `uv run pytest tests/test_golden.py` on the NVIDIA model before switching the demo to it. If the pass rate drops below the fallback provider, keep the fallback and disclose both.
