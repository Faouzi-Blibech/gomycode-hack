"""One OpenAI-compatible client. Provider chosen by env vars only."""
from __future__ import annotations

import base64
import json
import os
import time
from collections.abc import Callable
from pathlib import Path

Chat = Callable[[list[dict]], str]


class VLMClient:
    def __init__(
        self,
        chat: Chat | None = None,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        log_path: str | Path = "logs/vlm.jsonl",
        temperature: float = 0.0,
    ):
        self.model = model or os.environ.get("VLM_MODEL", "unset")
        self.base_url = base_url or os.environ.get("VLM_BASE_URL", "unset")
        self.log_path = Path(log_path)
        self.temperature = temperature
        # Instance attribute, not a class attribute: a dict on the class would be
        # shared by every VLMClient instance, so two clients would overwrite each
        # other's token counts.
        self._last_usage: dict = {}
        self._chat = chat or self._openai_chat(api_key or os.environ.get("VLM_API_KEY", ""))

    @classmethod
    def from_env(cls) -> VLMClient:
        return cls()

    def _openai_chat(self, api_key: str) -> Chat:
        from openai import OpenAI

        client = OpenAI(base_url=self.base_url, api_key=api_key or "missing")

        def chat(messages: list[dict]) -> str:
            resp = client.chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature, max_tokens=800,
            )
            usage = getattr(resp, "usage", None)
            self._last_usage = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
            }
            return resp.choices[0].message.content or ""

        return chat

    def complete_json(self, system: str, user: str, image_bytes: bytes, mime: str = "image/jpeg") -> str:
        data_url = f"data:{mime};base64," + base64.b64encode(image_bytes).decode()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": user},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
        ]
        t0 = time.perf_counter()
        text = self._chat(messages)
        self._log({
            "ts": time.time(), "provider": self.base_url, "model": self.model,
            "latency_ms": round((time.perf_counter() - t0) * 1000),
            "chars_out": len(text), **self._last_usage,
        })
        return text

    def _log(self, record: dict) -> None:
        # The log is a disclosure artifact, not part of the call's contract: a
        # filesystem failure here must never break the caller's request.
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError:
            pass
