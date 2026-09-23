"""Vision model labels each image: which face, which holes are blind, optional hole-depth guesses. Spec 6.1.
The model returns JSON only. It never sets the envelope and never returns code."""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from s2c.multiview.spec import MvAbstain

log = logging.getLogger(__name__)
Chat = Callable[[list[dict]], str]
_ESTIMATE_KEY = re.compile(r"^holes\[\d+\]\.depth_mm$")


class LabelHole(BaseModel):
    model_config = ConfigDict(extra="forbid")
    u: float = Field(ge=0, le=1)
    v: float = Field(ge=0, le=1)
    blind: bool = False


class MvLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    face: Literal["front", "back", "left", "right", "top", "bottom", "unknown"]
    input_kind: Literal["sketch", "photo", "drawing"]
    holes: list[LabelHole] = []
    description: str = ""
    estimates: dict[str, float] = {}
    confidence: float = Field(ge=0, le=1)


SYSTEM_PROMPT = """You label one image of a mechanical part for a CAD tool. Reply with JSON only, no prose, matching this schema:
{schema}
Rules:
- face: the side of the part the image shows (front, back, left, right, top, bottom), or unknown.
- input_kind: sketch (hand drawn), photo (real part) or drawing (clean printed drawing).
- holes: every round hole, as u, v fractions of the part's bounding box (u to the right, v upward); blind is true if it does not go through.
- estimates: optional hole depth guesses in millimetres, only with keys like "holes[0].depth_mm".
- Never estimate the overall width, height or depth. Never output code."""


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        t = t.rsplit("```", 1)[0]
    return t.strip()


def label_image(image_bytes: bytes, chat: Chat, face_hint: str | None = None,
                kind_hint: str | None = None) -> MvLabel | MvAbstain:
    b64 = base64.b64encode(image_bytes).decode()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(schema=json.dumps(MvLabel.model_json_schema()))},
        {"role": "user", "content": [
            {"type": "text", "text": "Label this image."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]},
    ]
    label = None
    for _ in range(2):
        raw = chat(messages)
        try:
            label = MvLabel.model_validate_json(_strip_fences(raw))
            break
        except (ValidationError, ValueError) as e:
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": f"Your previous output failed validation: {e}. "
                                                     "Return corrected JSON only."}]
    if label is None:
        return MvAbstain(stage="label", reason="label_invalid",
                         remedy="The model could not describe this photo. Try a cleaner photo.")
    dropped = [k for k in label.estimates if not _ESTIMATE_KEY.match(k)]
    if dropped:
        log.warning("discarded model estimates %s: only hole depths may be estimated", dropped)
    label = label.model_copy(update={
        "estimates": {k: v for k, v in label.estimates.items() if _ESTIMATE_KEY.match(k) and v > 0},
        "face": face_hint or label.face, "input_kind": kind_hint or label.input_kind})
    if label.face == "unknown":
        return MvAbstain(stage="label", reason="face_unknown", remedy="Tell us which face this photo shows.")
    return label


def hint_label(face: str, kind: str = "sketch") -> MvLabel:
    """A label from the user's tags alone, when no vision model is configured."""
    return MvLabel(face=face, input_kind=kind, confidence=0.9)


def env_chat(log_path: str | Path = "logs/vlm.jsonl", stage: str = "mv_label") -> Chat | None:
    """OpenAI-compatible chat from VLM_BASE_URL, VLM_MODEL, VLM_API_KEY; None when not configured.
    Swap for the integrator's VLMClient when s2c/vision/ lands."""
    base, model, key = (os.environ.get(k) for k in ("VLM_BASE_URL", "VLM_MODEL", "VLM_API_KEY"))
    if not (base and model and key):
        return None
    from openai import OpenAI
    client, path = OpenAI(base_url=base, api_key=key), Path(log_path)

    def chat(messages: list[dict]) -> str:
        t0 = time.perf_counter()
        r = client.chat.completions.create(model=model, messages=messages, temperature=0)
        usage = r.usage
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"provider": base, "model": model, "stage": stage,
                                "latency_ms": round((time.perf_counter() - t0) * 1000),
                                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                                "completion_tokens": getattr(usage, "completion_tokens", None)}) + "\n")
        return r.choices[0].message.content or ""

    return chat
