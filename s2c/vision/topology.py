from __future__ import annotations

import re

from pydantic import ValidationError

from s2c.partspec.models import Abstain, SourceInput, Topology
from s2c.vision.client import VLMClient
from s2c.vision.prompts import RETRY_SUFFIX, SYSTEM, USER_BY_KIND

_FENCE = re.compile(r"`{3}(?:json)?\s*(.*?)`{3}", re.DOTALL)  # code fences some models wrap JSON in


def extract_json(text: str) -> str:
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    start, end = text.find("{"), text.rfind("}")
    return text[start: end + 1] if start != -1 and end != -1 else text.strip()


def topology_from_image(image_bytes: bytes, client: VLMClient, input_kind: SourceInput) -> Topology | Abstain:
    user = USER_BY_KIND[input_kind]
    raw = client.complete_json(SYSTEM, user, image_bytes)
    try:
        return Topology.model_validate_json(extract_json(raw))
    except (ValidationError, ValueError) as first:
        raw2 = client.complete_json(SYSTEM, user + RETRY_SUFFIX.format(error=str(first)[:600]), image_bytes)
        try:
            return Topology.model_validate_json(extract_json(raw2))
        except (ValidationError, ValueError):
            return Abstain(
                stage="vision", reason="schema_failed",
                remedy="The model could not describe this part. Try a cleaner, better-lit photo.",
            )
