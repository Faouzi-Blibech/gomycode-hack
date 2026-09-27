"""What every handwriting reader shares. Readers only transcribe; they never estimate a size (rule 2)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from pydantic import BaseModel, Field

# A calibrated read below this is not evidence: phantom reads scored 0.31-0.51, real ones 0.99+
# (project review 2026-09-24, measured facts)
MIN_CONFIDENCE = 0.7


@dataclass
class Crop:
    image: np.ndarray                   # BGR
    box: tuple[int, int, int, int]      # x, y, w, h on the source image


class ReaderResult(BaseModel):
    text: str
    confidence: float = Field(ge=0, le=1)


class Reader(Protocol):
    """One result per crop, same order; None means the reader failed.
    `calibrated`: the confidence is a real probability (local models), not a VLM's constant.
    Optional: `cache_key` (str; readers without one are never cached) and `warm()` (load the model)."""
    name: str
    calibrated: bool
    timeout_s: float

    def read(self, crops: list[Crop]) -> list[ReaderResult] | None: ...


def read_timeout_s(default: float = 20.0) -> float:
    """Time budget of a remote reader, from READ_TIMEOUT_S."""
    try:
        return float(os.environ.get("READ_TIMEOUT_S", default))
    except ValueError:
        return default
