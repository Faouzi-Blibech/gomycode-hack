"""Readers transcribe handwriting in crops. They never estimate a size. Swappable by env var."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
from pydantic import BaseModel, Field, ValidationError

from s2c.vision.client import VLMClient

log = logging.getLogger(__name__)


@dataclass
class Crop:
    image: np.ndarray
    box: tuple[int, int, int, int]


class ReaderResult(BaseModel):
    text: str
    confidence: float = Field(ge=0, le=1)


class Reader(Protocol):
    name: str

    def read(self, crops: list[Crop]) -> list[ReaderResult] | None: ...


SYSTEM = (
    "You transcribe handwriting cropped from an engineering sketch. Each numbered tile is one crop. "
    'Return JSON only: {"reads": [{"i": 1, "text": "...", "confidence": 0.0}]}, one entry per tile. '
    "Copy exactly the characters written. Use Ø for a diameter sign, R for a radius, ° for degrees. "
    "Keep a leading decimal point exactly as written, for example .50. "
    "Return an empty text when a tile is unreadable. "
    "Never estimate, measure or guess a value that is not written."
)


class _Read(BaseModel):
    i: int
    text: str
    confidence: float = Field(default=0.8, ge=0, le=1)


class _Reply(BaseModel):
    reads: list[_Read]


def _json_block(raw: str) -> str:
    start, end = raw.find("{"), raw.rfind("}")
    return raw[start: end + 1] if start >= 0 and end > start else raw


def tile_grid(crops: list[Crop], cols: int = 4, cell_h: int = 80, max_w: int = 320) -> bytes:
    """One image with every crop in a numbered cell; the number sits in a header, not on the ink."""
    header = 26
    tiles = []
    for c in crops:
        h, w = c.image.shape[:2]
        k = cell_h / max(h, 1)
        tile = cv2.resize(c.image, (min(max_w, max(1, round(w * k))), cell_h))
        tiles.append(tile)
    cell_w = max_w + 16
    rows = (len(tiles) + cols - 1) // cols
    grid = np.full((rows * (cell_h + header + 12), cols * cell_w, 3), 255, np.uint8)
    for i, tile in enumerate(tiles):
        r, col = divmod(i, cols)
        y, x = r * (cell_h + header + 12), col * cell_w
        cv2.putText(grid, f"#{i + 1}", (x + 4, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 0, 0), 2)
        grid[y + header: y + header + cell_h, x + 8: x + 8 + tile.shape[1]] = tile
        cv2.rectangle(grid, (x + 4, y + header - 2), (x + cell_w - 4, y + header + cell_h + 2),
                      (180, 180, 180), 1)
    _, buf = cv2.imencode(".png", grid)
    return buf.tobytes()


class VlmReader:
    name = "vlm"

    def __init__(self, client: VLMClient, batch: int = 24):
        self.client, self.batch = client, batch

    def read(self, crops: list[Crop]) -> list[ReaderResult] | None:
        out: list[ReaderResult] = []
        for start in range(0, len(crops), self.batch):
            part = self._read_batch(crops[start: start + self.batch])
            if part is None:
                return None
            out += part
        return out

    def _read_batch(self, crops: list[Crop]) -> list[ReaderResult] | None:
        n = len(crops)
        image = tile_grid(crops)
        user = f"There are {n} tiles, numbered 1 to {n}."
        reply = None
        for _ in range(2):
            try:
                raw = self.client.complete_json(SYSTEM, user, image, mime="image/png")
            except Exception as exc:  # noqa: BLE001 transport errors must not break the pipeline
                log.warning("vlm reader failed: %s", exc)
                return None
            try:
                reply = _Reply.model_validate_json(_json_block(raw))
                break
            except ValidationError as exc:
                user += f"\nYour previous reply was invalid ({exc.error_count()} errors). Return JSON only."
        if reply is None:
            return None
        by_i = {r.i: r for r in reply.reads}
        return [ReaderResult(text=by_i[i].text, confidence=by_i[i].confidence) if i in by_i
                else ReaderResult(text="", confidence=0.0) for i in range(1, n + 1)]


class PaddleReader:
    """PP-OCRv5 text recognition. Optional dependency: `uv sync --extra ocr`.
    Checked against PaddleOCR 3.x (`TextRecognition(...).predict(...)` returning `rec_text`,
    `rec_score`)."""
    name = "paddle"

    def __init__(self, model_name: str | None = None):
        from paddleocr import TextRecognition

        self._model = TextRecognition(
            model_name=model_name or os.environ.get("SKETCH_PADDLE_MODEL", "PP-OCRv5_server_rec"))

    def read(self, crops: list[Crop]) -> list[ReaderResult] | None:
        try:
            out = []
            for c in crops:
                res = next(self._model.predict(input=c.image, batch_size=1))
                data = res.json.get("res", res.json) if hasattr(res, "json") else dict(res)
                out.append(ReaderResult(
                    text=str(data.get("rec_text", "")),
                    confidence=float(np.clip(data.get("rec_score", 0.0), 0, 1))))
            return out
        except Exception as exc:  # noqa: BLE001
            log.warning("paddle reader failed: %s", exc)
            return None


class TrocrReader:
    """TrOCR handwritten. Optional dependency: `uv sync --extra trocr`."""
    name = "trocr"

    def __init__(self, model_id: str | None = None, device: str | None = None):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        mid = model_id or os.environ.get("SKETCH_TROCR_MODEL", "microsoft/trocr-base-handwritten")
        self._proc = TrOCRProcessor.from_pretrained(mid)
        self._model = VisionEncoderDecoderModel.from_pretrained(mid)
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device).eval()

    def read(self, crops: list[Crop]) -> list[ReaderResult] | None:
        if not crops:
            return []
        try:
            import torch
            from PIL import Image

            images = [Image.fromarray(cv2.cvtColor(c.image, cv2.COLOR_BGR2RGB)) for c in crops]
            pixels = self._proc(images=images, return_tensors="pt").pixel_values.to(self._device)
            with torch.no_grad():
                gen = self._model.generate(pixels, max_new_tokens=16, output_scores=True,
                                           return_dict_in_generate=True)
            texts = self._proc.batch_decode(gen.sequences, skip_special_tokens=True)
            scores = self._model.compute_transition_scores(gen.sequences, gen.scores,
                                                           normalize_logits=True)
            conf = torch.exp(scores.mean(dim=1)).clamp(0, 1).tolist()
            return [ReaderResult(text=t.strip(), confidence=float(c)) for t, c in zip(texts, conf)]
        except Exception as exc:  # noqa: BLE001
            log.warning("trocr reader failed: %s", exc)
            return None


def readers_from_env(spec: str | None = None) -> list[Reader]:
    names = [n.strip() for n in (spec or os.environ.get("SKETCH_READERS", "paddle,vlm")).split(",")]
    out: list[Reader] = []
    for name in filter(None, names):
        try:
            if name == "vlm":
                out.append(VlmReader(VLMClient.from_env()))
            elif name == "paddle":
                out.append(PaddleReader())
            elif name == "trocr":
                out.append(TrocrReader())
            else:
                log.warning("unknown reader %r skipped", name)
        except Exception as exc:  # noqa: BLE001 a missing optional dependency must not stop
            log.warning("reader %s unavailable: %s", name, exc)
    return out
