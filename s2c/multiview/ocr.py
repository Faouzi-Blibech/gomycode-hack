"""Handwritten dimension values: find the text, read it, link it to a face axis or a hole. Spec section 4.2.
The reader only reads what the user wrote; it never estimates a size."""
from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import cv2
import numpy as np

from s2c.multiview.outline import PixelOutline, foreground

log = logging.getLogger(__name__)

Reader = Callable[[np.ndarray], tuple[str, float]]  # BGR crop -> (text, confidence)
BatchReader = Callable[[list[np.ndarray]], list[tuple[str, float]] | None]  # every crop of one image; None on failure
_DIAMETER_SIGNS = "⌀ØøΦφ∅"
_VALUE = re.compile(r"^(⌀|D|R)?(\d+(?:\.\d+)?)$")
STROKE_PX = 25


@dataclass
class Reading:
    value_mm: float
    kind: Literal["linear", "diameter", "radius"]
    bbox: tuple[int, int, int, int]
    confidence: float
    text: str


@dataclass
class Linked:
    reading: Reading
    axis: Literal["a", "b", "ab"] | None  # face axis; "ab" is the diameter of a round outline
    hole_index: int | None


def parse_value(text: str) -> tuple[float, str] | None:
    t = text.strip().replace(" ", "").replace(",", ".")
    for sign in _DIAMETER_SIGNS:
        t = t.replace(sign, "⌀")
    if t.lower().endswith("mm"):
        t = t[:-2]
    t = t.rstrip(".")
    if len(t) > 1 and t[0] in "oO" and t[1].isdigit():
        t = "⌀" + t[1:]  # a handwritten ⌀ is often read as o
    if len(t) > 1:
        t = t[0] + t[1:].replace("O", "0").replace("o", "0")
    m = _VALUE.match(t)
    if not m:
        return None
    value = float(m.group(2))
    if not 0 < value < 2000:
        return None
    return value, {"⌀": "diameter", "D": "diameter", "R": "radius"}.get(m.group(1) or "", "linear")


def text_regions(image_bgr: np.ndarray, outline: PixelOutline) -> list[tuple[int, int, int, int]]:
    """Boxes of written text: ink left after erasing the outline, the openings and long straight lines."""
    ink = foreground(image_bgr)
    erase = np.zeros_like(ink)
    cv2.drawContours(erase, [outline.outer.reshape(-1, 1, 2).astype(np.int32)], -1, 255, STROKE_PX)
    for loop in outline.inner:
        cv2.drawContours(erase, [loop.reshape(-1, 1, 2).astype(np.int32)], -1, 255, STROKE_PX)
    for c in outline.circles:
        cv2.circle(erase, (round(c.cx), round(c.cy)), round(c.d / 2), 255, STROKE_PX)
    text = cv2.bitwise_and(ink, cv2.bitwise_not(erase))
    lines = cv2.bitwise_or(cv2.morphologyEx(text, cv2.MORPH_OPEN, np.ones((1, 60), np.uint8)),
                           cv2.morphologyEx(text, cv2.MORPH_OPEN, np.ones((60, 1), np.uint8)))
    words = cv2.dilate(cv2.subtract(text, lines), np.ones((9, 25), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(words)
    boxes = []
    for i in range(1, n):
        x, y, w, h = (int(v) for v in stats[i, :4])
        if 15 <= h <= 220 and w >= 12 and w / h <= 8:
            boxes.append((x, y, w, h))
    return boxes


def read_values(image_bgr: np.ndarray, outline: PixelOutline, reader: Reader | None,
                batch: BatchReader | None = None) -> list[Reading]:
    """One batch call for every crop when a batch reader is given; the single-crop reader is the fallback."""
    boxes = text_regions(image_bgr, outline)
    crops = [image_bgr[max(y - 6, 0): y + h + 6, max(x - 6, 0): x + w + 6] for x, y, w, h in boxes]
    reads = batch(crops) if batch is not None and crops else None
    if reads is None:
        if reader is None:
            return []
        reads = [reader(crop) for crop in crops]
    out = []
    for box, (text, confidence) in zip(boxes, reads):
        parsed = parse_value(text)
        if parsed is None:
            log.info("dropped OCR read %r at %s", text, box)
            continue
        out.append(Reading(parsed[0], parsed[1], box, float(confidence), text))
    return out


def link(readings: list[Reading], outline: PixelOutline) -> list[Linked]:
    """Below or above the outline: axis a. Left or right: axis b. Diameters: the nearest hole."""
    bx, by, bw, bh = outline.bbox
    out = []
    for r in readings:
        x, y, w, h = r.bbox
        cx, cy = x + w / 2, y + h / 2
        inside = bx <= cx <= bx + bw and by <= cy <= by + bh
        if r.kind != "linear":
            if outline.circles and (inside or not outline.circular):
                k = min(range(len(outline.circles)),
                        key=lambda i: (outline.circles[i].cx - cx) ** 2 + (outline.circles[i].cy - cy) ** 2)
                out.append(Linked(r, None, k))
            else:
                out.append(Linked(r, "ab" if outline.circular else None, None))
        elif cy < by or cy > by + bh:
            out.append(Linked(r, "a", None))
        elif cx < bx or cx > bx + bw:
            out.append(Linked(r, "b", None))
        else:
            out.append(Linked(r, None, None))  # written inside the part: kept for the lab view, not linked
    return out


@lru_cache(maxsize=1)
def _trocr(model_name: str):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import RobertaTokenizer, TrOCRProcessor, VisionEncoderDecoderModel, ViTImageProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # transformers 5 does not fetch vocab.json and merges.txt for this repo by itself; take the small files explicitly
    local = snapshot_download(model_name, allow_patterns=["*.json", "*.txt"])
    processor = TrOCRProcessor(image_processor=ViTImageProcessor.from_pretrained(local),
                               tokenizer=RobertaTokenizer.from_pretrained(local))
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device).eval()
    return processor, model, device


def trocr_reader(model_name: str = "microsoft/trocr-base-handwritten") -> Reader:
    """Fallback reader until the numbers owner's reader lands. The model loads on the first read."""
    import transformers  # noqa: F401  fail early when the ai extra is missing

    def read(crop_bgr: np.ndarray) -> tuple[str, float]:
        import torch
        from PIL import Image
        processor, model, device = _trocr(model_name)
        image = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        pixels = processor(images=image, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            out = model.generate(pixels, max_new_tokens=10, num_beams=1, output_scores=True,
                                 return_dict_in_generate=True)
        text = processor.batch_decode(out.sequences, skip_special_tokens=True)[0]
        scores = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
        confidence = float(torch.exp(scores.mean()).item()) if scores.numel() else 0.0
        return text, confidence

    return read
