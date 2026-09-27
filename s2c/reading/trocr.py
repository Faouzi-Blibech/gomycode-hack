"""TrOCR handwritten, the one local reader shared by the Studio and the sketch package.
Optional dependency: the ai extra (torch, transformers). The model loads once per process and device,
under a lock, on warm() or on the first read."""
from __future__ import annotations

import os
import threading

import cv2
import numpy as np

from s2c.reading.base import Crop, ReaderResult

DEFAULT_MODEL = "microsoft/trocr-base-handwritten"
_LOADED: dict[tuple[str, str], tuple] = {}
_LOCK = threading.Lock()
_READ_LOCK = threading.Lock()  # concurrent batches queue instead of thrashing the same model and device


def token_confidences(logprobs: np.ndarray, mask: np.ndarray) -> list[float]:
    """exp(mean log-probability) over each row's real tokens. Batched generation pads finished rows with
    tokens at log-probability 0; counting them would push a short read's confidence toward 1."""
    out = []
    for row, keep in zip(np.asarray(logprobs, float), np.asarray(mask).astype(bool)):
        real = row[keep]
        out.append(float(np.clip(np.exp(real.mean()), 0.0, 1.0)) if real.size else 0.0)
    return out


def _load(model_id: str, device: str) -> tuple:
    with _LOCK:
        if (model_id, device) not in _LOADED:
            from huggingface_hub import snapshot_download
            from transformers import RobertaTokenizer, TrOCRProcessor, VisionEncoderDecoderModel, ViTImageProcessor
            # transformers 5 does not fetch vocab.json and merges.txt for this repo by itself; take the small files
            local = snapshot_download(model_id, allow_patterns=["*.json", "*.txt"])
            processor = TrOCRProcessor(image_processor=ViTImageProcessor.from_pretrained(local),
                                       tokenizer=RobertaTokenizer.from_pretrained(local))
            model = VisionEncoderDecoderModel.from_pretrained(model_id).to(device).eval()
            if device == "cuda":
                model = model.half()  # halves GPU memory; CPU stays float32 (no fp16 kernels there)
            _LOADED[(model_id, device)] = (processor, model)
        return _LOADED[(model_id, device)]


class TrocrReader:
    name = "trocr"
    calibrated = True

    def __init__(self, model_id: str | None = None, device: str | None = None, timeout_s: float = 60.0,
                 max_new_tokens: int = 12):
        import torch
        import transformers  # noqa: F401 - fail early when the ai extra is missing

        self.model_id = model_id or os.environ.get("TROCR_MODEL", DEFAULT_MODEL)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.timeout_s, self.max_new_tokens = timeout_s, max_new_tokens
        self.cache_key = f"trocr:{self.model_id}"

    def warm(self) -> None:
        _load(self.model_id, self.device)

    def read(self, crops: list[Crop]) -> list[ReaderResult] | None:
        if not crops:
            return []
        import torch
        from PIL import Image

        processor, model = _load(self.model_id, self.device)
        images = [Image.fromarray(cv2.cvtColor(c.image, cv2.COLOR_BGR2RGB)) for c in crops]
        with _READ_LOCK:
            pixels = processor(images=images, return_tensors="pt").pixel_values.to(self.device, dtype=model.dtype)
            with torch.no_grad():
                out = model.generate(pixels, max_new_tokens=self.max_new_tokens, num_beams=1, output_scores=True,
                                     return_dict_in_generate=True)
            texts = processor.batch_decode(out.sequences, skip_special_tokens=True)
            scores = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
        generated = out.sequences[:, -scores.shape[1]:]  # the tokens the scores describe
        pad = model.generation_config.pad_token_id
        if pad is None:
            pad = processor.tokenizer.pad_token_id
        mask = (generated != pad).cpu().numpy()
        confidences = token_confidences(scores.float().cpu().numpy(), mask)
        return [ReaderResult(text=t.strip(), confidence=c) for t, c in zip(texts, confidences)]
