"""Qwen-Image: reference images and a prompt -> one image, on DashScope or the Hugging Face Space.
Spec 2026-09-23 section 6. Our code only reads the image back as an outline; it is never exported."""
from __future__ import annotations

import base64
import concurrent.futures as cf
import json
import logging
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import httpx
import numpy as np

from s2c.multiview.outline import resize_long_side

log = logging.getLogger(__name__)
ImageGen = Callable[[list[np.ndarray], str, int, str], np.ndarray]  # refs, prompt, seed, log stage -> BGR image
MAX_REFS = 10
REF_LONG_SIDE = 1024
DASHSCOPE_TIMEOUT_S = 90
SPACE_TIMEOUT_S = 120
_pool = cf.ThreadPoolExecutor(max_workers=2)  # a timed-out Space call finishes here while the caller moves on


class ImageGenError(RuntimeError):
    pass


def log_call(log_path, provider: str, model: str, stage: str, t0: float, ok: bool) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"provider": provider, "model": model, "stage": stage, "ok": ok,
                            "latency_ms": round((time.perf_counter() - t0) * 1000)}) + "\n")


def _ref(img: np.ndarray) -> np.ndarray:
    return img if max(img.shape[:2]) <= REF_LONG_SIDE else resize_long_side(img, REF_LONG_SIDE)


def _png_b64(img: np.ndarray) -> str:
    return base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode()


def _decode(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ImageGenError("the reply is not an image")
    return img


def _image_url(output: dict) -> str | None:
    for choice in output.get("choices") or []:
        for item in (choice.get("message") or {}).get("content") or []:
            if isinstance(item, dict) and item.get("image"):
                return item["image"]
    results = output.get("results") or []
    return results[0].get("url") if results else None


def dashscope_gen(base_url: str, model: str, key: str, client: httpx.Client | None = None, poll_s: float = 2.0,
                  timeout_s: float = DASHSCOPE_TIMEOUT_S, log_path="logs/vlm.jsonl") -> ImageGen:
    http = client or httpx.Client(timeout=30)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def call(refs: list[np.ndarray], prompt: str, seed: int) -> np.ndarray:
        start = time.monotonic()
        content = [{"text": prompt}]
        content += [{"image": f"data:image/png;base64,{_png_b64(_ref(r))}"} for r in refs[:MAX_REFS]]
        payload = {"model": model, "input": {"messages": [{"role": "user", "content": content}]},
                   "parameters": {"prompt_extend": False, "watermark": False, "seed": int(seed)}}
        r = http.post(f"{base_url}/services/aigc/multimodal-generation/generation", json=payload, headers=headers)
        r.raise_for_status()
        out = r.json().get("output") or {}
        while _image_url(out) is None and out.get("task_id"):
            if out.get("task_status") in ("FAILED", "CANCELED", "UNKNOWN"):
                raise ImageGenError(f"task {out['task_status']}: {out.get('message', '')}")
            if time.monotonic() - start > timeout_s:
                raise ImageGenError("timed out")
            time.sleep(poll_s)
            r = http.get(f"{base_url}/tasks/{out['task_id']}", headers=headers)
            r.raise_for_status()
            out = r.json().get("output") or {}
        url = _image_url(out)
        if url is None:
            raise ImageGenError("no image in the reply")
        got = http.get(url)
        got.raise_for_status()
        return _decode(got.content)

    def gen(refs: list[np.ndarray], prompt: str, seed: int, stage: str) -> np.ndarray:
        t0, ok = time.perf_counter(), False
        try:
            img = call(refs, prompt, seed)
            ok = True
            return img
        except ImageGenError:
            raise
        except Exception as e:
            raise ImageGenError(f"DashScope: {e}") from e
        finally:
            log_call(log_path, "dashscope", model, stage, t0, ok)

    return gen


def space_gen(space: str, token: str | None = None, client_factory=None, timeout_s: float = SPACE_TIMEOUT_S,
              log_path="logs/vlm.jsonl") -> ImageGen:
    def run(refs: list[np.ndarray], prompt: str, seed: int) -> np.ndarray:
        from gradio_client import Client, handle_file
        client = (client_factory or Client)(space, token=token)
        with tempfile.TemporaryDirectory() as d:
            files = []
            for k, img in enumerate(refs[:MAX_REFS]):
                path = Path(d) / f"ref{k}.png"
                cv2.imwrite(str(path), _ref(img))
                files.append({"image": handle_file(str(path)), "caption": None})
            result = client.predict(input_images=files, original_prompt=prompt, enable_extend=False,
                                    seed=int(seed), randomize_seed=False, api_name="/generate_with_enhance")
        image = result[0] if isinstance(result, (list, tuple)) else result
        path = image.get("path") if isinstance(image, dict) else image
        if not path:
            raise ImageGenError("the Space returned no image")
        return _decode(Path(path).read_bytes())

    def gen(refs: list[np.ndarray], prompt: str, seed: int, stage: str) -> np.ndarray:
        t0, ok = time.perf_counter(), False
        try:
            img = _pool.submit(run, refs, prompt, seed).result(timeout=timeout_s)
            ok = True
            return img
        except ImageGenError:
            raise
        except Exception as e:  # includes the timeout
            raise ImageGenError(f"Space {space}: {e!r}") from e
        finally:
            log_call(log_path, "hf-space", space, stage, t0, ok)

    return gen


def default_gen() -> ImageGen | None:
    """QWEN_IMAGE_BACKEND=dashscope with its settings, else the Space in QWEN_IMAGE_SPACE, else None."""
    backend = os.environ.get("QWEN_IMAGE_BACKEND", "space")
    base, model, key = (os.environ.get(k) for k in ("QWEN_IMAGE_BASE_URL", "QWEN_IMAGE_MODEL", "VLM_API_KEY"))
    if backend == "dashscope":
        if base and model and key:
            return dashscope_gen(base, model, key)
        log.warning("QWEN_IMAGE_BACKEND=dashscope needs QWEN_IMAGE_BASE_URL, QWEN_IMAGE_MODEL and VLM_API_KEY; "
                    "trying the Space")
    space = os.environ.get("QWEN_IMAGE_SPACE")
    return space_gen(space, os.environ.get("HF_TOKEN")) if space else None
