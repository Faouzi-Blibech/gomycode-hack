import json
import tempfile
import time
from pathlib import Path
from typing import ClassVar

import cv2
import httpx
import numpy as np
import pytest

from s2c.multiview import qwen_image as Q

PNG = cv2.imencode(".png", np.zeros((8, 8, 3), np.uint8))[1].tobytes()
REF = np.full((20, 30, 3), 255, np.uint8)
BASE = "https://dash.example/api/v1"


def client(handler, seen):
    def record(request):
        seen.append(request)
        return handler(request)

    return httpx.Client(transport=httpx.MockTransport(record))


def test_dashscope_sync_reply(tmp_path):
    seen = []

    def handler(req):
        if req.url.path.endswith("/generation"):
            content = [{"image": "https://img.example/a.png"}]
            return httpx.Response(200, json={"output": {"choices": [{"message": {"content": content}}]}})
        return httpx.Response(200, content=PNG)

    gen = Q.dashscope_gen(BASE, "qwen-image-x", "k", client(handler, seen), log_path=tmp_path / "log.jsonl")
    assert gen([REF, REF], "draw the top view", 7, "mv_face").shape == (8, 8, 3)
    body = json.loads(seen[0].content)
    assert body["model"] == "qwen-image-x"
    assert body["parameters"] == {"prompt_extend": False, "watermark": False, "seed": 7}
    content = body["input"]["messages"][0]["content"]
    assert content[0] == {"text": "draw the top view"} and len(content) == 3
    assert seen[0].headers["authorization"] == "Bearer k"
    assert json.loads((tmp_path / "log.jsonl").read_text())["stage"] == "mv_face"


def test_dashscope_polls_a_task(tmp_path):
    states = iter(["RUNNING", "SUCCEEDED"])

    def handler(req):
        if req.url.path.endswith("/generation"):
            return httpx.Response(200, json={"output": {"task_id": "t1", "task_status": "PENDING"}})
        if req.url.path.endswith("/tasks/t1"):
            status = next(states)
            out = {"task_id": "t1", "task_status": status}
            if status == "SUCCEEDED":
                out["results"] = [{"url": "https://img.example/b.png"}]
            return httpx.Response(200, json={"output": out})
        return httpx.Response(200, content=PNG)

    gen = Q.dashscope_gen(BASE, "m", "k", client(handler, []), poll_s=0, log_path=tmp_path / "l")
    assert gen([REF], "p", 7, "mv_face").shape == (8, 8, 3)


def test_dashscope_failures_raise_image_gen_error(tmp_path):
    def fail(req):
        return httpx.Response(500, text="boom")

    def running(req):
        return httpx.Response(200, json={"output": {"task_id": "t", "task_status": "RUNNING"}})

    with pytest.raises(Q.ImageGenError):
        Q.dashscope_gen(BASE, "m", "k", client(fail, []), log_path=tmp_path / "l")([REF], "p", 7, "mv_face")
    t0 = time.monotonic()
    with pytest.raises(Q.ImageGenError, match="timed out"):
        Q.dashscope_gen(BASE, "m", "k", client(running, []), poll_s=0, timeout_s=0.05,
                        log_path=tmp_path / "l")([REF], "p", 7, "mv_face")
    assert time.monotonic() - t0 < 2


class FakeSpace:
    calls: ClassVar[list] = []

    def __init__(self, src, token=None):
        self.src, self.token = src, token

    def predict(self, **kw):
        FakeSpace.calls.append((self.src, self.token, kw))
        path = Path(tempfile.mkdtemp()) / "out.png"
        path.write_bytes(PNG)
        return {"path": str(path)}, 7.0, "rewritten"


def test_the_space_gets_every_reference_and_a_fixed_seed(tmp_path):
    FakeSpace.calls.clear()
    gen = Q.space_gen("Qwen/Qwen-Image-2.1", "tok", FakeSpace, log_path=tmp_path / "l")
    assert gen([REF, REF], "p", 8, "mv_rescue").shape == (8, 8, 3)
    src, token, kw = FakeSpace.calls[0]
    assert (src, token) == ("Qwen/Qwen-Image-2.1", "tok")
    assert len(kw["input_images"]) == 2 and kw["seed"] == 8 and kw["randomize_seed"] is False
    assert kw["enable_extend"] is False and kw["api_name"] == "/generate_with_enhance"


def test_a_slow_space_times_out_quickly(tmp_path):
    class Slow(FakeSpace):
        def predict(self, **kw):
            time.sleep(2)
            return super().predict(**kw)

    t0 = time.monotonic()
    with pytest.raises(Q.ImageGenError):
        Q.space_gen("s", None, Slow, timeout_s=0.2, log_path=tmp_path / "l")([REF], "p", 7, "mv_face")
    assert time.monotonic() - t0 < 1.5


def test_the_backend_comes_from_the_environment(monkeypatch):
    for key in ("QWEN_IMAGE_BACKEND", "QWEN_IMAGE_SPACE", "QWEN_IMAGE_BASE_URL", "QWEN_IMAGE_MODEL", "VLM_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    assert Q.default_gen() is None
    monkeypatch.setenv("QWEN_IMAGE_SPACE", "Qwen/Qwen-Image-2.1")
    assert Q.default_gen().__qualname__.startswith("space_gen")
    for key, value in {"QWEN_IMAGE_BACKEND": "dashscope", "QWEN_IMAGE_BASE_URL": BASE, "QWEN_IMAGE_MODEL": "m",
                       "VLM_API_KEY": "k"}.items():
        monkeypatch.setenv(key, value)
    assert Q.default_gen().__qualname__.startswith("dashscope_gen")
