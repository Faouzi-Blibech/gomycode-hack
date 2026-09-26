"""Real calls to the hosted models. Needs a filled .env:
    NETWORK_TESTS=1 uv run pytest tests/test_mv_network.py -v"""
import os

import cv2
import numpy as np
import pytest
from dotenv import load_dotenv

pytestmark = [pytest.mark.network,
              pytest.mark.skipif(os.environ.get("NETWORK_TESTS") != "1", reason="set NETWORK_TESTS=1")]


@pytest.fixture(autouse=True)
def env():
    load_dotenv()


def written(text):
    img = np.full((80, 200, 3), 255, np.uint8)
    cv2.putText(img, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    return img


def test_qwen_vl_reads_written_values():
    from s2c.multiview.label import env_chat
    from s2c.multiview.ocr import parse_value
    from s2c.multiview.qwen_reader import qwen_batch_reader
    chat = env_chat(stage="mv_read")
    assert chat, "set VLM_BASE_URL, VLM_MODEL and VLM_API_KEY"
    reads = qwen_batch_reader(chat)([written("60"), written("R3")])
    assert [parse_value(text) for text, _ in reads] == [(60.0, "linear"), (3.0, "radius")]


def test_qwen_image_draws_a_top_view():
    from s2c.multiview.qwen_faces import face_prompt, outline_from_image
    from s2c.multiview.qwen_image import default_gen
    from s2c.multiview.spec import Envelope
    gen = default_gen()
    assert gen, "set QWEN_IMAGE_SPACE or the DashScope settings"
    front = np.full((800, 1000, 3), 255, np.uint8)
    cv2.rectangle(front, (200, 250), (800, 550), (0, 0, 0), 4)
    img = gen([front], face_prompt(["front"], "top"), 7, "mv_face")
    assert outline_from_image(img, "top", Envelope(x_mm=60.0, y_mm=30.0, z_mm=20.0)) is not None


def test_solaria_returns_a_depth_map():
    from s2c.multiview.depth import solaria_depth
    from tests.test_mv_depth import plate_photo
    space = os.environ.get("SOLARIA_SPACE")
    assert space, "set SOLARIA_SPACE"
    depth = solaria_depth(space, os.environ.get("HF_TOKEN"))(plate_photo())
    assert depth.shape == (1200, 1600) and np.isfinite(depth).mean() > 0.5
