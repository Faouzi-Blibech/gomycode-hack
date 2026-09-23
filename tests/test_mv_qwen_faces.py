import cv2
import numpy as np

from s2c.multiview.complete import complete
from s2c.multiview.qwen_faces import consistent, face_prompt
from s2c.multiview.qwen_image import ImageGenError
from s2c.multiview.raster import iou, outline_mask
from s2c.multiview.spec import Envelope, Outline
from tests.mv_helpers import outline, rect

ENV = Envelope(x_mm=50.0, y_mm=30.0, z_mm=20.0)
FRONT_L = [(0, 0), (50, 0), (50, 30), (44, 30), (44, 6), (0, 6)]
TOP_TAPER = [(0, 5), (50, 0), (50, 20), (0, 15)]
RIGHT_L = [(0, 0), (20, 0), (20, 6), (6, 6), (6, 30), (0, 30)]    # above y = 6 the part is only at z 14..20
BAD_TOP = [(25, 0), (50, 0), (50, 20), (0, 20), (0, 12), (25, 12)]  # for x < 25 only z 0..8: cuts the front
IMAGE = np.zeros((4, 4, 3), np.uint8)


def silhouette(points_mm, a_len, b_len, px=10):
    """What a good Qwen answer looks like: a black silhouette on white, 10 px per mm, with a margin."""
    img = np.full((round(b_len * px) + 200, round(a_len * px) + 200, 3), 255, np.uint8)
    pts = np.array([(100 + a * px, 100 + (b_len - b) * px) for a, b in points_mm], np.int32)
    cv2.fillPoly(img, [pts], (0, 0, 0))
    return img


def fake_gen(*answers):
    calls = []

    def gen(refs, prompt, seed, stage):
        calls.append((len(refs), prompt, seed, stage))
        answer = answers[min(len(calls), len(answers)) - 1]
        if isinstance(answer, Exception):
            raise answer
        return answer

    gen.calls = calls
    return gen


def ol(points):
    return Outline.model_validate(outline(points))


def front_only():
    return {"front": ol(FRONT_L)}


def front_and_right():
    return {"front": ol(rect(50, 30)), "right": ol(RIGHT_L)}


def test_the_prompt_names_every_reference_and_the_camera():
    p = face_prompt(["front", "right"], "top")
    assert "image 1 is the front view, image 2 is the right view" in p
    assert "looking down" in p and "No text" in p


def test_consistent_rejects_a_top_that_removes_photographed_material():
    assert consistent({**front_and_right(), "top": ol(rect(50, 20))}, ENV, ["front", "right"])
    assert not consistent({**front_and_right(), "top": ol(BAD_TOP)}, ENV, ["front", "right"])


def test_qwen_fills_the_missing_faces_and_says_so():
    gen = fake_gen(silhouette(TOP_TAPER, 50, 20), silhouette(rect(20, 30), 20, 30))
    filled = {}
    result, _, _ = complete(front_only(), ENV, "front", None, IMAGE, None, gen=gen, refs=[("front", IMAGE)],
                            filled_by=filled)
    assert filled == {"front": "observed", "top": "qwen-image", "right": "qwen-image"}
    top = result["top"]
    assert top.source == "inferred"
    assert iou(outline_mask(top.outer, top.inner, 50, 20), outline_mask(TOP_TAPER, [], 50, 20)) > 0.95
    assert [c[2:] for c in gen.calls] == [(7, "mv_face"), (7, "mv_face")] and "top view" in gen.calls[0][1]


def test_a_rejected_qwen_top_is_retried_with_the_next_seed():
    gen = fake_gen(silhouette(BAD_TOP, 50, 20), silhouette(rect(50, 20), 50, 20))
    filled = {}
    complete(front_and_right(), ENV, "front", None, IMAGE, None, gen=gen, refs=[("front", IMAGE)], filled_by=filled)
    assert filled["top"] == "qwen-image" and [c[2] for c in gen.calls] == [7, 8]


def test_two_rejected_answers_fall_back_and_warn():
    def broken(img):
        raise RuntimeError("no GPU")

    filled = {}
    _, warnings, _ = complete(front_and_right(), ENV, "front", np.zeros((512, 512), np.uint8), IMAGE, broken,
                              gen=fake_gen(silhouette(BAD_TOP, 50, 20)), refs=[("front", IMAGE)], filled_by=filled)
    assert filled["top"] == "assumed"
    assert "top: Qwen-Image view rejected, assumed used" in warnings


def test_an_empty_or_failed_answer_is_not_used():
    for answer in (np.zeros((300, 300, 3), np.uint8), ImageGenError("503")):
        filled = {}
        complete(front_only(), ENV, "front", None, IMAGE, None, gen=fake_gen(answer), refs=[("front", IMAGE)],
                 filled_by=filled)
        assert filled["top"] == "assumed" and filled["right"] == "assumed"


def test_the_answer_is_cached_so_an_edit_never_calls_qwen_again():
    gen = fake_gen(silhouette(TOP_TAPER, 50, 20), silhouette(rect(20, 30), 20, 30))
    cache = {}
    complete(front_only(), ENV, "front", None, IMAGE, None, gen=gen, refs=[("front", IMAGE)], qwen_cache=cache)
    wider = Envelope(x_mm=60.0, y_mm=30.0, z_mm=20.0)
    front = {"front": ol([(a * 1.2, b) for a, b in FRONT_L])}
    result, _, _ = complete(front, wider, "front", None, None, None, gen=gen, refs=[], qwen_cache=cache)
    assert len(gen.calls) == 2 and max(a for a, _ in result["top"].outer) == 60.0


def test_a_rejected_face_skips_qwen():
    gen = fake_gen(silhouette(TOP_TAPER, 50, 20))
    filled = {}
    complete(front_only(), ENV, "front", None, IMAGE, None, rejected=("top", "right"), gen=gen,
             refs=[("front", IMAGE)], filled_by=filled)
    assert gen.calls == [] and filled["top"] == "assumed"
