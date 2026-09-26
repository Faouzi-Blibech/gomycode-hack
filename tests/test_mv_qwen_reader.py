import json

import numpy as np

from s2c.multiview.ocr import read_values
from s2c.multiview.outline import extract
from s2c.multiview.qwen_reader import qwen_batch_reader
from tests.test_mv_ocr import sketch_with_values

CROP = np.full((40, 60, 3), 255, np.uint8)


def chat_returning(*outputs):
    calls = []

    def chat(messages):
        calls.append(messages)
        return outputs[len(calls) - 1]

    chat.calls = calls
    return chat


def reads(*pairs):
    return json.dumps({"reads": [{"i": i, "text": t} for i, t in pairs]})


def test_reads_come_back_in_crop_order():
    chat = chat_returning(reads((2, "40"), (1, "⌀6")))
    assert qwen_batch_reader(chat)([CROP, CROP]) == [("⌀6", 0.9), ("40", 0.9)]


def test_one_message_holds_every_crop():
    chat = chat_returning(reads())
    qwen_batch_reader(chat)([CROP, CROP, CROP])
    content = chat.calls[0][0]["content"]
    assert sum(part["type"] == "image_url" for part in content) == 3
    assert "Never guess" in content[0]["text"]


def test_a_missing_crop_reads_as_empty():
    chat = chat_returning(reads((1, "60")))
    assert qwen_batch_reader(chat)([CROP, CROP]) == [("60", 0.9), ("", 0.9)]


def test_bad_json_is_retried_once_with_the_error():
    chat = chat_returning("not json", reads((1, "60")))
    assert qwen_batch_reader(chat)([CROP]) == [("60", 0.9)]
    assert "failed validation" in chat.calls[1][-1]["content"]


def test_an_index_out_of_range_is_retried():
    chat = chat_returning(reads((5, "60")), reads((1, "60")))
    assert qwen_batch_reader(chat)([CROP]) == [("60", 0.9)]
    assert len(chat.calls) == 2


def test_two_failures_or_an_error_return_none():
    assert qwen_batch_reader(chat_returning("nope", "{}"))([CROP]) is None

    def broken(messages):
        raise RuntimeError("503")

    assert qwen_batch_reader(broken)([CROP]) is None


def test_no_crops_makes_no_call():
    chat = chat_returning()
    assert qwen_batch_reader(chat)([]) == [] and chat.calls == []


def test_read_values_uses_the_batch_once_and_falls_back_to_the_reader():
    img = sketch_with_values()
    o = extract(img)
    calls = []

    def batch(crops):
        calls.append(len(crops))
        return [("60", 0.9), ("40 mm", 0.9)]

    def failing(crops):
        return None

    assert sorted(r.value_mm for r in read_values(img, o, None, batch)) == [40.0, 60.0] and calls == [2]
    assert [r.value_mm for r in read_values(img, o, lambda crop: ("60", 0.8), failing)] == [60.0, 60.0]
    assert read_values(img, o, None, failing) == []
