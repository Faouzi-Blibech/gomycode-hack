import json

from s2c.multiview.label import MvLabel, hint_label, label_image
from s2c.multiview.spec import MvAbstain

GOOD = json.dumps({"face": "front", "input_kind": "sketch", "holes": [{"u": 0.2, "v": 0.3, "blind": True}],
                   "description": "plate", "estimates": {"holes[0].depth_mm": 3.0, "envelope.x_mm": 60},
                   "confidence": 0.8})


def chat_returning(*outputs):
    calls = []

    def chat(messages):
        calls.append(messages)
        return outputs[len(calls) - 1]

    chat.calls = calls
    return chat


def test_valid_label_keeps_hole_depths_and_drops_envelope_estimates():
    label = label_image(b"jpeg", chat_returning(GOOD))
    assert label.face == "front" and label.holes[0].blind
    assert label.estimates == {"holes[0].depth_mm": 3.0}


def test_retry_once_with_the_validation_error():
    chat = chat_returning("not json", GOOD)
    assert isinstance(label_image(b"jpeg", chat), MvLabel)
    assert "failed validation" in chat.calls[1][-1]["content"]


def test_two_failures_abstain():
    res = label_image(b"jpeg", chat_returning("nope", "{}"))
    assert isinstance(res, MvAbstain) and res.reason == "label_invalid"


def test_the_users_face_tag_wins_and_unknown_abstains():
    unknown = GOOD.replace('"front"', '"unknown"')
    assert label_image(b"jpeg", chat_returning(unknown), face_hint="top").face == "top"
    res = label_image(b"jpeg", chat_returning(unknown))
    assert isinstance(res, MvAbstain) and res.reason == "face_unknown"


def test_code_fences_are_stripped():
    assert label_image(b"jpeg", chat_returning(f"```json\n{GOOD}\n```")).face == "front"


def test_hint_label_without_a_model():
    label = hint_label("right", "photo")
    assert (label.face, label.input_kind) == ("right", "photo")


def test_the_chat_client_gives_up_in_a_minute(monkeypatch):
    import openai

    from s2c.multiview.label import env_chat
    made = {}

    class Recorder:
        def __init__(self, **kwargs):
            made.update(kwargs)

    monkeypatch.setattr(openai, "OpenAI", Recorder)
    for key, value in {"VLM_BASE_URL": "http://localhost:9/v1", "VLM_MODEL": "m", "VLM_API_KEY": "k"}.items():
        monkeypatch.setenv(key, value)
    assert env_chat() is not None
    assert made["timeout"] == 60 and made["max_retries"] == 1
