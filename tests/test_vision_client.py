import json

from s2c.vision.client import VLMClient


def test_complete_json_sends_image_and_returns_text(tmp_path):
    seen = {}

    def fake_chat(messages):
        seen["messages"] = messages
        return '{"ok": true}'

    client = VLMClient(chat=fake_chat, log_path=tmp_path / "vlm.jsonl", model="fake-model")
    out = client.complete_json("SYS", "USER", b"\xff\xd8bytes")
    assert out == '{"ok": true}'
    assert seen["messages"][0] == {"role": "system", "content": "SYS"}
    parts = seen["messages"][1]["content"]
    assert parts[0] == {"type": "text", "text": "USER"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_every_call_is_logged(tmp_path):
    client = VLMClient(chat=lambda m: "{}", log_path=tmp_path / "vlm.jsonl", model="fake-model")
    client.complete_json("s", "u", b"x")
    client.complete_json("s", "u", b"x")
    lines = (tmp_path / "vlm.jsonl").read_text().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["model"] == "fake-model"
    assert "latency_ms" in rec


def test_call_survives_when_log_write_fails(tmp_path):
    # Point the log path at a location that cannot be created: a file used as
    # a path segment, so mkdir(parents=True) on its "parent" fails on Windows
    # and POSIX alike.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    unwritable_log_path = blocker / "nested" / "vlm.jsonl"

    client = VLMClient(chat=lambda m: "{}", log_path=unwritable_log_path, model="fake-model")
    out = client.complete_json("s", "u", b"x")
    assert out == "{}"
    assert not unwritable_log_path.exists()


def test_instances_do_not_share_last_usage():
    # Each VLMClient must track its own usage; _last_usage must not be a
    # class-level dict shared across instances.
    client_a = VLMClient(chat=lambda m: "{}", model="model-a")
    client_b = VLMClient(chat=lambda m: "{}", model="model-b")
    client_a._last_usage = {"prompt_tokens": 111, "completion_tokens": 222}
    assert client_b._last_usage == {}
