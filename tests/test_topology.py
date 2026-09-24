from s2c.partspec.models import Abstain, Topology
from s2c.vision.client import VLMClient
from s2c.vision.topology import extract_json, topology_from_image

GOOD = '{"part_type": "plate", "holes": [{"u": 0.2, "v": 0.5, "kind": "through"}], "confidence": 0.9}'
WITH_DIMS = '{"part_type": "plate", "width_mm": 60, "confidence": 0.9}'


def client_returning(*replies, log_path):
    it = iter(replies)
    calls = []

    def chat(messages):
        calls.append(messages)
        return next(it)

    return VLMClient(chat=chat, log_path=log_path, model="fake"), calls


def test_extract_json_strips_code_fences():
    fence = "`" * 3
    assert extract_json(fence + "json\n{\"a\": 1}\n" + fence) == '{"a": 1}'
    assert extract_json("Sure! {\"a\": 1} done") == '{"a": 1}'


def test_valid_reply_becomes_topology(tmp_path):
    client, calls = client_returning(GOOD, log_path=tmp_path / "l")
    out = topology_from_image(b"img", client, "sketch")
    assert isinstance(out, Topology)
    assert out.holes[0].u == 0.2
    assert len(calls) == 1
    assert "never output a length" in calls[0][0]["content"].lower()


def test_dimension_in_reply_triggers_one_retry_with_error(tmp_path):
    client, calls = client_returning(WITH_DIMS, GOOD, log_path=tmp_path / "l")
    out = topology_from_image(b"img", client, "sketch")
    assert isinstance(out, Topology)
    assert len(calls) == 2
    retry_text = calls[1][1]["content"][0]["text"]
    assert "failed validation" in retry_text and "width_mm" in retry_text


def test_two_failures_abstain(tmp_path):
    client, _calls = client_returning(WITH_DIMS, "not json at all", log_path=tmp_path / "l")
    out = topology_from_image(b"img", client, "sketch")
    assert isinstance(out, Abstain)
    assert out.stage == "vision" and out.reason == "schema_failed"
