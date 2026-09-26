"""Stand-in for the vision integrator's VLM chat transport (s2c/vision/client.py's
`Chat`). Returns a hardcoded Topology JSON string and never touches the network.
Delete once a real `Chat` callable exists.
"""


def chat(messages: list[dict]) -> str:
    return ('{"part_type": "plate", "view": "front", '
            '"holes": [{"u": 0.17, "v": 0.25, "kind": "through"}, {"u": 0.83, "v": 0.75, "kind": "through"}], '
            '"annotation_count": 4, "confidence": 0.9, "notes": "fake"}')
