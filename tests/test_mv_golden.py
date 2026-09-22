"""Real captures in tests/golden_mv/<name>/: face images plus expected.json. Skips while the folder is empty.
GOLDEN_AI=1 runs the full pipeline (vision model, TrOCR, TripoSR) without the typed user_values."""
import json
import os
from pathlib import Path

import pytest

from s2c.multiview.pipeline import ImageInput, MvPipeline, default_pipeline
from s2c.multiview.spec import MvAbstain

GOLDEN = Path(__file__).parent / "golden_mv"
CASES = sorted(p for p in GOLDEN.iterdir() if (p / "expected.json").exists()) if GOLDEN.exists() else []


def within(got: float, want: float) -> bool:
    return abs(got - want) <= max(0.05 * abs(want), 1.0)


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_golden_mv_case(case: Path, tmp_path):
    e = json.loads((case / "expected.json").read_text())
    ai = os.environ.get("GOLDEN_AI") == "1"
    pipe = default_pipeline() if ai else MvPipeline()
    images = [ImageInput((case / i["file"]).read_bytes(), i["face"], i["kind"]) for i in e["images"]]
    observed = pipe.observe(images, e.get("reference"))
    assert not isinstance(observed, MvAbstain), observed
    spec = pipe.fuse(observed, {} if ai else e.get("user_values", {}))
    assert not isinstance(spec, MvAbstain), spec
    for axis in "xyz":
        got, want = getattr(spec.envelope, f"{axis}_mm"), e["envelope"][f"{axis}_mm"]
        assert within(got, want), f"{axis}: got {got}, expected {want}"
    assert len(spec.features) == e["hole_count"]
    built = pipe.build(spec, tmp_path, observed.masks)
    assert not isinstance(built, MvAbstain), built
    assert min(built.iou.values()) >= 0.85, built.iou
