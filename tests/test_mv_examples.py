from pathlib import Path

import pytest

from s2c.multiview.build import build, volume
from s2c.multiview.spec import MultiViewSpec

EXAMPLES = sorted((Path(__file__).parents[1] / "examples" / "mv").glob("*.json"))


@pytest.mark.parametrize("path", EXAMPLES, ids=[p.stem for p in EXAMPLES])
def test_example_specs_build(path):
    assert volume(build(MultiViewSpec.model_validate_json(path.read_text()))) > 0
