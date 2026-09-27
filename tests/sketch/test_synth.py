"""Smoke test for synthetic sheets."""
from s2c.reading import Crop
from tests.sketch.synth import Sheet, TruthReader, bridge_block


def test_bridge_block_creates_synthetic_sheet():
    sh = bridge_block(Sheet())
    # Assert 21 texts recorded: 18 values + 3 labels
    assert len(sh.texts) == 21
    # Assert ink is present
    assert sh.ink().max() > 0
    # Assert TruthReader can read one known box
    first_text, first_box = sh.texts[0]
    x, y, w, h = first_box
    crop = Crop(sh.bgr()[y:y + h, x:x + w], first_box)
    reader = TruthReader(sh.texts)
    result = reader.read([crop])
    assert len(result) == 1
    assert result[0].text == first_text
