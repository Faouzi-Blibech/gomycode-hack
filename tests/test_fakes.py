from pathlib import Path

import numpy as np

from s2c.fakes import builder, metrology, ocr, views, vision
from s2c.partspec.models import Annotations, Measurements, PartSpec, Topology

DATA_DIR = Path(__file__).resolve().parent / "data"


def test_fake_stages_produce_valid_contracts(tmp_path):
    img = np.zeros((10, 10, 3), np.uint8)
    topo = Topology.model_validate_json(vision.chat([]))
    assert topo.part_type == "plate"
    assert isinstance(ocr.read_annotations(img, topo), Annotations)
    assert isinstance(metrology.measure(img), Measurements)
    spec = PartSpec.model_validate_json((DATA_DIR / "plate_60x40x5.json").read_text())
    solid = builder.build(spec)
    step, stl = builder.export(solid, tmp_path)
    assert step.exists() and stl.exists() and stl.read_text().startswith("solid")
    masks = views.silhouettes(solid)
    assert set(masks) == {"front", "back", "left", "right", "top", "bottom"}
    assert masks["front"].shape == (512, 512)
