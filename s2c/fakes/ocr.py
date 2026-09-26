"""Stand-in for the numbers owner's handwriting reader (s2c/ocr.py's
`read_annotations`). Returns hardcoded Annotation values that match the shared
test spec in tests/data/plate_60x40x5.json. Delete once the real OCR module lands.
"""
import numpy as np

from s2c.partspec.models import Annotation, Annotations, Topology


def read_annotations(image_bgr: np.ndarray, topology: Topology) -> Annotations:
    def a(value, kind, linked_to, hole_index=None):
        return Annotation(value_mm=value, kind=kind, bbox_px=(0.0, 0.0, 10.0, 10.0),
                          linked_to=linked_to, hole_index=hole_index, confidence=0.9)
    return Annotations(items=[
        a(60.0, "linear", "width"), a(40.0, "linear", "height"), a(5.0, "linear", "thickness"),
        a(6.0, "diameter", "hole_diameter"),
    ], confidence=0.9)
