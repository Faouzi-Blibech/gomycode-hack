"""Stand-in for the numbers owner's coin metrology reader (s2c/metrology.py's
`measure`). Returns hardcoded Measurements that match the shared test spec in
tests/data/plate_60x40x5.json. Delete once the real metrology module lands.
"""
import numpy as np

from s2c.partspec.models import Abstain, Bbox, Circle, Coin, Measurements


def measure(image_bgr: np.ndarray) -> Measurements | Abstain:
    return Measurements(
        mm_per_px=0.1,
        coin=Coin(name="1 TND", pixel_diameter=250.0, eccentricity=0.05, confidence=0.95),
        outer_contour_mm=[(0.0, 0.0), (60.0, 0.0), (60.0, 40.0), (0.0, 40.0)],
        bbox_mm=Bbox(width=60.0, height=40.0),
        circles_mm=[Circle(x=10.0, y=10.0, diameter=6.0), Circle(x=50.0, y=30.0, diameter=6.0)],
        confidence=0.9,
    )
