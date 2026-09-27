"""A real phone photo, no reading models: capture and view splitting only."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from s2c.sketch.capture import Captured, capture

PHOTO = Path(__file__).resolve().parents[1] / "golden_sketch" / "real_bracket_1" / "image.jpg"


@pytest.fixture(scope="module")
def cap() -> Captured:
    out = capture(cv2.imread(str(PHOTO)))
    assert isinstance(out, Captured)
    return out


def test_paper_is_cropped_so_the_pen_width_is_the_drawn_one(cap):
    assert cap.stroke_px == pytest.approx(3.0, abs=0.6)


def test_spiral_and_desk_ink_are_dropped(cap):
    h, w = cap.ink.shape
    assert not cap.ink[:12].any()  # the other notebook's cover above the sheet
    rows = cap.ink[:, w - 40:].any(1).nonzero()[0]  # the binding ran the whole height; only "Ø8" is this far right
    assert rows.size == 0 or np.ptp(rows) < 0.25 * h
