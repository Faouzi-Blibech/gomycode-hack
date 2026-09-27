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


def test_views_split_into_top_front_and_right_despite_the_miter_line_and_bridging_dimensions(cap):
    from s2c.sketch.views import split_views

    views, _ = split_views(cap.ink, [], cap.stroke_px)
    by_name = {v.name: v.bbox for v in views}
    assert sorted(by_name) == ["front", "right", "top"]
    top, front, right = by_name["top"], by_name["front"], by_name["right"]
    assert top[1] + top[3] <= front[1] + 0.2 * front[3]      # top above front
    assert right[0] >= front[0] + 0.8 * front[2]              # right beside front, not merged into it
