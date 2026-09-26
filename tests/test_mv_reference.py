import cv2
import numpy as np
import pytest

from s2c.multiview.outline import extract
from s2c.multiview.reference import find_reference
from s2c.multiview.spec import MvAbstain


def scene():
    img = np.full((1200, 1600, 3), 255, np.uint8)
    cv2.rectangle(img, (200, 200), (900, 800), (60, 60, 60), -1)  # the part
    return img


def test_coin_gives_the_scale():
    img = scene()
    cv2.circle(img, (1300, 600), 100, (40, 40, 40), -1)  # a 1 TND coin, 200 px across
    r = find_reference(img, "1 TND")
    assert abs(r.mm_per_px - 0.125) / 0.125 < 0.02
    assert r.bbox[0] <= 1200 and r.bbox[0] + r.bbox[2] >= 1400


def test_tilted_coin_abstains():
    img = scene()
    cv2.ellipse(img, (1300, 600), (100, 60), 0, 0, 360, (40, 40, 40), -1)
    res = find_reference(img, "1 TND")
    assert isinstance(res, MvAbstain) and res.reason == "coin_tilted"


def test_missing_coin_abstains():
    res = find_reference(scene(), "2 EUR")
    assert isinstance(res, MvAbstain) and res.reason == "coin_not_found"


def test_card_gives_the_scale():
    img = scene()
    cv2.rectangle(img, (1000, 300), (1428, 570), (50, 50, 50), -1)  # 428 x 270 px, a card at 5 px/mm
    r = find_reference(img, "card")
    assert abs(r.mm_per_px - 0.2) < 0.004


def test_a4_sheet_is_rectified():
    img = np.full((1200, 1600, 3), 40, np.uint8)
    cv2.rectangle(img, (250, 150), (1350, 928), (245, 245, 245), -1)   # A4 at 3.704 px/mm
    cv2.rectangle(img, (600, 400), (822, 548), (30, 30, 30), -1)       # a 60 x 40 mm part on it
    r = find_reference(img, "a4")
    assert r.mm_per_px == pytest.approx(0.2)
    o = extract(r.image)
    assert abs(o.bbox[2] * r.mm_per_px - 60) < 1.5 and abs(o.bbox[3] * r.mm_per_px - 40) < 1.5


def test_unknown_reference_name():
    with pytest.raises(ValueError):
        find_reference(scene(), "banana")
