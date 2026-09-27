import cv2
import numpy as np

from s2c.sketch.capture import LONG_SIDE, Captured, capture, sheet_to_original
from s2c.sketch.models import SketchAbstain


def page(h=1130, w=800):
    img = np.full((h, w, 3), 245, np.uint8)
    cv2.rectangle(img, (200, 300), (600, 700), (20, 20, 20), 4)
    cv2.line(img, (100, 900), (700, 900), (20, 20, 20), 4)
    return img


def on_table(pg, corners):
    h, w = pg.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, np.float32(corners))
    canvas = cv2.warpPerspective(pg, M, (1600, 1200), borderValue=(70, 70, 70))
    return canvas


def test_rectifies_a_tilted_sheet():
    tl = (300, 150)
    img = on_table(page(), [tl, (1150, 190), (1120, 1150), (270, 1100)])
    out = capture(img)
    assert isinstance(out, Captured)
    assert max(out.sheet.shape[:2]) == LONG_SIDE
    frac = (out.ink > 0).mean()
    assert 0.002 < frac < 0.1
    back = sheet_to_original(np.float32([[0, 0]]), out.to_original)[0]
    assert abs(back[0] - tl[0]) < 8 and abs(back[1] - tl[1]) < 8
    assert 2 <= out.stroke_px <= 12


def test_full_frame_sheet_falls_back_to_whole_image():
    img = cv2.resize(page(), (1131, 1600))
    out = capture(img)
    assert isinstance(out, Captured)
    back = sheet_to_original(np.float32([[500, 700]]), out.to_original)[0]
    assert np.allclose(back, [500 * 1131 / out.sheet.shape[1], 700 * 1600 / out.sheet.shape[0]], atol=3)


def test_shadow_keeps_lines_and_ignores_the_shadow():
    img = cv2.resize(page(), (1131, 1600)).astype(np.float32)
    gradient = np.linspace(0.45, 1.0, img.shape[1])[None, :, None]
    img = (img * gradient).astype(np.uint8)
    out = capture(img)
    assert isinstance(out, Captured)
    ink = out.ink > 0
    sx = out.sheet.shape[1] / 1131
    # the rectangle's left side at x=200*1131/800 sits in the dark half
    x = int(200 * 1131 / 800 * sx)
    assert ink[int(500 * 1600 / 1130 * sx), x - 6: x + 6].any()
    blank = ink[int(150 * sx): int(350 * sx), int(40 * sx): int(240 * sx)]
    assert blank.mean() < 0.01


def test_dark_photo_abstains():
    out = capture(np.full((1200, 1600, 3), 25, np.uint8))
    assert isinstance(out, SketchAbstain) and out.reason == "image_quality"


def test_no_sheet_abstains():
    img = np.full((1200, 1600, 3), 110, np.uint8)
    cv2.circle(img, (800, 600), 200, (40, 40, 40), -1)
    out = capture(img)
    assert isinstance(out, SketchAbstain) and out.reason == "sheet_not_found"


def test_blurry_photo_abstains():
    img = cv2.GaussianBlur(cv2.resize(page(), (1131, 1600)), (61, 61), 20)
    out = capture(img)
    assert isinstance(out, SketchAbstain) and out.reason == "image_quality"


def test_large_photo_is_resized():
    img = cv2.resize(page(), (2830, 4000))
    out = capture(img)
    assert isinstance(out, Captured) and max(out.sheet.shape[:2]) == LONG_SIDE
