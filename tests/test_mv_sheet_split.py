"""One image holding several orthographic views is split into drawings and views (spec 3.1)."""
import cv2
import numpy as np
import pytest

from s2c.multiview.build import build
from s2c.multiview.sheet import is_sheet, split_sheet
from s2c.multiview.spec import Envelope
from tests.mv_helpers import box_mesh, make_spec, outline
from tests.sheet_helpers import draw_sheet, part_views

ENV = Envelope(x_mm=80.0, y_mm=60.0, z_mm=40.0)
FIVE = ("front", "top", "right", "left", "back")
# An L-bracket: an upright plate at the front with a base leg running back, an L seen from the side.
L_SIDE = [(0, 0), (40, 0), (40, 8), (8, 8), (8, 60), (0, 60)]
HOLE = {"type": "hole", "face": "front", "a_mm": 40.0, "b_mm": 36.0, "diameter_mm": 12.0}


@pytest.fixture(scope="module")
def bracket():
    views = part_views(build(make_spec((ENV.x_mm, ENV.y_mm, ENV.z_mm), right=outline(L_SIDE))), ENV)
    return {f: views[f] for f in FIVE}


@pytest.fixture(scope="module")
def holed_bracket():
    spec = make_spec((ENV.x_mm, ENV.y_mm, ENV.z_mm), right=outline(L_SIDE), features=[HOLE])
    views = part_views(build(spec), ENV)
    return {f: views[f] for f in FIVE}


def off(a, b) -> int:
    return max(abs(p - q) for p, q in zip(a, b))


def assert_views_match(views, truth, faces, labels=True):
    """Every drawn view has exactly one split view within 3 px of its box, and its label."""
    assert len(views) == len(faces)
    found = []
    for face in faces:
        view = min(views, key=lambda v: off(v.box, truth[face]))
        assert off(view.box, truth[face]) <= 3, (face, view.box, truth[face])
        if labels:
            assert view.label_box is not None, face
            assert off(view.label_box, truth[f"label:{face}"]) <= 3, (face, view.label_box)
        found.append(id(view))
    assert len(set(found)) == len(faces)


def test_a_first_angle_sheet_splits_into_its_views(bracket):
    img, truth = draw_sheet(bracket)
    sheet = split_sheet(img)
    assert sheet.shape == img.shape[:2]
    assert len(sheet.drawings) == 1
    assert_views_match(sheet.drawings[0].views, truth, FIVE)
    assert is_sheet(sheet)


def test_centre_lines_and_inner_circles_stay_inside_their_view(holed_bracket):
    img, truth = draw_sheet(holed_bracket, centre_lines=True)
    sheet = split_sheet(img)
    assert len(sheet.drawings) == 1
    assert_views_match(sheet.drawings[0].views, truth, FIVE)


def test_the_border_frame_is_removed(bracket):
    img, truth = draw_sheet(bracket, border=True)
    sheet = split_sheet(img)
    assert len(sheet.drawings) == 1
    assert_views_match(sheet.drawings[0].views, truth, FIVE)


def test_two_drawings_are_two_drawings(bracket):
    img, truth = draw_sheet(bracket, symbol="first")
    sheet = split_sheet(img)
    assert sorted(len(d.views) for d in sheet.drawings) == [2, 5]
    part = max(sheet.drawings, key=lambda d: len(d.views))
    symbol = min(sheet.drawings, key=lambda d: len(d.views))
    assert_views_match(part.views, truth, FIVE)
    assert_views_match(symbol.views, truth, ("symbol:side", "symbol:end"), labels=False)
    assert is_sheet(sheet)


def photo(blobs, shape=(600, 900)) -> np.ndarray:
    """A grey, noisy, unevenly lit background with dark filled blobs at (x, y)."""
    rng = np.random.default_rng(7)
    h, w = shape
    img = 170 + np.linspace(-12, 12, w)[None, :] + rng.normal(0, 4, (h, w))
    for mask, (x, y) in blobs:
        mh, mw = mask.shape
        img[y: y + mh, x: x + mw][mask > 127] = 60 + rng.normal(0, 6, int((mask > 127).sum()))
    return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def outlined(mask) -> np.ndarray:
    """A mask's outline only: the same shape as a line drawing."""
    lines = np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(lines, contours, -1, 255, 2)
    return lines


def test_a_single_part_photo_is_not_a_sheet(bracket):
    front, right = bracket["front"], bracket["right"]
    one = split_sheet(photo([(front, (300, 200))]))
    assert sum(len(d.views) for d in one.drawings) == 1
    assert not is_sheet(one)
    apart = split_sheet(photo([(front, (60, 40)), (right, (560, 380))]))
    assert sum(len(d.views) for d in apart.drawings) == 2
    assert not is_sheet(apart)
    drawn_apart = split_sheet(photo([(outlined(front), (60, 40)), (outlined(right), (560, 380))]))
    assert not is_sheet(drawn_apart)


def test_filled_blobs_in_a_row_are_not_a_sheet(bracket):
    """A part and its shadow, or a coin as tall as the part, line up without being views (Review Focus 1)."""
    front = bracket["front"]
    coin = np.zeros((front.shape[0], front.shape[0]), np.uint8)
    cv2.circle(coin, (coin.shape[1] // 2, coin.shape[0] // 2), coin.shape[0] // 2, 255, -1)
    sheet = split_sheet(photo([(front, (60, 200)), (coin, (500, 200))]))
    assert sum(len(d.views) for d in sheet.drawings) == 2
    assert not is_sheet(sheet)
    lines = split_sheet(photo([(outlined(front), (60, 200)), (outlined(coin), (500, 200))]))
    assert is_sheet(lines)


def stack(top: np.ndarray, bottom: np.ndarray, band: np.ndarray) -> np.ndarray:
    w = max(top.shape[1], bottom.shape[1])
    pad = [np.pad(im, ((0, 0), (0, w - im.shape[1]), (0, 0)), constant_values=255) for im in (top, bottom)]
    return np.vstack([pad[0], np.broadcast_to(band, (band.shape[0], w, 3)), pad[1]])


def test_a_black_separator_band_splits_drawings(bracket):
    sheet_img, _ = draw_sheet(bracket)
    pair_img, _ = draw_sheet({f: bracket[f] for f in ("front", "top")})
    black, white = np.zeros((25, 1, 3), np.uint8), np.full((25, 1, 3), 255, np.uint8)
    banded = split_sheet(stack(sheet_img, pair_img, black))
    assert sorted(len(d.views) for d in banded.drawings) == [2, 5]
    together = split_sheet(stack(sheet_img, pair_img, white))  # the band alone keeps them apart
    assert [len(d.views) for d in together.drawings] == [7]


def page(h=600, w=800) -> np.ndarray:
    return np.full((h, w, 3), 255, np.uint8)


def test_a_label_in_separate_words_is_one_label():
    """The words sit farther apart than the dilation joins, and "A" alone is too narrow to pass as a label."""
    img = page()
    cv2.rectangle(img, (200, 100), (600, 380), (0, 0, 0), 2)
    cv2.putText(img, "VIEW A", (310, 440), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    views = [v for d in split_sheet(img).drawings for v in d.views]
    assert len(views) == 1
    x, y, w, _ = views[0].label_box
    assert y > 380 and x < 320 and x + w > 440  # "VIEW" starts at 311, "A" ends at 444


def test_a_label_under_a_thin_view_is_its_label():
    """A plate's top view is as flat as a line of text; its label is still its label, not the other way round."""
    env = Envelope(x_mm=80.0, y_mm=60.0, z_mm=4.0)
    views = part_views(box_mesh(80.0, 60.0, 4.0), env)
    faces = ("front", "top", "right")
    img, truth = draw_sheet({f: views[f] for f in faces})
    sheet = split_sheet(img)
    assert len(sheet.drawings) == 1
    assert_views_match(sheet.drawings[0].views, truth, faces)


def test_a_round_view_filling_the_image_is_not_a_border():
    """A tight crop of a flange: thin and near all four sides, but round, so its holes never pass for views."""
    img = page(500, 500)
    cv2.circle(img, (250, 250), 240, (0, 0, 0), 2)
    cv2.circle(img, (250, 250), 60, (0, 0, 0), 2)
    for dx, dy in ((0, -150), (150, 0), (0, 150), (-150, 0)):
        cv2.circle(img, (250 + dx, 250 + dy), 30, (0, 0, 0), 2)
    sheet = split_sheet(img)
    assert [len(d.views) for d in sheet.drawings] == [1]
    assert not is_sheet(sheet)


def test_a_view_as_wide_as_the_image_is_not_cut_apart():
    """Its top and bottom edges ink over 90 % of their rows, but nothing lies beyond them, so they separate nothing."""
    img = page(500, 800)
    cv2.rectangle(img, (10, 150), (789, 350), (0, 0, 0), 2)
    cv2.circle(img, (200, 250), 40, (0, 0, 0), 2)
    cv2.circle(img, (600, 250), 40, (0, 0, 0), 2)
    sheet = split_sheet(img)
    assert [len(d.views) for d in sheet.drawings] == [1]
    assert not is_sheet(sheet)


def test_notes_beside_a_sketch_do_not_make_a_sheet():
    """Dimension numbers and item balloons line up with each other, but they are far smaller than the view."""
    img = page()
    cv2.rectangle(img, (150, 100), (600, 400), (0, 0, 0), 3)
    for text, org in (("80", (300, 470)), ("25", (420, 470)), ("50", (640, 260))):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    for text, x in (("1", 680), ("2", 740)):
        cv2.circle(img, (x, 150), 16, (0, 0, 0), 2)
        cv2.putText(img, text, (x - 6, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    sheet = split_sheet(img)
    assert sum(len(d.views) for d in sheet.drawings) == 6
    assert not is_sheet(sheet)
