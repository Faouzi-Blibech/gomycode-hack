import cv2
import numpy as np
import pytest

from s2c.multiview.fuse import Observation
from s2c.multiview.outline import extract, find_hidden_lines, ink_mask, resize_long_side
from s2c.multiview.pipeline import ImageInput, MvPipeline

HIDDEN = (16, 8)          # dash, gap
CHAIN = (30, 6, 6, 6)     # long dash, gap, short dash, gap


def page(h=1200, w=1600, color=255):
    return np.full((h, w, 3), color, np.uint8)


def pattern_line(img, p0, p1, pattern, thickness=2):
    """An axis-parallel line drawn as ink / gap lengths in px, repeated from p0 to p1."""
    (x0, y0), (x1, y1) = p0, p1
    length = max(abs(x1 - x0), abs(y1 - y0))
    ux, uy = int(np.sign(x1 - x0)), int(np.sign(y1 - y0))
    t, k = 0, 0
    while t < length:
        if k % 2 == 0:
            e = min(t + pattern[k % len(pattern)], length)
            cv2.line(img, (x0 + ux * t, y0 + uy * t), (x0 + ux * e, y0 + uy * e), (0, 0, 0), thickness)
        t += pattern[k % len(pattern)]
        k += 1


def centre_cross(img, cx, cy, r, beyond=15):
    """A centre mark: two continuous lines through the centre, a little longer than the circle."""
    cv2.line(img, (cx - r - beyond, cy), (cx + r + beyond, cy), (0, 0, 0), 2)
    cv2.line(img, (cx, cy - r - beyond), (cx, cy + r + beyond), (0, 0, 0), 2)


def dashed_view():
    img = page()
    cv2.rectangle(img, (400, 300), (1000, 700), (0, 0, 0), 2)
    for x in (550, 850):
        pattern_line(img, (x, 300), (x, 700), HIDDEN)
    return img


def digest(o):
    return (tuple(o.bbox), tuple(map(tuple, np.asarray(o.outer).tolist())),
            tuple(tuple(map(tuple, np.asarray(i).tolist())) for i in o.inner),
            tuple((round(c.cx, 2), round(c.cy, 2), round(c.d, 2)) for c in o.circles), o.circular)


def test_a_centre_line_does_not_stretch_the_outline():
    img = page()
    cv2.rectangle(img, (600, 500), (1000, 700), (0, 0, 0), 2)
    pattern_line(img, (520, 600), (1080, 600), CHAIN)
    o = extract(img, drawing=True)
    _, _, w, h = o.bbox
    assert o.line_art
    assert abs(w - 400) <= 6 and abs(h - 200) <= 6
    assert o.hidden == []  # a chain line is a centre line, not a hidden edge


def test_inner_visible_edges_are_not_openings():
    img = page()
    cv2.rectangle(img, (400, 300), (1200, 800), (0, 0, 0), 2)
    cv2.line(img, (800, 300), (800, 800), (0, 0, 0), 2)
    cv2.rectangle(img, (500, 400), (700, 600), (0, 0, 0), 2)
    o = extract(img, drawing=True)
    assert o.line_art and o.inner == [] and o.circles == []


def test_a_drawn_circle_is_still_a_circle():
    img = page()
    cv2.rectangle(img, (400, 300), (1200, 800), (0, 0, 0), 2)
    cv2.circle(img, (700, 550), 60, (0, 0, 0), 2)
    o = extract(img, drawing=True)
    assert len(o.circles) == 1
    c = o.circles[0]
    assert abs(c.cx - 700) <= 2 and abs(c.cy - 550) <= 2 and 112 <= c.d <= 122


def test_a_drawn_ellipse_is_not_a_circle():
    img = page()
    cv2.rectangle(img, (400, 300), (1200, 800), (0, 0, 0), 2)
    cv2.ellipse(img, (700, 550), (100, 65), 0, 0, 360, (0, 0, 0), 2)
    assert extract(img, drawing=True).circles == []


def test_a_circle_with_a_centre_cross_is_one_circle():
    img = page()
    cv2.rectangle(img, (400, 300), (1200, 800), (0, 0, 0), 2)
    cv2.circle(img, (700, 550), 60, (0, 0, 0), 2)
    centre_cross(img, 700, 550, 60)
    o = extract(img, drawing=True)
    assert len(o.circles) == 1
    c = o.circles[0]
    assert abs(c.cx - 700) <= 2 and abs(c.cy - 550) <= 2 and 110 <= c.d <= 122


def test_a_bore_in_a_boss_with_centre_lines_gives_both_circles():
    img = page()
    cv2.rectangle(img, (300, 200), (1300, 1000), (0, 0, 0), 2)
    cv2.circle(img, (800, 600), 200, (0, 0, 0), 2)
    cv2.circle(img, (800, 600), 80, (0, 0, 0), 2)
    centre_cross(img, 800, 600, 200)
    o = extract(img, drawing=True)
    small, large = sorted(c.d for c in o.circles)
    assert 150 <= small <= 162 and 390 <= large <= 402
    assert all(abs(c.cx - 800) <= 2 and abs(c.cy - 600) <= 2 for c in o.circles)


def test_a_round_view_with_a_bore_and_centre_lines():
    img = page()
    cv2.circle(img, (800, 600), 300, (0, 0, 0), 2)
    cv2.circle(img, (800, 600), 100, (0, 0, 0), 2)
    centre_cross(img, 800, 600, 300, beyond=40)
    o = extract(img, drawing=True)
    _, _, w, h = o.bbox
    assert o.circular and abs(w - 600) <= 6 and abs(h - 600) <= 6
    assert len(o.circles) == 1 and 190 <= o.circles[0].d <= 202


def test_a_crossed_circle_beside_a_split_face_is_found_once():
    img = page()
    cv2.rectangle(img, (200, 100), (1400, 1100), (0, 0, 0), 2)
    u = np.array([[400, 300], [1200, 300], [1200, 900], [1000, 900], [1000, 500], [600, 500], [600, 900], [400, 900]])
    cv2.polylines(img, [u], True, (0, 0, 0), 2)
    cv2.line(img, (800, 300), (800, 500), (0, 0, 0), 2)
    cv2.circle(img, (800, 720), 60, (0, 0, 0), 2)
    centre_cross(img, 800, 720, 60)
    o = extract(img, drawing=True)
    assert len(o.circles) == 1 and abs(o.circles[0].cy - 720) <= 2


def test_dashed_lines_are_found_as_hidden_lines():
    o = extract(dashed_view(), drawing=True)
    assert o.line_art and o.inner == [] and o.circles == []
    assert [h[0] for h in o.hidden] == ["v", "v"]
    assert sorted(h[1] for h in o.hidden) == [pytest.approx(0.25, abs=0.03), pytest.approx(0.75, abs=0.03)]
    assert all(0 <= s < e <= 1 and e - s > 0.8 for _, _, s, e in o.hidden)


def test_a_dashed_horizontal_line_is_measured_from_the_top_of_the_bbox():
    img = page()
    cv2.rectangle(img, (400, 300), (1000, 700), (0, 0, 0), 2)
    pattern_line(img, (400, 400), (1000, 400), HIDDEN)
    o = extract(img, drawing=True)
    assert len(o.hidden) == 1 and o.hidden[0][0] == "h"
    assert o.hidden[0][1] == pytest.approx(0.25, abs=0.03)


@pytest.mark.parametrize("line", [1, 2, 3])
def test_a_small_view_blown_up_to_full_size_reads_the_same(line):
    """A view cropped from a sheet is small; the pipeline scales it up about four times, lines and all."""
    img = page(280, 380)
    cv2.rectangle(img, (40, 40), (340, 240), (0, 0, 0), line)
    pattern_line(img, (20, 140), (360, 140), (12 * line, 3 * line, 3 * line, 3 * line), line)
    for x in (120, 260):
        pattern_line(img, (x, 40), (x, 240), (6 * line, 3 * line), line)
    cv2.circle(img, (190, 140), 25, (0, 0, 0), line)
    big = resize_long_side(img)
    s = big.shape[1] / 380
    o = extract(big, drawing=True)
    _, _, w, h = o.bbox
    assert abs(w - 300 * s) <= 0.02 * 300 * s and abs(h - 200 * s) <= 0.02 * 300 * s
    assert len(o.circles) == 1 and abs(o.circles[0].d - 50 * s) <= 0.08 * 50 * s
    assert sorted(round(v[1], 1) for v in o.hidden) == [0.3, 0.7] and {v[0] for v in o.hidden} == {"v"}


def test_find_hidden_lines_on_a_bare_ink_mask():
    ink = ink_mask(dashed_view())
    hidden = find_hidden_lines(ink, (399, 299, 603, 403))
    assert sorted(round(h[1], 2) for h in hidden) == [pytest.approx(0.25, abs=0.03), pytest.approx(0.75, abs=0.03)]


def filled_part(tones=((90, 100, 120),)):
    img = page()
    for k, tone in enumerate(tones):  # one band per shaded face, top to bottom, like a render
        y0 = 300 + k * 400 // len(tones)
        cv2.rectangle(img, (400, y0), (1000, 300 + (k + 1) * 400 // len(tones)), tone, -1)
    cv2.rectangle(img, (600, 480), (800, 520), (255, 255, 255), -1)  # see-through slot
    cv2.circle(img, (500, 400), 30, (255, 255, 255), -1)             # through hole
    return img


@pytest.mark.parametrize("tones", [((90, 100, 120),), ((180, 170, 150), (120, 110, 95))])
def test_a_filled_render_is_unchanged_in_drawing_mode(tones):
    img = filled_part(tones)
    plain, drawn = extract(img), extract(img, drawing=True)
    assert not drawn.line_art and drawn.hidden == []
    assert digest(drawn) == digest(plain)
    assert len(drawn.inner) == 1 and len(drawn.circles) == 1


def test_a_thin_rim_render_keeps_its_outline_in_drawing_mode():
    """A render of a thin toothed rim has line-art fill, but nothing sticks out of it: its outline stays as found."""
    img = page(1600, 1600)
    rim = (160, 140, 128)
    cv2.circle(img, (800, 800), 720, rim, -1)
    for a in np.linspace(0, 2 * np.pi, 64, endpoint=False):
        cv2.circle(img, (round(800 + 722 * np.cos(a)), round(800 + 722 * np.sin(a))), 6, rim, -1)
    cv2.circle(img, (800, 800), 700, (255, 255, 255), -1)
    plain, drawn = extract(img), extract(img, drawing=True)
    assert drawn.line_art and digest(drawn) == digest(plain)


def sketch_fixtures():
    """The sketches and photos of tests/test_mv_outline.py, with their outlines from before line art existed."""
    a = page()
    cv2.rectangle(a, (400, 300), (1000, 700), (0, 0, 0), 4)
    b = a.copy()
    cv2.circle(b, (500, 400), 40, (0, 0, 0), 3)
    cv2.circle(b, (900, 600), 40, (0, 0, 0), 3)
    c = page()
    cv2.rectangle(c, (400, 300), (1000, 700), (60, 60, 60), -1)
    cv2.circle(c, (500, 400), 30, (255, 255, 255), -1)
    d = page()
    cv2.rectangle(d, (400, 300), (1000, 700), (60, 60, 60), -1)
    cv2.rectangle(d, (600, 480), (800, 520), (255, 255, 255), -1)
    e = page(color=30)
    cv2.rectangle(e, (400, 300), (1000, 700), (220, 220, 220), -1)
    f = page(1600, 1600)
    cv2.rectangle(f, (75, 785), (75 + 1450, 785 + 30), (128, 128, 128), -1)
    drawn = ((398, 298, 605, 405), ((398, 300), (400, 702), (1002, 700), (1000, 298)))
    filled = ((400, 300, 601, 401), ((400, 301), (401, 700), (1000, 699), (999, 300)))
    return [
        (a, (*drawn, (), (), False)),
        (b, (*drawn, (), ((499.99, 399.99, 74.66), (899.99, 599.99, 74.66)), False)),
        (c, (*filled, (), ((500.0, 400.0, 59.89),), False)),
        (d, (*filled, (((600, 481), (601, 520), (800, 519), (799, 480)),), (), False)),
        (e, (*filled, (), (), False)),
        (f, ((75, 785, 1451, 31), ((75, 786), (76, 815), (1525, 814), (1524, 785)), (), (), False)),
    ]


def test_sketches_are_unchanged():
    for img, expected in sketch_fixtures():
        o = extract(img)
        assert digest(o) == expected
        assert not o.line_art and o.hidden == []


def png(img):
    return cv2.imencode(".png", img)[1].tobytes()


def test_only_kind_drawing_reads_line_art_and_the_observation_carries_it():
    data = png(dashed_view())
    drawn = MvPipeline().observe([ImageInput(data, "front", "drawing")]).observations[0]
    sketched = MvPipeline().observe([ImageInput(data, "front", "sketch")]).observations[0]
    assert drawn.line_art and len(drawn.hidden) == 2
    assert not sketched.line_art and sketched.hidden == []
    direct = Observation(face="front", kind="drawing", outline=extract(dashed_view(), drawing=True))
    assert direct.line_art and direct.hidden == direct.outline.hidden
