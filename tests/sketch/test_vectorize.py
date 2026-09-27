from itertools import pairwise

import numpy as np

from s2c.sketch.vectorize import circle_fit, trace, vectorize
from tests.sketch.synth import Sheet


def prims_of(draw):
    sh = Sheet()
    draw(sh)
    return vectorize(sh.ink(), 3.0, "front")


def near(p, q, tol=5):
    return np.linalg.norm(np.asarray(p, float) - np.asarray(q, float)) <= tol


def test_rectangle_gives_four_lines_at_the_corners():
    def draw(sh):
        for p, q in [((200, 200), (600, 200)), ((600, 200), (600, 450)),
                     ((600, 450), (200, 450)), ((200, 450), (200, 200))]:
            sh.line(p, q)
    prims = prims_of(draw)
    lines = [p for p in prims if p.kind == "line"]
    assert len(lines) == 4
    corners = [(200, 200), (600, 200), (600, 450), (200, 450)]
    for c in corners:
        assert sum(near(l.p0, c) or near(l.p1, c) for l in lines) == 2


def test_circle():
    prims = prims_of(lambda sh: sh.circle((500, 400), 40))
    assert len(prims) == 1 and prims[0].kind == "circle"
    assert near(prims[0].center, (500, 400), 2) and abs(prims[0].radius - 40) < 3


def test_arc():
    import cv2

    def draw(sh):
        cv2.ellipse(sh.img, (500, 400), (80, 80), 0, 0, 120, (25, 25, 25), 3, cv2.LINE_AA)
    prims = prims_of(draw)
    assert len(prims) == 1 and prims[0].kind == "arc" and abs(prims[0].radius - 80) < 4


def test_dashed_line_becomes_short_collinear_pieces():
    prims = prims_of(lambda sh: sh.dashed((200, 300), (600, 300)))
    assert len(prims) >= 10
    assert all(p.kind == "line" and p.length < 20 for p in prims)
    assert all(abs(p.p0[1] - 300) < 3 and abs(p.p1[1] - 300) < 3 for p in prims)


def test_t_junction_splits_into_three_lines():
    def draw(sh):
        sh.line((200, 300), (600, 300))
        sh.line((400, 300), (400, 500))
    prims = prims_of(draw)
    assert len([p for p in prims if p.kind == "line" and p.length > 50]) == 3


def test_smooth_free_curve_is_a_curve():
    def draw(sh):
        xs = np.linspace(200, 800, 200)
        pts = np.stack([xs, 400 + 60 * np.sin((xs - 200) / 600 * 2 * np.pi)], 1)
        for p, q in pairwise(pts):
            sh.line(p, q)
    prims = prims_of(draw)
    assert [p.kind for p in prims] == ["curve"]


def test_circle_fit_is_exact_on_a_circle():
    t = np.linspace(0, 2 * np.pi, 50)
    c, r, rms = circle_fit(np.stack([10 + 5 * np.cos(t), 20 + 5 * np.sin(t)], 1))
    assert np.allclose(c, [10, 20]) and abs(r - 5) < 1e-9 and rms < 1e-9


def test_trace_of_empty_mask_is_empty():
    assert trace(np.zeros((50, 50), np.uint8)) == []
