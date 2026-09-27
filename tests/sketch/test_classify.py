import cv2
import numpy as np

from s2c.sketch.classify import classify
from s2c.sketch.text import erase_mask, find_text_boxes, read_texts
from s2c.sketch.vectorize import vectorize
from tests.sketch.synth import Sheet, TruthReader


def run(draw):
    sh = Sheet()
    draw(sh)
    ink = sh.ink()
    texts = read_texts(sh.bgr(), find_text_boxes(ink, 3.0),
                       [TruthReader(sh.texts, "a"), TruthReader(sh.texts, "b")])
    geo = cv2.bitwise_and(ink, cv2.bitwise_not(erase_mask(texts, ink.shape)))
    return classify(vectorize(geo, 3.0, "front"), geo, 3.0, texts)


def box(sh, x0, y0, x1, y1):
    for p, q in [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]:
        sh.line(p, q)


def test_horizontal_dimension_with_extension_lines():
    def draw(sh):
        box(sh, 300, 300, 700, 500)
        sh.hdim(300, 700, 500, 560, "100")
    c = run(draw)
    assert len(c.dimlines) == 1
    d = c.dimlines[0]
    assert d.horizontal and len(d.arrows) == 2
    tips = sorted(a.tip[0] for a in d.arrows)
    assert abs(tips[0] - 300) < 5 and abs(tips[1] - 700) < 5
    assert set(c.extensions) == {f"{d.id}:0", f"{d.id}:1"}
    assert len([p for p in c.visible if p.kind == "line"]) == 4


def test_dashed_line_is_one_hidden_line():
    c = run(lambda sh: (box(sh, 300, 300, 700, 500), sh.dashed((400, 300), (400, 500))))
    assert len(c.hidden) == 1
    h = c.hidden[0]
    assert abs(h.p0[0] - 400) < 4 and abs(h.p1[0] - 400) < 4
    assert abs(min(h.p0[1], h.p1[1]) - 300) < 12 and abs(max(h.p0[1], h.p1[1]) - 500) < 12


def test_leader_to_a_circle():
    def draw(sh):
        box(sh, 300, 300, 700, 500)
        sh.circle((500, 400), 30)
        tip = np.array([500, 400]) + 30 * np.array([0.707, -0.707])
        sh.leader(tip, (tip[0] + 60, tip[1] - 60), "Ø15")
    c = run(draw)
    assert len(c.leaders) == 1
    tip = c.leaders[0].tip
    assert abs(np.hypot(tip[0] - 500, tip[1] - 400) - 30) < 6
    assert any(p.kind == "circle" for p in c.visible)


def test_chained_dimensions_share_their_middle_tip():
    def draw(sh):
        box(sh, 300, 300, 700, 500)
        sh.hdim(300, 450, 300, 250, "30")
        sh.hdim(450, 700, 300, 250, "50")
    c = run(draw)
    assert len(c.dimlines) == 2
    tips = sorted(round(a.tip[0]) for d in c.dimlines for a in d.arrows)
    assert len(tips) == 4 and abs(tips[1] - 450) < 6 and abs(tips[2] - 450) < 6


def test_open_v_arrowheads_count():
    def draw(sh):
        box(sh, 300, 300, 700, 500)
        for x in (300, 700):
            sh.line((x, 506), (x, 570), 1)
        sh.line((300, 560), (700, 560), 1)
        for tip, s in (((300, 560), 1), ((700, 560), -1)):
            sh.line(tip, (tip[0] + s * 18, tip[1] - 7), 1)
            sh.line(tip, (tip[0] + s * 18, tip[1] + 7), 1)
        sh.text("100", (500, 540))
    c = run(draw)
    assert len(c.dimlines) == 1 and len(c.dimlines[0].arrows) == 2


def test_a_lone_short_edge_is_visible_not_hidden():
    c = run(lambda sh: box(sh, 300, 300, 330, 320))
    assert c.hidden == [] and len(c.visible) == 4
