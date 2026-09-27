import cv2
import numpy as np

from s2c.sketch.classify import classify
from s2c.sketch.text import erase_mask, find_text_boxes, read_texts
from s2c.sketch.vectorize import vectorize
from tests.sketch.synth import FRONT_OUTLINE, F, R, Sheet, T, TruthReader, bridge_block


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


def visible_lines(c):
    return [p for p in c.visible if p.kind == "line"]


def test_stacked_dimensions_share_an_extension_line_without_leaving_an_edge():
    """The overall value's extension line crosses the inner dimension line; both pieces are extension
    line, none is an edge of the part (found end to end: the piece left over widened the view)."""
    def draw(sh):
        box(sh, 300, 300, 700, 500)
        sh.hdim(300, 450, 500, 545, "37.5")
        sh.hdim(300, 700, 500, 590, "100")
    c = run(draw)
    assert len(c.dimlines) == 2
    assert len(visible_lines(c)) == 4


def test_a_leader_drawn_across_an_outline_corner_stays_one_leader():
    """The SIDE view's leader crosses the outline at its corner: the piece beyond the crossing is
    still the leader, and the corner is neither an arrowhead nor a dimension line."""
    def draw(sh):
        box(sh, 300, 300, 500, 500)
        sh.circle((350, 350), 25)
        tip = np.array([350.0, 350.0]) + 25 * np.array([-0.707, -0.707])
        sh.leader(tip, (tip[0] - 64, tip[1] - 60), "Ø12.5", scale=0.6)
    c = run(draw)
    assert len(c.leaders) == 1 and c.dimlines == []
    tail = c.leaders[0].tail
    assert np.hypot(tail[0] - (332.3 - 64), tail[1] - (332.3 - 60)) < 10
    assert all(290 <= q[0] <= 510 and 290 <= q[1] <= 510 for p in visible_lines(c) for q in (p.p0, p.p1))


def test_a_dimension_that_lost_one_end_leaves_its_extension_line_out_of_the_edges():
    """Text written across a dimension line can cut off one end (the SIDE "50"); the line crossing
    the remaining tip is still an extension line, not an edge of the part."""
    def draw(sh):
        box(sh, 300, 300, 500, 500)
        sh.line((506, 500), (568, 500), 1)
        sh.line((560, 380), (560, 500), 1)
        sh.arrowhead((560, 500), (0, 1))
        sh.text("50", (590, 380))
    c = run(draw)
    assert len(visible_lines(c)) == 4


def outline_corners():
    """Corners of the part outline in the three views of `bridge_block`."""
    front = [F(*p) for p in FRONT_OUTLINE]
    top = [T(x, b) for x in (0, 25, 37.5, 62.5, 75, 100) for b in (0, 25)]
    side = [R(a, y) for a in (0, 25) for y in (0, 12.5, 50)]
    return np.float64(front + top + side)


def test_full_sheet_keeps_its_short_dimensions():
    """Text boxes that swallowed the short dimensions next to their text erased them: 5 of the 16
    drawn dimensions came out. Counted by matching both arrow tips, so a false dimension line does
    not make up for a lost one. The SIDE "12.5" is written across the "50" line, so erasing it cuts
    that line: the one accepted loss."""
    drawn = bridge_block(Sheet()).dims
    c = run(bridge_block)

    def found(p, q):
        return any(len(d.arrows) == 2 and min(
            max(np.hypot(*(a.tip - p)), np.hypot(*(b.tip - q))),
            max(np.hypot(*(a.tip - q)), np.hypot(*(b.tip - p)))) < 6
            for d in c.dimlines for a, b in [d.arrows])

    assert len(drawn) == 16
    assert sum(found(np.float64(p), np.float64(q)) for p, q in drawn) >= 15
    corners = outline_corners()
    for d in c.dimlines:  # a corner of the outline is not an arrowhead
        for a in d.arrows:
            assert np.min(np.hypot(*(corners - a.tip).T)) > 6, (d.id, a.tip)
