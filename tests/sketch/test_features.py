import numpy as np
import pytest

from s2c.sketch.classify import Classified
from s2c.sketch.features import find_holes
from s2c.sketch.grammar import parse_text
from s2c.sketch.link import Link, build_refs
from s2c.sketch.models import Reading
from s2c.sketch.solve import solve
from s2c.sketch.text import TextItem
from s2c.sketch.vectorize import Prim


def seg(pid, p, q):
    return Prim(pid, "line", np.array([p, q], float), 3.0)


def rect(prefix, x0, y0, x1, y1):
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return [seg(f"{prefix}{k}", p, q) for k, (p, q) in enumerate(zip(corners, corners[1:] + corners[:1]))]


def item(tid, s):
    p = parse_text(s)
    return TextItem(tid, (0, 0, 9, 9), [Reading(reader="a", text=s, confidence=0.95)], p, None,
                    "dimension", "written", 0.95, [p.value])


def lin(refs, view, axis, px0, px1, s):
    rs = refs[view][axis]
    r0 = min(rs, key=lambda r: abs(r.px - px0))
    r1 = min(rs, key=lambda r: abs(r.px - px1))
    return Link(item(f"{view}{axis}{s}", s), view, "linear", axis, (r0.id, r1.id), [], [],
                "dimension_line", 0.95)


def setup(pair_y=(820, 900), with_pair=True):
    """FRONT 100 x 20 mm, TOP 100 x 40 mm, 4 px per mm; a Ø10 hole at X 50, 20 mm from the front face."""
    front = Classified(visible=rect("front-v", 200, 820, 600, 900))
    if with_pair:
        front.hidden = [seg("front-h0", (380, pair_y[0]), (380, pair_y[1])),
                        seg("front-h1", (420, pair_y[0]), (420, pair_y[1]))]
    circle = Prim("top-c", "circle", np.zeros((3, 2)), 3.0, (400.0, 440.0), 20.0)
    top = Classified(visible=rect("top-v", 200, 360, 600, 520) + [circle])
    cls = {"front": front, "top": top}
    refs = {v: build_refs(v, c) for v, c in cls.items()}
    links = [lin(refs, "front", "a", 200, 600, "100"), lin(refs, "front", "b", -900, -820, "20"),
             lin(refs, "top", "b", -520, -360, "40"),
             Link(item("dia", "Ø10"), "top", "diameter", None, None, ["top-c"], ["top-c"], "leader", 0.95)]
    solved = solve(refs, links, {"top-c": circle})
    return cls, refs, solved


def test_through_hole_from_a_circle_and_a_full_hidden_pair():
    cls, refs, solved = setup()
    feats, issues = find_holes(cls, refs, solved)
    assert len(feats) == 1 and issues == []
    h = feats[0]
    assert h.type == "hole" and h.axis == "y" and h.through is True and h.depth is None
    assert h.diameter == pytest.approx(10) and h.badge == "written"
    assert h.position_mm == pytest.approx((50, 20, 20), abs=0.3)
    assert set(h.evidence) == {"top-c", "front-h0", "front-h1"}


def test_blind_hole_depth_from_the_hidden_pair():
    cls, refs, solved = setup(pair_y=(820, 860))
    feats, _ = find_holes(cls, refs, solved)
    assert feats[0].through is False and feats[0].depth == pytest.approx(10, abs=0.3)


def test_circle_without_hidden_lines_is_a_predicted_through_hole():
    cls, refs, solved = setup(with_pair=False)
    feats, issues = find_holes(cls, refs, solved)
    assert feats[0].badge == "predicted" and feats[0].through is True
    assert [i.kind for i in issues] == ["predicted"]
