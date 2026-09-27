import numpy as np
import pytest

from s2c.sketch.grammar import parse_text
from s2c.sketch.link import Link, Ref
from s2c.sketch.models import Reading
from s2c.sketch.solve import px_to_mm, solve
from s2c.sketch.text import TextItem
from s2c.sketch.vectorize import Prim


def refs_of(view, axis, pxs, sources=None):
    return [Ref(f"{view}.{axis}{k}", view, axis, float(p), (sources or {}).get(k, []))
            for k, p in enumerate(pxs)]


def lin(view, axis, refs, i, j, s, badge="written", candidates=None):
    p = parse_text(s)
    t = TextItem(f"{view}.{axis}.{i}-{j}", (0, 0, 10, 10),
                 [Reading(reader="a", text=s, confidence=0.95)], p, None, "dimension", badge, 0.95,
                 candidates or [p.value])
    return Link(t, view, "linear", axis, (refs[i].id, refs[j].id), [], [], "dimension_line", 0.95)


FRONT_A = [200, 300, 350, 450, 500, 600]      # 4 px per mm: 0 25 37.5 62.5 75 100
FRONT_B = [-900, -850, -800, -700]            # 0 12.5 25 50


def front(values=None, badges=None, candidates=None):
    a, b = refs_of("front", "a", FRONT_A), refs_of("front", "b", FRONT_B)
    spec = [(0, 5, "100"), (0, 2, "37.5"), (3, 5, "37.5"), (1, 2, "12.5"), (2, 3, "25"), (3, 4, "12.5")]
    values, badges, candidates = values or {}, badges or {}, candidates or {}
    links = [lin("front", "a", a, i, j, values.get(k, s), badges.get(k, "written"), candidates.get(k))
             for k, (i, j, s) in enumerate(spec)]
    links += [lin("front", "b", b, 1, 3, "37.5"), lin("front", "b", b, 2, 3, "25"),
              lin("front", "b", b, 1, 2, "12.5"), lin("front", "b", b, 0, 1, "12.5")]
    return {"front": {"a": a, "b": b}}, links


def test_front_solves_exactly_and_the_upright_start_is_derived():
    views, links = front()
    s = solve(views, links, {})
    got = [s.mm[f"front.a{k}"] for k in range(6)]
    assert got == pytest.approx([0, 25, 37.5, 62.5, 75, 100], abs=1e-3)
    assert all(d.badge == "written" for d in s.dims.values())
    assert s.gaps == []                                  # every gap follows from written values
    assert s.envelope["x"].value == pytest.approx(100) and s.envelope["x"].badge == "written"
    assert s.envelope["y"].value == pytest.approx(50) and s.envelope["y"].badge == "derived"


def test_a_conflict_marks_the_value_that_disagrees_with_the_drawing():
    views, links = front(values={4: "35"})              # 37.5 + 35 + 37.5 != 100
    s = solve(views, links, {})
    bad = [k for k, d in s.dims.items() if d.badge == "conflict"]
    assert bad == ["front.a.2-3"]
    assert any(i.kind == "conflict" and i.severity == "red" for i in s.issues)


def test_an_uncertain_value_gets_the_candidate_the_geometry_implies():
    views, links = front(badges={4: "uncertain"}, candidates={4: [85.0, 25.0]}, values={4: "85"})
    s = solve(views, links, {})
    d = s.dims["front.a.2-3"]
    assert d.badge == "uncertain" and d.value == pytest.approx(25)
    assert d.implied == pytest.approx(25, abs=0.5)


def test_the_decimal_trap_suggests_the_scaled_value():
    a = refs_of("front", "a", [0, 100, 200, 300])
    links = [lin("front", "a", a, 0, 1, "50"), lin("front", "a", a, 1, 2, "50"),
             lin("front", "a", a, 2, 3, "5")]
    s = solve({"front": {"a": a, "b": refs_of("front", "b", [-100, 0])}}, links, {})
    issue = next(i for i in s.issues if i.kind == "decimal")
    assert "50" in issue.message and issue.targets == ["front.a.2-3"]


def test_the_top_width_is_derived_across_views():
    views, links = front()
    views["top"] = {"a": refs_of("top", "a", [200, 250, 300, 350, 450, 500, 550, 600]),
                    "b": refs_of("top", "b", [-520, -470, -420])}
    s = solve(views, links, {})
    assert s.mm["top.a7"] == pytest.approx(100, abs=1e-3)
    assert s.mm["top.a1"] == pytest.approx(12.5, abs=0.5)     # not written: proportion only
    assert any(g.refs == ("top.a0", "top.a1") for g in s.gaps)


def test_an_unwritten_axis_is_predicted_from_proportions():
    a = refs_of("front", "a", [200, 600])
    b = refs_of("front", "b", [-900, -700])
    s = solve({"front": {"a": a, "b": b}}, [lin("front", "a", a, 0, 1, "100")], {})
    assert s.envelope["y"].badge == "predicted" and s.envelope["y"].evidence == "proportion"
    assert s.envelope["y"].value == pytest.approx(50)
    assert "z" not in s.envelope and any(i.kind == "predicted" for i in s.issues)


def test_a_circle_diameter_fixes_its_width():
    c = Prim("top-p9", "circle", np.zeros((3, 2)), 3, (250.0, 470.0), 25.0)
    a = refs_of("top", "a", [200, 225, 250, 275, 600], {1: ["top-p9"], 2: ["top-p9"], 3: ["top-p9"]})
    b = refs_of("top", "b", [-520, -495, -470, -445, -420], {1: ["top-p9"], 2: ["top-p9"], 3: ["top-p9"]})
    t = TextItem("t", (0, 0, 9, 9), [Reading(reader="a", text="Ø12.5", confidence=0.9)],
                 parse_text("Ø12.5"), None, "dimension", "written", 0.9, [12.5])
    links = [Link(t, "top", "diameter", None, None, ["top-p9"], ["top-p9"], "leader", 0.9),
             lin("top", "a", a, 0, 4, "100")]
    s = solve({"top": {"a": a, "b": b}}, links, {"top-p9": c})
    assert s.mm["top.a3"] - s.mm["top.a1"] == pytest.approx(12.5, abs=1e-3)
    assert s.circle_d["top-p9"] == (pytest.approx(12.5), "written")


def test_a_mirrored_left_view_matches_the_right_view():
    right = refs_of("right", "a", [800, 850, 900])     # distance from the front face grows rightward
    left = refs_of("left", "a", [0, 50, 100])          # mirrored: the front face is on its right
    y = refs_of("right", "b", [-900, -700])
    ly = refs_of("left", "b", [-900, -700])
    links = [lin("right", "a", right, 0, 2, "25"), lin("left", "a", left, 1, 2, "15")]
    s = solve({"right": {"a": right, "b": y}, "left": {"a": left, "b": ly}}, links, {})
    # left a1..a2 (near the front face) mirrors right a0..a1
    assert s.mm["right.a1"] == pytest.approx(15, abs=1e-3)


def test_px_to_mm_interpolates_between_refs():
    views, links = front()
    s = solve(views, links, {})
    assert px_to_mm(s, views["front"]["a"], "front", "a", 400) == pytest.approx(50, abs=1e-3)
    assert px_to_mm(s, views["front"]["a"], "front", "a", 640) == pytest.approx(110, abs=0.5)
