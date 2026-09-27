import numpy as np

from s2c.sketch.classify import Arrow, Classified, DimLine, Leader, classify
from s2c.sketch.grammar import parse_text
from s2c.sketch.link import build_refs, link_view
from s2c.sketch.models import Reading
from s2c.sketch.text import TextItem, find_text_boxes, read_texts
from s2c.sketch.vectorize import Prim, vectorize
from s2c.sketch.views import split_views
from tests.sketch.synth import FRONT_OUTLINE, F, Sheet, TruthReader, bridge_block


def line(pid, p, q, width=3.0):
    return Prim(pid, "line", np.array([p, q], float), width)


def text(tid, s, centre, w=50, h=24):
    box = (int(centre[0] - w / 2), int(centre[1] - h / 2), w, h)
    return TextItem(tid, box, [Reading(reader="a", text=s, confidence=0.95)], parse_text(s),
                    None, "dimension", "written", 0.95, [parse_text(s).value])


def hdim(did, x1, x2, y_obj, y_line):
    d = DimLine(did, np.array([x1, y_line], float), np.array([x2, y_line], float),
                [Arrow(np.array([x1, y_line], float), np.array([-1.0, 0]), did),
                 Arrow(np.array([x2, y_line], float), np.array([1.0, 0]), did)], [did])
    ext = {f"{did}:0": line(did + "e0", (x1, y_obj + 6), (x1, y_line + 8), 1),
           f"{did}:1": line(did + "e1", (x2, y_obj + 6), (x2, y_line + 8), 1)}
    return d, ext


def front_outline():
    pts = [F(*p) for p in FRONT_OUTLINE]
    return [line(f"v{k}", p, q) for k, (p, q) in enumerate(zip(pts, pts[1:] + pts[:1]))]


def test_refs_merge_positions_closer_than_the_tolerance():
    c = Classified(visible=[line("a", (300, 100), (300, 400)), line("b", (302, 100), (302, 400)),
                            line("c", (300, 400), (600, 400))])
    refs = build_refs("front", c)
    assert [round(r.px) for r in refs["a"]] == [301, 600]
    assert set(refs["a"][0].sources) == {"a", "b", "c"}
    assert [round(r.px) for r in refs["b"]] == [-400, -100]


def test_overall_and_chain_values_follow_their_extension_lines():
    d1, e1 = hdim("d1", 200, 600, 900, 990)
    d2, e2 = hdim("d2", 300, 350, 700, 660)
    d3, e3 = hdim("d3", 350, 450, 700, 660)
    c = Classified(visible=front_outline(), dimlines=[d1, d2, d3], extensions={**e1, **e2, **e3})
    texts = [text("t1", "100", (400, 974)), text("t2", "12.5", (325, 644)), text("t3", "25", (400, 644))]
    refs, links, issues = link_view("front", c, texts, 3.0)
    px = {r.id: r.px for r in refs["a"]}
    by = {l.text.id: l for l in links}
    assert [round(px[r]) for r in by["t1"].refs] == [200, 600]
    assert [round(px[r]) for r in by["t2"].refs] == [300, 350]
    assert [round(px[r]) for r in by["t3"].refs] == [350, 450]
    assert by["t2"].refs[1] == by["t3"].refs[0]          # the chain shares a position
    assert all(l.how == "dimension_line" and l.axis == "a" for l in links) and issues == []


def test_vertical_value_links_on_axis_b():
    d = DimLine("dv", np.array([150, 700.0]), np.array([150, 850.0]),
                [Arrow(np.array([150, 700.0]), np.array([0, -1.0]), "dv"),
                 Arrow(np.array([150, 850.0]), np.array([0, 1.0]), "dv")], ["dv"])
    ext = {"dv:0": line("x0", (194, 700), (142, 700), 1), "dv:1": line("x1", (194, 850), (142, 850), 1)}
    c = Classified(visible=front_outline(), dimlines=[d], extensions=ext)
    refs, links, _ = link_view("front", c, [text("t", "37.5", (114, 775))], 3.0)
    px = {r.id: r.px for r in refs["b"]}
    assert links[0].axis == "b" and [round(px[r]) for r in links[0].refs] == [-850, -700]


def test_leader_links_a_diameter_and_its_count_to_same_size_circles():
    circles = [Prim("c1", "circle", np.zeros((3, 2)), 3, (250, 470), 25),
               Prim("c2", "circle", np.zeros((3, 2)), 3, (550, 470), 25),
               Prim("c3", "circle", np.zeros((3, 2)), 3, (400, 470), 10)]
    tip = np.array([550, 470]) + 25 * np.array([0.707, -0.707])
    lead = Leader("L", tip, tip + np.array([45, -45]), ["L"])
    c = Classified(visible=front_outline() + circles, leaders=[lead])
    t = text("t", "2xØ12.5", (tip[0] + 45, tip[1] - 61), w=90)
    _, links, _ = link_view("top", c, [t], 3.0)
    assert links[0].how == "leader" and links[0].kind == "diameter"
    assert sorted(links[0].circles) == ["c1", "c2"]


def test_value_without_a_dimension_line_links_by_position_with_an_issue():
    c = Classified(visible=front_outline())
    _, links, issues = link_view("front", c, [text("t", "100", (400, 960))], 3.0)
    assert links[0].how == "position" and links[0].axis == "a" and links[0].confidence < 0.95
    assert [i.kind for i in issues] == ["unplaced"]


def test_a_value_far_from_everything_is_kept_unplaced():
    c = Classified(visible=front_outline())
    _, links, issues = link_view("front", c, [text("t", "7", (1500, 100))], 3.0)
    assert links[0].how == "unplaced" and links[0].refs is None
    assert issues[0].kind == "unplaced"


def test_bridge_front_values_all_link_through_real_strokes():
    sh = bridge_block(Sheet())
    ink = sh.ink()
    texts = read_texts(sh.bgr(), find_text_boxes(ink, 3.0),
                       [TruthReader(sh.texts, "a"), TruthReader(sh.texts, "b")])
    views, _ = split_views(ink, texts, 3.0)
    front = next(v for v in views if v.name == "front")
    c = classify(vectorize(front.ink, 3.0, "front"), front.ink, 3.0, front.texts)
    refs, links, issues = link_view("front", c, front.texts, 3.0)
    assert len(links) == 10 and all(l.how == "dimension_line" for l in links), issues
    px = {r.id: r.px for rs in refs.values() for r in rs}
    spans = sorted(round(abs(px[l.refs[1]] - px[l.refs[0]])) for l in links)
    # vectorize's approxPolyDP corner offset shifts one build_refs cluster by ~0.4 px, which can flip
    # a span's rounding by 1 px; written values, not raw pixel spans, set the actual sizes.
    expected = [50, 50, 50, 50, 100, 100, 150, 150, 150, 400]
    assert len(spans) == len(expected) and all(abs(s - e) <= 1 for s, e in zip(spans, expected))
