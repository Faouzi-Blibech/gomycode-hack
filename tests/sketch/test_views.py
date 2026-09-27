import numpy as np

from s2c.sketch.models import SketchAbstain
from s2c.sketch.text import find_text_boxes, read_texts
from s2c.sketch.views import split_views
from tests.sketch.synth import F, Sheet, TruthReader, bridge_block


def run(sh):
    items = read_texts(sh.bgr(), find_text_boxes(sh.ink(), 3.0),
                       [TruthReader(sh.texts, "a"), TruthReader(sh.texts, "b")])
    return split_views(sh.ink(), items, 3.0)


def test_bridge_block_has_three_views_named_by_their_labels():
    views, issues = run(bridge_block(Sheet()))
    by = {v.name: v for v in views}
    assert set(by) == {"top", "front", "right"}
    assert all(v.named_by == "label" for v in views) and issues == []
    x, y = (round(c) for c in F(0, 0))
    assert by["front"].ink[y - 2:y + 3, x - 2:x + 3].any()
    assert not by["top"].ink[y - 2:y + 3, x - 2:x + 3].any()
    assert len([t for t in by["front"].texts if t.role == "dimension"]) == 10
    assert len([t for t in by["right"].texts if t.role == "dimension"]) == 5


def test_views_without_labels_are_named_from_the_third_angle_layout():
    views, issues = run(bridge_block(Sheet(), labels=False))
    assert {v.name for v in views} == {"top", "front", "right"}
    assert all(v.named_by == "layout" for v in views)
    assert len(issues) == 3 and all(i.kind == "label" for i in issues)


def test_two_close_views_stay_apart():
    sh = Sheet()
    for x0 in (200, 560):
        sh.line((x0, 300), (x0 + 300, 300))
        sh.line((x0 + 300, 300), (x0 + 300, 600))
        sh.line((x0 + 300, 600), (x0, 600))
        sh.line((x0, 600), (x0, 300))
    sh.vdim(300, 600, 500, 520, "75", rotate=True)   # A's dimension fills the 60 px gap
    sh.hdim(560, 860, 600, 645, "75")
    sh.text("FRONT", (350, 700), scale=1.0)
    sh.text("SIDE", (710, 720), scale=1.0)
    views, _ = run(sh)
    assert sorted(v.name for v in views) == ["front", "right"]


def test_blank_sheet_abstains():
    out = split_views(np.zeros((1131, 1600), np.uint8), [], 3.0)
    assert isinstance(out, SketchAbstain) and out.reason == "no_views_found"
