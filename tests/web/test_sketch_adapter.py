"""sketch_adapter turns a hand-built SketchReading into Observation objects MvPipeline.fuse() accepts.
No models, no network: every SketchReading here is built by hand."""
from __future__ import annotations

from s2c.multiview.fuse import fuse_envelope
from s2c.multiview.pipeline import MvPipeline
from s2c.sketch.models import Dimension, Feature, Reading, Size, SketchReading, View
from s2c.web.sketch_adapter import observations_from_sketch, observed_from_sketch

FRONT = View(name="front", label_text="FRONT", bbox_px=(0, 0, 200, 100), size_mm=(50.0, 25.0))
TOP = View(name="top", label_text="TOP", bbox_px=(0, 120, 200, 80), size_mm=(50.0, 20.0))


def _dim(view, axis, value, badge, kind="linear", id_="d1") -> Dimension:
    return Dimension(id=id_, view=view, kind=kind, value=value, text_raw=str(value),
                     readings=[Reading(reader="qwen", text=str(value), confidence=0.9)], measures=[], axis=axis,
                     badge=badge, evidence="reader", bbox_px=(0, 0, 10, 10))


def _reading(views, dims=(), feats=()) -> SketchReading:
    return SketchReading(image_size_px=(200, 200), views=views, entities=[], dimensions=list(dims),
                         features=list(feats), envelope={a: Size(value=1.0, badge="derived", evidence="geometry")
                                                          for a in "xyz"}, issues=[], timings_ms={})


def test_one_view_gives_one_observation_with_rectangle_outline_and_page_scale():
    obs = observations_from_sketch(_reading([FRONT]))
    assert len(obs) == 1
    o = obs[0]
    assert o.face == "front" and o.kind == "sketch"
    assert o.mm_per_px == 50.0 / 200
    assert len(o.outline.outer) == 4 and not o.outline.circles


def test_confirmed_dimension_is_written_unconfirmed_is_not():
    written = _dim("front", "a", 50.0, "written", id_="w")
    uncertain = _dim("front", "b", 25.0, "uncertain", id_="u")
    obs = observations_from_sketch(_reading([FRONT], dims=[written, uncertain]))
    linked = {lv.axis: lv for lv in obs[0].values}
    assert linked["a"].reading.confirmed is True
    assert linked["b"].reading.confirmed is False


def test_written_dimension_wins_the_envelope_axis_over_page_scale():
    written = _dim("front", "a", 48.0, "written")
    obs = observations_from_sketch(_reading([FRONT, TOP], dims=[written]))
    env, prov, _ = fuse_envelope(obs)
    assert env.x_mm == 48.0 and prov["envelope.x_mm"] == "user_written"
    # y (front's b axis) and z (top's b axis) had no dimension at all: the page scale still gives a value.
    assert prov["envelope.y_mm"] == "measured"
    assert prov["envelope.z_mm"] == "measured"


def test_unconfirmed_dimension_never_outranks_the_page_scale():
    uncertain = _dim("front", "a", 999.0, "uncertain")  # wildly wrong; must not win over the page's own scale
    obs = observations_from_sketch(_reading([FRONT, TOP], dims=[uncertain]))
    env, prov, _ = fuse_envelope(obs)
    assert prov["envelope.x_mm"] == "measured"
    assert env.x_mm != 999.0


def test_confirmed_hole_feature_becomes_a_written_circle_on_its_view():
    hole = Feature(id="h1", type="hole", axis="z", position_mm=(25.0, 12.5, 0.0), diameter=6.0, through=True,
                   evidence=["d1"], badge="written")
    obs = observations_from_sketch(_reading([FRONT], feats=[hole]))
    o = obs[0]
    assert len(o.outline.circles) == 1
    circle = o.outline.circles[0]
    assert circle.cx == 100.0 and circle.cy == 50.0  # centred: 25/50*200, (1-12.5/25)*100
    linked = [lv for lv in o.values if lv.hole_index == 0]
    assert len(linked) == 1 and linked[0].reading.confirmed is True and linked[0].reading.value_mm == 6.0


def test_uncertain_hole_feature_gets_a_circle_but_no_written_value():
    hole = Feature(id="h1", type="hole", axis="z", position_mm=(25.0, 12.5, 0.0), diameter=6.0, through=True,
                   evidence=["d1"], badge="predicted")
    obs = observations_from_sketch(_reading([FRONT], feats=[hole]))
    o = obs[0]
    assert len(o.outline.circles) == 1
    assert not any(lv.hole_index == 0 for lv in o.values)


def test_feeds_straight_into_mv_pipeline_fuse():
    """The whole point: sheet mode's Observation list is exactly what MvPipeline.fuse() expects, with no
    other bridging -- the same call the per-face photo mode makes after MvPipeline.observe()."""
    written = _dim("front", "a", 50.0, "written")
    observed = observed_from_sketch(_reading([FRONT, TOP], dims=[written]))
    result = MvPipeline().fuse(observed)
    assert result.envelope.x_mm == 50.0
