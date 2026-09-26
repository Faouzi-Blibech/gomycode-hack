import numpy as np
import pytest

from s2c.multiview.fuse import (
    Observation,
    assemble,
    attach_label,
    canonical_outlines,
    features_from,
    fuse_envelope,
    snap_coord,
    snap_diameter,
)
from s2c.multiview.label import LabelHole, MvLabel
from s2c.multiview.ocr import Linked, Reading
from s2c.multiview.outline import PixelCircle, PixelOutline
from s2c.multiview.spec import Envelope, MvAbstain, Outline


def px_outline(x=400, y=300, w=601, h=401, circles=()):
    outer = np.array([[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]])
    return PixelOutline(outer=outer, circles=list(circles), bbox=(x, y, w, h), shape=(1200, 1600))


def written(value, axis=None, hole=None, kind="linear", conf=0.9):
    return Linked(Reading(float(value), kind, (0, 0, 10, 10), conf, str(value)), axis, hole)


def front_obs(**kw):
    return Observation(face="front", kind="sketch", outline=px_outline(**kw),
                       values=[written(60, "a"), written(40, "b")])


def test_gate_asks_for_the_missing_depth():
    res = fuse_envelope([front_obs()])
    assert isinstance(res, MvAbstain) and res.reason == "missing_z" and res.remedy == "Enter the depth in mm."
    assert res.partial["known"] == {"envelope.x_mm": 60, "envelope.y_mm": 40}
    assert res.partial["missing"] == ["envelope.z_mm"]


def test_a_top_view_without_a_depth_value_gives_a_suggestion():
    top = Observation(face="top", kind="sketch", outline=px_outline(h=101), values=[written(60, "a")])
    res = fuse_envelope([front_obs(), top])
    assert isinstance(res, MvAbstain) and res.partial["suggested"] == {"envelope.z_mm": 10.0}


def test_a_typed_value_passes_the_gate():
    env, prov, _ = fuse_envelope([front_obs()], {"envelope.z_mm": 5})
    assert (env.x_mm, env.y_mm, env.z_mm) == (60, 40, 5)
    assert prov == {"envelope.x_mm": "user_written", "envelope.y_mm": "user_written", "envelope.z_mm": "user_edited"}


def test_written_beats_measured_with_a_warning():
    photo = Observation(face="front", kind="photo", outline=px_outline(), mm_per_px=0.11, values=[written(60, "a")])
    env, prov, warnings = fuse_envelope([photo], {"envelope.z_mm": 5})
    assert env.x_mm == 60 and prov["envelope.x_mm"] == "user_written"
    assert env.y_mm == pytest.approx(44.0) and prov["envelope.y_mm"] == "measured"
    assert any("using the written value" in w for w in warnings)


def test_nothing_written_and_nothing_measured_abstains_on_x_first():
    res = fuse_envelope([Observation(face="front", kind="sketch", outline=px_outline())])
    assert isinstance(res, MvAbstain) and res.reason == "missing_x"


def test_a_back_view_is_mirrored_into_the_front():
    env = Envelope(x_mm=60, y_mm=40, z_mm=5)
    tri = PixelOutline(outer=np.array([[400, 700], [1000, 700], [400, 300]]), bbox=(400, 300, 601, 401),
                       shape=(1200, 1600))
    outlines, _ = canonical_outlines([Observation(face="back", kind="sketch", outline=tri)], env)
    ol, prov = outlines["front"]
    assert ol.source == "mirrored" and prov == "scaled"
    assert sorted(ol.outer) == sorted([(60.0, 0.0), (0.0, 0.0), (60.0, 40.0)])


def test_written_diameter_wins_and_a_through_hole_seen_twice_is_kept_once():
    env = Envelope(x_mm=60, y_mm=40, z_mm=5)
    front = Observation(face="front", kind="sketch", outline=px_outline(circles=[PixelCircle(500, 600, 58)]),
                        values=[written(6, None, 0, "diameter")])
    back = Observation(face="back", kind="sketch", outline=px_outline(circles=[PixelCircle(900, 600, 60)]))
    feats, prov = features_from([front, back], env)
    assert len(feats) == 1
    assert feats[0]["diameter_mm"] == 6 and prov["features[0].diameter_mm"] == "user_written"
    assert feats[0]["a_mm"] == pytest.approx(10) and feats[0]["b_mm"] == pytest.approx(10)
    assert prov["features[0].a_mm"] == "scaled"


def test_blind_hole_depth_defaults_to_half_the_axis():
    env = Envelope(x_mm=60, y_mm=40, z_mm=6)
    o = Observation(face="front", kind="sketch", outline=px_outline(circles=[PixelCircle(500, 600, 60)]),
                    blind={0: True})
    feats, prov = features_from([o], env)
    assert feats[0]["depth_mm"] == 3 and prov["features[0].depth_mm"] == "default"


def test_a_label_blind_flag_and_depth_estimate_never_reach_a_feature():
    """Rule 2: only Solaria may mark a hole blind. A vision-model label saying blind, with a depth guess
    attached, must leave the feature a through hole with no model millimetres anywhere."""
    env = Envelope(x_mm=60, y_mm=40, z_mm=6)
    o = Observation(face="front", kind="sketch", outline=px_outline(circles=[PixelCircle(500, 600, 60)]))
    label = MvLabel(face="front", input_kind="sketch", holes=[LabelHole(u=100 / 601, v=101 / 401, blind=True)],
                    estimates={"holes[0].depth_mm": 4.0}, confidence=0.9)
    attach_label(o, label)
    feats, prov = features_from([o], env)
    assert feats[0]["depth_mm"] is None
    assert "features[0].depth_mm" not in prov


def test_a_solaria_depth_ratio_still_makes_a_hole_blind_and_estimated():
    env = Envelope(x_mm=60, y_mm=40, z_mm=6)
    o = Observation(face="front", kind="sketch", outline=px_outline(circles=[PixelCircle(500, 600, 60)]),
                    blind={0: True}, depth_ratio={0: 0.5}, depth_from_image={0})
    feats, prov = features_from([o], env)
    assert feats[0]["depth_mm"] == pytest.approx(3.0) and prov["features[0].depth_mm"] == "estimated"


def test_snapping_tables():
    assert snap_diameter(5.37) == 5.5 and snap_diameter(7.2) == 7.0
    assert snap_coord(0.3, 60) == 0.0 and snap_coord(59.8, 60) == 60
    assert snap_coord(2.85, 60) == 3.0 and snap_coord(57.1, 60) == 57.0
    assert snap_coord(31.26, 60) == 31.5


def test_assemble_snaps_only_untrusted_values_and_applies_edits():
    env = Envelope(x_mm=60, y_mm=40, z_mm=5)
    o = Observation(face="front", kind="sketch", outline=px_outline(circles=[PixelCircle(500, 600, 53.7)]))
    outlines, _ = canonical_outlines([o], env)
    outlines["top"] = (Outline(outer=[(0, 0), (60, 0), (60, 5), (0, 5)], source="assumed", confidence=0.3), "default")
    outlines["right"] = (Outline(outer=[(0, 0), (5, 0), (5, 40), (0, 40)], source="assumed", confidence=0.3), "default")
    feats, fprov = features_from([o], env)
    env_prov = {"envelope.x_mm": "user_written", "envelope.y_mm": "user_written", "envelope.z_mm": "user_edited"}
    spec = assemble(env, env_prov, outlines, feats, fprov, [], user_values={"features[0].b_mm": 12.3})
    assert spec.features[0].diameter_mm == 5.5 and "features[0].diameter_mm" in spec.snapped
    assert spec.features[0].b_mm == 12.3 and spec.provenance["features[0].b_mm"] == "user_edited"
    assert spec.confidence == 0.3
