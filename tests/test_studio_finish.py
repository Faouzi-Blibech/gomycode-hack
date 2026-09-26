from pathlib import Path

import pytest

from s2c.multiview import build as build_mod
from s2c.multiview.build import BuildError, build, volume
from s2c.multiview.finish import apply_geometry, largest_finish, max_finish_mm
from s2c.multiview.settings import GeometrySettings
from s2c.multiview.spec import MultiViewSpec

SPEC = MultiViewSpec.model_validate_json(
    (Path(__file__).parents[1] / "examples" / "mv" / "l_bracket.json").read_text())


def test_no_finish_returns_the_spec_unchanged():
    spec, warnings = apply_geometry(SPEC, GeometrySettings())
    assert spec is SPEC and warnings == []


def test_a_fillet_is_added_as_a_user_value_and_removes_material():
    spec, warnings = apply_geometry(SPEC, GeometrySettings(finish="fillet", finish_mm=1.0))
    assert warnings == [] and spec.finishes[-1].type == "fillet" and spec.finishes[-1].radius_mm == 1.0
    assert spec.finishes[-1].edges == "all_vertical"
    assert spec.provenance[f"finishes[{len(spec.finishes) - 1}].radius_mm"] == "user_edited"
    assert volume(build(spec)) < volume(build(SPEC))


def test_a_chamfer_on_all_edges():
    spec, _ = apply_geometry(SPEC, GeometrySettings(finish="chamfer", finish_mm=0.5, finish_edges="all"))
    assert (spec.finishes[-1].type, spec.finishes[-1].edges) == ("chamfer", "all")


def test_the_size_is_clamped_to_the_part():
    assert max_finish_mm(SPEC) == pytest.approx(0.45 * 20)
    spec, warnings = apply_geometry(SPEC, GeometrySettings(finish="fillet", finish_mm=10.0))
    assert spec.finishes[-1].radius_mm == pytest.approx(9.0)
    assert warnings == ["Fillet reduced to 9 mm to fit the part"]


def test_a_finish_that_cannot_be_built_still_raises_a_build_error():
    spec, _ = apply_geometry(SPEC, GeometrySettings(finish="fillet", finish_mm=8.0))
    with pytest.raises(BuildError) as e:
        build(spec)
    assert e.value.reason == "fillet_failed"


def test_largest_finish_that_builds(monkeypatch):
    real_build = build_mod.build
    calls = []

    def counting(spec):
        calls.append(spec)
        return real_build(spec)

    monkeypatch.setattr(build_mod, "build", counting)
    geometry = GeometrySettings(finish="fillet", finish_mm=8.0)  # known to fail, from the test above
    size = largest_finish(SPEC, geometry)
    assert size is not None and len(calls) <= 7
    trial, _ = apply_geometry(SPEC, geometry.model_copy(update={"finish_mm": size}))
    build(trial)  # does not raise


def test_largest_finish_is_none_when_even_the_floor_fails(monkeypatch):
    monkeypatch.setattr(build_mod, "build", lambda spec: (_ for _ in ()).throw(BuildError("fillet_failed", "x")))
    assert largest_finish(SPEC, GeometrySettings(finish="fillet", finish_mm=8.0), lo=0.2) is None


def test_largest_finish_bisects_to_the_boundary_in_at_most_seven_builds(monkeypatch):
    calls = []

    def fake_build(spec):
        calls.append(1)
        if spec.finishes[-1].radius_mm > 3.35:
            raise BuildError("fillet_failed", "too big")
        return object()

    monkeypatch.setattr(build_mod, "build", fake_build)
    size = largest_finish(SPEC, GeometrySettings(finish="fillet", finish_mm=8.0), lo=0.2)
    assert size == pytest.approx(3.3) and len(calls) <= 7
