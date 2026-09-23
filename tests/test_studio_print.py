import pytest
import trimesh

from s2c.multiview import slice as slicing
from s2c.multiview.build import build
from s2c.multiview.print_settings import slicer_flags
from s2c.multiview.settings import PrintSettings
from s2c.multiview.slice import find_slicer, slice_solid
from s2c.multiview.spec import MvAbstain
from tests.mv_helpers import make_spec


def value(flags, name):
    return flags[flags.index(name) + 1]


def test_defaults_become_pla_flags():
    f = slicer_flags(PrintSettings())
    assert value(f, "--fill-density") == "20%" and value(f, "--layer-height") == "0.2"
    assert value(f, "--temperature") == "210" and value(f, "--bed-temperature") == "60"
    assert "--support-material" in f and "--support-material-buildplate-only" in f


def test_every_choice_maps_to_a_flag():
    f = slicer_flags(PrintSettings(material="PETG", nozzle_mm=0.6, layer_mm=0.3, infill_pct=45,
                                   infill_pattern="gyroid", perimeters=5, supports="off", brim_mm=4))
    assert value(f, "--filament-type") == "PETG" and value(f, "--temperature") == "240"
    assert value(f, "--nozzle-diameter") == "0.6" and value(f, "--first-layer-height") == "0.3"
    assert value(f, "--fill-pattern") == "gyroid" and value(f, "--perimeters") == "5"
    assert value(f, "--brim-width") == "4" and "--no-support-material" in f and "--support-material" not in f
    everywhere = slicer_flags(PrintSettings(supports="everywhere"))
    assert "--no-support-material-buildplate-only" in everywhere


def test_the_first_layer_fits_a_small_nozzle():
    f = slicer_flags(PrintSettings(nozzle_mm=0.2, layer_mm=0.1))
    assert float(value(f, "--first-layer-height")) <= 0.15 + 1e-9


def fake_slicer(calls, write=True):
    def run(cmd, timeout_s, log_path, cwd=None):
        calls.append(cmd)
        if write:
            out = cmd[cmd.index("--output") + 1]
            with open(out, "w", encoding="utf-8") as g:
                g.write("G1 X1\n; filament used [g] = 3.21\n; estimated printing time (normal mode) = 12m 5s\n")
        return 0
    return run


def test_overrides_threads_and_datadir_reach_the_command(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(slicing, "run", fake_slicer(calls))
    res = slice_solid(build(make_spec((60.0, 40.0, 5.0))), tmp_path, slicer=tmp_path / "slicer.exe",
                      overrides=slicer_flags(PrintSettings(infill_pct=35)))
    cmd = calls[0]
    assert value(cmd, "--fill-density") == "35%" and value(cmd, "--threads") == "4" and "--datadir" in cmd
    assert cmd.index("--load") < cmd.index("--fill-density")
    assert (res.print_time_s, res.filament_g) == (725.0, 3.21)


def test_scale_applies_before_the_bed_check(tmp_path, monkeypatch):
    monkeypatch.setattr(slicing, "run", fake_slicer([]))
    res = slice_solid(build(make_spec((60.0, 40.0, 5.0))), tmp_path, slicer=tmp_path / "s.exe", scale=2.0)
    mesh = trimesh.load(str(res.print_stl), force="mesh")
    assert sorted(mesh.extents) == pytest.approx([10.0, 80.0, 120.0], abs=0.05)
    too_big = slice_solid(build(make_spec((150.0, 40.0, 5.0))), tmp_path / "b", slicer=tmp_path / "s.exe", scale=2.0)
    assert isinstance(too_big, MvAbstain) and too_big.reason == "too_big_for_bed"


def test_a_hung_slicer_is_a_clean_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(slicing, "run", lambda cmd, timeout_s, log_path, cwd=None: None)
    res = slice_solid(build(make_spec((60.0, 40.0, 5.0))), tmp_path, slicer=tmp_path / "s.exe")
    assert isinstance(res, MvAbstain) and res.reason == "slicer_failed"


@pytest.mark.slicer
@pytest.mark.skipif(find_slicer() is None, reason="PrusaSlicer not installed")
def test_thicker_layers_mean_fewer_layers(tmp_path):
    solid = build(make_spec((60.0, 40.0, 6.0)))
    thin = slice_solid(solid, tmp_path / "a", overrides=slicer_flags(PrintSettings(layer_mm=0.2)))
    thick = slice_solid(solid, tmp_path / "b", overrides=slicer_flags(PrintSettings(layer_mm=0.3)))
    count = [r.gcode.read_text(encoding="utf-8").count(";LAYER_CHANGE") for r in (thin, thick)]
    assert count[1] < count[0]
