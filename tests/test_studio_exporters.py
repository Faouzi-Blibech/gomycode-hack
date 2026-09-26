import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cadquery as cq
import pytest
import trimesh

from s2c.multiview import exporters
from s2c.multiview.build import build, volume
from s2c.multiview.exporters import FILE_NAMES, MESH_FORMATS, export_mesh_formats, mesh_of
from s2c.multiview.spec import MultiViewSpec

SPEC = MultiViewSpec.model_validate_json(
    (Path(__file__).parents[1] / "examples" / "mv" / "l_bracket.json").read_text())


@pytest.fixture(scope="module")
def part():
    return build(SPEC)


@pytest.fixture(scope="module")
def files(part, tmp_path_factory):
    return export_mesh_formats(part, tmp_path_factory.mktemp("exports"), MESH_FORMATS, "normal")


def test_every_format_is_written_with_its_name(files):
    assert set(files) == set(MESH_FORMATS)
    for fmt, path in files.items():
        assert path.name == FILE_NAMES[fmt] and path.stat().st_size > 0


def test_stl_is_watertight_with_the_right_volume(part, files):
    mesh = trimesh.load(str(files["stl"]), force="mesh")
    assert mesh.is_watertight and mesh.volume == pytest.approx(volume(part), rel=0.005)


def test_step_and_brep_reload_exactly(part, files):
    assert cq.importers.importStep(str(files["step"])).val().Volume() == pytest.approx(volume(part), rel=0.001)
    assert cq.Shape.importBrep(str(files["brep"])).Volume() == pytest.approx(volume(part), rel=0.001)


def test_3mf_holds_a_model(files):
    with zipfile.ZipFile(files["3mf"]) as z:
        assert "3D/3dmodel.model" in z.namelist()


def test_obj_and_ply_reload_with_the_right_volume(part, files):
    for fmt in ("obj", "ply"):
        mesh = trimesh.load(str(files[fmt]), force="mesh")
        assert mesh.volume == pytest.approx(volume(part), rel=0.005), fmt


def test_glb_is_in_metres(files):
    extents = sorted(trimesh.load(str(files["glb"])).extents)
    assert extents == pytest.approx(sorted([0.050, 0.030, 0.020]), rel=0.01)


def test_quality_is_honoured_even_after_a_finer_mesh(part):
    fine = len(mesh_of(part, "fine").faces)
    draft = len(mesh_of(part, "draft").faces)
    assert draft < fine


def test_an_unknown_format_is_refused(part, tmp_path):
    with pytest.raises(ValueError):
        export_mesh_formats(part, tmp_path, ["xyz"])


def test_all_occt_writes_hold_the_lock(part, tmp_path, monkeypatch):
    locked = []
    real_export, real_export_stl = cq.exporters.export, cq.Shape.exportStl

    def fake_export(*args, **kwargs):
        locked.append(exporters._OCCT_LOCK.locked())
        return real_export(*args, **kwargs)

    def fake_export_stl(self, *args, **kwargs):
        locked.append(exporters._OCCT_LOCK.locked())
        return real_export_stl(self, *args, **kwargs)

    monkeypatch.setattr(cq.exporters, "export", fake_export)
    monkeypatch.setattr(cq.Shape, "exportStl", fake_export_stl)
    export_mesh_formats(part, tmp_path, ["stl", "step", "3mf", "brep"])
    assert locked and all(locked)


def test_failed_write_leaves_no_file(tmp_path, monkeypatch):
    box = cq.Workplane("XY").box(10, 10, 10)

    def bad_writer(solid, path, quality="normal"):
        Path(path).write_bytes(b"0123456789")
        raise RuntimeError("boom")

    monkeypatch.setitem(exporters.WRITERS, "obj", bad_writer)
    with pytest.raises(RuntimeError):
        export_mesh_formats(box, tmp_path, ["obj"])
    assert not (tmp_path / "part.obj").exists()
    assert not list(tmp_path.glob("*.tmp*"))


def test_concurrent_exports(tmp_path):
    box = cq.Workplane("XY").box(10, 10, 10)

    def do_export(i):
        return export_mesh_formats(box, tmp_path / f"t{i}", ["stl", "step", "3mf", "glb"])

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(do_export, range(4)))
    for files in results:
        mesh = trimesh.load(str(files["stl"]), force="mesh")
        assert mesh.volume == pytest.approx(1000.0, rel=0.01)
        assert files["step"].read_text(encoding="utf-8", errors="replace").startswith("ISO-10303-21")
