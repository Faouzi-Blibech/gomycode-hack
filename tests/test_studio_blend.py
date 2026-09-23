import os
import sys
import zipfile

import pytest

from s2c.multiview import blend
from s2c.multiview.blend import KIT_NAME, KIT_WARNING, blender_runner, write_blend


@pytest.fixture
def obj(tmp_path):
    path = tmp_path / "part.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    return path


def test_without_blender_a_kit_is_written(obj, tmp_path, monkeypatch):
    monkeypatch.setattr(blend, "blender_runner", lambda: None)
    path, warnings = write_blend(obj, tmp_path / "out")
    assert path.name == KIT_NAME and warnings == [KIT_WARNING]
    with zipfile.ZipFile(path) as z:
        assert sorted(z.namelist()) == ["open_in_blender.py", "part.obj"]
        assert "obj_import" in z.read("open_in_blender.py").decode()


def test_with_a_blender_python_the_script_runs_with_both_paths(obj, tmp_path, monkeypatch):
    fake = ("import sys\nargv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]\n"
            "open(argv[1], 'wb').write(b'BLENDER-v402' + open(argv[0], 'rb').read()[:1])\n")
    monkeypatch.setattr(blend, "SCRIPT", fake)
    monkeypatch.setattr(blend, "blender_runner", lambda: [sys.executable])
    path, warnings = write_blend(obj, tmp_path / "out")
    assert path.name == "part.blend" and warnings == []
    assert path.read_bytes().startswith(b"BLENDER")


def test_a_failing_blender_falls_back_to_the_kit(obj, tmp_path, monkeypatch):
    monkeypatch.setattr(blend, "SCRIPT", "raise SystemExit(3)\n")
    monkeypatch.setattr(blend, "blender_runner", lambda: [sys.executable])
    path, warnings = write_blend(obj, tmp_path / "out")
    assert path.name == KIT_NAME and any("Blender failed" in w for w in warnings)


def test_the_runner_comes_from_the_environment(tmp_path, monkeypatch):
    for key in ("BLENDER_PATH", "BLENDER_PYTHON"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(blend, "VENDOR_PYTHON", tmp_path / "missing.exe")
    assert blender_runner() is None
    exe = tmp_path / "blender.exe"
    exe.write_text("")
    monkeypatch.setenv("BLENDER_PATH", str(exe))
    assert blender_runner() == [str(exe), "-b", "--factory-startup", "--python"]


@pytest.mark.blender
@pytest.mark.skipif(blender_runner() is None, reason="Blender or bpy not configured")
def test_a_real_blend_file(obj, tmp_path):
    path, warnings = write_blend(obj, tmp_path / "out")
    data = path.read_bytes()
    assert path.suffix == ".blend" and warnings == []
    assert data[:7] == b"BLENDER" or data[:4] == b"\x28\xb5\x2f\xfd"  # plain, or zstd-compressed (Blender 4.2)
    assert os.path.getsize(path) > 1000
