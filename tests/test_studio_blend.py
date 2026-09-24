import os
import sys
import zipfile
from pathlib import Path

import pytest

from s2c.multiview import blend
from s2c.multiview.blend import KIT_NAME, KIT_WARNING, SCRIPT, blender_runner, write_blend


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


def test_cached_blend_is_checked(obj, tmp_path, monkeypatch):
    monkeypatch.setattr(blend, "blender_runner", lambda: None)
    for i, bad in enumerate((b"", b"garbage")):
        out = tmp_path / f"bad{i}"
        out.mkdir()
        (out / "part.blend").write_bytes(bad)
        path, warnings = write_blend(obj, out)
        assert path.name == KIT_NAME and warnings == [KIT_WARNING]
        assert not (out / "part.blend").exists()
    for i, good in enumerate((b"BLENDER-v402", b"\x28\xb5\x2f\xfd\x00\x01")):
        out = tmp_path / f"good{i}"
        out.mkdir()
        target = out / "part.blend"
        target.write_bytes(good)
        path, warnings = write_blend(obj, out)
        assert path == target and warnings == [] and path.read_bytes() == good


def test_real_blend_removes_stale_kit(obj, tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    kit = out / KIT_NAME
    kit.write_bytes(b"stale")
    monkeypatch.setattr(blend, "blender_runner", lambda: ["python"])

    def fake_run(cmd, timeout_s, log_path, cwd=None):
        Path(cmd[cmd.index("--") + 2]).write_bytes(b"BLENDER-v402")
        return 0

    monkeypatch.setattr(blend, "run", fake_run)
    path, warnings = write_blend(obj, out)
    assert path.name == "part.blend" and warnings == []
    assert not kit.exists()


def test_script_usage_line():
    assert "--factory-startup" in SCRIPT.splitlines()[1]


@pytest.mark.blender
@pytest.mark.skipif(blender_runner() is None, reason="Blender or bpy not configured")
def test_a_real_blend_file(obj, tmp_path):
    path, warnings = write_blend(obj, tmp_path / "out")
    data = path.read_bytes()
    assert path.suffix == ".blend" and warnings == []
    assert data[:7] == b"BLENDER" or data[:4] == b"\x28\xb5\x2f\xfd"  # plain, or zstd-compressed (Blender 4.2)
    assert os.path.getsize(path) > 1000
