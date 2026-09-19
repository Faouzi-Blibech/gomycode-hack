import os
import time
from pathlib import Path

from s2c.store import FileStore


def test_put_and_get(tmp_path):
    s = FileStore(tmp_path, ttl_s=3600)
    fid = s.put(b"hello", ".stl")
    assert fid.endswith(".stl") and "/" not in fid and ".." not in fid
    assert s.path(fid).read_bytes() == b"hello"


def test_unknown_or_traversal_ids_return_none(tmp_path):
    s = FileStore(tmp_path)
    assert s.path("nope.stl") is None
    assert s.path("../pyproject.toml") is None


def test_rejects_all_malicious_identifier_shapes(tmp_path):
    """The path lookup is security-sensitive: it must accept only identifiers it
    could itself have generated (32 lowercase hex chars + '.' + 1-5 lowercase
    alnum chars), and reject everything else by returning None rather than
    raising."""
    s = FileStore(tmp_path)
    valid_hex = "a" * 32
    candidates = [
        "../pyproject.toml",  # parent-directory traversal
        "../../etc/passwd",  # deeper traversal
        "/etc/passwd",  # absolute path, posix-style
        "C:/Windows/System32/drivers/etc/hosts",  # absolute path, forward slashes
        "C:\\Windows\\System32\\drivers\\etc\\hosts",  # absolute path, backslashes
        valid_hex + "/x.stl",  # forward-slash separator
        valid_hex + "\\x.stl",  # backslash separator
        valid_hex.upper() + ".stl",  # right shape, uppercase hex
        "",  # empty string
    ]
    for candidate in candidates:
        assert s.path(candidate) is None, f"expected None for {candidate!r}"


def test_rejected_and_unknown_ids_are_indistinguishable(tmp_path):
    """A rejected (malformed) identifier and a well-formed but unknown one must
    both return None with no exception raised, so a caller cannot learn
    anything about the filesystem from the difference."""
    s = FileStore(tmp_path)
    well_formed_but_unknown = ("b" * 32) + ".stl"
    malformed = "../secret.txt"
    assert s.path(well_formed_but_unknown) is None
    assert s.path(malformed) is None


def test_sweep_removes_expired(tmp_path):
    s = FileStore(tmp_path, ttl_s=1)
    fid = s.put(b"x", ".png")
    old = time.time() - 10
    os.utime(s.path(fid), (old, old))
    assert s.sweep() == 1
    assert s.path(fid) is None


def test_sweep_tolerates_file_vanishing_mid_sweep(tmp_path, monkeypatch):
    """Two requests can sweep concurrently, and on Windows a file may also be
    briefly locked. Simulate another process deleting the file between the
    directory listing and our stat/unlink call, and confirm sweep() does not
    raise."""
    s = FileStore(tmp_path, ttl_s=1)
    fid = s.put(b"x", ".png")
    target = s.path(fid)
    old = time.time() - 10
    os.utime(target, (old, old))

    real_stat = Path.stat
    triggered = {"done": False}

    def flaky_stat(self, *args, **kwargs):
        # Simulate a concurrent sweep (or the OS) removing the file the
        # instant before we stat it. Use os.remove (not self.unlink /
        # self.exists) so this doesn't recurse back into the patched stat.
        if self.name == fid and not triggered["done"]:
            triggered["done"] = True
            try:
                os.remove(self)
            except OSError:
                pass
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", flaky_stat)

    removed = s.sweep()  # must not raise FileNotFoundError/OSError
    assert isinstance(removed, int)
    assert s.path(fid) is None


def test_sweep_tolerates_unexpected_directory_entry(tmp_path):
    """A directory appearing inside the store root (not something the store
    itself ever creates) must not crash the sweep."""
    s = FileStore(tmp_path, ttl_s=0)
    stray_dir = tmp_path / "unexpected_dir"
    stray_dir.mkdir()
    removed = s.sweep()  # must not raise
    assert isinstance(removed, int)
    assert stray_dir.exists()


def test_store_root_creation_is_idempotent(tmp_path):
    """Several components construct a store independently; constructing more
    than one FileStore against the same root must not raise."""
    root = tmp_path / "shared_root"
    s1 = FileStore(root)
    s2 = FileStore(root)
    fid = s1.put(b"y", ".stl")
    assert s2.path(fid).read_bytes() == b"y"
