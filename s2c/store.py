"""Temp files with a TTL. Nothing here outlives an hour.

This module sits behind a public HTTP endpoint that serves a file by the
identifier a caller supplies, so `path()`'s identifier validation is the
only thing standing between a caller and reading arbitrary files off the
machine. It must accept only an identifier the store could itself have
generated (see `_SAFE`) and reject anything else by returning None -- never
by raising, and never with any signal that would let a caller distinguish a
rejected identifier from a well-formed but unknown one.

Sweeping is called explicitly by the HTTP layer on each request. There is no
background thread or scheduler here on purpose, to keep this module
dependency-free and easy to test.
"""

import re
import tempfile
import time
import uuid
from pathlib import Path

# Exactly what uuid4().hex + a short lowercase suffix looks like. No path
# separators, no "..", no uppercase -- anything outside this shape is
# rejected before it ever touches the filesystem.
_SAFE = re.compile(r"^[a-f0-9]{32}\.[a-z0-9]{1,5}$")


class FileStore:
    def __init__(self, root: Path | None = None, ttl_s: int = 3600):
        self.root = Path(root) if root else Path(tempfile.gettempdir()) / "s2c_store"
        # Safe to call repeatedly: several components construct a store
        # independently against the same root.
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl_s = ttl_s

    def put(self, data: bytes, suffix: str) -> str:
        file_id = uuid.uuid4().hex + suffix
        (self.root / file_id).write_bytes(data)
        return file_id

    def path(self, file_id: str) -> Path | None:
        if not _SAFE.match(file_id):
            return None
        p = self.root / file_id
        return p if p.exists() else None

    def sweep(self) -> int:
        """Delete every stored file older than ttl_s. Returns the count removed.

        Two requests can sweep concurrently, and on Windows a file can be
        briefly locked by another process. This tolerates a file vanishing
        between listing and stat-ing or unlinking it, and tolerates an
        unexpected directory entry showing up inside the store root -- it
        never raises because of either.
        """
        now, removed = time.time(), 0
        try:
            entries = list(self.root.iterdir())
        except OSError:
            return removed
        for p in entries:
            try:
                if p.is_dir():
                    continue
                if now - p.stat().st_mtime > self.ttl_s:
                    p.unlink(missing_ok=True)
                    removed += 1
            except OSError:
                # The entry vanished (or became briefly inaccessible)
                # between listing, stat-ing and unlinking it. Not our file
                # to worry about any more either way.
                continue
        return removed
