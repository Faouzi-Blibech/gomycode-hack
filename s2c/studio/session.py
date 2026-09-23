"""Per-browser-session state, kept on the server and keyed by an id: gr.State only holds the id, because the
pipeline's Observed (images, masks, meshes) is too big to deep-copy on every event. Spec 2026-09-23-studio §8."""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

from s2c.multiview.artifacts import ExportResult, Part
from s2c.multiview.pipeline import Observed
from s2c.multiview.settings import AiSettings, GeometrySettings
from s2c.multiview.spec import MultiViewSpec


@dataclass
class Item:
    id: str
    path: str
    name: str
    face: str = "auto"
    kind: str = "auto"


@dataclass
class Session:
    id: str
    items: list[Item] = field(default_factory=list)
    reference: str = "none"
    ai: AiSettings = field(default_factory=AiSettings)
    geometry: GeometrySettings = field(default_factory=GeometrySettings)
    observed: Observed | None = None
    spec: MultiViewSpec | None = None
    edits: dict[str, float] = field(default_factory=dict)
    rejected: tuple[str, ...] = ()
    row_paths: list[str] = field(default_factory=list)
    shown: dict[str, float | None] = field(default_factory=dict)
    part: Part | None = None
    exported: ExportResult | None = None
    touched: float = field(default_factory=time.time)


class SessionStore:
    def __init__(self, ttl_s: float = 3600):
        self.ttl_s, self._items, self._lock = ttl_s, {}, threading.Lock()

    def new(self) -> str:
        sid = uuid.uuid4().hex
        with self._lock:
            self._sweep()
            self._items[sid] = Session(sid)
        return sid

    def get(self, sid: str | None) -> Session:
        """The session, or a fresh one under that id (after a server restart the browser still holds its id)."""
        with self._lock:
            session = self._items.get(sid or "")
            if session is None:
                session = self._items[sid or uuid.uuid4().hex] = Session(sid or uuid.uuid4().hex)
            session.touched = time.time()
            return session

    def drop(self, sid: str | None) -> None:
        with self._lock:
            self._items.pop(sid or "", None)

    def _sweep(self) -> None:
        now = time.time()
        for sid in [s for s, v in self._items.items() if now - v.touched > self.ttl_s]:
            del self._items[sid]
