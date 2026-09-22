"""HTTP surface of the multi-view path. The integrator mounts `router` in s2c/api.py. Spec section 7.1.
Files live under STORE_ROOT for one hour."""
from __future__ import annotations

import json
import re
import shutil
import time
import uuid
from functools import lru_cache
from pathlib import Path

import cv2
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from s2c.multiview.pipeline import ImageInput, MvPipeline, Observed, default_pipeline
from s2c.multiview.spec import MultiViewSpec, MvAbstain

router = APIRouter(prefix="/mv", tags=["multiview"])
STORE_ROOT = Path("tmp/mv")
TTL_S = 3600
_ID = re.compile(r"^[0-9a-f]{32}$")
_NAME = re.compile(r"^[\w.-]+$")
_requests: dict[str, tuple[float, Observed]] = {}


@lru_cache(maxsize=1)
def get_pipeline() -> MvPipeline:
    return default_pipeline()


def _sweep() -> None:
    now = time.time()
    for rid in [r for r, (t, _) in _requests.items() if now - t > TTL_S]:
        del _requests[rid]
    if STORE_ROOT.exists():
        for d in STORE_ROOT.iterdir():
            if d.is_dir() and now - d.stat().st_mtime > TTL_S:
                shutil.rmtree(d, ignore_errors=True)


def _result(res) -> dict:
    if isinstance(res, MvAbstain):
        return {"abstain": res.model_dump()}
    return {"spec": res.model_dump(mode="json")}


def _tag(items: list, i: int) -> str | None:
    value = items[i] if i < len(items) else None
    return None if value in (None, "", "auto") else value


@router.post("/analyze")
def analyze(files: list[UploadFile] = File(...), faces: str = Form("[]"), kinds: str = Form("[]"),
            reference: str | None = Form(None), pipe: MvPipeline = Depends(get_pipeline)) -> dict:
    _sweep()
    face_tags, kind_tags = json.loads(faces), json.loads(kinds)
    images = [ImageInput(f.file.read(), _tag(face_tags, i), _tag(kind_tags, i)) for i, f in enumerate(files)]
    rid = uuid.uuid4().hex
    observed = pipe.observe(images, reference or None)
    if isinstance(observed, MvAbstain):
        return {"request_id": rid, "abstain": observed.model_dump()}
    _requests[rid] = (time.time(), observed)
    return {"request_id": rid, **_result(pipe.fuse(observed)), "labels": [l.model_dump() for l in observed.labels]}


class MergeBody(BaseModel):
    request_id: str
    user_values: dict[str, float] = {}
    accepted: list[str] = []
    rejected: list[str] = []


@router.post("/merge")
def merge(body: MergeBody, pipe: MvPipeline = Depends(get_pipeline)) -> dict:
    entry = _requests.get(body.request_id)
    if entry is None:
        raise HTTPException(404, "Unknown or expired request. Analyze the images again.")
    return _result(pipe.fuse(entry[1], body.user_values, body.accepted, body.rejected))


class BuildBody(BaseModel):
    spec: MultiViewSpec
    request_id: str | None = None


@router.post("/build")
def build_part(body: BuildBody, pipe: MvPipeline = Depends(get_pipeline)) -> dict:
    _sweep()
    bid = uuid.uuid4().hex
    out = STORE_ROOT / bid
    entry = _requests.get(body.request_id or "")
    res = pipe.build(body.spec, out, entry[1].masks if entry else None)
    if isinstance(res, MvAbstain):
        return {"abstain": res.model_dump()}
    base = f"/mv/files/{bid}"
    views = {}
    for face, mask in res.views.items():
        cv2.imwrite(str(out / f"view_{face}.png"), mask)
        views[face] = f"{base}/view_{face}.png"
    return {"stl_url": f"{base}/part.stl", "step_url": f"{base}/part.step",
            "gcode_url": f"{base}/part.gcode" if res.gcode else None,
            "print_time_s": res.print_time_s, "filament_g": res.filament_g,
            "views": views, "iou": res.iou, "warnings": res.warnings}


@router.get("/files/{bid}/{name}")
def files(bid: str, name: str) -> FileResponse:
    path = STORE_ROOT / bid / name
    if not _ID.match(bid) or not _NAME.match(name) or not path.is_file():
        raise HTTPException(404, "File not found or expired.")
    return FileResponse(path)
