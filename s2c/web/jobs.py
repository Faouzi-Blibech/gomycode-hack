"""Analysis jobs for the web app: one record per /api/analyze, filled by the pipeline's progress events.
Jobs live in memory for one hour. The job id is also the request id that /api/merge and /api/model use."""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field

from s2c.multiview.pipeline import ImageInput, MvPipeline, Observed
from s2c.multiview.routes import _forget_images
from s2c.multiview.spec import FACES, MvAbstain

log = logging.getLogger(__name__)
TTL_S = 3600
STAGES = ("label", "outline", "read", "draw", "fuse")
TOOLS = {"label": ("Vision model", True), "outline": ("OpenCV", False), "read": ("Qwen-VL", True),
         "draw": ("Qwen-Image", True), "fuse": ("CadQuery", False)}
DRAW_TOOLS = {"qwen-image": ("Qwen-Image", True), "triposr": ("TripoSR", True), "mirrored": ("mirror", False),
              "assumed": ("assumed", False)}
DRAW_WORDS = {"qwen-image": "drawn — check it", "triposr": "predicted — check it", "mirrored": "mirrored",
              "assumed": "assumed rectangular — check it"}
FAILED = "Analysis failed. Try again or use different photos."


class JobCancelled(Exception):
    pass


@dataclass
class Job:
    job_id: str
    stages: list[dict]
    images: list[dict]
    status: str = "running"
    coverage: dict[str, str] = field(default_factory=lambda: dict.fromkeys(FACES, "empty"))
    result: dict | None = None
    error: str | None = None
    observed: Observed | None = None
    created: float = field(default_factory=time.time)
    cancel: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)
    merge_lock: threading.Lock = field(default_factory=threading.Lock)  # fuse mutates the cached Observed

    def stage(self, key: str) -> dict:
        return next(s for s in self.stages if s["key"] == key)

    def to_json(self) -> dict:
        with self.lock:
            return {"job_id": self.job_id, "status": self.status,
                    "stages": [dict(s) for s in self.stages],
                    "images": [{**i, "circles": list(i["circles"]), "reads": list(i["reads"])} for i in self.images],
                    "coverage": dict(self.coverage), "result": self.result, "error": self.error}


JOBS: dict[str, Job] = {}
_registry_lock = threading.Lock()


def new_job(n_images: int, pipe: MvPipeline) -> Job:
    stages = []
    for key in STAGES:
        tool, ai = TOOLS[key]
        stages.append({"key": key, "state": "pending", "tool": tool, "ai": ai, "detail": "",
                       "started": None, "ended": None})
    job = Job(uuid.uuid4().hex, stages, [
        {"index": i, "width": 0, "height": 0, "face": None, "kind": None, "outline": None, "circles": [], "reads": []}
        for i in range(n_images)])
    if pipe.chat is None:
        job.stage("label").update(tool="Your face tags", ai=False)
    if pipe.reader is None and pipe.batch_reader is None:
        job.stage("read").update(state="skipped", detail="No reader configured: type the sizes")
    if pipe.image_gen is None or not pipe.draw_faces:
        job.stage("draw").update(tool="TripoSR" if pipe.mesh_provider is not None else "assumed",
                                 ai=pipe.mesh_provider is not None)
    with _registry_lock:
        JOBS[job.job_id] = job
    return job


def get_job(job_id: str) -> Job | None:
    with _registry_lock:
        return JOBS.get(job_id)


def sweep_jobs(ttl: float = TTL_S) -> None:
    now = time.time()
    with _registry_lock:
        for jid in [j for j, job in JOBS.items() if now - job.created > ttl]:
            del JOBS[jid]


def _mm(value: float) -> str:
    return f"{round(value, 2):g}"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _details(job: Job, key: str) -> str:
    done = [i for i in job.images if i["face"] is not None]
    if key == "label":
        return ", ".join(f"{i['face']} · {i['kind']}" for i in done)
    if key == "outline":
        return _plural(sum(1 for i in job.images if i["outline"]), "closed outline")
    if key == "read":
        values = [r["value_mm"] for i in job.images for r in i["reads"]]
        return " · ".join(_mm(v) for v in values) + " mm" if values else "No numbers found"
    return ""


def _draw_done(job: Job, filled_by: dict) -> None:
    stage = job.stage("draw")
    for face, by in filled_by.items():
        if face in job.coverage and job.coverage[face] != "observed":
            job.coverage[face] = by
    filled = {f: by for f, by in filled_by.items() if by != "observed"}
    if not filled:
        stage.update(state="skipped", detail="Every face was observed")
        return
    for by in ("qwen-image", "triposr", "mirrored", "assumed"):
        if by in filled.values():
            tool, ai = DRAW_TOOLS[by]
            stage.update(tool=tool, ai=ai)
            break
    groups = {}
    for face, by in filled.items():
        groups.setdefault(by, []).append(face)
    stage["detail"] = "; ".join(f"{' and '.join(faces)} {DRAW_WORDS.get(by, by)}" for by, faces in groups.items())


def reduce(job: Job, name: str, data: dict) -> None:
    """Apply one progress event to the job."""
    if name != "stage" or data.get("key") not in STAGES:
        return
    key, state, now = data["key"], data["state"], time.time()
    with job.lock:
        stage = job.stage(key)
        index = data.get("index")
        image = job.images[index] if index is not None and 0 <= index < len(job.images) else None
        if state == "running":
            stage["state"] = "running"
            stage["started"] = stage["started"] or now
            stage["ended"] = None
            return
        if state == "skipped":
            if stage["state"] != "done":
                stage.update(state="skipped", ended=now)
                if key == "read" and not stage["detail"]:
                    stage["detail"] = "No numbers to read on a photo"
            return
        if state != "done":
            stage.update(state=state, ended=now)
            return
        stage.update(state="done", ended=now)
        stage["started"] = stage["started"] or now
        if key == "label" and image is not None:
            image.update(face=data.get("face"), kind=data.get("kind"), width=data.get("width", 0),
                         height=data.get("height", 0))
            if data.get("face") in job.coverage:
                job.coverage[data["face"]] = "observed"
        elif key == "outline" and image is not None:
            image["outline"] = data.get("outline")
            image["circles"] = [{"cx": float(c["cx"]), "cy": float(c["cy"]), "d": float(c["d"])}
                                for c in data.get("circles", [])]
        elif key == "read" and image is not None:
            image["reads"] = [{**r, "bbox": list(r["bbox"])} for r in data.get("reads", [])]
        if key == "draw":
            _draw_done(job, data.get("filled_by", {}))
        elif key != "fuse":
            stage["detail"] = _details(job, key)


def _analysis(job: Job, res, filled_by: dict) -> dict:
    spec = None if isinstance(res, MvAbstain) else res.model_dump(mode="json")
    abstain = res.model_dump() if isinstance(res, MvAbstain) else None
    return {"request_id": job.job_id, "spec": spec, "abstain": abstain, "filled_by": dict(filled_by)}


def _finish(job: Job, res, filled_by: dict) -> None:
    now = time.time()
    with job.lock:
        if isinstance(res, MvAbstain):
            for stage in job.stages:
                if stage["state"] == "running":
                    stage.update(state="failed", ended=now, detail=res.remedy)
                elif stage["state"] == "pending":
                    failed = stage["key"] == "fuse"
                    stage.update(state="failed" if failed else "skipped", detail=res.remedy if failed else "")
        else:
            env = res.envelope
            job.stage("fuse")["detail"] = f"one part, {_mm(env.x_mm)} × {_mm(env.y_mm)} × {_mm(env.z_mm)} mm"
        job.result = _analysis(job, res, filled_by)
        job.status = "done"


def run(job: Job, pipe: MvPipeline, images: list[ImageInput], reference: str | None) -> None:
    def progress(name: str, data: dict) -> None:
        if job.cancel:
            raise JobCancelled()
        reduce(job, name, data)

    try:
        observed = pipe.observe(images, reference, progress=progress)
        if isinstance(observed, MvAbstain):
            _finish(job, observed, {})
            return
        job.observed = observed
        res = pipe.fuse(observed, progress=progress)
        _forget_images(observed, res)
        _finish(job, res, observed.filled_by)
    except JobCancelled:
        with job.lock:
            job.status = "cancelled"
    except Exception:
        log.exception("analysis %s failed", job.job_id)
        with job.lock:
            job.status, job.error = "failed", FAILED
            for stage in job.stages:
                if stage["state"] == "running":
                    stage.update(state="failed", ended=time.time())


def start(job: Job, pipe: MvPipeline, images: list[ImageInput], reference: str | None) -> None:
    threading.Thread(target=run, args=(job, pipe, images, reference), daemon=True,
                     name=f"analysis-{job.job_id[:8]}").start()


def merge(job: Job, pipe: MvPipeline, user_values: dict, accepted: list, rejected: list) -> dict:
    observed = job.observed
    with job.merge_lock:
        res = pipe.fuse(observed, user_values, accepted, rejected)
        _forget_images(observed, res)
    return _analysis(job, res, observed.filled_by)
