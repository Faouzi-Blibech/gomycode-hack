# Sketch-to-CAD Web App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the Claude Design screens (Analyzing, Review, Model & Export) as a working React web app wired to the real multi-view pipeline through a new FastAPI surface, demoable on 27 September 2026 before 16:00.

**Architecture:** A new `s2c/web/` package serves `/api/*` on one FastAPI app (`s2c.web.server:app`) and, when built, the React app from `web/dist`. Analysis runs as a background job that records real pipeline events (labels, traced outlines, read numbers, filled faces) through a new optional `progress` callback on `MvPipeline.observe` and `MvPipeline.fuse`. The frontend polls the job and plays those real events on the Analyzing screen. Review edits go back through `/api/merge`, and Model & Export uses the existing cached `artifacts.build_part` and `export_part`.

**Tech Stack:** Python 3.11, FastAPI, the existing `s2c.multiview` pipeline, pytest. Frontend: Vite, React 18, TypeScript, three (GLTFLoader), Vitest.

**Spec:** Visual source of truth = the Claude Design exports at the repo root: `Analyzing v2.dc.html`, `Review v2.dc.html`, `Model v2.dc.html` (templates with `{{ }}` bindings plus a `<script type="text/x-dc">` logic block). Product rules = `CLAUDE.md` and `design/claude-design/01..04-*.md` (local, untracked). Data contracts = `s2c/multiview/spec.py`, `settings.py`, `routes.py`.

## Global Constraints

- Every millimetre shown carries its provenance badge. Mapping copied verbatim from `Review v2.dc.html` `BADGE`: `user_written` "Written by you" ✓ trusted solid; `measured` "Measured" ✓ trusted solid; `user_edited` "Edited by you" ✓ trusted solid; `scaled` "Scaled from reference" ✓ trusted dotted; `inferred` "AI-drawn — check" ◇ ai dashed; `estimated` "Estimated — check" ! check dashed; `default` "Default — check" ! check dashed; `required` "Required" • stop dashed.
- Colour tokens, fonts (Geist, Geist Mono, Silkscreen) and the dark and light themes are copied verbatim from the `:root` and `:root[data-theme=light]` blocks of `Analyzing v2.dc.html`.
- The AI always wears `var(--ai)`. Stage chips say `AI` or `CODE`.
- Privacy timer: 60-minute countdown stored in `localStorage` key `s2c_deadline`, restarted on a new analysis. Theme in `s2c_theme`, sidebar state in `s2c_sb`.
- The Analyzing animation plays only real data from the job (outlines, read values, filled faces). Pacing may slow events for legibility; it never invents a value.
- The review screen sends back as `user_values` only numbers the user typed or confirmed, never `abstain.partial` echoed.
- Stop cards show `remedy` verbatim, with one action.
- No exception text or traceback reaches the browser. Errors become `{"error": "<plain sentence>"}` with a 4xx or 5xx status.
- Files and jobs expire after 3600 s. Every `/api` request sweeps.
- Never hard-code a model or provider name in source; show what `/api/status` reports.
- Commits: plain messages, **no `Co-Authored-By`, no "Generated with", no AI attribution**. Commit only your own files with explicit `git add <paths>`. If `.git/index.lock` exists, wait 3 s and retry; never delete it.
- Desktop 1440 × 900 is the target (the designs). Below 1024 px wide the layout must stack and stay usable, but pixel parity is not required.

## Review Focus

- A sketch with no readable numbers produces a `missing_x` stop card. Review must show the Width box as Required and focus it from the card, and typing a value must re-merge and clear the stop.
- The pipeline runs with no AI providers configured (no chat, no reader, no image generator). Faces come from the user's tags. The app must still complete with honest "CODE" stages and "skipped" AI stages.
- A slow provider (Qwen-Image 20–40 s) must not freeze the UI. Polling continues, the current stage shows its running text, and Cancel returns to Capture.
- An expired `request_id` or artifact (after one hour, or a server restart) returns 404 with a plain message. The UI shows a card with "Analyze again".
- A GLB that fails to load in the viewer must leave the rest of the Model screen usable, with a "3D preview unavailable" note.

---

## File structure

| Path | Responsibility |
| --- | --- |
| `s2c/multiview/pipeline.py` (modify) | Optional `progress` callback on `observe` and `fuse`; no behaviour change when it is `None` |
| `s2c/web/__init__.py` | Package marker |
| `s2c/web/jobs.py` | `Job` record, event reducer into the job JSON, in-memory registry with TTL, background runner |
| `s2c/web/api.py` | `APIRouter(prefix="/api")`: status, examples, analyze, jobs, merge, model, export, artifacts |
| `s2c/web/server.py` | `app`: FastAPI with the router, CORS for the Vite dev origin, and `web/dist` static mount when present |
| `tests/test_pipeline_progress.py`, `tests/test_web_api.py` | Backend tests |
| `web/` | Vite React TS app (see Task 3) |

## The API contract (both tracks build against this; do not change it without the controller)

All JSON. Base path `/api`. The server is `uv run uvicorn s2c.web.server:app --port 8000`, and Vite proxies `/api` to it.

```ts
// web/src/api/types.ts — the single source of truth for the frontend
export type Face = 'front' | 'back' | 'left' | 'right' | 'top' | 'bottom';
export type Provenance = 'user_written' | 'measured' | 'user_edited' | 'scaled' | 'inferred' | 'estimated' | 'default';
export type StageKey = 'label' | 'outline' | 'read' | 'draw' | 'fuse';
export type StageState = 'pending' | 'running' | 'done' | 'skipped' | 'failed';
export type FilledBy = 'observed' | 'qwen-image' | 'triposr' | 'mirrored' | 'assumed';

export interface Status { providers: { vision: boolean; reader: boolean; qwen_image: boolean; triposr: boolean; solaria: boolean; slicer: boolean; blender: boolean }; ttl_s: number }
export interface Example { name: string; url: string; face: Face; kind: 'sketch' | 'photo' | 'drawing' }

export interface Stage { key: StageKey; state: StageState; tool: string; ai: boolean; detail: string; started: number | null; ended: number | null }
export interface ReadValue { text: string; value_mm: number; kind: 'linear' | 'diameter' | 'radius'; bbox: [number, number, number, number]; confidence: number }
export interface JobImage { index: number; width: number; height: number; face: Face | 'unknown' | null; kind: 'sketch' | 'photo' | 'drawing' | null;
  outline: [number, number][] | null; circles: { cx: number; cy: number; d: number }[]; reads: ReadValue[] }
export interface Job { job_id: string; status: 'running' | 'done' | 'failed' | 'cancelled'; stages: Stage[]; images: JobImage[];
  coverage: Record<Face, FilledBy | 'empty'>; result: Analysis | null; error: string | null }

export interface Outline { outer: [number, number][]; inner: [number, number][][]; source: 'observed' | 'mirrored' | 'inferred' | 'assumed'; confidence: number }
export interface Hole { type: 'hole'; face: Face; a_mm: number; b_mm: number; diameter_mm: number; depth_mm?: number | null }
export interface Slot { type: 'slot'; face: Face; a_mm: number; b_mm: number; width_mm: number; length_mm: number; angle_deg: number; depth_mm?: number | null }
export interface Spec { version: 'mv1'; envelope: { x_mm: number; y_mm: number; z_mm: number }; views: { front: Outline; top: Outline; right: Outline };
  features: (Hole | Slot)[]; finishes: { type: 'fillet' | 'chamfer'; edges: string; radius_mm: number }[];
  provenance: Record<string, Provenance>; snapped: string[]; warnings: string[]; confidence: number }
export interface Abstain { stage: string; reason: string; remedy: string; partial: Record<string, number> | null }
export interface Analysis { request_id: string; spec: Spec | null; abstain: Abstain | null; filled_by: Partial<Record<Face, FilledBy>> }

export interface AiSettings { use_reader: boolean; use_qwen_image: boolean; use_rescue: boolean; use_triposr: boolean; use_solaria: boolean; seed: number; randomize_seed: boolean; attempts: number }
export interface GeometrySettings { snap: boolean; clearance: 'fine' | 'medium' | 'coarse'; finish: 'none' | 'fillet' | 'chamfer'; finish_mm: number; finish_edges: 'all_vertical' | 'top' | 'bottom' | 'all' }
export interface PrintSettings { material: 'PLA' | 'PETG' | 'ABS' | 'ASA' | 'TPU'; nozzle_mm: number; layer_mm: number; infill_pct: number;
  infill_pattern: 'grid' | 'gyroid' | 'rectilinear' | 'honeycomb' | 'cubic' | 'lightning'; perimeters: number; supports: 'off' | 'buildplate' | 'everywhere'; brim_mm: number; scale_pct: number }
export interface ModelResult { key: string | null; glb_url: string | null; volume_cm3: number | null; bbox_mm: [number, number, number] | null;
  iou: Partial<Record<Face, number>>; iou_mean: number | null; views: Partial<Record<Face, string>>; warnings: string[]; abstain: Abstain | null }
export interface ExportFile { url: string; name: string; size_bytes: number }
export interface ExportResult { files: Record<string, ExportFile>; zip_url: string | null; print_time_s: number | null; filament_g: number | null; warnings: string[]; abstain: Abstain | null }
```

| Method and path | Request | Response |
| --- | --- | --- |
| `GET /api/status` | — | `Status` |
| `GET /api/examples` | — | `Example[]`, from `examples/mv/sketches/examples.json`; `url` = `/api/examples/<file>` |
| `GET /api/examples/{file}` | — | the PNG (name must match `^[\w.-]+\.png$` and exist in the folder) |
| `POST /api/analyze` | multipart: `files` (1–6 images), `faces` JSON list (`auto` or a face), `kinds` JSON list (`auto`/`sketch`/`photo`/`drawing`), `reference` (`""`, `1 TND`, `1 EUR`, `2 EUR`, `card`, `a4`), `ai` JSON `AiSettings` (optional) | `{"job_id": str}`, 202. 413 over 10 MB per file, 415 not an image, 400 for 0 or more than 6 files |
| `GET /api/jobs/{job_id}` | — | `Job`. 404 `{"error": "Unknown or expired analysis. Analyze again."}` |
| `POST /api/jobs/{job_id}/cancel` | — | `{"ok": true}`. The worker stops at the next progress event |
| `POST /api/merge` | `{"request_id", "user_values": {path: float}, "accepted": [face], "rejected": [face]}` | `Analysis`; 404 as above |
| `POST /api/model` | `{"request_id": str \| null, "spec": Spec, "geometry": GeometrySettings}` | `ModelResult` |
| `POST /api/export` | `{"spec": Spec, "settings": StudioSettings}` | `ExportResult` |
| `GET /api/artifacts/{key}/{path}` | — | the file, with the same guards as `routes.artifact` |

`user_values` paths: `envelope.x_mm`, `envelope.y_mm`, `envelope.z_mm`, and `features[i].<field>` (e.g. `features[0].diameter_mm`). Confirming a "check" value sends its current value, which stamps it `user_edited`.

---

### Task 1: Pipeline progress events

**Files:**
- Modify: `s2c/multiview/pipeline.py` (`observe` lines 121–160, `fuse` lines 198–238)
- Test: `tests/test_pipeline_progress.py`

**Interfaces:**
- Produces: `Progress = Callable[[str, dict], None]`; `MvPipeline.observe(images, reference=None, progress: Progress | None = None)` and `MvPipeline.fuse(observed, user_values=None, accepted=(), rejected=(), geometry=None, progress: Progress | None = None)`.
- Events (name, data), in call order:
  - `("stage", {"key": "label", "state": "running"|"done", "index": i})` around `_label` for image i. Done carries `"face"`, `"kind"`, `"confidence"`, and `"width"`/`"height"` of the resized `bgr` (after `resize_long_side`).
  - `("stage", {"key": "outline", "state": "running"|"done", "index": i})` around `_outline`. Done carries `"outline": outer.tolist()` (int px pairs) and `"circles": [{"cx","cy","d"}]`.
  - `("stage", {"key": "read", "state": "running"|"done"|"skipped", "index": i})` around `read_values`. Done carries `"reads": [{"text","value_mm","kind","bbox","confidence"}]` from each `Linked.reading`. Skipped when `reads` is false or the kind is photo.
  - `("stage", {"key": "draw", "state": "running"|"done"})` around `complete(...)` in `fuse`. Done carries `"filled_by": dict(observed.filled_by)`.
  - `("stage", {"key": "fuse", "state": "running"|"done"})` around `assemble(...)`.
- A progress callback that raises `JobCancelled` (defined in `s2c/web/jobs.py`, a plain `Exception` subclass) must propagate out of `observe`/`fuse` untouched. Do not catch it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline_progress.py
from pathlib import Path

from s2c.multiview.pipeline import ImageInput, MvPipeline

SK = Path(__file__).resolve().parents[1] / "examples" / "mv" / "sketches"


def images():
    return [ImageInput((SK / "front.png").read_bytes(), "front", "sketch"),
            ImageInput((SK / "top.png").read_bytes(), "top", "sketch")]


def test_observe_and_fuse_emit_stage_events_in_order():
    events = []
    pipe = MvPipeline()
    observed = pipe.observe(images(), None, progress=lambda n, d: events.append((n, d)))
    pipe.fuse(observed, {"envelope.x_mm": 50, "envelope.y_mm": 30, "envelope.z_mm": 20},
              progress=lambda n, d: events.append((n, d)))
    keys = [(d["key"], d["state"]) for n, d in events if n == "stage"]
    assert keys[:2] == [("label", "running"), ("label", "done")]
    assert ("outline", "done") in keys and ("draw", "done") in keys and keys[-1] == ("fuse", "done")
    outline = next(d for n, d in events if d.get("key") == "outline" and d["state"] == "done")
    assert len(outline["outline"]) >= 3 and all(len(p) == 2 for p in outline["outline"])
    label = next(d for n, d in events if d.get("key") == "label" and d["state"] == "done")
    assert label["face"] == "front" and label["width"] > 0 and label["height"] > 0


def test_no_progress_callback_changes_nothing():
    pipe = MvPipeline()
    a = pipe.fuse(pipe.observe(images()), {"envelope.x_mm": 50, "envelope.y_mm": 30, "envelope.z_mm": 20})
    b = pipe.fuse(pipe.observe(images(), progress=lambda n, d: None),
                  {"envelope.x_mm": 50, "envelope.y_mm": 30, "envelope.z_mm": 20}, progress=lambda n, d: None)
    assert a.model_dump() == b.model_dump()


class Stop(Exception):
    pass


def test_a_raising_callback_stops_the_pipeline():
    def cb(n, d):
        if d.get("key") == "outline":
            raise Stop()
    try:
        MvPipeline().observe(images(), progress=cb)
    except Stop:
        return
    raise AssertionError("progress exception was swallowed")
```

- [ ] **Step 2:** Run `uv run pytest tests/test_pipeline_progress.py -v`. Expected: FAIL (`unexpected keyword argument 'progress'`).
- [ ] **Step 3:** Implement. Add `Progress = Callable[[str, dict], None]` and a helper `def _emit(progress, **data): progress and progress("stage", data)`. Wrap each call site as listed. Keep every existing return path; for early `return` abstains, emit nothing extra.
- [ ] **Step 4:** Run the new tests plus `uv run pytest tests -q -k "mv or multiview or pipeline"`. Expected: PASS, no regression.
- [ ] **Step 5:** Commit `git add s2c/multiview/pipeline.py tests/test_pipeline_progress.py && git commit -m "Report pipeline stage progress through an optional callback"`.

### Task 2: Web API and server

**Files:**
- Create: `s2c/web/__init__.py`, `s2c/web/jobs.py`, `s2c/web/api.py`, `s2c/web/server.py`
- Test: `tests/test_web_api.py`

**Interfaces:**
- Consumes: Task 1 events; `MvPipeline`, `ImageInput`, `Observed` from `s2c.multiview.pipeline`; `artifacts.build_part(spec, geometry, root)`, `export_part(part, formats, mesh, printing)`, `bundle(part, result, settings)`, `sweep(root)`; `silhouette.iou`; `StudioSettings`, `GeometrySettings`, `AiSettings`.
- Produces: `s2c.web.server:app`, the contract above; `s2c.web.api.get_pipeline()` (lru_cached `default_pipeline()`), overridable in tests with `app.dependency_overrides[get_pipeline] = lambda: MvPipeline()`.

Behaviour details:
- **Jobs** (`jobs.py`): `Job` dataclass holding the contract fields, plus `observed`, `created`, `cancel` flag and a lock. `reduce(job, name, data)` applies one event:
  - stage state updates, with `started`/`ended` from `time.time()`
  - `label` done → `images[i]` face/kind/width/height
  - `outline` done → outline/circles
  - `read` done → reads
  - `draw` done → coverage
  - observed faces set coverage to `observed`
  - faces never filled remain `empty`
  The stage `tool` and `ai` fields are fixed: label → ("Vision model", true), outline → ("OpenCV", false), read → ("Qwen-VL", true), draw → ("Qwen-Image", true), fuse → ("CadQuery", false). Stage `detail`, when done, is a plain sentence, e.g. "front · sketch", "2 closed outlines", "50 · 30 · 20 mm", "right drawn — check it", "one part, 50 × 30 × 20 mm".
  - If the pipeline has no chat, the label stage `tool` becomes "Your face tags" with `ai: false`.
  - If no reader is configured, read is `skipped`.
  - If `image_gen` is None or drawing is off, draw's `tool` is the provider that actually filled (`TripoSR`, `mirror`, `assumed`) with `ai` true only for Qwen-Image or TripoSR.
- Registry `JOBS: dict[str, Job]` with `sweep_jobs(ttl=3600)` and a `threading.Thread(daemon=True)` runner. The runner calls `observe(..., progress=cb)` then `fuse(observed, progress=cb)`, and stores `request_id = job_id` (the same id serves `/api/merge`). It sets `result = Analysis` and `status = "done"`. On `JobCancelled` it sets `cancelled`. On any other exception it logs the exception and sets `failed` with `error = "Analysis failed. Try again or use different photos."`.
- After a successful fuse, `observed.images.clear()` (same privacy rule as `routes._forget_images`).
- **Upload checks:** at most 10 MB per file, and the bytes must start with JPEG `\xff\xd8\xff` or PNG `\x89PNG\r\n\x1a\n` magic.
- **`/api/model`**: `build_part(spec, geometry, ARTIFACT_ROOT)` returns `ModelResult`:
  - `glb_url = /api/artifacts/{key}/{preview name}`, `volume_cm3 = volume_mm3/1000` rounded to 2 dp, `bbox_mm` from the part.
  - If the request's `observed.masks` exist, `iou[face] = round(iou(part.views[face], mask), 3)` for faces in both, and `iou_mean` is their mean.
  - `views[face]` is a PNG written to `part.folder / f"view_{face}.png"`, served through artifacts.
  - An abstain puts all other fields null and empty.
- **`/api/export`**: as `routes.export_files`, plus `name` and `size_bytes` per file.
- **Errors**: HTTPException details render as `{"error": ...}` through an exception handler on the router's app; a generic handler returns 500 `{"error": "Something went wrong on our side."}` and logs.
- **`server.py`**:
  - CORS allows `http://localhost:5173` and `http://127.0.0.1:5173`, plus any extra origins from the comma-separated `ALLOWED_ORIGINS` environment variable.
  - Mount `StaticFiles(directory="web/dist", html=True)` at `/` only if `web/dist/index.html` exists, after the router.
  - Also include `s2c.multiview.routes.router` so `/mv/*` keeps working.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_web_api.py
import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from s2c.multiview.pipeline import MvPipeline
from s2c.web.api import get_pipeline
from s2c.web.server import app

SK = Path(__file__).resolve().parents[1] / "examples" / "mv" / "sketches"
app.dependency_overrides[get_pipeline] = lambda: MvPipeline()
c = TestClient(app, raise_server_exceptions=False)


def analyze(values=True):
    files = [("files", ("front.png", (SK / "front.png").read_bytes(), "image/png")),
             ("files", ("top.png", (SK / "top.png").read_bytes(), "image/png"))]
    r = c.post("/api/analyze", files=files, data={"faces": json.dumps(["front", "top"]),
                                                  "kinds": json.dumps(["sketch", "sketch"]), "reference": ""})
    assert r.status_code == 202, r.text
    jid = r.json()["job_id"]
    for _ in range(200):
        job = c.get(f"/api/jobs/{jid}").json()
        if job["status"] != "running":
            return job
        time.sleep(0.05)
    raise AssertionError("job never finished")


def test_status_and_examples():
    assert "providers" in c.get("/api/status").json()
    ex = c.get("/api/examples").json()
    assert {e["face"] for e in ex} == {"front", "top"}
    assert c.get(ex[0]["url"]).status_code == 200
    assert c.get("/api/examples/..%2Fpyproject.toml").status_code == 404


def test_analyze_job_reports_real_stages_and_images():
    job = analyze()
    assert job["status"] == "done", job
    assert [s["key"] for s in job["stages"]] == ["label", "outline", "read", "draw", "fuse"]
    assert job["images"][0]["outline"] and job["images"][0]["width"] > 0
    assert job["coverage"]["front"] == "observed"
    res = job["result"]
    assert res["request_id"] == job["job_id"]
    assert res["spec"] or res["abstain"]


def test_merge_with_sizes_gives_a_spec_then_model_and_export():
    job = analyze()
    m = c.post("/api/merge", json={"request_id": job["job_id"], "accepted": [], "rejected": [],
                                    "user_values": {"envelope.x_mm": 50, "envelope.y_mm": 30, "envelope.z_mm": 20}}).json()
    assert m["spec"], m
    assert m["spec"]["provenance"]["envelope.x_mm"] in ("user_written", "user_edited")
    geo = {"snap": True, "clearance": "medium", "finish": "none", "finish_mm": 1.0, "finish_edges": "all_vertical"}
    model = c.post("/api/model", json={"request_id": job["job_id"], "spec": m["spec"], "geometry": geo}).json()
    assert model["glb_url"] and c.get(model["glb_url"]).status_code == 200
    assert model["volume_cm3"] > 0 and len(model["bbox_mm"]) == 3
    ex = c.post("/api/export", json={"spec": m["spec"], "settings": {"export": {"formats": ["stl", "step"]}}}).json()
    assert set(ex["files"]) >= {"stl", "step"} and ex["files"]["stl"]["size_bytes"] > 0
    assert c.get(ex["files"]["stl"]["url"]).status_code == 200


def test_unknown_job_and_request_are_plain_404s():
    r = c.get("/api/jobs/" + "0" * 32)
    assert r.status_code == 404 and "error" in r.json()
    r = c.post("/api/merge", json={"request_id": "0" * 32, "user_values": {}, "accepted": [], "rejected": []})
    assert r.status_code == 404 and "Traceback" not in r.text


def test_upload_limits():
    files = [("files", ("x.png", b"#!/bin/sh\n", "image/png"))]
    assert c.post("/api/analyze", files=files, data={"faces": "[]", "kinds": "[]"}).status_code == 415
    assert c.post("/api/analyze", files=[], data={"faces": "[]", "kinds": "[]"}).status_code in (400, 422)


def test_artifact_traversal_is_404():
    assert c.get("/api/artifacts/" + "a" * 20 + "/..%2F..%2Fpyproject.toml").status_code == 404
```

- [ ] **Step 2:** Run `uv run pytest tests/test_web_api.py -v`. Expected: FAIL (`No module named 's2c.web'`).
- [ ] **Step 3:** Implement `jobs.py`, `api.py` and `server.py` per the behaviour details. Reuse `routes._ID`, `routes._NAME`, `routes._KEY` and the `artifact` guard logic by importing them, not copying.
- [ ] **Step 4:** Run `uv run pytest tests/test_web_api.py tests/test_pipeline_progress.py -v`, then the full `uv run pytest -q` and `uv run ruff check s2c/web tests/test_web_api.py`. Expected: PASS.
- [ ] **Step 5:** Start `uv run uvicorn s2c.web.server:app --port 8000` for 5 seconds, `curl -s localhost:8000/api/status`, stop it. Paste the output in the report.
- [ ] **Step 6:** Commit `git add s2c/web tests/test_web_api.py && git commit -m "Add the web API with analysis jobs, merge, model and export"`.

### Task 3: Frontend scaffold, tokens, shell and API client

**Files:**
- Create: `web/package.json`, `web/tsconfig.json`, `web/vite.config.ts`, `web/index.html`, `web/src/main.tsx`, `web/src/App.tsx`, `web/src/styles/tokens.css`, `web/src/styles/base.css`, `web/src/api/types.ts` (verbatim from the contract), `web/src/api/client.ts`, `web/src/state/store.tsx`, `web/src/lib/provenance.ts`, `web/src/lib/provenance.test.ts`, `web/src/components/Shell.tsx`, `web/src/components/Badge.tsx`, `web/src/components/StopCard.tsx`, and placeholder screens `web/src/screens/{Capture,Analyzing,Review,Model}.tsx` (each exports a component rendering its title, replaced by Tasks 4–7).
- Modify: `.gitignore` (add `web/node_modules/`, `web/dist/`).

**Interfaces (Tasks 4–7 import exactly these):**
- `client.ts`:
  - `getStatus(): Promise<Status>`
  - `getExamples(): Promise<Example[]>`
  - `startAnalysis(files: File[], faces: string[], kinds: string[], reference: string, ai?: Partial<AiSettings>): Promise<string>`
  - `getJob(id): Promise<Job>`
  - `cancelJob(id): Promise<void>`
  - `merge(body: {request_id: string; user_values: Record<string, number>; accepted: Face[]; rejected: Face[]}): Promise<Analysis>`
  - `buildModel(body: {request_id: string | null; spec: Spec; geometry: GeometrySettings}): Promise<ModelResult>`
  - `exportFiles(body: {spec: Spec; settings: {geometry: GeometrySettings; mesh: {quality: 'draft'|'normal'|'fine'}; printing: PrintSettings; export: {formats: string[]}}}): Promise<ExportResult>`

  Every call throws `ApiError(status, message)`, where `message` is the server's `error` or a plain fallback.
- `store.tsx`: a `StoreProvider` and `useStore()` returning `{state, dispatch}`. The state is `{screen: 'capture'|'analyzing'|'review'|'model'; items: CaptureItem[]; reference: string; ai: AiSettings; jobId: string|null; job: Job|null; analysis: Analysis|null; typed: Record<string, number>; rejected: Face[]; geometry: GeometrySettings; model: ModelResult|null}`, where `CaptureItem = {id: string; file: File; url: string; face: Face|'auto'; kind: 'auto'|'sketch'|'photo'|'drawing'}`. Actions:
  - `ADD_FILES`, `SET_ITEM`, `REMOVE_ITEM`, `SET_REFERENCE`, `SET_AI`
  - `START_JOB {jobId}`, `JOB_UPDATE {job}`, `ANALYSIS {analysis}`
  - `TYPE_VALUE {path, value}`, `TOGGLE_REJECT {face}`
  - `SET_GEOMETRY {patch}`, `MODEL {model}`, `GOTO {screen}`, `RESET`
- `provenance.ts` exports:
  - `BADGE: Record<Provenance|'required', {label, icon, fg, line, bg}>`, copied from the design
  - `CHECK = ['inferred','estimated','default']`
  - `envelopeRows(spec)`, returning `{key:'x'|'y'|'z'; label:'Width'|'Height'|'Depth'; axis:'X'|'Y'|'Z'; path; value; prov}[]`
  - `featureRows(spec)`, returning one row per numeric feature field with `{path, name, face, value, prov, snapped}` (name like "Hole 1 · Ø", "Hole 1 · a")
  - `countChecks(spec)`
- `Shell.tsx` props: `{nav: {num, label, st: 'current'|'done'|'open'|'locked', sub, subC?, spinning?}[]; onNav(screen); children}`. It renders the sidebar (collapsible 248/76 px), the theme segmented toggle and the privacy timer ring exactly as in the design's `<aside>`. `useDeadline()` hook: starts or reads `s2c_deadline`; `resetDeadline()` is called when a new analysis starts.

- [ ] **Step 1:** In `web/`, write `package.json` with deps `react@18`, `react-dom@18`, `three@^0.160`, and devDeps `vite@^5`, `@vitejs/plugin-react`, `typescript`, `@types/react`, `@types/react-dom`, `@types/three`, `vitest`. Scripts: `dev`, `build` (`tsc -b && vite build`), `test` (`vitest run`). Then run `npm install`.
- [ ] **Step 2:** In `vite.config.ts`, add a proxy `'/api': 'http://127.0.0.1:8000'`, and `server.host: true`.
- [ ] **Step 3:** Write the failing `provenance.test.ts`, using `design/claude-design/samples/analyze-success.json` copied to `web/src/lib/__fixtures__/analyze-success.json`. Assert:
  - `envelopeRows` returns Width 50 / `user_written`, Height 30, Depth 20 / `user_edited`.
  - `featureRows` includes "Hole 2 · Ø" with prov `default` and `snapped: true`.
  - `countChecks === 3`, counting rows (hole 1 position counts once for a+b, hole 2 Ø once, hole 2 position once).
- [ ] **Step 4:** `npm test`, which must FAIL. Then implement `provenance.ts`, and `npm test` must PASS.
- [ ] **Step 5:** Copy the `:root` and light-theme blocks into `tokens.css` verbatim. Load the fonts link in `index.html`. Port the `<aside>` shell into `Shell.tsx` with React state for sidebar, theme and timer, using the same inline style values. Port the sidebar, the theme toggle and the privacy timer from `Analyzing v2.dc.html` lines 1–60.
- [ ] **Step 6:** `App.tsx` switches on `state.screen` inside `Shell`. Nav states:
  - Capture is current until a job starts.
  - Review is locked until `analysis` exists.
  - Model is locked until a spec exists and a model was requested.
- [ ] **Step 7:** Run `npm run build`, which must be clean, and `npm test`, which must pass.
- [ ] **Step 8:** Commit `git add web .gitignore -- ':!web/node_modules' && git commit -m "Scaffold the web app with design tokens, shell and API client"`.

### Task 4: Capture screen

**Files:** Modify `web/src/screens/Capture.tsx`; create `web/src/components/CoverageCube.tsx`.

The design has no Capture screen. Build it in the same language: the Shell, a `var(--surface)` panel with a 1 px `var(--line)` border and 14–18 px radius, `var(--shadow)`, Geist and Geist Mono, Silkscreen for the small uppercase tags, and the accent button style from the design's primary CTA.

- Left 60 %: a drop zone (dashed `var(--line)`, "Drop sketches or photos, or click to choose", accepting `image/*` with `multiple`), and a card grid of items. Each card shows the thumbnail, a face select (auto + 6 faces), kind chips (auto/sketch/photo/drawing) and Remove. Maximum 6 items, with a note when full.
- Right 40%:
  - **Coverage cube**: an unfolded net of the 6 faces with the design's `F` styles from `Analyzing v2.dc.html` line 455 (photo ✓×N, empty —). It is built from item face tags; an `auto` tag does not count.
  - Scale reference select: none, 1 TND coin, 1 EUR, 2 EUR, bank card, A4.
  - **Reading & AI** collapsible: toggles for use_reader, use_qwen_image, use_rescue, use_triposr, use_solaria (with a "+60–180 s" hint), a seed number input, and attempts 1–4.
  - **Try an example**: fetches `/api/examples`, converts each to a `File`, and adds them with their face and kind.
  - The primary **Analyze** button is disabled with 0 items. It calls `startAnalysis`, `resetDeadline()`, dispatches `START_JOB` and goes to `analyzing`. On `ApiError`, a `StopCard` shows the message.
- Keyboard: every control is focusable and labelled.
- [ ] Steps:
  1. Implement.
  2. `npm run build` must be clean.
  3. Run `npm run dev` together with the API, and load the example to check it by eye.
  4. Commit `git add web/src/screens/Capture.tsx web/src/components/CoverageCube.tsx && git commit -m "Add the capture screen"`.

### Task 5: Analyzing screen driven by real job events

**Files:** Modify `web/src/screens/Analyzing.tsx`; create `web/src/lib/pacing.ts`, `web/src/lib/pacing.test.ts`.

Port `Analyzing v2.dc.html` (template lines 1–297, logic 298–505) to React, faithful in layout, styles and motion. Replace the scripted timeline with real data.

**Polling:** `getJob` every 400 ms while running; dispatch `JOB_UPDATE`. On `done`, dispatch `ANALYSIS`. On `failed`/`cancelled`, show a `StopCard` with `job.error` and "Back to capture".

**`pacing.ts`:** `pace(job, startedAt, now): Playback`. It turns job events into a display clock with minimum durations, so a fast local run is still watchable. Each finished stage gets at least: label 0.9 s, outline 1.5 s, read 0.7 s per read (min 0.9 s), draw 1.2 s, fuse 0.9 s. Playback never runs ahead of real events: a stage is shown `done` only if the job says done **and** its minimum elapsed. It returns:
- `stages` with a display state
- `trace` 0..1 per image
- `readsShown` per image
- `faces` coverage per face
- `allDone`

Test `pacing.test.ts`:
- (a) A job with all stages done at t=0 shows the label stage running at 0.3 s and all done at the sum of the minimums.
- (b) A job still running `read` never shows `draw` as started.

**Mapping to the design:**

| Design element | Real data |
| --- | --- |
| `FRONT`/`TOP` polylines | `job.images[i].outline` in image px, scaled into the image box using `width`/`height` |
| `READS` boxes and chips | `job.images[i].reads[*].bbox`/`value_mm`; chip label "Qwen-VL reading" then "✓ Written by you" |
| Hole circle | `job.images[i].circles` |
| `STAGES` | `job.stages` with `tool`/`ai`/`detail` (`AI`/`CODE` chips) |
| Coverage cube states | `job.coverage` (`observed` → photo, `qwen-image`/`triposr` → ai, `mirrored` → mirror, `assumed` → dotted "assumed") |
| Ledger | Once `result.spec` exists: envelope values with badges from `provenance.ts`; while reading: the read values |
| Images | The local object URLs of the capture items, matched by index |

- If `result.abstain` exists, the CTA reads "Fix 1 value →" and goes to Review, which shows the stop card.
- The CTA is "Review N values →" when `allDone`.
- Cancel calls `cancelJob` and goes to Capture.
- Respect `prefers-reduced-motion` as the design does.
- [ ] Steps:
  1. Write `pacing.test.ts` and run it to see it fail.
  2. Implement `pacing.ts`, and the test must pass.
  3. Port the screen.
  4. `npm run build` must be clean.
  5. Commit `git add web/src/screens/Analyzing.tsx web/src/lib/pacing.ts web/src/lib/pacing.test.ts && git commit -m "Add the analyzing screen driven by real pipeline events"`.

### Task 6: Review screen with real data

**Files:** Modify `web/src/screens/Review.tsx`; create `web/src/components/FaceCard.tsx`.

Port `Review v2.dc.html` (template lines 1–287, logic 288–444) to React, faithful in layout and styles, with this data binding:
- **Envelope** from `envelopeRows(analysis.spec)`. With an abstain whose reason is `missing_x`/`missing_y`/`missing_z`, that box is `required` (red dashed, REQUIRED, glow), pre-filled from `abstain.partial` where present. The stop card shows `abstain.remedy` and a button that focuses the box.
- **Ledger** from `featureRows(spec)`. Editing a field dispatches `TYPE_VALUE {path, value}`, and the badge shows "Edited by you" at once. **Confirm** on a check row dispatches `TYPE_VALUE` with the current value.
- **Re-merge:** debounce 500 ms after the last edit, then call `merge({request_id, user_values: state.typed, accepted: [], rejected: state.rejected})` and dispatch `ANALYSIS`. While it runs, show a small "updating…" tag.
- **Face cards** (front, top, right): an SVG of `spec.views[face].outer` and `inner` (mm, flip y) with the badge from `analysis.filled_by[face]` (Observed, AI-drawn (Qwen-Image), 3D fallback (TripoSR), Mirrored, Assumed). AI faces get a Reject/Undo toggle that dispatches `TOGGLE_REJECT` and triggers a re-merge.
- **Handwriting read:** crops from `job.images[i].reads`. Draw the local image in a canvas cropped to `bbox` scaled from `width`/`height`, showing the value and "front sketch".
- **Warnings** from `spec.warnings`: icon ◇ when the text contains "Qwen-Image" or "drawn", ! otherwise, ✓ once every value for that feature has been edited.
- **Build button:** "Build anyway (N unchecked) →" when checks remain and nothing was typed, "Build part →" otherwise. It is disabled when there is an abstain. It calls `buildModel({request_id, spec, geometry: state.geometry})`, dispatches `MODEL` and goes to `model`. A `ModelResult.abstain` shows a StopCard here.
- [ ] Steps:
  1. Implement.
  2. `npm run build` must be clean.
  3. Commit `git add web/src/screens/Review.tsx web/src/components/FaceCard.tsx && git commit -m "Add the review screen with provenance badges and live re-merge"`.

### Task 7: Model and export screen

**Files:** Modify `web/src/screens/Model.tsx`; create `web/src/components/Viewer.tsx`, `web/src/components/MatchRing.tsx`.

Port `Model v2.dc.html` (template lines 1–255, logic 256–447) to React, faithful in layout and styles. Differences from the design:

**Viewer** uses `three` from npm (not unpkg).
- `GLTFLoader` loads `state.model.glb_url`. GLB units are metres, so scale by 1000 to show millimetres.
- Keep the design's lights, grid, orbit, auto-rotate, section-cut clipping plane, grow-in animation and dimension-label anchors.
- The anchors are placed at the loaded mesh's bounding-box edges, with labels showing `bbox_mm` values.
- On a load error, show "3D preview unavailable" in the viewer box and keep the page usable.

**Stats:**
- Size: `bbox_mm` joined with " × ", with the badge text from the envelope provenance.
- Volume: `volume_cm3`.
- Print time and filament: after export, from `ExportResult`, otherwise "after export".

**MatchRing:** `iou_mean` as a ring, green when ≥ 0.85, otherwise check colour with "check the dimensions". The six mini views come from `model.views` URLs, with each face's IoU as the caption when present.

**Shape controls:** finish (none/fillet/chamfer), finish size (0.2–10), edges ("outline corners" → `all_vertical`, "front face" → `top`, "back face" → `bottom`, "all" → `all`), clearance, snap. Each change dispatches `SET_GEOMETRY`, then calls `buildModel` (debounced 300 ms, last one wins) and dispatches `MODEL`. The "REBUILT ✓" tag shows for 900 ms.

**Export:**
- A format grid of the design's 12 formats.
- Mesh quality segmented control.
- A print drawer mapping to `PrintSettings` ("build plate" → `buildplate`).
- "Export N selected →" calls `exportFiles`. The download card lists `ExportResult.files` with name, size (KB or MB) and "Download" links, plus a "Download all (.zip)" primary link to `zip_url`.
- `warnings` are shown as check chips; `abstain` as a StopCard.
- [ ] Steps:
  1. Implement.
  2. `npm run build` must be clean.
  3. Commit `git add web/src/screens/Model.tsx web/src/components/Viewer.tsx web/src/components/MatchRing.tsx && git commit -m "Add the model and export screen with a live 3D viewer"`.

### Task 8: Integration, end-to-end check and run docs

**Files:**
- Modify: `README.md` (the "Run it" section)
- Create: `scripts/dev.ps1`, which starts the API and Vite together

Steps:
- [ ] `cd web && npm run build`, then start `uv run uvicorn s2c.web.server:app --port 8000`. Open `http://localhost:8000` in the browser (Playwright MCP) and drive the happy path:
  1. Capture → Try an example → Analyze.
  2. Watch the stages.
  3. Review: enter any missing size, confirm checks, Build.
  4. Model: see the 3D part, change the fillet, Export STL+STEP, download.
  5. Screenshot each screen.
- [ ] Fix any wiring bug found, with a failing test first where the bug is in pure logic.
- [ ] Run the full `uv run pytest -q`, `cd web && npm test && npm run build`.
- [ ] Update the README "Run it" section with `uv run uvicorn s2c.web.server:app --host 0.0.0.0 --port 8000`, and with `cd web && npm install && npm run build` (single-port demo) or `npm run dev` (hot reload).
- [ ] Commit `git add README.md scripts/dev.ps1 <fixed files> && git commit -m "Wire the web app end to end and document how to run it"`.

## Execution order and parallelism

- **Wave 1 (parallel, disjoint trees):** Task 1 (`s2c/multiview/pipeline.py`) and Task 3 (`web/`).
- **Wave 2 (parallel):** Task 2 (`s2c/web/`, after Task 1), and Tasks 4, 5, 6, 7 (each owns its own screen and component files, after Task 3).
- **Wave 3:** Task 8.
