# Qwen-Image faces, Qwen-VL reading, photo merging and Solaria depth: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the multi-view path read handwriting with Qwen-VL, merge several photos of one face, let Qwen-Image draw missing faces and rescue broken sketches, measure hole depth with Solaria, and put it all behind a Gradio lab app.

**Architecture:** The deterministic visual hull (`build.py`) and slicer stay the only geometry. New modules feed it better inputs: `qwen_reader.py` (batch transcription), `merge_views.py` (align, vote, median), `qwen_image.py` (DashScope or Space client), `qwen_faces.py` (face prompt, voxel consistency gate, sketch rescue), `depth.py` (Solaria point cloud to depth ratios). `pipeline.py` wires them; every provider is optional and falls back with a warning.

**Tech Stack:** Python 3.11, uv, pytest, ruff, OpenCV, NumPy, pydantic 2, httpx, gradio 6 and gradio_client 2, CadQuery.

**Spec:** `docs/superpowers/specs/2026-09-23-qwen-solaria-design.md` (extends `docs/superpowers/specs/2026-09-22-multiview-gcode-design.md`). Read both before starting.

## Global Constraints

- Work in `gomycode-hack/`. Run everything with `uv run`. Line length 120 (ruff). No lambda assignments (ruff E731).
- No model name or Space name in source. They come from `.env`; defaults live only in `.env.example`. (spec 3)
- The model never writes code. Qwen-VL returns JSON validated by pydantic, Qwen-Image returns images, Solaria returns a point cloud. (spec 2)
- Provenance: Qwen-VL reads are `user_written`; Qwen-Image outlines are `inferred`; Solaria depths are `estimated`. Only `user_written`, `measured` or `user_edited` pass the envelope gate. (spec 2)
- Do not modify `s2c/multiview/spec.py`, `s2c/multiview/build.py`, `s2c/multiview/slice.py`, `s2c/partspec/`, or `fuse_envelope` in `s2c/multiview/fuse.py`. (spec 1)
- Every external call has a timeout; a failure falls back to the existing behaviour and adds a warning. No new abstention reasons. (spec 2, 12)
- Every Qwen-VL, Qwen-Image and Solaria call appends one line to `logs/vlm.jsonl`. (spec 12)
- Unit tests never touch the network. Live tests are marked `network` and run only with `NETWORK_TESTS=1`. (spec 13)
- Write the failing test first. (CLAUDE.md)
- Commit messages are plain, in the team's voice. No `Co-Authored-By`, no "Generated with", no AI attribution. (CLAUDE.md)
- **Never push.** Commits stay local until the human partner decides.

## Review Focus

These inputs are implied by the spec but easy to break. Each one has a pinned test in the task named.

1. Several photos of one face where one photo is upside down: its holes must not add extra holes (Task 3, `test_a_photo_turned_upside_down_does_not_add_holes`).
2. Qwen-Image answers with an empty or all-black image, or the call fails: the face must fall back, never be used (Task 5, `test_an_empty_or_failed_answer_is_not_used`).
3. Qwen-VL returns fewer reads than crops, or an index out of range: missing crops read as empty, bad indices are retried (Task 2, `test_a_missing_crop_reads_as_empty`, `test_an_index_out_of_range_is_retried`).
4. A Hugging Face Space hangs: the call must give up at its timeout, not block the app (Task 4, `test_a_slow_space_times_out_quickly`; Task 8, `test_a_slow_solaria_times_out`).
5. The user types text, zero or a negative number in the Gradio values table: a message, no exception (Task 6, `test_bad_numbers_are_reported_not_raised`).

## File structure

| Path | Responsibility | Task |
| --- | --- | --- |
| `pyproject.toml`, `.env.example` | `httpx` dependency, `network` marker, provider settings | 1 |
| `s2c/multiview/ocr.py` | TrOCR tokenizer fix; `BatchReader`; `read_values` accepts a batch reader | 1, 2 |
| `s2c/multiview/label.py` | `env_chat(stage=...)` for log lines | 2 |
| `s2c/multiview/qwen_reader.py` | Qwen-VL reads every crop of one image in one call | 2 |
| `s2c/multiview/fuse.py` | `Observation.depth_ratio`, `Observation.depth_from_image`; ratio to millimetres in `features_from` | 3, 8 |
| `s2c/multiview/merge_views.py` | Several photos of one face to one observation | 3 |
| `s2c/multiview/qwen_image.py` | `ImageGen` on DashScope or the Space; `log_call` | 4 |
| `s2c/multiview/qwen_faces.py` | Face prompt, outline from a generated image, voxel gate, sketch rescue | 5, 7 |
| `s2c/multiview/complete.py` | Qwen-Image first, then TripoSR, then assumed; `filled_by` | 5 |
| `s2c/multiview/depth.py` | Solaria client, PLY to depth, hole ratios | 8 |
| `s2c/multiview/pipeline.py` | Wiring: batch reader, merge, image gen, rescue, depth | 2, 3, 5, 7, 8 |
| `s2c/multiview/routes.py`, `scripts/mv.py` | Report which provider filled each face | 5 |
| `app_mv_gradio.py` | Gradio lab app | 6 |
| `tests/test_mv_*.py` | One test file per new module, plus additions | all |
| `README.md`, `docs/models.md`, `docs/disclosure.md` | Docs and disclosure | 9 |

---

### Task 1: Setup, TrOCR fix, settings

**Files:**
- Modify: `pyproject.toml`, `.env.example`, `s2c/multiview/ocr.py` (function `_trocr`)
- Test: `tests/test_mv_ocr.py::test_trocr_reads_printed_digits` (existing, currently failing)

**Interfaces:**
- Consumes: nothing.
- Produces: the `network` pytest marker; `httpx` as a direct dependency; the environment variables every later task reads.

- [ ] **Step 1: Make sure there is a local repository (never push)**

Run: `git -C . rev-parse --is-inside-work-tree`

If it prints `true`, go to step 2. If it fails, ask your human partner: "Fresh repository here, or clone the team's repository?" For a fresh one:

```bash
git init -b main
git add -A
git commit -m "Import the multi-view path, specs and plans"
```

`.gitignore` already excludes `.env`, `.venv`, `tmp/`, `logs/` and `vendor/`. Do not add a remote and do not push.

- [ ] **Step 2: Confirm the TrOCR failure**

Run: `uv run pytest tests/test_mv_ocr.py::test_trocr_reads_printed_digits -v`
Expected: FAIL with `ValueError: Couldn't instantiate the backend tokenizer`. The cached `microsoft/trocr-base-handwritten` snapshot has `tokenizer_config.json` but no `vocab.json` or `merges.txt`, so transformers 5 cannot build the tokenizer. `sentencepiece` does not fix this.

- [ ] **Step 3: Load the tokenizer files explicitly**

In `s2c/multiview/ocr.py`, replace `_trocr` with:

```python
@lru_cache(maxsize=1)
def _trocr(model_name: str):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import RobertaTokenizer, TrOCRProcessor, ViTImageProcessor, VisionEncoderDecoderModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # transformers 5 does not fetch vocab.json and merges.txt for this repo by itself; take the small files explicitly
    local = snapshot_download(model_name, allow_patterns=["*.json", "*.txt"])
    processor = TrOCRProcessor(image_processor=ViTImageProcessor.from_pretrained(local),
                               tokenizer=RobertaTokenizer.from_pretrained(local))
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device).eval()
    return processor, model, device
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/test_mv_ocr.py::test_trocr_reads_printed_digits -v`
Expected: PASS. If it still fails, stop and use superpowers:systematic-debugging on the new error before changing anything else.

- [ ] **Step 5: Add `httpx` and the `network` marker**

In `pyproject.toml`, add `"httpx>=0.27",` to `[project] dependencies` after `"python-dotenv>=1.0",`, and add the marker:

```toml
markers = [
  "gpu: needs a CUDA GPU and the ai extra",
  "slicer: needs PrusaSlicer installed",
  "network: calls a hosted model; runs only with NETWORK_TESTS=1",
]
```

Run: `uv lock` then `uv sync --extra ai`
Expected: no package removed (httpx is already installed through gradio).

- [ ] **Step 6: Provider settings**

Replace `.env.example` with:

```text
# Qwen-VL on DashScope (OpenAI-compatible): face labels and handwriting
VLM_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
VLM_MODEL=replace-with-a-current-qwen-vl-model
VLM_API_KEY=replace-me
# Qwen-Image: "space" (default) or "dashscope"
QWEN_IMAGE_BACKEND=space
QWEN_IMAGE_SPACE=Qwen/Qwen-Image-2.1
# QWEN_IMAGE_BASE_URL=https://dashscope-intl.aliyuncs.com/api/v1
# QWEN_IMAGE_MODEL=replace-with-a-qwen-image-edit-model
# Solaria depth (Marigold V2) and TripoSR on Hugging Face Spaces
SOLARIA_SPACE=CronosSa/Solaria1.0
TRIPOSR_SPACE=stabilityai/TripoSR
# HF_TOKEN=hf_...   raises the ZeroGPU quota
# SLICER_PATH=C:/Program Files/Prusa3D/PrusaSlicer/prusa-slicer-console.exe
# SLICER_PROFILE=profiles/fdm_default.ini
# NETWORK_TESTS=1   runs tests/test_mv_network.py against the hosted models
```

- [ ] **Step 7: Full suite and lint**

Run: `uv run pytest -q` then `uv run ruff check s2c tests`
Expected: `98 passed, 2 skipped` (the TrOCR test now passes); ruff reports nothing new in the files you touched.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock .env.example s2c/multiview/ocr.py docs/superpowers
git commit -m "Fix TrOCR tokenizer loading and add Qwen and Solaria settings"
```

---

### Task 2: Qwen-VL batch reader

**Files:**
- Create: `s2c/multiview/qwen_reader.py`, `tests/test_mv_qwen_reader.py`
- Modify: `s2c/multiview/ocr.py` (`BatchReader`, `read_values`), `s2c/multiview/label.py` (`env_chat`), `s2c/multiview/pipeline.py` (`MvPipeline.__init__`, `observe`, `default_pipeline`), `tests/test_mv_pipeline.py` (`fake_reads`, two new tests)

**Interfaces:**
- Consumes: `Chat = Callable[[list[dict]], str]` and `_strip_fences(text) -> str` from `s2c/multiview/label.py`; `parse_value`, `text_regions`, `Reading` from `ocr.py`.
- Produces:
  - `ocr.BatchReader = Callable[[list[np.ndarray]], list[tuple[str, float]] | None]`
  - `ocr.read_values(image_bgr, outline, reader: Reader | None, batch: BatchReader | None = None) -> list[Reading]`
  - `qwen_reader.qwen_batch_reader(chat: Chat) -> BatchReader`, `qwen_reader.READ_CONFIDENCE = 0.9`
  - `label.env_chat(log_path="logs/vlm.jsonl", stage="mv_label") -> Chat | None`
  - `MvPipeline(..., batch_reader: BatchReader | None = None)`; attribute `batch_reader`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mv_qwen_reader.py
import json

import numpy as np

from s2c.multiview.ocr import read_values
from s2c.multiview.outline import extract
from s2c.multiview.qwen_reader import qwen_batch_reader
from tests.test_mv_ocr import sketch_with_values

CROP = np.full((40, 60, 3), 255, np.uint8)


def chat_returning(*outputs):
    calls = []

    def chat(messages):
        calls.append(messages)
        return outputs[len(calls) - 1]

    chat.calls = calls
    return chat


def reads(*pairs):
    return json.dumps({"reads": [{"i": i, "text": t} for i, t in pairs]})


def test_reads_come_back_in_crop_order():
    chat = chat_returning(reads((2, "40"), (1, "⌀6")))
    assert qwen_batch_reader(chat)([CROP, CROP]) == [("⌀6", 0.9), ("40", 0.9)]


def test_one_message_holds_every_crop():
    chat = chat_returning(reads())
    qwen_batch_reader(chat)([CROP, CROP, CROP])
    content = chat.calls[0][0]["content"]
    assert sum(part["type"] == "image_url" for part in content) == 3
    assert "Never guess" in content[0]["text"]


def test_a_missing_crop_reads_as_empty():
    chat = chat_returning(reads((1, "60")))
    assert qwen_batch_reader(chat)([CROP, CROP]) == [("60", 0.9), ("", 0.9)]


def test_bad_json_is_retried_once_with_the_error():
    chat = chat_returning("not json", reads((1, "60")))
    assert qwen_batch_reader(chat)([CROP]) == [("60", 0.9)]
    assert "failed validation" in chat.calls[1][-1]["content"]


def test_an_index_out_of_range_is_retried():
    chat = chat_returning(reads((5, "60")), reads((1, "60")))
    assert qwen_batch_reader(chat)([CROP]) == [("60", 0.9)]
    assert len(chat.calls) == 2


def test_two_failures_or_an_error_return_none():
    assert qwen_batch_reader(chat_returning("nope", "{}"))([CROP]) is None

    def broken(messages):
        raise RuntimeError("503")

    assert qwen_batch_reader(broken)([CROP]) is None


def test_no_crops_makes_no_call():
    chat = chat_returning()
    assert qwen_batch_reader(chat)([]) == [] and chat.calls == []


def test_read_values_uses_the_batch_once_and_falls_back_to_the_reader():
    img = sketch_with_values()
    o = extract(img)
    calls = []

    def batch(crops):
        calls.append(len(crops))
        return [("60", 0.9), ("40 mm", 0.9)]

    def failing(crops):
        return None

    assert sorted(r.value_mm for r in read_values(img, o, None, batch)) == [40.0, 60.0] and calls == [2]
    assert [r.value_mm for r in read_values(img, o, lambda crop: ("60", 0.8), failing)] == [60.0, 60.0]
    assert read_values(img, o, None, failing) == []
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_mv_qwen_reader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 's2c.multiview.qwen_reader'`.

- [ ] **Step 3: Batch support in `ocr.py`**

Below the `Reader` line add:

```python
BatchReader = Callable[[list[np.ndarray]], list[tuple[str, float]] | None]  # every crop of one image; None on failure
```

Replace `read_values` with:

```python
def read_values(image_bgr: np.ndarray, outline: PixelOutline, reader: Reader | None,
                batch: BatchReader | None = None) -> list[Reading]:
    """One batch call for every crop when a batch reader is given; the single-crop reader is the fallback."""
    boxes = text_regions(image_bgr, outline)
    crops = [image_bgr[max(y - 6, 0): y + h + 6, max(x - 6, 0): x + w + 6] for x, y, w, h in boxes]
    reads = batch(crops) if batch is not None and crops else None
    if reads is None:
        if reader is None:
            return []
        reads = [reader(crop) for crop in crops]
    out = []
    for box, (text, confidence) in zip(boxes, reads):
        parsed = parse_value(text)
        if parsed is None:
            log.info("dropped OCR read %r at %s", text, box)
            continue
        out.append(Reading(parsed[0], parsed[1], box, float(confidence), text))
    return out
```

- [ ] **Step 4: The reader**

```python
# s2c/multiview/qwen_reader.py
"""Qwen-VL reads every handwritten crop of one image in a single call. Spec 2026-09-23 section 4.
The model only transcribes what the user wrote; parse_value and link in ocr.py decide what it means."""
from __future__ import annotations

import base64
import logging

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict

from s2c.multiview.label import Chat, _strip_fences
from s2c.multiview.ocr import BatchReader

log = logging.getLogger(__name__)
READ_CONFIDENCE = 0.9
PROMPT = ("Each image is a crop of handwriting from a mechanical sketch. Return exactly what is written in each crop, "
          'as JSON only: {"reads": [{"i": 1, "text": "⌀6"}, ...]}. Use ⌀ for a diameter sign and R for a radius. '
          'Return "" when a crop is unreadable. Never guess a value that is not written.')


class _Read(BaseModel):
    model_config = ConfigDict(extra="forbid")
    i: int
    text: str


class _Reads(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reads: list[_Read]


def _png_url(crop: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", crop)[1].tobytes()).decode()


def qwen_batch_reader(chat: Chat) -> BatchReader:
    def read(crops: list[np.ndarray]) -> list[tuple[str, float]] | None:
        if not crops:
            return []
        content = [{"type": "text", "text": PROMPT}]
        for k, crop in enumerate(crops, 1):
            content += [{"type": "text", "text": f"Crop {k}:"},
                        {"type": "image_url", "image_url": {"url": _png_url(crop)}}]
        messages = [{"role": "user", "content": content}]
        for _ in range(2):
            try:
                raw = chat(messages)
            except Exception as e:
                log.warning("Qwen-VL read failed: %s", e)
                return None
            try:
                parsed = _Reads.model_validate_json(_strip_fences(raw))
                bad = [r.i for r in parsed.reads if not 1 <= r.i <= len(crops)]
                if bad:
                    raise ValueError(f"crop numbers {bad} do not exist; use 1 to {len(crops)}")
            except ValueError as e:  # pydantic's ValidationError is a ValueError
                messages = messages + [{"role": "assistant", "content": raw},
                                       {"role": "user", "content": f"Your previous output failed validation: {e}. "
                                                                   "Return corrected JSON only."}]
                continue
            by_crop = {r.i: r.text for r in parsed.reads}
            return [(by_crop.get(k, ""), READ_CONFIDENCE) for k in range(1, len(crops) + 1)]
        return None

    return read
```

- [ ] **Step 5: Run the reader tests**

Run: `uv run pytest tests/test_mv_qwen_reader.py tests/test_mv_ocr.py -v`
Expected: PASS.

- [ ] **Step 6: Log stage in `env_chat`**

In `s2c/multiview/label.py`, change the signature and the log line of `env_chat`:

```python
def env_chat(log_path: str | Path = "logs/vlm.jsonl", stage: str = "mv_label") -> Chat | None:
```

and in the logged dict replace `"stage": "mv_label"` with `"stage": stage`.

- [ ] **Step 7: Write the failing pipeline tests**

In `tests/test_mv_pipeline.py`, change the inner fake in `fake_reads` to accept the batch argument:

```python
    def read_values(bgr, outline, reader, batch=None):
```

and add:

```python
def test_the_batch_reader_feeds_ocr(monkeypatch):
    seen = []

    def fake(bgr, outline, reader, batch=None):
        seen.append((reader, batch))
        return []

    def batch(crops):
        return []

    monkeypatch.setattr(pipeline, "read_values", fake)
    observed = MvPipeline(batch_reader=batch).observe([ImageInput(sketch(600, 400), "front", "sketch")])
    assert seen == [(None, batch)]
    assert "OCR unavailable: enter the dimensions by hand" not in observed.warnings


def test_default_pipeline_reads_with_qwen_vl_when_configured(monkeypatch):
    for key, value in {"VLM_BASE_URL": "http://localhost:9/v1", "VLM_MODEL": "m", "VLM_API_KEY": "k"}.items():
        monkeypatch.setenv(key, value)
    assert pipeline.default_pipeline().batch_reader is not None
```

Run: `uv run pytest tests/test_mv_pipeline.py -v`
Expected: FAIL with `TypeError: MvPipeline.__init__() got an unexpected keyword argument 'batch_reader'`.

- [ ] **Step 8: Wire the pipeline**

In `s2c/multiview/pipeline.py`:

Imports: replace `from s2c.multiview.ocr import Reader, link, read_values` with

```python
from s2c.multiview.ocr import BatchReader, Reader, link, read_values
from s2c.multiview.qwen_reader import qwen_batch_reader
```

Constructor:

```python
    def __init__(self, chat: Chat | None = None, reader: Reader | None = None,
                 mesh_provider: MeshProvider | None = None, slicer: Path | None = None, profile: Path | None = None,
                 batch_reader: BatchReader | None = None):
        self.chat, self.reader, self.mesh_provider = chat, reader, mesh_provider
        self.slicer, self.profile = slicer, profile
        self.batch_reader = batch_reader
```

In `observe`, replace

```python
        if self.reader is None:
            observed.warnings.append("OCR unavailable: enter the dimensions by hand")
```

with

```python
        reads = self.reader is not None or self.batch_reader is not None
        if not reads:
            observed.warnings.append("OCR unavailable: enter the dimensions by hand")
```

and replace

```python
            if self.reader is not None and label.input_kind != "photo":
                values = link(read_values(bgr, outline, self.reader), outline)
```

with

```python
            if reads and label.input_kind != "photo":
                values = link(read_values(bgr, outline, self.reader, self.batch_reader), outline)
```

Replace `default_pipeline` with:

```python
def default_pipeline() -> MvPipeline:
    """Qwen-VL from the environment; TrOCR and TripoSR when the ai extra is installed."""
    reader = provider = None
    try:
        from s2c.multiview.ocr import trocr_reader
        reader = trocr_reader()
    except Exception as e:  # transformers missing
        log.warning("TrOCR unavailable: %s", e)
    try:
        from s2c.multiview.hf3d import default_provider
        provider = default_provider()
    except Exception as e:
        log.warning("TripoSR unavailable: %s", e)
    read_chat = env_chat(stage="mv_read")
    return MvPipeline(chat=env_chat(), reader=reader, mesh_provider=provider,
                      batch_reader=qwen_batch_reader(read_chat) if read_chat else None)
```

- [ ] **Step 9: Run the suite**

Run: `uv run pytest -q` then `uv run ruff check s2c tests`
Expected: all pass; ruff clean for the touched files.

- [ ] **Step 10: Commit**

```bash
git add s2c/multiview/qwen_reader.py s2c/multiview/ocr.py s2c/multiview/label.py s2c/multiview/pipeline.py tests/test_mv_qwen_reader.py tests/test_mv_pipeline.py
git commit -m "Read handwritten values with Qwen-VL, one call per image"
```

---

### Task 3: Merge several photos of one face

**Files:**
- Create: `s2c/multiview/merge_views.py`, `tests/test_mv_merge_views.py`
- Modify: `s2c/multiview/fuse.py` (`Observation`), `s2c/multiview/pipeline.py` (`observe`, new `_merge`), `tests/test_mv_pipeline.py` (one new test)

**Interfaces:**
- Consumes: `Observation` (fuse), `Linked`, `Reading` (ocr), `PixelCircle`, `PixelOutline`, `extract` (outline), `iou`, `polygon_mask` (raster), `MvAbstain` (spec).
- Produces:
  - `Observation.depth_ratio: dict[int, float]` and `Observation.depth_from_image: set[int]` (filled by Task 8)
  - `merge_views.merge_same_face(observations: list[Observation], images: list[np.ndarray]) -> tuple[list[Observation], list[np.ndarray], list[str]]`
  - `merge_views.PAD = 16`, `merge_views.GRID = 512`
  - `MvPipeline._merge(observed: Observed) -> Observed` (static); `observe` now returns one observation per face

- [ ] **Step 1: Add the depth fields to `Observation`**

In `s2c/multiview/fuse.py`, add two fields to `Observation`, after `depth_estimates`:

```python
    depth_ratio: dict[int, float] = field(default_factory=dict)      # circle index -> blind depth / axis length, Solaria
    depth_from_image: set[int] = field(default_factory=set)          # circles whose blind flag came from Solaria
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_mv_merge_views.py
import cv2
import numpy as np
import pytest

from s2c.multiview.fuse import Observation
from s2c.multiview.merge_views import PAD, merge_same_face
from s2c.multiview.ocr import Linked, Reading
from s2c.multiview.outline import extract
from s2c.multiview.raster import iou, polygon_mask

HOLES = [(100, 100), (450, 300)]  # centres in plate pixels, radius 30; not symmetric under a half turn
SX, SY = 511 / 600, 340 / 400     # plate bbox (601 x 401 px) onto the 512 x 341 grid


def photo(shift=(0, 0), angle=0.0, size=(600, 400), holes=HOLES, notch=False, seed=0):
    """A dark plate with light holes on a light table, like a top-down photo."""
    img = np.full((1200, 1600, 3), 200, np.uint8)
    x0, y0 = 500 + shift[0], 400 + shift[1]
    cv2.rectangle(img, (x0, y0), (x0 + size[0], y0 + size[1]), (40, 40, 40), -1)
    if notch:
        cv2.rectangle(img, (x0 + 150, y0), (x0 + size[0], y0 + 300), (200, 200, 200), -1)
    for cx, cy in holes:
        cv2.circle(img, (x0 + cx, y0 + cy), 30, (200, 200, 200), -1)
    if angle:
        m = cv2.getRotationMatrix2D((800, 600), angle, 1.0)
        img = cv2.warpAffine(img, m, (1600, 1200), borderValue=(200, 200, 200))
    noise = np.random.default_rng(seed).normal(0, 6, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def obs(img, confidence=0.9, values=(), mm_per_px=None):
    return Observation(face="top", kind="photo", outline=extract(img), values=list(values), mm_per_px=mm_per_px,
                       confidence=confidence)


def linear(value, axis="a"):
    return Linked(Reading(float(value), "linear", (800, 850, 40, 30), 0.9, str(value)), axis, None)


def test_a_single_photo_passes_through_unchanged():
    img = photo()
    o = obs(img)
    merged, images, warnings = merge_same_face([o], [img])
    assert merged[0] is o and images[0] is img and warnings == []


def test_three_photos_vote_one_clean_outline_and_drop_the_outlier():
    imgs = [photo(), photo(shift=(20, -15), seed=1), photo(angle=3, seed=2), photo(notch=True, holes=[], seed=3)]
    observations = [obs(imgs[0], 0.95), obs(imgs[1]), obs(imgs[2]), obs(imgs[3])]
    (merged,), (image,), warnings = merge_same_face(observations, imgs)
    assert image is imgs[0]
    assert "top: photo 4 disagrees with the others, ignored" in warnings
    assert any(w.startswith("top: merged 3 photos") for w in warnings)
    shape = merged.outline.shape
    truth = polygon_mask([(PAD, PAD), (PAD + 511, PAD), (PAD + 511, PAD + 340), (PAD, PAD + 340)], (), shape)
    assert iou(polygon_mask(merged.outline.outer, merged.outline.inner, shape), truth) >= 0.97
    want = [(PAD + cx * SX, PAD + cy * SY) for cx, cy in HOLES]
    got = [(c.cx, c.cy) for c in merged.outline.circles]
    diag = float(np.hypot(512, 341))
    assert len(got) == 2
    assert all(np.hypot(g[0] - w[0], g[1] - w[1]) < 0.01 * diag for g, w in zip(got, want))
    ref_d = observations[0].outline.circles[0].d * np.sqrt(SX * SY)
    assert merged.outline.circles[0].d == pytest.approx(ref_d, rel=0.03)


def test_a_photo_of_the_wrong_proportions_is_dropped():
    imgs = [photo(), photo(shift=(20, 0), seed=1), photo(size=(600, 150), holes=[], seed=2)]
    (merged,), _, warnings = merge_same_face([obs(imgs[0], 0.95), obs(imgs[1]), obs(imgs[2])], imgs)
    assert "top: photo 3 disagrees with the others, ignored" in warnings
    assert len(merged.outline.circles) == 2


def test_a_photo_turned_upside_down_does_not_add_holes():
    imgs = [photo(), photo(shift=(20, 0), seed=1), photo(angle=180, seed=2)]
    (merged,), _, _ = merge_same_face([obs(imgs[0], 0.95), obs(imgs[1]), obs(imgs[2])], imgs)
    assert len(merged.outline.circles) == 2


def test_values_keep_what_most_photos_read_and_warn_about_the_rest():
    imgs = [photo(), photo(shift=(20, 0), seed=1), photo(shift=(-20, 0), seed=2)]
    observations = [obs(imgs[0], 0.95, [linear(60), linear(40, "b")]), obs(imgs[1], values=[linear(60)]),
                    obs(imgs[2], values=[linear(66)])]
    (merged,), _, warnings = merge_same_face(observations, imgs)
    assert [lv.reading.value_mm for lv in merged.values if lv.axis == "a"] == [60.0]
    assert [lv.reading.value_mm for lv in merged.values if lv.axis == "b"] == [40.0]
    assert any("66" in w for w in warnings)


def test_scale_is_the_median_and_a_spread_warns():
    imgs = [photo(), photo(shift=(20, 0), seed=1), photo(shift=(-20, 0), seed=2)]
    observations = [obs(imgs[0], 0.95, mm_per_px=0.1), obs(imgs[1], mm_per_px=0.1), obs(imgs[2], mm_per_px=0.104)]
    (merged,), _, warnings = merge_same_face(observations, imgs)
    _, _, w, _ = merged.outline.bbox
    assert (w - 1) * merged.mm_per_px == pytest.approx(60.0, rel=0.01)
    assert "top: scale varies between photos, check the reference object" in warnings


def test_a_solaria_result_wins_over_the_labels():
    imgs = [photo(), photo(shift=(20, 0), seed=1), photo(shift=(-20, 0), seed=2)]
    observations = [obs(imgs[0], 0.95), obs(imgs[1]), obs(imgs[2])]
    for o in observations[1:]:
        o.blind = {0: False, 1: False}
    observations[0].blind, observations[0].depth_ratio, observations[0].depth_from_image = {0: True}, {0: 0.4}, {0}
    (merged,), _, _ = merge_same_face(observations, imgs)
    upper = min(range(2), key=lambda k: merged.outline.circles[k].cy)
    assert merged.blind[upper] and merged.depth_ratio[upper] == 0.4 and upper in merged.depth_from_image
```

- [ ] **Step 3: Run them to see them fail**

Run: `uv run pytest tests/test_mv_merge_views.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 's2c.multiview.merge_views'`.

- [ ] **Step 4: Write `merge_views.py`**

```python
# s2c/multiview/merge_views.py
"""Several photos of one face -> one observation: aligned masks, a per-pixel vote, median holes, values and scale.
Spec 2026-09-23 section 5. Only our own image processing; no model is called here."""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import replace

import cv2
import numpy as np

from s2c.multiview.fuse import Observation
from s2c.multiview.ocr import Linked
from s2c.multiview.outline import PixelCircle, extract
from s2c.multiview.raster import iou, polygon_mask
from s2c.multiview.spec import MvAbstain

log = logging.getLogger(__name__)
GRID = 512
PAD = 16
OUTLIER_IOU = 0.7
ASPECT_TOL = 0.10
SCALE_SPREAD = 0.03
VALUE_TOL = 0.05
CIRCLE_TOL = 0.05


def _solid_mask(o: Observation) -> np.ndarray:
    """Outer outline minus non-circular openings. Circles stay filled: they are merged separately."""
    return polygon_mask(o.outline.outer, o.outline.inner, o.outline.shape)


def _to_grid(o: Observation, gw: int, gh: int) -> np.ndarray:
    """3 x 3 affine: image pixels -> grid pixels, stretching the outline's bounding box onto the grid."""
    x, y, w, h = o.outline.bbox
    sx, sy = (gw - 1) / max(w - 1, 1), (gh - 1) / max(h - 1, 1)
    return np.array([[sx, 0.0, PAD - x * sx], [0.0, sy, PAD - y * sy], [0.0, 0.0, 1.0]])


def _warp(mask: np.ndarray, t: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.warpAffine(mask, t[:2].astype(np.float64), size, flags=cv2.INTER_NEAREST)


def _blur(mask: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 8)


def _align(ref: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """3 x 3 affine moving `mask` onto `ref`; identity when ECC does not converge."""
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(_blur(ref), _blur(mask), warp, cv2.MOTION_AFFINE,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), None, 5)
    except cv2.error as e:
        log.info("ECC did not converge: %s", e)
        return np.eye(3)
    # ECC maps ref coordinates into mask coordinates; the inverse moves mask points onto ref
    return np.vstack([cv2.invertAffineTransform(warp), [0.0, 0.0, 1.0]])


def _apply(t: np.ndarray, pts) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    return pts @ t[:2, :2].T + t[:2, 2]


def _map_box(t: np.ndarray, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x, y, w, h = box
    pts = _apply(t, [[x, y], [x + w, y + h]])
    (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
    return int(x0), int(y0), max(int(x1 - x0), 1), max(int(y1 - y0), 1)


def _linear_scale(t: np.ndarray) -> float:
    return float(np.sqrt(abs(np.linalg.det(t[:2, :2]))))


def _vote(masks: list[np.ndarray]) -> np.ndarray:
    stack = np.stack([m > 127 for m in masks])
    return np.where(stack.sum(axis=0) * 2 >= len(masks), 255, 0).astype(np.uint8)


def _merge_circles(group, transforms, diag: float) -> tuple[list[PixelCircle], dict[tuple[int, int], int]]:
    """Clusters of circle centres; a cluster seen on at least half of the photos becomes one median circle."""
    clusters: list[list[tuple[int, int, float, float, float]]] = []
    for k, (o, t) in enumerate(zip(group, transforms)):
        s = _linear_scale(t)
        for i, c in enumerate(o.outline.circles):
            (cx, cy), = _apply(t, [[c.cx, c.cy]])
            member = (k, i, float(cx), float(cy), c.d * s)
            home = next((cl for cl in clusters if np.hypot(cl[0][2] - cx, cl[0][3] - cy) < CIRCLE_TOL * diag), None)
            if home is None:
                clusters.append([member])
            else:
                home.append(member)
    circles, index = [], {}
    for cl in clusters:
        if 2 * len({m[0] for m in cl}) < len(group):
            continue
        for m in cl:
            index[(m[0], m[1])] = len(circles)
        circles.append(PixelCircle(*(float(np.median([m[j] for m in cl])) for j in (2, 3, 4))))
    return circles, index


def _clusters(values: list[float]) -> list[list[int]]:
    groups: list[list[int]] = []
    for i, v in enumerate(values):
        home = next((g for g in groups if abs(v - values[g[0]]) <= VALUE_TOL * max(abs(values[g[0]]), 1e-9)), None)
        if home is None:
            groups.append([i])
        else:
            home.append(i)
    return groups


def _merge_values(face: str, group, transforms, circle_index) -> tuple[list[Linked], list[str]]:
    """Per axis or merged hole: keep every value read on at least half of the photos that read one there."""
    buckets: dict[tuple, list[tuple[int, Linked]]] = defaultdict(list)
    out: list[Linked] = []
    for k, (o, t) in enumerate(zip(group, transforms)):
        for lv in o.values:
            reading = replace(lv.reading, bbox=_map_box(t, lv.reading.bbox))
            if lv.hole_index is not None:
                hole = circle_index.get((k, lv.hole_index))
                if hole is not None:
                    buckets[("hole", hole)].append((k, Linked(reading, None, hole)))
            elif lv.axis is not None:
                buckets[("axis", lv.axis)].append((k, Linked(reading, lv.axis, None)))
            else:
                out.append(Linked(reading, None, None))
    warnings = []
    for items in buckets.values():
        photos = len({k for k, _ in items})
        for g in _clusters([lv.reading.value_mm for _, lv in items]):
            members = [items[i][1] for i in g]
            support = len({items[i][0] for i in g})
            value = float(np.median([lv.reading.value_mm for lv in members]))
            if 2 * support < photos:
                warnings.append(f"{face}: {value:g} mm was read on only {support} of {photos} photos, ignored")
                continue
            best = max(members, key=lambda lv: lv.reading.confidence)
            out.append(Linked(replace(best.reading, value_mm=value), best.axis, best.hole_index))
    return out, warnings


def _merge_labels(group, circle_index, n_circles: int):
    """Blind flags and depths per merged circle. A Solaria result beats every vision-model label."""
    blind, estimates, ratio, from_image = {}, {}, {}, set()
    for j in range(n_circles):
        members = [(k, i) for (k, i), c in circle_index.items() if c == j]
        solaria = [(k, i) for k, i in members if i in group[k].depth_from_image]
        if solaria:
            k, i = solaria[0]
            blind[j] = group[k].blind.get(i, False)
            from_image.add(j)
            if i in group[k].depth_ratio:
                ratio[j] = group[k].depth_ratio[i]
            continue
        votes = [group[k].blind.get(i, False) for k, i in members]
        blind[j] = 2 * sum(votes) > len(votes)
        guesses = [group[k].depth_estimates[i] for k, i in members if i in group[k].depth_estimates]
        if guesses:
            estimates[j] = float(np.median(guesses))
    return blind, estimates, ratio, from_image


def _merged_scale(face: str, group, transforms) -> tuple[float | None, list[str]]:
    scales = [o.mm_per_px / _linear_scale(t) for o, t in zip(group, transforms) if o.mm_per_px]
    if not scales:
        return None, []
    median = float(np.median(scales))
    if (max(scales) - min(scales)) / median > SCALE_SPREAD:
        return median, [f"{face}: scale varies between photos, check the reference object"]
    return median, []


def _merge_group(face: str, group: list[Observation], images: list[np.ndarray]):
    ratios = [o.outline.bbox[2] / o.outline.bbox[3] for o in group]
    median = float(np.median(ratios))
    gw, gh = (GRID, max(round(GRID / median), 8)) if median >= 1 else (max(round(GRID * median), 8), GRID)
    size = (gw + 2 * PAD, gh + 2 * PAD)
    ref = max(range(len(group)), key=lambda k: group[k].confidence)
    to_grid = [_to_grid(o, gw, gh) for o in group]
    solids = [_solid_mask(o) for o in group]
    grids = [_warp(s, g, size) for s, g in zip(solids, to_grid)]
    transforms = [g if k == ref else _align(grids[ref], grids[k]) @ g for k, g in enumerate(to_grid)]
    masks = [_warp(s, t, size) for s, t in zip(solids, transforms)]
    first = _vote(masks)
    keep = [k for k in range(len(group))
            if abs(ratios[k] / median - 1) <= ASPECT_TOL and iou(masks[k], first) >= OUTLIER_IOU]
    warnings = [f"{face}: photo {k + 1} disagrees with the others, ignored" for k in range(len(group)) if k not in keep]
    if len(keep) < 2:
        best = max(keep or [ref], key=lambda k: group[k].confidence)
        return group[best], images[best], warnings
    if ref not in keep:
        merged, image, more = _merge_group(face, [group[k] for k in keep], [images[k] for k in keep])
        return merged, image, warnings + more
    kept, kept_t = [group[k] for k in keep], [transforms[k] for k in keep]
    vote = _vote([masks[k] for k in keep])
    agreement = float(np.mean([iou(masks[k], vote) for k in keep]))
    outline = extract(cv2.cvtColor(255 - vote, cv2.COLOR_GRAY2BGR))
    if isinstance(outline, MvAbstain):
        return group[ref], images[ref], warnings + [f"{face}: photos could not be merged, using the clearest one"]
    circles, circle_index = _merge_circles(kept, kept_t, float(np.hypot(gw, gh)))
    outline = replace(outline, circles=circles)
    values, more = _merge_values(face, kept, kept_t, circle_index)
    blind, estimates, ratio, from_image = _merge_labels(kept, circle_index, len(circles))
    scale, scale_warnings = _merged_scale(face, kept, kept_t)
    merged = Observation(face=face, kind=group[ref].kind, outline=outline, values=values, mm_per_px=scale,
                         blind=blind, depth_estimates=estimates, depth_ratio=ratio, depth_from_image=from_image,
                         confidence=round(max(o.confidence for o in kept) * agreement, 3))
    warnings += more + scale_warnings + [f"{face}: merged {len(kept)} photos, agreement {agreement:.2f}"]
    return merged, images[ref], warnings


def merge_same_face(observations: list[Observation], images: list[np.ndarray]):
    """One observation per face tag, with the reference photo of each, and the warnings."""
    by_face: dict[str, list[int]] = defaultdict(list)
    for k, o in enumerate(observations):
        by_face[o.face].append(k)
    out_obs, out_images, warnings = [], [], []
    for face, idx in by_face.items():
        if len(idx) == 1:
            out_obs.append(observations[idx[0]])
            out_images.append(images[idx[0]])
            continue
        merged, image, more = _merge_group(face, [observations[k] for k in idx], [images[k] for k in idx])
        out_obs.append(merged)
        out_images.append(image)
        warnings += more
    return out_obs, out_images, warnings
```

- [ ] **Step 5: Run the merge tests**

Run: `uv run pytest tests/test_mv_merge_views.py -v`
Expected: PASS. If `test_three_photos_vote...` fails on the rotated photo only, print the ECC result for it before touching thresholds; the spec values (0.7, 10 percent, 5 percent) are fixed.

- [ ] **Step 6: Write the failing pipeline test**

Add to `tests/test_mv_pipeline.py`:

```python
def test_two_sketches_of_one_face_merge_into_one_observation(tmp_path):
    pipe = MvPipeline()
    observed = pipe.observe([ImageInput(sketch(600, 400), "front", "sketch"),
                             ImageInput(sketch(606, 404), "front", "sketch")])
    assert len(observed.observations) == 1 and len(observed.images) == 1
    assert any(w.startswith("front: merged 2 photos") for w in observed.warnings)
    spec = pipe.fuse(observed, {"envelope.x_mm": 60, "envelope.y_mm": 40, "envelope.z_mm": 5})
    assert pipe.build(spec, tmp_path, observed.masks).iou["front"] > 0.85
```

Run: `uv run pytest tests/test_mv_pipeline.py::test_two_sketches_of_one_face_merge_into_one_observation -v`
Expected: FAIL on `len(observed.observations) == 1` (it is 2).

- [ ] **Step 7: Merge in `observe`**

In `s2c/multiview/pipeline.py`, add the import `from s2c.multiview.merge_views import merge_same_face`. In `observe`, delete the line `observed.masks[label.face] = input_mask(outline)` and replace the final `return observed` with `return self._merge(observed)`. Add the method to `MvPipeline`:

```python
    @staticmethod
    def _merge(observed: Observed) -> Observed:
        """One observation per face: several photos of a face are merged (spec 2026-09-23 section 5)."""
        merged, images, warnings = merge_same_face(observed.observations, observed.images)
        observed.observations, observed.images = merged, images
        observed.warnings += warnings
        observed.masks = {o.face: input_mask(o.outline) for o in merged}
        return observed
```

- [ ] **Step 8: Run the suite**

Run: `uv run pytest -q` then `uv run ruff check s2c tests`
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add s2c/multiview/merge_views.py s2c/multiview/fuse.py s2c/multiview/pipeline.py tests/test_mv_merge_views.py tests/test_mv_pipeline.py
git commit -m "Merge several photos of one face by aligned voting"
```

---

### Task 4: Qwen-Image client

**Files:**
- Create: `s2c/multiview/qwen_image.py`, `tests/test_mv_qwen_image.py`

**Interfaces:**
- Consumes: `resize_long_side(image, long_side)` from `outline.py`.
- Produces:
  - `ImageGen = Callable[[list[np.ndarray], str, int, str], np.ndarray]`, called as `gen(refs, prompt, seed, stage)`
  - `ImageGenError(RuntimeError)`
  - `dashscope_gen(base_url, model, key, client=None, poll_s=2.0, timeout_s=90, log_path="logs/vlm.jsonl") -> ImageGen`
  - `space_gen(space, token=None, client_factory=None, timeout_s=120, log_path="logs/vlm.jsonl") -> ImageGen`
  - `default_gen() -> ImageGen | None`
  - `log_call(log_path, provider, model, stage, t0, ok) -> None` (Task 8 reuses it)
  - `MAX_REFS = 10`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mv_qwen_image.py
import json
import tempfile
import time
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest

from s2c.multiview import qwen_image as Q

PNG = cv2.imencode(".png", np.zeros((8, 8, 3), np.uint8))[1].tobytes()
REF = np.full((20, 30, 3), 255, np.uint8)
BASE = "https://dash.example/api/v1"


def client(handler, seen):
    def record(request):
        seen.append(request)
        return handler(request)

    return httpx.Client(transport=httpx.MockTransport(record))


def test_dashscope_sync_reply(tmp_path):
    seen = []

    def handler(req):
        if req.url.path.endswith("/generation"):
            content = [{"image": "https://img.example/a.png"}]
            return httpx.Response(200, json={"output": {"choices": [{"message": {"content": content}}]}})
        return httpx.Response(200, content=PNG)

    gen = Q.dashscope_gen(BASE, "qwen-image-x", "k", client(handler, seen), log_path=tmp_path / "log.jsonl")
    assert gen([REF, REF], "draw the top view", 7, "mv_face").shape == (8, 8, 3)
    body = json.loads(seen[0].content)
    assert body["model"] == "qwen-image-x"
    assert body["parameters"] == {"prompt_extend": False, "watermark": False, "seed": 7}
    content = body["input"]["messages"][0]["content"]
    assert content[0] == {"text": "draw the top view"} and len(content) == 3
    assert seen[0].headers["authorization"] == "Bearer k"
    assert json.loads((tmp_path / "log.jsonl").read_text())["stage"] == "mv_face"


def test_dashscope_polls_a_task(tmp_path):
    states = iter(["RUNNING", "SUCCEEDED"])

    def handler(req):
        if req.url.path.endswith("/generation"):
            return httpx.Response(200, json={"output": {"task_id": "t1", "task_status": "PENDING"}})
        if req.url.path.endswith("/tasks/t1"):
            status = next(states)
            out = {"task_id": "t1", "task_status": status}
            if status == "SUCCEEDED":
                out["results"] = [{"url": "https://img.example/b.png"}]
            return httpx.Response(200, json={"output": out})
        return httpx.Response(200, content=PNG)

    gen = Q.dashscope_gen(BASE, "m", "k", client(handler, []), poll_s=0, log_path=tmp_path / "l")
    assert gen([REF], "p", 7, "mv_face").shape == (8, 8, 3)


def test_dashscope_failures_raise_image_gen_error(tmp_path):
    def fail(req):
        return httpx.Response(500, text="boom")

    def running(req):
        return httpx.Response(200, json={"output": {"task_id": "t", "task_status": "RUNNING"}})

    with pytest.raises(Q.ImageGenError):
        Q.dashscope_gen(BASE, "m", "k", client(fail, []), log_path=tmp_path / "l")([REF], "p", 7, "mv_face")
    t0 = time.monotonic()
    with pytest.raises(Q.ImageGenError, match="timed out"):
        Q.dashscope_gen(BASE, "m", "k", client(running, []), poll_s=0, timeout_s=0.05,
                        log_path=tmp_path / "l")([REF], "p", 7, "mv_face")
    assert time.monotonic() - t0 < 2


class FakeSpace:
    calls = []

    def __init__(self, src, token=None):
        self.src, self.token = src, token

    def predict(self, **kw):
        FakeSpace.calls.append((self.src, self.token, kw))
        path = Path(tempfile.mkdtemp()) / "out.png"
        path.write_bytes(PNG)
        return {"path": str(path)}, 7.0, "rewritten"


def test_the_space_gets_every_reference_and_a_fixed_seed(tmp_path):
    FakeSpace.calls.clear()
    gen = Q.space_gen("Qwen/Qwen-Image-2.1", "tok", FakeSpace, log_path=tmp_path / "l")
    assert gen([REF, REF], "p", 8, "mv_rescue").shape == (8, 8, 3)
    src, token, kw = FakeSpace.calls[0]
    assert (src, token) == ("Qwen/Qwen-Image-2.1", "tok")
    assert len(kw["input_images"]) == 2 and kw["seed"] == 8 and kw["randomize_seed"] is False
    assert kw["enable_extend"] is False and kw["api_name"] == "/generate_with_enhance"


def test_a_slow_space_times_out_quickly(tmp_path):
    class Slow(FakeSpace):
        def predict(self, **kw):
            time.sleep(2)
            return super().predict(**kw)

    t0 = time.monotonic()
    with pytest.raises(Q.ImageGenError):
        Q.space_gen("s", None, Slow, timeout_s=0.2, log_path=tmp_path / "l")([REF], "p", 7, "mv_face")
    assert time.monotonic() - t0 < 1.5


def test_the_backend_comes_from_the_environment(monkeypatch):
    for key in ("QWEN_IMAGE_BACKEND", "QWEN_IMAGE_SPACE", "QWEN_IMAGE_BASE_URL", "QWEN_IMAGE_MODEL", "VLM_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    assert Q.default_gen() is None
    monkeypatch.setenv("QWEN_IMAGE_SPACE", "Qwen/Qwen-Image-2.1")
    assert Q.default_gen().__qualname__.startswith("space_gen")
    for key, value in {"QWEN_IMAGE_BACKEND": "dashscope", "QWEN_IMAGE_BASE_URL": BASE, "QWEN_IMAGE_MODEL": "m",
                       "VLM_API_KEY": "k"}.items():
        monkeypatch.setenv(key, value)
    assert Q.default_gen().__qualname__.startswith("dashscope_gen")
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_mv_qwen_image.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 's2c.multiview.qwen_image'`.

- [ ] **Step 3: Write `qwen_image.py`**

```python
# s2c/multiview/qwen_image.py
"""Qwen-Image: reference images and a prompt -> one image, on DashScope or the Hugging Face Space.
Spec 2026-09-23 section 6. Our code only reads the image back as an outline; it is never exported."""
from __future__ import annotations

import base64
import concurrent.futures as cf
import json
import logging
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import httpx
import numpy as np

from s2c.multiview.outline import resize_long_side

log = logging.getLogger(__name__)
ImageGen = Callable[[list[np.ndarray], str, int, str], np.ndarray]  # refs, prompt, seed, log stage -> BGR image
MAX_REFS = 10
REF_LONG_SIDE = 1024
DASHSCOPE_TIMEOUT_S = 90
SPACE_TIMEOUT_S = 120
_pool = cf.ThreadPoolExecutor(max_workers=2)  # a timed-out Space call finishes here while the caller moves on


class ImageGenError(RuntimeError):
    pass


def log_call(log_path, provider: str, model: str, stage: str, t0: float, ok: bool) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"provider": provider, "model": model, "stage": stage, "ok": ok,
                            "latency_ms": round((time.perf_counter() - t0) * 1000)}) + "\n")


def _ref(img: np.ndarray) -> np.ndarray:
    return img if max(img.shape[:2]) <= REF_LONG_SIDE else resize_long_side(img, REF_LONG_SIDE)


def _png_b64(img: np.ndarray) -> str:
    return base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode()


def _decode(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ImageGenError("the reply is not an image")
    return img


def _image_url(output: dict) -> str | None:
    for choice in output.get("choices") or []:
        for item in (choice.get("message") or {}).get("content") or []:
            if isinstance(item, dict) and item.get("image"):
                return item["image"]
    results = output.get("results") or []
    return results[0].get("url") if results else None


def dashscope_gen(base_url: str, model: str, key: str, client: httpx.Client | None = None, poll_s: float = 2.0,
                  timeout_s: float = DASHSCOPE_TIMEOUT_S, log_path="logs/vlm.jsonl") -> ImageGen:
    http = client or httpx.Client(timeout=30)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def call(refs: list[np.ndarray], prompt: str, seed: int) -> np.ndarray:
        start = time.monotonic()
        content = [{"text": prompt}]
        content += [{"image": f"data:image/png;base64,{_png_b64(_ref(r))}"} for r in refs[:MAX_REFS]]
        payload = {"model": model, "input": {"messages": [{"role": "user", "content": content}]},
                   "parameters": {"prompt_extend": False, "watermark": False, "seed": int(seed)}}
        r = http.post(f"{base_url}/services/aigc/multimodal-generation/generation", json=payload, headers=headers)
        r.raise_for_status()
        out = r.json().get("output") or {}
        while _image_url(out) is None and out.get("task_id"):
            if out.get("task_status") in ("FAILED", "CANCELED", "UNKNOWN"):
                raise ImageGenError(f"task {out['task_status']}: {out.get('message', '')}")
            if time.monotonic() - start > timeout_s:
                raise ImageGenError("timed out")
            time.sleep(poll_s)
            r = http.get(f"{base_url}/tasks/{out['task_id']}", headers=headers)
            r.raise_for_status()
            out = r.json().get("output") or {}
        url = _image_url(out)
        if url is None:
            raise ImageGenError("no image in the reply")
        got = http.get(url)
        got.raise_for_status()
        return _decode(got.content)

    def gen(refs: list[np.ndarray], prompt: str, seed: int, stage: str) -> np.ndarray:
        t0, ok = time.perf_counter(), False
        try:
            img = call(refs, prompt, seed)
            ok = True
            return img
        except ImageGenError:
            raise
        except Exception as e:
            raise ImageGenError(f"DashScope: {e}") from e
        finally:
            log_call(log_path, "dashscope", model, stage, t0, ok)

    return gen


def space_gen(space: str, token: str | None = None, client_factory=None, timeout_s: float = SPACE_TIMEOUT_S,
              log_path="logs/vlm.jsonl") -> ImageGen:
    def run(refs: list[np.ndarray], prompt: str, seed: int) -> np.ndarray:
        from gradio_client import Client, handle_file
        client = (client_factory or Client)(space, token=token)
        with tempfile.TemporaryDirectory() as d:
            files = []
            for k, img in enumerate(refs[:MAX_REFS]):
                path = Path(d) / f"ref{k}.png"
                cv2.imwrite(str(path), _ref(img))
                files.append({"image": handle_file(str(path)), "caption": None})
            result = client.predict(input_images=files, original_prompt=prompt, enable_extend=False,
                                    seed=int(seed), randomize_seed=False, api_name="/generate_with_enhance")
        image = result[0] if isinstance(result, (list, tuple)) else result
        path = image.get("path") if isinstance(image, dict) else image
        if not path:
            raise ImageGenError("the Space returned no image")
        return _decode(Path(path).read_bytes())

    def gen(refs: list[np.ndarray], prompt: str, seed: int, stage: str) -> np.ndarray:
        t0, ok = time.perf_counter(), False
        try:
            img = _pool.submit(run, refs, prompt, seed).result(timeout=timeout_s)
            ok = True
            return img
        except ImageGenError:
            raise
        except Exception as e:  # includes the timeout
            raise ImageGenError(f"Space {space}: {e!r}") from e
        finally:
            log_call(log_path, "hf-space", space, stage, t0, ok)

    return gen


def default_gen() -> ImageGen | None:
    """QWEN_IMAGE_BACKEND=dashscope with its settings, else the Space in QWEN_IMAGE_SPACE, else None."""
    backend = os.environ.get("QWEN_IMAGE_BACKEND", "space")
    base, model, key = (os.environ.get(k) for k in ("QWEN_IMAGE_BASE_URL", "QWEN_IMAGE_MODEL", "VLM_API_KEY"))
    if backend == "dashscope":
        if base and model and key:
            return dashscope_gen(base, model, key)
        log.warning("QWEN_IMAGE_BACKEND=dashscope needs QWEN_IMAGE_BASE_URL, QWEN_IMAGE_MODEL and VLM_API_KEY; "
                    "trying the Space")
    space = os.environ.get("QWEN_IMAGE_SPACE")
    return space_gen(space, os.environ.get("HF_TOKEN")) if space else None
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_mv_qwen_image.py -v`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

Run: `uv run ruff check s2c tests`

```bash
git add s2c/multiview/qwen_image.py tests/test_mv_qwen_image.py
git commit -m "Add the Qwen-Image client for DashScope and the Hugging Face Space"
```

---

### Task 5: Qwen-Image draws missing faces, with the voxel gate

**Files:**
- Create: `s2c/multiview/qwen_faces.py`, `tests/test_mv_qwen_faces.py`
- Modify: `s2c/multiview/complete.py` (docstring, imports, `complete`), `s2c/multiview/pipeline.py` (`Observed`, `MvPipeline.__init__`, `fuse`, `default_pipeline`), `s2c/multiview/routes.py` (`analyze`, `merge`), `scripts/mv.py`, `tests/test_mv_pipeline.py`, `tests/test_mv_routes.py`

**Interfaces:**
- Consumes: `ImageGen`, `ImageGenError`, `MAX_REFS`, `default_gen` (Task 4); `extract`, `resize_long_side`, `to_face_mm` (outline); `polygon_mask` (raster); `CANONICAL_FACES`, `Envelope`, `MvAbstain`, `Outline`, `face_size` (spec).
- Produces:
  - `qwen_faces.face_prompt(ref_faces: list[str], face: str) -> str`
  - `qwen_faces.outline_from_image(img, face, env) -> Outline | None`
  - `qwen_faces.grid_mask(outline, a_len, b_len, n=128) -> np.ndarray` (bool `[ib, ia]`, `ib` upward)
  - `qwen_faces.carve(front, top, right) -> np.ndarray` (bool `[ix, iy, iz]`), `qwen_faces.project(vox, face)`
  - `qwen_faces.consistent(outlines: dict[str, Outline], env, observed: list[str], n=128) -> bool`
  - `qwen_faces.qwen_face(outlines, env, face, observed, refs, gen, cache) -> Outline | None`; `SEED = 7`, `TRIES = 2`
  - `complete(..., gen=None, refs=(), qwen_cache=None, filled_by=None)`: same 3-tuple return; fills `filled_by[face]` with `observed | mirrored | qwen-image | triposr | assumed`
  - `Observed.qwen_cache: dict`, `Observed.filled_by: dict[str, str]`; `MvPipeline(..., image_gen=None)`
  - `/mv/analyze` and `/mv/merge` responses gain `filled_by`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mv_qwen_faces.py
import cv2
import numpy as np

from s2c.multiview.complete import complete
from s2c.multiview.qwen_faces import consistent, face_prompt
from s2c.multiview.qwen_image import ImageGenError
from s2c.multiview.raster import iou, outline_mask
from s2c.multiview.spec import Envelope, Outline
from tests.mv_helpers import outline, rect

ENV = Envelope(x_mm=50.0, y_mm=30.0, z_mm=20.0)
FRONT_L = [(0, 0), (50, 0), (50, 30), (44, 30), (44, 6), (0, 6)]
TOP_TAPER = [(0, 5), (50, 0), (50, 20), (0, 15)]
RIGHT_L = [(0, 0), (20, 0), (20, 6), (6, 6), (6, 30), (0, 30)]    # above y = 6 the part is only at z 14..20
BAD_TOP = [(25, 0), (50, 0), (50, 20), (0, 20), (0, 12), (25, 12)]  # for x < 25 only z 0..8: cuts the front
IMAGE = np.zeros((4, 4, 3), np.uint8)


def silhouette(points_mm, a_len, b_len, px=10):
    """What a good Qwen answer looks like: a black silhouette on white, 10 px per mm, with a margin."""
    img = np.full((round(b_len * px) + 200, round(a_len * px) + 200, 3), 255, np.uint8)
    pts = np.array([(100 + a * px, 100 + (b_len - b) * px) for a, b in points_mm], np.int32)
    cv2.fillPoly(img, [pts], (0, 0, 0))
    return img


def fake_gen(*answers):
    calls = []

    def gen(refs, prompt, seed, stage):
        calls.append((len(refs), prompt, seed, stage))
        answer = answers[min(len(calls), len(answers)) - 1]
        if isinstance(answer, Exception):
            raise answer
        return answer

    gen.calls = calls
    return gen


def ol(points):
    return Outline.model_validate(outline(points))


def front_only():
    return {"front": ol(FRONT_L)}


def front_and_right():
    return {"front": ol(rect(50, 30)), "right": ol(RIGHT_L)}


def test_the_prompt_names_every_reference_and_the_camera():
    p = face_prompt(["front", "right"], "top")
    assert "image 1 is the front view, image 2 is the right view" in p
    assert "looking down" in p and "No text" in p


def test_consistent_rejects_a_top_that_removes_photographed_material():
    assert consistent({**front_and_right(), "top": ol(rect(50, 20))}, ENV, ["front", "right"])
    assert not consistent({**front_and_right(), "top": ol(BAD_TOP)}, ENV, ["front", "right"])


def test_qwen_fills_the_missing_faces_and_says_so():
    gen = fake_gen(silhouette(TOP_TAPER, 50, 20), silhouette(rect(20, 30), 20, 30))
    filled = {}
    result, _, _ = complete(front_only(), ENV, "front", None, IMAGE, None, gen=gen, refs=[("front", IMAGE)],
                            filled_by=filled)
    assert filled == {"front": "observed", "top": "qwen-image", "right": "qwen-image"}
    top = result["top"]
    assert top.source == "inferred"
    assert iou(outline_mask(top.outer, top.inner, 50, 20), outline_mask(TOP_TAPER, [], 50, 20)) > 0.95
    assert [c[2:] for c in gen.calls] == [(7, "mv_face"), (7, "mv_face")] and "top view" in gen.calls[0][1]


def test_a_rejected_qwen_top_is_retried_with_the_next_seed():
    gen = fake_gen(silhouette(BAD_TOP, 50, 20), silhouette(rect(50, 20), 50, 20))
    filled = {}
    complete(front_and_right(), ENV, "front", None, IMAGE, None, gen=gen, refs=[("front", IMAGE)], filled_by=filled)
    assert filled["top"] == "qwen-image" and [c[2] for c in gen.calls] == [7, 8]


def test_two_rejected_answers_fall_back_and_warn():
    def broken(img):
        raise RuntimeError("no GPU")

    filled = {}
    _, warnings, _ = complete(front_and_right(), ENV, "front", np.zeros((512, 512), np.uint8), IMAGE, broken,
                              gen=fake_gen(silhouette(BAD_TOP, 50, 20)), refs=[("front", IMAGE)], filled_by=filled)
    assert filled["top"] == "assumed"
    assert "top: Qwen-Image view rejected, assumed used" in warnings


def test_an_empty_or_failed_answer_is_not_used():
    for answer in (np.zeros((300, 300, 3), np.uint8), ImageGenError("503")):
        filled = {}
        complete(front_only(), ENV, "front", None, IMAGE, None, gen=fake_gen(answer), refs=[("front", IMAGE)],
                 filled_by=filled)
        assert filled["top"] == "assumed" and filled["right"] == "assumed"


def test_the_answer_is_cached_so_an_edit_never_calls_qwen_again():
    gen = fake_gen(silhouette(TOP_TAPER, 50, 20), silhouette(rect(20, 30), 20, 30))
    cache = {}
    complete(front_only(), ENV, "front", None, IMAGE, None, gen=gen, refs=[("front", IMAGE)], qwen_cache=cache)
    wider = Envelope(x_mm=60.0, y_mm=30.0, z_mm=20.0)
    front = {"front": ol([(a * 1.2, b) for a, b in FRONT_L])}
    result, _, _ = complete(front, wider, "front", None, None, None, gen=gen, refs=[], qwen_cache=cache)
    assert len(gen.calls) == 2 and max(a for a, _ in result["top"].outer) == 60.0


def test_a_rejected_face_skips_qwen():
    gen = fake_gen(silhouette(TOP_TAPER, 50, 20))
    filled = {}
    complete(front_only(), ENV, "front", None, IMAGE, None, rejected=("top", "right"), gen=gen,
             refs=[("front", IMAGE)], filled_by=filled)
    assert gen.calls == [] and filled["top"] == "assumed"
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_mv_qwen_faces.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 's2c.multiview.qwen_faces'`.

- [ ] **Step 3: Write `qwen_faces.py`**

```python
# s2c/multiview/qwen_faces.py
"""Qwen-Image draws a face nobody photographed; a voxel check keeps it only if it never removes material that a
photographed face shows. Spec 2026-09-23 section 7. Holes come only from photographed faces, where they are
editable features, so circles on a drawn face are ignored."""
from __future__ import annotations

import logging

import numpy as np

from s2c.multiview.outline import extract, resize_long_side, to_face_mm
from s2c.multiview.qwen_image import ImageGen, ImageGenError
from s2c.multiview.raster import polygon_mask
from s2c.multiview.spec import CANONICAL_FACES, Envelope, MvAbstain, Outline, face_size

log = logging.getLogger(__name__)
VOXELS = 128
GATE_IOU = 0.95
TRIES = 2
SEED = 7
CONFIDENCE = 0.6
VIEWER = {
    "front": "the front, looking straight along the depth axis, with the top of the part at the top of the image",
    "top": "directly above, looking down, with the front edge of the part at the bottom of the image",
    "right": "the right side, looking straight at it, with the top of the part at the top of the image",
}
FACE_PROMPT = ("Images 1 to {n} show one mechanical part: {which}. Draw the orthographic {face} view of the same "
               "part, as seen from {viewer}. Solid black silhouette on a pure white background. Through-holes are "
               "white. No text, no dimension lines, no shading, no perspective, no background objects.")


def face_prompt(ref_faces: list[str], face: str) -> str:
    which = ", ".join(f"image {k} is the {f} view" for k, f in enumerate(ref_faces, 1))
    return FACE_PROMPT.format(n=len(ref_faces), which=which, face=face, viewer=VIEWER[face])


def outline_from_image(img: np.ndarray, face: str, env: Envelope) -> Outline | None:
    """A drawn silhouette, stretched onto the face's envelope rectangle."""
    found = extract(resize_long_side(img))
    if isinstance(found, MvAbstain):
        return None
    a_len, b_len = face_size(face, env)
    _, _, w, h = found.bbox
    sa, sb = a_len / max(w - 1, 1), b_len / max(h - 1, 1)

    def mm(points):
        return [(min(max(a, 0.0), a_len), min(max(b, 0.0), b_len)) for a, b in to_face_mm(points, found.bbox, sa, sb)]

    outer = mm(found.outer)
    if len(set(outer)) < 3:
        return None
    return Outline(outer=outer, inner=[mm(loop) for loop in found.inner], source="inferred", confidence=CONFIDENCE)


def grid_mask(outline: Outline, a_len: float, b_len: float, n: int = VOXELS) -> np.ndarray:
    """Boolean [ib, ia] samples of an outline on an n x n grid over its face rectangle; ib grows upward."""
    def px(points):
        return [(a / a_len * n, n - b / b_len * n) for a, b in points]

    img = polygon_mask(px(outline.outer), [px(loop) for loop in outline.inner], (n, n))
    return img[::-1] > 127


def carve(front: np.ndarray, top: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Voxels [ix, iy, iz] inside all three extruded outlines. Face frames as in the multi-view spec section 3:
    front (a=X, b=Y), top (a=X, b=depth-Z), right (a=depth-Z, b=Y)."""
    return front.T[:, :, None] & top[::-1, :].T[:, None, :] & right[:, ::-1][None, :, :]


def project(vox: np.ndarray, face: str) -> np.ndarray:
    """Silhouette of the voxels seen from a canonical face, as [ib, ia] like grid_mask."""
    if face == "front":
        return vox.any(axis=2).T
    if face == "top":
        return vox.any(axis=1).T[::-1]
    return vox.any(axis=0)[:, ::-1]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def consistent(outlines: dict[str, Outline], env: Envelope, observed: list[str], n: int = VOXELS) -> bool:
    """True when carving with every outline still shows each observed face as it was photographed."""
    masks = {f: grid_mask(outlines[f], *face_size(f, env), n) if f in outlines else np.ones((n, n), bool)
             for f in CANONICAL_FACES}
    vox = carve(masks["front"], masks["top"], masks["right"])
    return all(_iou(project(vox, f), masks[f]) >= GATE_IOU for f in observed)


def _drawn(gen: ImageGen | None, refs, face: str, cache: dict, seed: int) -> np.ndarray | None:
    """The generated image for (face, seed), from the cache when possible. A failure is cached as None."""
    key = (face, seed)
    if key not in cache:
        if gen is None or not refs:
            return None
        try:
            cache[key] = gen([img for _, img in refs], face_prompt([f for f, _ in refs], face), seed, "mv_face")
        except ImageGenError as e:
            log.warning("Qwen-Image %s failed: %s", face, e)
            cache[key] = None
    return cache[key]


def qwen_face(outlines: dict[str, Outline], env: Envelope, face: str, observed: list[str], refs,
              gen: ImageGen | None, cache: dict) -> Outline | None:
    """The first drawing of `face` that passes the voxel check, trying seeds SEED and SEED + 1."""
    for attempt in range(TRIES):
        img = _drawn(gen, refs, face, cache, SEED + attempt)
        candidate = None if img is None else outline_from_image(img, face, env)
        if candidate is not None and consistent({**outlines, face: candidate}, env, observed):
            return candidate
    return None
```

- [ ] **Step 4: Put Qwen-Image first in `complete.py`**

Replace the module docstring and imports of `s2c/multiview/complete.py` with:

```python
"""Fill the canonical faces nobody photographed: drawn by Qwen-Image, else predicted from a TripoSR mesh, else
assumed rectangular. The mirror rule already ran in fuse. Spec 2026-09-22 section 6.3, spec 2026-09-23 section 7.
Neither the drawn image nor the mesh is exported; both are only read back as outlines."""
from __future__ import annotations

import itertools
import logging
from collections.abc import Callable

import numpy as np

from s2c.multiview.qwen_faces import qwen_face
from s2c.multiview.qwen_image import ImageGen
from s2c.multiview.raster import Mesh, face_mask, iou, mask_to_mm, normalize_mask
from s2c.multiview.spec import CANONICAL_FACES, Envelope, Outline, face_size
```

Keep every function between the imports and `complete` unchanged. Replace `complete` with:

```python
def complete(outlines: dict[str, Outline], env: Envelope, target_face: str, target_mask: np.ndarray | None,
             image: np.ndarray | None, provider: MeshProvider | None, mesh: Mesh | None = None, rejected=(),
             gen: ImageGen | None = None, refs=(), qwen_cache: dict | None = None,
             filled_by: dict | None = None) -> tuple[dict[str, Outline], list[str], Mesh | None]:
    """All three canonical outlines, the warnings, and the mesh so the caller can cache it.
    filled_by, when given, receives who filled each face: observed, mirrored, qwen-image, triposr or assumed."""
    result, warnings = dict(outlines), []
    filled_by = {} if filled_by is None else filled_by
    qwen_cache = {} if qwen_cache is None else qwen_cache
    filled_by.update({f: ol.source for f, ol in outlines.items()})
    observed = list(outlines)
    missing = [f for f in CANONICAL_FACES if f not in outlines]
    tried_qwen = []
    for face in missing:
        if face in rejected:
            continue
        drawn = qwen_face(result, env, face, observed, list(refs), gen, qwen_cache)
        if drawn is not None:
            result[face], filled_by[face] = drawn, "qwen-image"
        elif gen is not None or any(key[0] == face for key in qwen_cache):
            tried_qwen.append(face)
    wanted = [f for f in missing if f not in rejected and f not in result]
    if wanted and mesh is None and provider is not None and image is not None:
        try:
            mesh = provider(image)
        except Exception as e:
            log.warning("3D predictor failed: %s", e)
            warnings.append("3D predictor unavailable")
    fitted, score = None, 0.0
    if wanted and mesh is not None:
        r, score = orient(mesh, target_face, target_mask)
        if score >= MIN_ORIENTATION_IOU:
            fitted = fit_to_envelope(mesh, r, env)
        else:
            warnings.append("predicted view unreliable")
    for face in missing:
        if face in result:
            continue
        if fitted is not None and face in wanted:
            try:
                result[face], filled_by[face] = predicted_outline(fitted, face, env, score), "triposr"
            except ValueError:
                log.warning("predicted %s view was empty", face)
        if face not in result:
            result[face], filled_by[face] = assumed_outline(face, env), "assumed"
            warnings.append(f"assumed rectangular {face}, check it")
        if face in tried_qwen:
            warnings.append(f"{face}: Qwen-Image view rejected, {filled_by[face]} used")
    return result, warnings, mesh
```

- [ ] **Step 5: Run the face and completion tests**

Run: `uv run pytest tests/test_mv_qwen_faces.py tests/test_mv_complete.py -v`
Expected: PASS (the five existing completion tests are unchanged).

- [ ] **Step 6: Write the failing pipeline and route tests**

Add to `tests/test_mv_pipeline.py`:

```python
def test_qwen_draws_the_missing_faces(tmp_path):
    from tests.mv_helpers import rect
    from tests.test_mv_qwen_faces import fake_gen, silhouette
    gen = fake_gen(silhouette(rect(60, 10), 60, 10), silhouette(rect(10, 40), 10, 40))
    pipe = MvPipeline(image_gen=gen)
    observed = pipe.observe([ImageInput(sketch(600, 400), "front", "sketch")])
    spec = pipe.fuse(observed, {"envelope.x_mm": 60, "envelope.y_mm": 40, "envelope.z_mm": 10})
    assert observed.filled_by == {"front": "observed", "top": "qwen-image", "right": "qwen-image"}
    assert spec.views.top.source == "inferred" and spec.provenance["views.top.outer"] == "inferred"
    assert pipe.build(spec, tmp_path, observed.masks).iou["front"] > 0.85
    assert len(gen.calls) == 2 and gen.calls[0][0] == 1
```

In `tests/test_mv_routes.py`, in `test_analyze_merge_build_and_download`, replace

```python
    spec = c.post("/mv/merge", json={"request_id": body["request_id"], "user_values": values}).json()["spec"]
```

with

```python
    merged = c.post("/mv/merge", json={"request_id": body["request_id"], "user_values": values}).json()
    assert merged["filled_by"] == {"front": "observed", "top": "observed", "right": "assumed"}
    spec = merged["spec"]
```

Run: `uv run pytest tests/test_mv_pipeline.py tests/test_mv_routes.py -v`
Expected: FAIL (`unexpected keyword argument 'image_gen'`, and `KeyError: 'filled_by'`).

- [ ] **Step 7: Wire the pipeline, routes and script**

In `s2c/multiview/pipeline.py`:

Add the import `from s2c.multiview.qwen_image import MAX_REFS, ImageGen, default_gen`.

Add two fields at the end of `Observed`:

```python
    qwen_cache: dict = field(default_factory=dict)            # (face, seed) -> drawn image, or None after a failure
    filled_by: dict[str, str] = field(default_factory=dict)   # canonical face -> who filled it
```

Constructor: add the parameter `image_gen: ImageGen | None = None` after `batch_reader`, and the line `self.image_gen = image_gen`.

In `fuse`, replace the `complete(...)` call with:

```python
        pairs = sorted(zip(observed.observations, observed.images), key=lambda p: -p[0].confidence)
        refs = [(o.face, img) for o, img in pairs][:MAX_REFS]
        full, more, observed.mesh = complete({f: ol for f, (ol, _) in outlines.items()}, env, target.face,
                                             observed.masks[target.face], image, self.mesh_provider,
                                             observed.mesh, tuple(rejected), gen=self.image_gen, refs=refs,
                                             qwen_cache=observed.qwen_cache, filled_by=observed.filled_by)
```

In `default_pipeline`, pass `image_gen=default_gen()` to `MvPipeline(...)` and change its docstring to `"""Qwen-VL and Qwen-Image from the environment; TrOCR and TripoSR when the ai extra is installed."""`.

In `s2c/multiview/routes.py`, `analyze` returns

```python
    return {"request_id": rid, **_result(res), "labels": [l.model_dump() for l in observed.labels],
            "filled_by": observed.filled_by}
```

and `merge` returns `{**_result(res), "filled_by": entry[1].filled_by}`.

In `scripts/mv.py`, replace the face print line with:

```python
        print(f"{face:<7} {getattr(spec.views, face).source} ({observed.filled_by.get(face, '')})")
```

- [ ] **Step 8: Run the suite**

Run: `uv run pytest -q` then `uv run ruff check s2c tests scripts`
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add s2c/multiview/qwen_faces.py s2c/multiview/complete.py s2c/multiview/pipeline.py s2c/multiview/routes.py scripts/mv.py tests/test_mv_qwen_faces.py tests/test_mv_pipeline.py tests/test_mv_routes.py
git commit -m "Draw missing faces with Qwen-Image behind a voxel consistency check"
```

---

### Task 6: Gradio lab app

**Files:**
- Create: `app_mv_gradio.py` (repository root), `tests/test_mv_gradio.py`

**Interfaces:**
- Consumes: `MvPipeline`, `ImageInput`, `Observed`, `default_pipeline` (pipeline); `Observed.filled_by` (Task 5); `outline_mask` (raster); `CANONICAL_FACES`, `FACES`, `MultiViewSpec`, `MvAbstain`, `face_size` (spec).
- Produces: `build_app(pipe) -> gr.Blocks`; `Handlers(pipe)` with `analyze(paths, tags, reference) -> tuple` and `rebuild(rows, rejected, state) -> tuple`; `OUTPUTS` naming the tuple slots; `MISSING`; `OUT_ROOT`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mv_gradio.py
import gradio as gr

import app_mv_gradio as A
from s2c.multiview.pipeline import MvPipeline
from tests.test_mv_pipeline import sketch

ENVELOPE = {"envelope.x_mm": 60, "envelope.y_mm": 40, "envelope.z_mm": 10}


def analyzed(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "OUT_ROOT", tmp_path / "out")
    handlers = A.Handlers(MvPipeline())
    paths = []
    for name, (w, h) in {"front.png": (600, 400), "top.png": (600, 100)}.items():
        path = tmp_path / name
        path.write_bytes(sketch(w, h))
        paths.append(str(path))
    tags = [["front.png", "front", "sketch"], ["top.png", "top", "sketch"]]
    return handlers, dict(zip(A.OUTPUTS, handlers.analyze(paths, tags, "none")))


def test_the_app_builds():
    assert isinstance(A.build_app(MvPipeline()), gr.Blocks)


def test_analyze_asks_for_the_envelope_and_rebuild_makes_the_files(tmp_path, monkeypatch):
    handlers, out = analyzed(tmp_path, monkeypatch)
    assert "missing_x" in out["message"]
    assert [r[0] for r in out["values"]] == list(ENVELOPE) and out["values"][0][2] == A.MISSING
    rows = [[path, ENVELOPE[path], source] for path, _, source in out["values"]]
    done = dict(zip(A.OUTPUTS, handlers.rebuild(rows, [], out["state"])))
    assert done["model"].endswith("part.stl") and len(done["files"]) >= 2
    assert [r[2] for r in done["values"][:3]] == ["user_edited"] * 3
    assert [caption.split(":")[0] for _, caption in done["faces"]] == ["front", "top", "right"]


def test_bad_numbers_are_reported_not_raised(tmp_path, monkeypatch):
    handlers, out = analyzed(tmp_path, monkeypatch)
    rows = [["envelope.x_mm", "abc", A.MISSING], ["envelope.y_mm", "-5", A.MISSING], ["envelope.z_mm", "0", A.MISSING]]
    done = dict(zip(A.OUTPUTS, handlers.rebuild(rows, [], out["state"])))
    assert "not a number" in done["message"] and "more than 0" in done["message"] and done["model"] is None


def test_a_wrong_face_tag_is_explained(tmp_path):
    path = tmp_path / "a.png"
    path.write_bytes(sketch(600, 400))
    out = dict(zip(A.OUTPUTS, A.Handlers(MvPipeline()).analyze([str(path)], [["a.png", "side", "sketch"]], "none")))
    assert "face must be one of" in out["message"]


def test_rebuild_before_analyze_explains():
    out = dict(zip(A.OUTPUTS, A.Handlers(MvPipeline()).rebuild([], [], None)))
    assert out["message"] == "Analyze images first."
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_mv_gradio.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app_mv_gradio'`.

- [ ] **Step 3: Write `app_mv_gradio.py`**

```python
"""Lab app for the multi-view path: upload face images, review, edit the numbers, rebuild, download.
Spec 2026-09-23 section 11. Run: uv run python app_mv_gradio.py, then open http://localhost:7860"""
from __future__ import annotations

import math
import re
import uuid
from pathlib import Path

import gradio as gr
import numpy as np
from dotenv import load_dotenv

from s2c.multiview.pipeline import ImageInput, MvPipeline, Observed, default_pipeline
from s2c.multiview.raster import outline_mask
from s2c.multiview.spec import CANONICAL_FACES, FACES, MultiViewSpec, MvAbstain, face_size

OUT_ROOT = Path("tmp/mv_gradio")
FACE_CHOICES = ("auto", *FACES)
KIND_CHOICES = ("auto", "sketch", "photo", "drawing")
REFERENCES = ["none", "1 TND", "1 EUR", "2 EUR", "card", "a4"]
AMBER = frozenset({"scaled", "inferred", "estimated", "default"})
MISSING = "missing: confirm or type a value"
BADGES = {"observed": "observed", "mirrored": "mirrored from the opposite face", "qwen-image": "drawn by Qwen-Image",
          "triposr": "predicted by TripoSR", "assumed": "assumed rectangle"}
NOTICE = ("Photos of real parts: shoot top-down, with the part lying flat on a plain surface. "
          "Images are sent to DashScope and Hugging Face Spaces to read the handwriting and predict missing faces.")
OUTPUTS = ("faces", "reads", "values", "warnings", "message", "model", "views", "files", "stats", "state")
_FEATURE = re.compile(r"(\w+)\[(\d+)\]\.(\w+)")


def new_state() -> dict:
    return {"observed": None, "shown": {}, "edits": {}}


def tag_rows(paths) -> list[list[str]]:
    return [[Path(p).name, "auto", "auto"] for p in paths or []]


def _pack(state: dict, **parts) -> tuple:
    out = {"faces": [], "reads": [], "values": [], "warnings": "", "message": "", "model": None, "views": [],
           "files": [], "stats": "", "state": state}
    out.update(parts)
    return tuple(out[k] for k in OUTPUTS)


def _abstain(a: MvAbstain) -> str:
    return f"**Stopped at {a.stage}: {a.reason}.** {a.remedy}"


def _bullets(lines) -> str:
    return "\n".join(f"- {line}" for line in lines)


def _value(spec: MultiViewSpec, path: str) -> float:
    data = spec.model_dump()
    m = _FEATURE.fullmatch(path)
    if m:
        return data[m.group(1)][int(m.group(2))][m.group(3)]
    head, name = path.split(".", 1)
    return data[head][name]


def value_rows(spec: MultiViewSpec | None, abstain: MvAbstain | None, edits: dict) -> list[list]:
    """Envelope first, then every numeric feature value, with its source. Outlines are not edited here."""
    if spec is not None:
        return [[p, _value(spec, p), f"{s} (check)" if s in AMBER else s]
                for p, s in spec.provenance.items() if not p.startswith("views.")]
    partial = (abstain.partial if abstain else None) or {}
    known, suggested = partial.get("known", {}), partial.get("suggested", {})
    rows = []
    for axis in "xyz":
        path = f"envelope.{axis}_mm"
        if path in edits:
            rows.append([path, edits[path], "user_edited"])
        elif path in known:
            rows.append([path, known[path], "found"])
        else:
            rows.append([path, suggested.get(path, ""), MISSING])
    return rows


def parse_edits(rows, shown: dict) -> tuple[dict[str, float], list[str]]:
    """Values the user changed or confirmed. Every bad entry is reported; none raises."""
    edits, errors = {}, []
    for row in rows or []:
        if len(row) < 2 or str(row[0]).strip() not in shown:
            continue
        path, text = str(row[0]).strip(), str(row[1]).strip()
        if text in ("", "None", "nan"):
            continue
        try:
            value = float(text.replace(",", "."))
        except ValueError:
            errors.append(f"{path}: '{text}' is not a number")
            continue
        if not (value > 0 and math.isfinite(value)):
            errors.append(f"{path}: must be more than 0")
            continue
        old, source = shown[path]
        if source == MISSING or old in ("", None) or abs(value - float(old)) > 1e-9:
            edits[path] = value
    return edits, errors


def face_gallery(observed: Observed, spec: MultiViewSpec | None) -> list[tuple[np.ndarray, str]]:
    if spec is None:
        return [(255 - mask, f"{face}: input") for face, mask in observed.masks.items()]
    items = []
    for face in CANONICAL_FACES:
        ol = getattr(spec.views, face)
        mask = outline_mask(ol.outer, ol.inner, *face_size(face, spec.envelope), px=256)
        badge = BADGES.get(observed.filled_by.get(face, ol.source), ol.source)
        merged = next((w.split(": ", 1)[1] for w in spec.warnings if w.startswith(f"{face}: merged")), "")
        items.append((255 - mask, f"{face}: {badge}" + (f", {merged}" if merged else "")))
    return items


def read_rows(observed: Observed) -> list[list]:
    rows = []
    for o in observed.observations:
        for lv in o.values:
            where = (f"hole {lv.hole_index + 1}" if lv.hole_index is not None
                     else f"{lv.axis} axis" if lv.axis else "not linked")
            rows.append([o.face, lv.reading.text, lv.reading.value_mm, where])
    return rows


class Handlers:
    """The two buttons, as plain methods so tests call them without a browser."""

    def __init__(self, pipe: MvPipeline):
        self.pipe = pipe

    def analyze(self, paths, tags, reference) -> tuple:
        state = new_state()
        if not paths:
            return _pack(state, message="Add at least one image.")
        images = []
        for i, path in enumerate(paths):
            row = list(tags[i]) if tags is not None and i < len(tags) else []
            face = str(row[1]).strip().lower() if len(row) > 1 and row[1] else "auto"
            kind = str(row[2]).strip().lower() if len(row) > 2 and row[2] else "auto"
            if face not in FACE_CHOICES or kind not in KIND_CHOICES:
                return _pack(state, message=f"Row {i + 1}: face must be one of {', '.join(FACE_CHOICES)}; "
                                            f"kind one of {', '.join(KIND_CHOICES)}.")
            images.append(ImageInput(Path(path).read_bytes(), None if face == "auto" else face,
                                     None if kind == "auto" else kind))
        observed = self.pipe.observe(images, None if reference in (None, "", "none") else reference)
        if isinstance(observed, MvAbstain):
            return _pack(state, message=_abstain(observed))
        state["observed"] = observed
        return _pack(state, **self._review(observed, self.pipe.fuse(observed), state))

    def rebuild(self, rows, rejected, state) -> tuple:
        state = state or new_state()
        observed = state.get("observed")
        if observed is None:
            return _pack(state, message="Analyze images first.")
        edits, errors = parse_edits(rows, state["shown"])
        if errors:
            return _pack(state, values=rows, message=_bullets(errors))
        state["edits"].update(edits)
        res = self.pipe.fuse(observed, dict(state["edits"]), rejected=tuple(rejected or ()))
        review = self._review(observed, res, state)
        if isinstance(res, MvAbstain):
            return _pack(state, **review)
        built = self.pipe.build(res, OUT_ROOT / uuid.uuid4().hex, observed.masks)
        if isinstance(built, MvAbstain):
            return _pack(state, **{**review, "message": _abstain(built)})
        views = [(255 - mask, face + (f", IoU {built.iou[face]:.2f}" if face in built.iou else ""))
                 for face, mask in built.views.items()]
        stats = (f"Print time {built.print_time_s / 60:.0f} min, filament {built.filament_g or 0:.1f} g"
                 if built.print_time_s else "G-code unavailable: slicer not installed")
        files = [str(p) for p in (built.stl, built.step, built.gcode) if p]
        return _pack(state, **{**review, "warnings": _bullets(built.warnings),
                               "message": "Built. Check every value marked (check)."},
                     model=str(built.stl), views=views, files=files, stats=stats)

    def _review(self, observed: Observed, res, state: dict) -> dict:
        spec = None if isinstance(res, MvAbstain) else res
        abstain = res if isinstance(res, MvAbstain) else None
        rows = value_rows(spec, abstain, state["edits"])
        state["shown"] = {r[0]: (r[1], r[2]) for r in rows}
        warnings = spec.warnings if spec is not None else observed.warnings
        message = _abstain(abstain) if abstain else "Review the faces and values, edit any number, then Rebuild."
        return {"faces": face_gallery(observed, spec), "reads": read_rows(observed), "values": rows,
                "warnings": _bullets(warnings), "message": message}


def build_app(pipe: MvPipeline) -> gr.Blocks:
    handlers = Handlers(pipe)
    with gr.Blocks(title="Sketch-to-CAD lab") as app:
        state = gr.State(new_state())
        gr.Markdown("# Sketch-to-CAD: multi-view lab\n\n" + NOTICE)
        with gr.Row():
            files = gr.File(label="Face images", file_count="multiple", file_types=["image"], type="filepath")
            with gr.Column():
                tags = gr.Dataframe(headers=["file", "face", "kind"], type="array", interactive=True,
                                    label=f"Face: {', '.join(FACE_CHOICES)}. Kind: {', '.join(KIND_CHOICES)}.")
                reference = gr.Dropdown(REFERENCES, value="none", label="Reference object in the photos")
                analyze = gr.Button("Analyze", variant="primary")
        message = gr.Markdown()
        with gr.Row():
            faces = gr.Gallery(label="Canonical faces", columns=3, height=280)
            reads = gr.Dataframe(headers=["face", "text", "value", "linked to"], type="array", interactive=False,
                                 label="Handwriting read")
        values = gr.Dataframe(headers=["field", "value (mm)", "source"], type="array", interactive=True,
                              label="Values: edit the value column")
        rejected = gr.CheckboxGroup(list(CANONICAL_FACES), label="Reject an AI-drawn face and use a rectangle")
        warnings = gr.Markdown()
        rebuild = gr.Button("Rebuild", variant="primary")
        with gr.Row():
            model = gr.Model3D(label="Part")
            views = gr.Gallery(label="Rendered views", columns=3, height=320)
        stats = gr.Markdown()
        downloads = gr.File(label="STL, STEP, G-code", file_count="multiple")
        outputs = [faces, reads, values, warnings, message, model, views, downloads, stats, state]
        files.change(tag_rows, files, tags)
        analyze.click(handlers.analyze, [files, tags, reference], outputs)
        rebuild.click(handlers.rebuild, [values, rejected, state], outputs)
    return app


if __name__ == "__main__":
    load_dotenv()
    build_app(default_pipeline()).launch()
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_mv_gradio.py -v`
Expected: PASS.

- [ ] **Step 5: See it run**

Run: `uv run python app_mv_gradio.py` and open http://localhost:7860. Upload `front` and `top` sketches (for example `tmp/mv_demo` inputs or two photos of paper sketches), tag them, Analyze, fill the three envelope values, Rebuild. Expected: a 3D part, six views, and STL and STEP downloads. Stop the server afterwards.

- [ ] **Step 6: Lint and commit**

Run: `uv run ruff check app_mv_gradio.py tests/test_mv_gradio.py`

```bash
git add app_mv_gradio.py tests/test_mv_gradio.py
git commit -m "Add the Gradio lab app for the multi-view path"
```

---

### Task 7: Sketch rescue

**Files:**
- Modify: `s2c/multiview/qwen_faces.py` (imports, rescue functions), `s2c/multiview/pipeline.py` (`observe`, new `_outline`)
- Create: `tests/test_mv_rescue.py`

**Interfaces:**
- Consumes: `ImageGen`, `ImageGenError` (Task 4); `extract`, `foreground`, `PixelOutline` (outline); `iou`, `polygon_mask` (raster); `SEED` (Task 5).
- Produces: `qwen_faces.raw_region(image_bgr) -> np.ndarray`; `qwen_faces.rescue_sketch(image_bgr, gen, seed=SEED) -> PixelOutline | None`; `qwen_faces.RESCUE_PENALTY = 0.8`; `MvPipeline._outline(bgr, mask_out, kind) -> tuple[PixelOutline | MvAbstain, bool]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mv_rescue.py
import cv2
import numpy as np
import pytest

from s2c.multiview.outline import extract
from s2c.multiview.pipeline import ImageInput, MvPipeline
from s2c.multiview.qwen_faces import rescue_sketch
from s2c.multiview.qwen_image import ImageGenError
from s2c.multiview.spec import MvAbstain
from tests.test_mv_qwen_faces import fake_gen


def broken_sketch(gap=30):
    """A 600 x 400 rectangle drawn in pen whose left side stops short: a gap at the top-left corner."""
    img = np.full((1200, 1600, 3), 255, np.uint8)
    x0, y0, x1, y1 = 500, 400, 1100, 800
    for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0 + gap))):
        cv2.line(img, a, b, (0, 0, 0), 4)
    return img


def clean(w=600, h=400):
    img = np.full((1200, 1600, 3), 255, np.uint8)
    cv2.rectangle(img, (500, 400), (500 + w, 400 + h), (0, 0, 0), -1)
    return img


def test_a_broken_sketch_has_no_outline():
    res = extract(broken_sketch())
    assert isinstance(res, MvAbstain) and res.reason == "no_outline"


def test_rescue_accepts_a_faithful_redraw():
    outline = rescue_sketch(broken_sketch(), fake_gen(clean()))
    assert outline is not None and abs(outline.bbox[2] - 601) <= 4


def test_rescue_rejects_a_distorted_redraw_or_a_failure():
    assert rescue_sketch(broken_sketch(), fake_gen(clean(600, 250))) is None
    assert rescue_sketch(broken_sketch(), fake_gen(ImageGenError("503"))) is None


def test_the_pipeline_rescues_a_broken_sketch_and_warns():
    data = cv2.imencode(".png", broken_sketch())[1].tobytes()
    observed = MvPipeline(image_gen=fake_gen(clean())).observe([ImageInput(data, "front", "sketch")])
    assert not isinstance(observed, MvAbstain)
    assert "front: sketch cleaned by Qwen-Image, check it" in observed.warnings
    assert observed.observations[0].confidence == pytest.approx(0.9 * 0.8)
    assert isinstance(MvPipeline().observe([ImageInput(data, "front", "sketch")]), MvAbstain)
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_mv_rescue.py -v`
Expected: FAIL with `ImportError: cannot import name 'rescue_sketch'`.

- [ ] **Step 3: Rescue in `qwen_faces.py`**

Replace the imports of `s2c/multiview/qwen_faces.py` with:

```python
import logging

import cv2
import numpy as np

from s2c.multiview.outline import PixelOutline, extract, foreground, resize_long_side, to_face_mm
from s2c.multiview.qwen_image import ImageGen, ImageGenError
from s2c.multiview.raster import iou, polygon_mask
from s2c.multiview.spec import CANONICAL_FACES, Envelope, MvAbstain, Outline, face_size
```

Append to the file:

```python
RESCUE_PROMPT = ("Redraw this hand sketch of a mechanical part face as a clean solid black silhouette on a pure "
                 "white background. Keep the proportions and position exactly. Remove all text, numbers, arrows "
                 "and dimension lines. Holes are white.")
RESCUE_IOU = 0.85
RESCUE_ASPECT = 0.05
RESCUE_PENALTY = 0.8
BRIDGE_FRACTION = 0.03  # of the long side: enough to close a gap in a pen line


def raw_region(image_bgr: np.ndarray) -> np.ndarray:
    """The pen strokes with small gaps bridged, largest region filled: what the sketch outline encloses."""
    ink = foreground(image_bgr)
    k = max(3, int(BRIDGE_FRACTION * max(ink.shape)) | 1)
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    region = np.zeros_like(ink)
    if contours:
        cv2.drawContours(region, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    return region


def rescue_sketch(image_bgr: np.ndarray, gen: ImageGen, seed: int = SEED) -> PixelOutline | None:
    """Qwen-Image redraws a sketch whose outline is not closed. Kept only if it matches the raw strokes."""
    h, w = image_bgr.shape[:2]
    try:
        drawn = gen([image_bgr], RESCUE_PROMPT, seed, "mv_rescue")
    except ImageGenError as e:
        log.warning("sketch rescue failed: %s", e)
        return None
    outline = extract(cv2.resize(drawn, (w, h), interpolation=cv2.INTER_AREA))
    if isinstance(outline, MvAbstain):
        return None
    raw = raw_region(image_bgr)
    if iou(polygon_mask(outline.outer, outline.inner, (h, w)), raw) < RESCUE_IOU:
        return None
    _, _, rw, rh = cv2.boundingRect(raw)
    _, _, cw, ch = outline.bbox
    if rh == 0 or ch == 0 or abs((cw / ch) / (rw / rh) - 1) > RESCUE_ASPECT:
        return None
    return outline
```

- [ ] **Step 4: Rescue in the pipeline**

In `s2c/multiview/pipeline.py`:

Add the import `from s2c.multiview.qwen_faces import RESCUE_PENALTY, rescue_sketch` (`PixelOutline` and `extract` are already imported from `s2c.multiview.outline`).

Add the method:

```python
    def _outline(self, bgr: np.ndarray, mask_out, kind: str) -> tuple[PixelOutline | S.MvAbstain, bool]:
        """The outline, and whether Qwen-Image had to redraw the sketch (spec 2026-09-23 section 8)."""
        outline = extract(bgr, mask_out)
        if (isinstance(outline, S.MvAbstain) and outline.reason == "no_outline" and kind != "photo"
                and self.image_gen is not None):
            fixed = rescue_sketch(bgr, self.image_gen)
            if fixed is not None:
                return fixed, True
        return outline, False
```

In `observe`, replace `outline = extract(bgr, mask_out)` with `outline, rescued = self._outline(bgr, mask_out, label.input_kind)`, change the `Observation(...)` confidence argument to `confidence=label.confidence * (RESCUE_PENALTY if rescued else 1.0)`, and after `attach_label(obs, label)` add:

```python
            if rescued:
                observed.warnings.append(f"{label.face}: sketch cleaned by Qwen-Image, check it")
```

- [ ] **Step 5: Run the suite**

Run: `uv run pytest -q` then `uv run ruff check s2c tests`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add s2c/multiview/qwen_faces.py s2c/multiview/pipeline.py tests/test_mv_rescue.py
git commit -m "Rescue sketches with an open outline by a checked Qwen-Image redraw"
```

---

### Task 8: Solaria depth

**Files:**
- Create: `s2c/multiview/depth.py`, `tests/test_mv_depth.py`
- Modify: `s2c/multiview/fuse.py` (`features_from`), `s2c/multiview/pipeline.py` (imports, `MvPipeline.__init__`, `observe`, new `_depths`, `default_pipeline`), `tests/test_mv_pipeline.py`

**Interfaces:**
- Consumes: `Observation` with `depth_ratio`, `depth_from_image` (Task 3); `log_call` (Task 4); `PixelOutline`, `resize_long_side` (outline); `polygon_mask` (raster).
- Produces:
  - `depth.DepthProvider = Callable[[np.ndarray], np.ndarray]` (BGR image to a depth array of the same height and width, NaN where unknown)
  - `depth.read_depth(ply_path, width, height, stride=4) -> np.ndarray`
  - `depth.solaria_depth(space, token=None, client_factory=None, timeout_s=180, log_path="logs/vlm.jsonl") -> DepthProvider`
  - `depth.hole_depths(depth, outline, exclude=()) -> tuple[dict[int, float | None], list[str]]` (None means through)
  - `depth.apply_depth(obs, depth, exclude=()) -> list[str]`
  - `MvPipeline(..., depth: DepthProvider | None = None)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mv_depth.py
import time

import cv2
import numpy as np
import pytest

from s2c.multiview.depth import apply_depth, read_depth, solaria_depth
from s2c.multiview.fuse import Observation, features_from
from s2c.multiview.outline import extract
from s2c.multiview.spec import Envelope


def solaria_ply(path, depth, stride=4):
    """A point cloud written exactly like Solaria's pointcloud.py: ASCII, normalised x and y, z scaled to 0..2."""
    h, w = depth.shape
    y, x = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[::stride, ::stride]
    z = (z - z.min()) / max(float(z.max() - z.min()), 1e-8) * 2.0
    xs, ys = (x - w / 2) / w, -(y - h / 2) / h
    header = ("ply\nformat ascii 1.0\nelement vertex {n}\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write(header.format(n=z.size))
        for a, b, c in zip(xs.ravel(), ys.ravel(), z.ravel()):
            f.write(f"{a:.6f} {b:.6f} {c:.6f} 128 128 128\n")


def plate_photo():
    img = np.full((1200, 1600, 3), 200, np.uint8)
    cv2.rectangle(img, (500, 400), (1100, 800), (40, 40, 40), -1)
    for cx in (650, 950):
        cv2.circle(img, (cx, 600), 40, (200, 200, 200), -1)
    return img


def scene(blind_ratio=0.4, table=10.0, face=0.0):
    """Depth of plate_photo: table, part face, a through hole at x 650 and a blind hole at x 950."""
    depth = np.full((1200, 1600), table, np.float32)
    depth[400:801, 500:1101] = face
    yy, xx = np.mgrid[0:1200, 0:1600]
    depth[(xx - 650) ** 2 + (yy - 600) ** 2 <= 40 ** 2] = table
    depth[(xx - 950) ** 2 + (yy - 600) ** 2 <= 40 ** 2] = face + blind_ratio * (table - face)
    return depth


def plate(**kw):
    return Observation(face="top", kind="photo", outline=extract(plate_photo()), **kw)


def test_the_point_cloud_turns_back_into_a_depth_map(tmp_path):
    depth = np.random.default_rng(0).random((120, 160)).astype(np.float32)
    solaria_ply(tmp_path / "c.ply", depth)
    got = read_depth(tmp_path / "c.ply", 160, 120)
    want = (depth[::4, ::4] - depth[::4, ::4].min()) / (depth[::4, ::4].max() - depth[::4, ::4].min()) * 2
    assert got.shape == (30, 40) and np.allclose(got, want, atol=1e-5)


def test_through_and_blind_holes_from_a_depth_map():
    o = plate()
    assert apply_depth(o, scene(0.4)) == []
    assert o.blind == {0: False, 1: True} and o.depth_ratio[1] == pytest.approx(0.4, abs=0.02)
    assert o.depth_from_image == {0, 1}


def test_ratios_do_not_depend_on_scale_shift_or_direction():
    o = plate()
    apply_depth(o, 3.0 - 0.25 * scene(0.4))
    assert o.depth_ratio[1] == pytest.approx(0.4, abs=0.02)


def test_a_flat_scene_is_not_measured():
    o = plate(blind={1: True})
    warnings = apply_depth(o, scene(0.4, table=0.0))
    assert o.depth_from_image == set() and o.blind == {1: True}
    assert warnings == ["top: depth: part too flat to measure, check hole depths"]


def test_a_shallow_circle_is_a_mark():
    o = plate()
    warnings = apply_depth(o, scene(0.05))
    assert 1 not in o.depth_from_image and "top: hole 2 looks like a mark, not a hole" in warnings


def test_fuse_turns_the_ratio_into_millimetres():
    o = Observation(face="front", kind="photo", outline=extract(plate_photo()), blind={1: True},
                    depth_ratio={1: 0.4})
    feats, prov = features_from([o], Envelope(x_mm=60.0, y_mm=40.0, z_mm=10.0))
    assert feats[0]["depth_mm"] is None and feats[1]["depth_mm"] == pytest.approx(4.0)
    assert prov["features[1].depth_mm"] == "estimated"


def test_the_solaria_provider_sends_a_white_mask_and_returns_depth_at_image_size(tmp_path):
    ply = tmp_path / "cloud.ply"

    class Fake:
        def __init__(self, src, token=None):
            pass

        def predict(self, image, mask, api_name):
            m = cv2.imread(mask["path"], cv2.IMREAD_GRAYSCALE)
            assert api_name == "/gerar_3d" and m.min() == 255
            h, w = m.shape
            solaria_ply(ply, np.tile(np.linspace(0, 1, w, dtype=np.float32), (h, 1)))
            return str(tmp_path / "d.png"), str(ply), "ok"

    depth = solaria_depth("CronosSa/Solaria1.0", None, Fake, log_path=tmp_path / "l")(np.zeros((1200, 1600, 3), np.uint8))
    assert depth.shape == (1200, 1600) and np.nanmax(depth) == pytest.approx(2.0, abs=0.01)


def test_a_failing_space_raises(tmp_path):
    class Down:
        def __init__(self, *a, **k):
            pass

        def predict(self, *a, **k):
            return None, None, "Erro durante a geração 3D"

    with pytest.raises(RuntimeError):
        solaria_depth("s", None, Down, log_path=tmp_path / "l")(np.zeros((100, 100, 3), np.uint8))


def test_a_slow_solaria_times_out(tmp_path):
    class Slow:
        def __init__(self, *a, **k):
            pass

        def predict(self, *a, **k):
            time.sleep(2)
            return None, None, "late"

    t0 = time.monotonic()
    with pytest.raises(Exception):
        solaria_depth("s", None, Slow, timeout_s=0.2, log_path=tmp_path / "l")(np.zeros((100, 100, 3), np.uint8))
    assert time.monotonic() - t0 < 1.5
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_mv_depth.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 's2c.multiview.depth'`.

- [ ] **Step 3: Write `depth.py`**

```python
# s2c/multiview/depth.py
"""Solaria (Marigold V2 depth on a Hugging Face Space) -> through or blind holes, and blind depth, on photos only.
Spec 2026-09-23 section 9. Marigold depth is right only up to scale and shift, and Solaria rescales it again, so
only ratios of depth differences are used: they do not depend on scale, shift or which way depth grows.
The point cloud is never exported."""
from __future__ import annotations

import concurrent.futures as cf
import logging
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from s2c.multiview.fuse import Observation
from s2c.multiview.outline import PixelOutline, resize_long_side
from s2c.multiview.qwen_image import log_call
from s2c.multiview.raster import polygon_mask

log = logging.getLogger(__name__)
DepthProvider = Callable[[np.ndarray], np.ndarray]  # BGR image -> depth at the same size, NaN where unknown
STRIDE = 4
SEND_LONG_SIDE = 1024
TIMEOUT_S = 180
THROUGH, MARK, FLAT = 0.9, 0.1, 0.05
NEAR, FAR = 0.05, 0.15  # table band, as fractions of the part's bounding-box diagonal
_pool = cf.ThreadPoolExecutor(max_workers=2)


def read_depth(ply_path, width: int, height: int, stride: int = STRIDE) -> np.ndarray:
    """Invert Solaria's point cloud: x = (col - W/2) / W, y = -(row - H/2) / H, z = depth, on a stride grid."""
    lines = Path(ply_path).read_text(encoding="utf-8").splitlines()
    body = lines[lines.index("end_header") + 1:]
    out = np.full((-(-height // stride), -(-width // stride)), np.nan, np.float32)
    if not body:
        return out
    pts = np.loadtxt(body, ndmin=2, usecols=(0, 1, 2))
    cols = np.rint((pts[:, 0] + 0.5) * width).astype(int) // stride
    rows = np.rint((0.5 - pts[:, 1]) * height).astype(int) // stride
    ok = (rows >= 0) & (rows < out.shape[0]) & (cols >= 0) & (cols < out.shape[1])
    out[rows[ok], cols[ok]] = pts[ok, 2]
    return out


def solaria_depth(space: str, token: str | None = None, client_factory=None, timeout_s: float = TIMEOUT_S,
                  log_path="logs/vlm.jsonl") -> DepthProvider:
    def run(img: np.ndarray) -> np.ndarray:
        from gradio_client import Client, handle_file
        h, w = img.shape[:2]
        client = (client_factory or Client)(space, token=token)
        with tempfile.TemporaryDirectory() as d:
            src, mask = Path(d) / "image.png", Path(d) / "mask.png"
            cv2.imwrite(str(src), img)
            cv2.imwrite(str(mask), np.full((h, w), 255, np.uint8))  # all white: depth for the whole frame
            _, ply, status = client.predict(handle_file(str(src)), handle_file(str(mask)), api_name="/gerar_3d")
        ply = ply.get("path") if isinstance(ply, dict) else ply
        if not ply:
            raise RuntimeError(f"Solaria returned no point cloud: {status}")
        return read_depth(ply, w, h)

    def provide(image_bgr: np.ndarray) -> np.ndarray:
        img = resize_long_side(image_bgr, SEND_LONG_SIDE) if max(image_bgr.shape[:2]) > SEND_LONG_SIDE else image_bgr
        t0, ok = time.perf_counter(), False
        try:
            small = _pool.submit(run, img).result(timeout=timeout_s)
            ok = True
        finally:
            log_call(log_path, "hf-space", space, "mv_depth", t0, ok)
        h, w = image_bgr.shape[:2]
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    return provide


def _median(depth: np.ndarray, where: np.ndarray) -> float | None:
    values = depth[where]
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else None


def hole_depths(depth: np.ndarray, outline: PixelOutline, exclude=()) -> tuple[dict[int, float | None], list[str]]:
    """Per circle: None when it goes through, else its depth as a fraction of the axis length. A mark, or a part
    too flat to measure, gets no entry and a warning. The part lies flat, so the table is one axis length down."""
    h, w = depth.shape
    filled = polygon_mask(outline.outer, (), (h, w))
    part = polygon_mask(outline.outer, outline.inner, (h, w)) > 127
    yy, xx = np.mgrid[0:h, 0:w]
    dist2 = [(xx - c.cx) ** 2 + (yy - c.cy) ** 2 for c in outline.circles]
    for c, d2 in zip(outline.circles, dist2):
        part &= d2 > (c.d / 2) ** 2
    diag = float(np.hypot(outline.bbox[2], outline.bbox[3]))
    away = cv2.distanceTransform(255 - filled, cv2.DIST_L2, 5)
    table = (away > NEAR * diag) & (away <= FAR * diag)
    for x, y, bw, bh in exclude:
        table[max(y, 0): y + bh, max(x, 0): x + bw] = False
    finite = depth[np.isfinite(depth)]
    span = float(np.percentile(finite, 98) - np.percentile(finite, 2)) if finite.size else 0.0
    d_table = _median(depth, table)
    found, warnings = {}, []
    for i, (c, d2) in enumerate(zip(outline.circles, dist2)):
        r = c.d / 2
        d_face = _median(depth, part & (d2 >= r * r) & (d2 <= (1.5 * r) ** 2))
        d_hole = _median(depth, d2 <= (0.7 * r) ** 2)
        if d_table is None or d_face is None or d_hole is None or abs(d_table - d_face) <= FLAT * span:
            warnings.append("depth: part too flat to measure, check hole depths")
            continue
        ratio = (d_hole - d_face) / (d_table - d_face)
        if ratio >= THROUGH:
            found[i] = None
        elif ratio >= MARK:
            found[i] = float(ratio)
        else:
            warnings.append(f"hole {i + 1} looks like a mark, not a hole")
    return found, list(dict.fromkeys(warnings))


def apply_depth(obs: Observation, depth: np.ndarray, exclude=()) -> list[str]:
    """Write Solaria's verdicts into the observation; they override the vision model's labels."""
    found, warnings = hole_depths(depth, obs.outline, exclude)
    for i, ratio in found.items():
        obs.depth_from_image.add(i)
        obs.blind[i] = ratio is not None
        obs.depth_estimates.pop(i, None)
        if ratio is None:
            obs.depth_ratio.pop(i, None)
        else:
            obs.depth_ratio[i] = ratio
    return [f"{obs.face}: {w}" for w in warnings]
```

- [ ] **Step 4: Ratio to millimetres in `fuse.features_from`**

In `s2c/multiview/fuse.py`, `features_from`, replace

```python
            if o.blind.get(i):
                if i in o.depth_estimates:
```

with

```python
            if o.blind.get(i):
                if i in o.depth_ratio:
                    depth, depth_prov = o.depth_ratio[i] * axis_len, "estimated"
                elif i in o.depth_estimates:
```

- [ ] **Step 5: Run the depth tests**

Run: `uv run pytest tests/test_mv_depth.py -v`
Expected: PASS.

- [ ] **Step 6: Write the failing pipeline tests**

Add to `tests/test_mv_pipeline.py`:

```python
def test_solaria_runs_once_per_face_and_its_ratio_becomes_a_depth():
    from tests.test_mv_depth import plate_photo, scene
    calls = []

    def depth(img):
        calls.append(img.shape)
        return scene(0.4)

    data = cv2.imencode(".png", plate_photo())[1].tobytes()
    pipe = MvPipeline(depth=depth)
    observed = pipe.observe([ImageInput(data, "front", "photo"), ImageInput(data, "front", "photo")])
    assert len(calls) == 1
    spec = pipe.fuse(observed, {"envelope.x_mm": 60, "envelope.y_mm": 40, "envelope.z_mm": 10})
    assert sorted(f.depth_mm for f in spec.features if f.depth_mm) == [4.0]


def test_a_failing_depth_provider_only_warns():
    from tests.test_mv_depth import plate_photo

    def down(img):
        raise RuntimeError("Space asleep")

    data = cv2.imencode(".png", plate_photo())[1].tobytes()
    observed = MvPipeline(depth=down).observe([ImageInput(data, "front", "photo")])
    assert "front: depth unavailable" in observed.warnings
```

Run: `uv run pytest tests/test_mv_pipeline.py -v`
Expected: FAIL with `unexpected keyword argument 'depth'`.

- [ ] **Step 7: Wire depth into the pipeline**

In `s2c/multiview/pipeline.py`:

Add `import os` to the standard-library imports and `from s2c.multiview.depth import DepthProvider, apply_depth, solaria_depth` to the package imports.

Constructor: add the parameter `depth: DepthProvider | None = None` after `image_gen`, and the line `self.depth = depth`.

In `observe`: before the `for item in images:` loop add `excluded: list[tuple] = []`; after `observed.labels.append(label)` add `excluded.append(mask_out)`; replace the final `return self._merge(observed)` with:

```python
        if self.depth is not None:
            observed.warnings += self._depths(observed, excluded)
        return self._merge(observed)
```

Add the method:

```python
    def _depths(self, observed: Observed, excluded: list[tuple]) -> list[str]:
        """Solaria once per face, on its most confident photo that shows a hole (spec 2026-09-23 section 9)."""
        best: dict[str, int] = {}
        for k, o in enumerate(observed.observations):
            if o.kind == "photo" and o.outline.circles:
                if o.face not in best or o.confidence > observed.observations[best[o.face]].confidence:
                    best[o.face] = k
        warnings = []
        for face, k in best.items():
            try:
                depth = self.depth(observed.images[k])
            except Exception as e:
                log.warning("Solaria failed on %s: %s", face, e)
                warnings.append(f"{face}: depth unavailable")
                continue
            warnings += apply_depth(observed.observations[k], depth, excluded[k])
        return warnings
```

In `default_pipeline`, before the `return`, add `space = os.environ.get("SOLARIA_SPACE")`, pass `depth=solaria_depth(space, os.environ.get("HF_TOKEN")) if space else None` to `MvPipeline(...)`, and change the docstring to `"""Qwen-VL, Qwen-Image and Solaria from the environment; TrOCR and TripoSR when the ai extra is installed."""`.

- [ ] **Step 8: Run the suite**

Run: `uv run pytest -q` then `uv run ruff check s2c tests`
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add s2c/multiview/depth.py s2c/multiview/fuse.py s2c/multiview/pipeline.py tests/test_mv_depth.py tests/test_mv_pipeline.py
git commit -m "Tell through from blind holes and estimate depth with Solaria"
```

---

### Task 9: Live tests, docs and disclosure

**Files:**
- Create: `tests/test_mv_network.py`, `docs/disclosure.md`
- Modify: `README.md` (multi-view section), `docs/models.md` (multi-view table)

**Interfaces:**
- Consumes: `env_chat`, `qwen_batch_reader`, `default_gen`, `face_prompt`, `outline_from_image`, `solaria_depth`, `parse_value`.
- Produces: the live check to run before the demo, and the disclosure the judges read.

- [ ] **Step 1: Write the live tests**

```python
# tests/test_mv_network.py
"""Real calls to the hosted models. Needs a filled .env:
    NETWORK_TESTS=1 uv run pytest tests/test_mv_network.py -v"""
import os

import cv2
import numpy as np
import pytest
from dotenv import load_dotenv

pytestmark = [pytest.mark.network,
              pytest.mark.skipif(os.environ.get("NETWORK_TESTS") != "1", reason="set NETWORK_TESTS=1")]


@pytest.fixture(autouse=True)
def env():
    load_dotenv()


def written(text):
    img = np.full((80, 200, 3), 255, np.uint8)
    cv2.putText(img, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    return img


def test_qwen_vl_reads_written_values():
    from s2c.multiview.label import env_chat
    from s2c.multiview.ocr import parse_value
    from s2c.multiview.qwen_reader import qwen_batch_reader
    chat = env_chat(stage="mv_read")
    assert chat, "set VLM_BASE_URL, VLM_MODEL and VLM_API_KEY"
    reads = qwen_batch_reader(chat)([written("60"), written("R3")])
    assert [parse_value(text) for text, _ in reads] == [(60.0, "linear"), (3.0, "radius")]


def test_qwen_image_draws_a_top_view():
    from s2c.multiview.qwen_faces import face_prompt, outline_from_image
    from s2c.multiview.qwen_image import default_gen
    from s2c.multiview.spec import Envelope
    gen = default_gen()
    assert gen, "set QWEN_IMAGE_SPACE or the DashScope settings"
    front = np.full((800, 1000, 3), 255, np.uint8)
    cv2.rectangle(front, (200, 250), (800, 550), (0, 0, 0), 4)
    img = gen([front], face_prompt(["front"], "top"), 7, "mv_face")
    assert outline_from_image(img, "top", Envelope(x_mm=60.0, y_mm=30.0, z_mm=20.0)) is not None


def test_solaria_returns_a_depth_map():
    from s2c.multiview.depth import solaria_depth
    from tests.test_mv_depth import plate_photo
    space = os.environ.get("SOLARIA_SPACE")
    assert space, "set SOLARIA_SPACE"
    depth = solaria_depth(space, os.environ.get("HF_TOKEN"))(plate_photo())
    assert depth.shape == (1200, 1600) and np.isfinite(depth).mean() > 0.5
```

Run: `uv run pytest tests/test_mv_network.py -v`
Expected: 3 skipped.

With a filled `.env`, run: `NETWORK_TESTS=1 uv run pytest tests/test_mv_network.py -v`
Expected: 3 passed. Record the date, the Qwen-VL model name and the Solaria Space commit (`f531e56` on 2026-09-23) in `docs/models.md`. A failure here is information, not a blocker: the pipeline falls back. Tell your human partner which one failed.

- [ ] **Step 2: `docs/models.md`**

Replace the `## Multi-view path` table with:

```markdown
## Multi-view path

| Model | Source | Use | Runs | Checked |
| --- | --- | --- | --- | --- |
| Qwen-VL | Alibaba Cloud Model Studio (DashScope), OpenAI-compatible | Face labels; transcribes handwritten values, one call per image | `VLM_BASE_URL`, `VLM_MODEL` | pending |
| Qwen-Image-2.1 | `Qwen/Qwen-Image-2.1` (Qwen Research License) | Draws faces nobody photographed; redraws sketches whose outline is open. Only read back as outlines | `QWEN_IMAGE_SPACE`, or DashScope with `QWEN_IMAGE_BACKEND=dashscope` | pending |
| Solaria 1.0 (Marigold V2 depth) | `CronosSa/Solaria1.0` Space, commit `f531e56` on 2026-09-23 | Depth map of a photo: through or blind holes, blind depth as a ratio | `SOLARIA_SPACE` (ZeroGPU, `HF_TOKEN` for quota) | pending |
| TripoSR | `stabilityai/TripoSR` (MIT) | Fallback when Qwen-Image is unavailable or rejected; only its silhouettes are used | Local CUDA, else `TRIPOSR_SPACE` | |
| TrOCR base handwritten | `microsoft/trocr-base-handwritten` | Fallback reader when Qwen-VL is unavailable | Local, CUDA or CPU | |
| rembg (u2net) | `rembg` | Removes the background before TripoSR | Local CPU | |
| PrusaSlicer | prusa3d.com (AGPL) | Slices the STL to G-code with `profiles/fdm_default.ini` | Local CLI | |

Before the demo: `NETWORK_TESTS=1 uv run pytest tests/test_mv_network.py -v`, then fill in the Checked column with the date.
```

- [ ] **Step 3: `docs/disclosure.md`**

```markdown
# Tools, models and data

## Models called at run time

| Model | Provider | What it does here | What it never does |
| --- | --- | --- | --- |
| Qwen-VL | Alibaba Cloud Model Studio (DashScope) | Says which face a photo shows; copies the handwritten numbers | Estimate a size; write code |
| Qwen-Image-2.1 | Hugging Face Space `Qwen/Qwen-Image-2.1` or DashScope | Draws the silhouette of a face nobody photographed; cleans up a sketch with an open outline | Produce a file we export; set a dimension |
| Marigold V2 via Solaria 1.0 | Hugging Face Space `CronosSa/Solaria1.0` | Depth map of a photo, to tell through from blind holes | Set a dimension without an amber "estimated" badge |
| TripoSR | Local GPU or Hugging Face Space | Fallback face prediction | Produce a file we export |
| TrOCR | Local | Fallback handwriting reader | |

Every call is logged to `logs/vlm.jsonl` with provider, model, stage and latency.

## Data

- Images are sent to DashScope and to Hugging Face Spaces for the calls above. The app says so next to the upload.
- Locally, images live only for the request; silhouettes and exported files stay in `tmp/` for one hour.
- No model output is executed. Geometry is built by our own CadQuery code from validated JSON.

## Licences

- Qwen-Image-2.1: Qwen Research License. Confirmed allowed for this hackathon demo on: pending.
- TripoSR: MIT. PrusaSlicer: AGPL, run as a separate program. rembg: MIT.
```

- [ ] **Step 4: `README.md`**

Replace the `## Multi-view path (pending team sign-off)` section body with:

```markdown
## Multi-view path (pending team sign-off)

Give one or more images per face, several of the same face if you have them: they are aligned and voted into one cleaner outline. Qwen-VL reads the numbers you wrote. Faces you did not give are drawn by Qwen-Image and kept only if they agree with the faces you did give; otherwise TripoSR, otherwise a rectangle. Solaria's depth map tells through holes from blind ones. The part is the intersection of the three extruded outlines, sliced to G-code. Designs: `docs/superpowers/specs/2026-09-22-multiview-gcode-design.md` and `docs/superpowers/specs/2026-09-23-qwen-solaria-design.md`.

    uv run python app_mv_gradio.py                                                           # lab app on :7860
    uv run python scripts/mv_build.py examples/mv/l_bracket.json --out tmp/mv_demo          # spec -> STEP, STL, G-code
    uv run python scripts/mv.py --image front.jpg@front@sketch --image top.jpg@top@sketch   # images -> the same
    uv run uvicorn s2c.multiview.app:app --port 8001                                        # /mv API
    NETWORK_TESTS=1 uv run pytest tests/test_mv_network.py -v                               # live check of the hosted models

Settings are in `.env.example`: Qwen-VL and Qwen-Image on DashScope or Hugging Face, Solaria on Hugging Face. Photos of real parts: shoot top-down with the part lying flat.

G-code needs PrusaSlicer: `winget install --id Prusa3D.PrusaSlicer -e` (needs admin), or unzip the portable zip from the PrusaSlicer GitHub release into `vendor/`. Without it you still get STL and STEP.
```

- [ ] **Step 5: Full check**

Run: `uv run pytest -q` then `uv run ruff check s2c tests scripts app_mv_gradio.py`
Expected: all pass, 3 network tests skipped.

- [ ] **Step 6: Commit (do not push)**

```bash
git add tests/test_mv_network.py docs/models.md docs/disclosure.md README.md
git commit -m "Add live model checks, model table and disclosure for the multi-view path"
```

- [ ] **Step 7: Hand over the golden captures**

Tell your human partner: the golden multi-view set in `tests/golden_mv/` is still empty. It needs real captures by a person (five parts, one and three views each, per `tests/golden_mv/README.md`), and one more case per part with two photos of the same face to exercise merging. Then run `GOLDEN_AI=1 uv run pytest tests/test_mv_golden.py -v`.
