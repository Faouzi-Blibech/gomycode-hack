"""What the Studio's buttons do, as plain methods that tests call without a browser. Each returns a small view model
(Review, Model, Exported) that app.py maps onto components. Spec 2026-09-23-studio section 8."""
from __future__ import annotations

import json
import math
import random
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from s2c.multiview import spec as S
from s2c.multiview.artifacts import ROOT, build_part, bundle, export_part, sweep
from s2c.multiview.fuse import fuse_envelope
from s2c.multiview.pipeline import ImageInput, MvPipeline
from s2c.multiview.raster import iou, outline_mask
from s2c.multiview.settings import (
    DENSITIES,
    AiSettings,
    ExportSettings,
    GeometrySettings,
    MeshSettings,
    PrintSettings,
    filament_metres,
)
from s2c.studio.session import Item, SessionStore
from s2c.studio.theme import FACE_BADGES, TRUSTED, bullet_html, card, chip, source_chip, stats_html

FACE_CHOICES = ["auto", *S.FACES]
KIND_CHOICES = ["auto", "sketch", "photo", "drawing"]
REFERENCES = ["none", "1 TND", "1 EUR", "2 EUR", "card", "a4"]
AXES = ("x", "y", "z")
AXIS_LABEL = {"x": "Width (X)", "y": "Height (Y)", "z": "Depth (Z)"}
EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "mv" / "sketches"
_FEATURE = re.compile(r"(features|finishes)\[(\d+)\]\.(\w+)")
_FIELD_WORDS = {"a_mm": "position a", "b_mm": "position b", "diameter_mm": "diameter", "depth_mm": "depth",
                "width_mm": "width", "length_mm": "length", "angle_deg": "angle", "radius_mm": "size"}


@dataclass
class Review:
    ok: bool
    stage: str  # "capture" when the images themselves failed, else "review"
    message_html: str
    sizes: dict[str, dict] = field(default_factory=dict)  # axis -> value, placeholder, info, required
    rows: list[list] = field(default_factory=list)       # field, value, source chip
    faces: list[tuple[np.ndarray, str]] = field(default_factory=list)
    ai_faces: list[str] = field(default_factory=list)
    warnings_html: str = ""
    reads: list[list] = field(default_factory=list)
    seed: int = 7
    unchecked: int = 0
    rejected: list[str] = field(default_factory=list)         # currently-rejected faces, for the checkbox value
    reject_choices: list[str] = field(default_factory=list)   # ai_faces plus rejected, canonical order


@dataclass
class Model:
    ok: bool
    message_html: str
    preview: str | None = None
    views: list[tuple[np.ndarray, str]] = field(default_factory=list)
    stats_html: str = ""
    open_step: int | None = None  # the step whose controls can fix a failure the user cannot fix where they are


@dataclass
class Exported:
    message_html: str
    files: list[str] = field(default_factory=list)
    zip_path: str | None = None
    stats_html: str = ""


def parse_size(text) -> float | None:
    """'42,5', '42 mm' and ' 42 ' are 42.5, 42 and 42; anything that is not a positive number is None."""
    t = str(text or "").strip().lower().removesuffix("mm").strip().replace(",", ".")
    try:
        value = float(t)
    except ValueError:
        return None
    return value if math.isfinite(value) and value > 0 else None


def duration_text(seconds: float) -> str:
    """2692 s is '45 min' and 6300 s is '1 h 45 min': hours are floored, never rounded up."""
    hours, minutes = divmod(round(seconds / 60), 60)
    return f"{hours} h {minutes} min" if hours else f"{minutes} min"


def field_label(spec: S.MultiViewSpec, path: str) -> str:
    m = _FEATURE.fullmatch(path)
    if not m:
        return path
    group, k, name = m.group(1), int(m.group(2)), m.group(3)
    if group == "finishes":
        return f"{spec.finishes[k].type.capitalize()} {_FIELD_WORDS.get(name, name)}"
    f = spec.features[k]
    return f"{f.type.capitalize()} {k + 1} ({f.face}) · {_FIELD_WORDS.get(name, name)}"


def _value(spec: S.MultiViewSpec, path: str) -> float:
    data = spec.model_dump()
    m = _FEATURE.fullmatch(path)
    if m:
        return data[m.group(1)][int(m.group(2))][m.group(3)]
    head, name = path.split(".", 1)
    return data[head][name]


_REASON_WORDS = {"bad_image": "Image not readable"}


def _reason_words(reason: str) -> str:
    return _REASON_WORDS.get(reason, reason.replace("_", " ").capitalize())


def _abstain_card(a: S.MvAbstain) -> str:
    return card(f"Stopped at {a.stage}: {_reason_words(a.reason)}", a.remedy, "stop")


class Studio:
    def __init__(self, pipe: MvPipeline, store: SessionStore | None = None, root: Path = ROOT):
        self.pipe, self.store, self.root = pipe, store or SessionStore(), Path(root)

    # ---- capture -----------------------------------------------------------------------------------------
    def add_images(self, sid: str, paths) -> None:
        session = self.store.get(sid)
        for p in paths or []:
            session.items.append(Item(uuid.uuid4().hex[:8], str(p), Path(p).name))

    def set_face(self, sid: str, item_id: str, face: str) -> None:
        for item in self.store.get(sid).items:
            if item.id == item_id and face in FACE_CHOICES:
                item.face = face

    def set_kind(self, sid: str, item_id: str, kind: str) -> None:
        for item in self.store.get(sid).items:
            if item.id == item_id and kind in KIND_CHOICES:
                item.kind = kind

    def remove(self, sid: str, item_id: str) -> None:
        session = self.store.get(sid)
        session.items = [i for i in session.items if i.id != item_id]

    def load_examples(self, sid: str) -> None:
        session = self.store.get(sid)
        session.items = []
        for entry in json.loads((EXAMPLES / "examples.json").read_text()):
            session.items.append(Item(uuid.uuid4().hex[:8], str(EXAMPLES / entry["file"]), entry["file"],
                                      entry["face"], entry["kind"]))

    def coverage_html(self, sid: str) -> str:
        items = self.store.get(sid).items
        if not items:
            return chip("Add at least one image", "info")
        counts = {face: sum(1 for i in items if i.face == face) for face in S.FACES}
        auto = sum(1 for i in items if i.face == "auto")
        chips = []
        for canon, opposite in (("front", "back"), ("top", "bottom"), ("right", "left")):
            n, m = counts[canon], counts[opposite]
            if n or m:
                chips.append(chip(f"{canon} ✓{n}" + (f" · {opposite} ✓{m}" if m else ""), "ok"))
            else:
                chips.append(chip(f"{canon}: AI will draw it", "ai"))
        if auto:
            chips.append(chip(f"{auto} image(s) face: auto", "check"))
        return "".join(chips)

    # ---- review --------------------------------------------------------------------------------------------
    def analyze(self, sid: str, reference: str, ai: AiSettings) -> Review:
        session = self.store.get(sid)
        if not session.items:
            return Review(False, "capture", card("No images yet", "Drop sketches or photos, then Analyze.", "check"))
        if ai.randomize_seed:
            ai = ai.model_copy(update={"seed": random.randint(0, 2**31 - 1)})
        session.ai, session.reference = ai, reference
        images = []
        for item in session.items:
            try:
                data = Path(item.path).read_bytes()
            except OSError:
                missing = card("Image missing", f"{item.name} is no longer available. Add it again.", "stop")
                return Review(False, "capture", missing)
            if cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) is None:
                bad = card("Image not readable", f"{item.name} is not an image we can read. Upload a JPEG or PNG.",
                          "stop")
                return Review(False, "capture", bad)
            images.append(ImageInput(data, None if item.face == "auto" else item.face,
                                     None if item.kind == "auto" else item.kind))
        pipe = self.pipe.configured(ai)
        observed = pipe.observe(images, None if reference in (None, "", "none") else reference)
        if isinstance(observed, S.MvAbstain):
            return Review(False, "capture", _abstain_card(observed), seed=ai.seed)
        session.observed, session.edits, session.rejected, session.part = observed, {}, (), None
        return self._review(session, pipe.fuse(observed, geometry=session.geometry))

    def redraw(self, sid: str) -> Review:
        session = self.store.get(sid)
        if session.observed is None:
            return Review(False, "capture", card("Nothing to redraw", "Analyze your images first.", "check"))
        seed = random.randint(0, 2**31 - 1) if session.ai.randomize_seed else session.ai.seed + session.ai.attempts
        session.ai = session.ai.model_copy(update={"seed": seed})
        return self._review(session, self._fuse(session))

    def suggested_sizes(self, sid: str) -> dict[str, str]:
        """The suggestion for each size the last review is missing, e.g. {"z": "10"}: a view that shows that axis
        next to a known one, scaled. The same fuse_envelope call the review made, so the same partial."""
        session = self.store.get(sid)
        if session.observed is None:
            return {}
        res = fuse_envelope(session.observed.observations, dict(session.edits))
        suggested = (res.partial or {}).get("suggested", {}) if isinstance(res, S.MvAbstain) else {}
        return {axis: f"{suggested[f'envelope.{axis}_mm']:g}" for axis in AXES if f"envelope.{axis}_mm" in suggested}

    def _fuse(self, session):
        return self.pipe.configured(session.ai).fuse(session.observed, dict(session.edits),
                                                     rejected=session.rejected, geometry=session.geometry)

    def _review(self, session, res) -> Review:
        spec = None if isinstance(res, S.MvAbstain) else res
        abstain = res if isinstance(res, S.MvAbstain) else None
        session.spec = spec
        observed = session.observed
        known = (abstain.partial or {}).get("known", {}) if abstain else {}
        suggested = ((abstain.partial or {}).get("suggested", {}) if abstain else {})
        sizes = {}
        for axis in AXES:
            path = f"envelope.{axis}_mm"
            if spec is not None:
                value, prov = getattr(spec.envelope, f"{axis}_mm"), spec.provenance[path]
            else:
                value = known.get(path, session.edits.get(path))
                prov = "user_edited" if path in session.edits else None
            hint = suggested.get(path)
            sizes[axis] = {"value": "" if value is None else f"{value:g}",
                           "placeholder": f"suggested {hint:g}" if hint else "mm",
                           "info": f"from: {prov.replace('_', ' ')}" if prov
                           else "Required: type it or use the suggestion",
                           "required": value is None}
            session.shown[path] = value
        rows, paths = [], []
        if spec is not None:
            for path, prov in spec.provenance.items():
                if path.startswith(("envelope.", "views.")):
                    continue
                paths.append(path)
                session.shown[path] = _value(spec, path)
                rows.append([field_label(spec, path), _value(spec, path), source_chip(prov)])
        session.row_paths = paths
        faces, ai_faces = [], []
        if spec is not None:
            for face in S.CANONICAL_FACES:
                ol = getattr(spec.views, face)
                a, b = S.face_size(face, spec.envelope)
                mask = 255 - outline_mask(ol.outer, ol.inner, a, b, px=256)
                who = observed.filled_by.get(face, ol.source)
                text, _ = FACE_BADGES.get(who, (who, "info"))
                merged = next((w.split(": ", 1)[1] for w in spec.warnings if w.startswith(f"{face}: merged")), "")
                faces.append((mask, f"{face}: {text}" + (f" · {merged}" if merged else "")))
                if who in ("qwen-image", "triposr"):
                    ai_faces.append(face)
        else:
            faces = [(255 - m, f"{face}: your image") for face, m in observed.masks.items()]
        rejected = [f for f in S.CANONICAL_FACES if f in session.rejected]
        reject_choices = [f for f in S.CANONICAL_FACES if f in ai_faces or f in rejected]
        warnings = spec.warnings if spec is not None else observed.warnings
        info = [w for w in warnings if ": merged " in w]
        check = [w for w in warnings if w not in info]
        reads = [[o.face, lv.reading.text, lv.reading.value_mm,
                  f"hole {lv.hole_index + 1}" if lv.hole_index is not None
                  else f"{lv.axis} axis" if lv.axis else "not linked"]
                 for o in observed.observations for lv in o.values]
        unchecked = sum(1 for p in paths if spec.provenance[p] not in TRUSTED)  # the amber rows of the table
        message = (_abstain_card(abstain) if abstain else
                   card("Ready to build", "Check the amber values, reject any AI face you do not trust, then Build.",
                        "ok"))
        return Review(spec is not None, "review", message, sizes, rows, faces, ai_faces,
                      bullet_html("Check", check, "check") + bullet_html("Info", info, "info"), reads,
                      session.ai.seed, unchecked, rejected, reject_choices)

    # ---- build ----------------------------------------------------------------------------------------------
    def build(self, sid: str, sizes: dict[str, str], rows, rejected,
              geometry: GeometrySettings) -> tuple[Review, Model]:
        session = self.store.get(sid)
        if session.observed is None:
            msg = card("Nothing to build", "Analyze your images first.", "check")
            return Review(False, "capture", msg), Model(False, msg)
        errors, edits = [], dict(session.edits)
        for axis in AXES:
            text = str(sizes.get(axis, "")).strip()
            path = f"envelope.{axis}_mm"
            if not text:
                edits.pop(path, None)
                continue
            value = parse_size(text)
            if value is None:
                errors.append(f"{AXIS_LABEL[axis]}: '{text}' is not a size in mm")
            elif session.shown.get(path) is None or abs(value - float(session.shown[path])) > 1e-9:
                edits[path] = value
        for path, row in zip(session.row_paths, rows or []):
            text = str(row[1]).strip() if len(row) > 1 else ""
            if not text:
                edits.pop(path, None)
                continue
            value = parse_size(text)
            shown = session.shown.get(path)
            if value is None:
                if text not in ("None", str(shown)):
                    errors.append(f"{row[0]}: '{row[1]}' is not a positive number")
            elif shown is None or abs(value - float(shown)) > 1e-9:
                edits[path] = value
        if errors:
            msg = card("Please fix these values", " · ".join(errors), "stop")
            review = self._review(session, self._fuse(session))
            review.ok, review.message_html = False, msg
            return review, Model(False, msg)
        session.edits, session.rejected, session.geometry = edits, tuple(rejected or ()), geometry
        review = self._review(session, self._fuse(session))
        if not review.ok:
            return review, Model(False, review.message_html)
        return review, self._model(session)

    def rebuild_geometry(self, sid: str, geometry: GeometrySettings) -> tuple[Review, Model]:
        """A finish/clearance/snap change re-fuses (so the Review table shows the new snapped values, not stale
        ones) and rebuilds; the caller maps the returned review onto the sizes and the values table."""
        session = self.store.get(sid)
        if session.observed is None:
            msg = card("Nothing to build", "Analyze your images first.", "check")
            return Review(False, "capture", msg), Model(False, msg)
        session.geometry = geometry
        review = self._review(session, self._fuse(session))
        if not review.ok:
            return review, Model(False, review.message_html)
        return review, self._model(session)

    def _model(self, session) -> Model:
        sweep(self.root)
        part = build_part(session.spec, session.geometry, self.root)
        if isinstance(part, S.MvAbstain):
            session.part = None
            # A finish that cannot be built is fixed with the Geometry controls, which live in step 3 (index 2)
            finish_failed = part.stage == "build" and session.geometry.finish != "none"
            return Model(False, _abstain_card(part), open_step=2 if finish_failed else None)
        session.part, session.exported = part, None
        masks = session.observed.masks
        views = []
        for face, mask in part.views.items():
            score = f" · match {iou(mask, masks[face]):.2f}" if face in masks else ""
            views.append((255 - mask, f"{face}{score}"))
        x, y, z = part.bbox_mm
        stats = stats_html([("Size", f"{x:.1f} × {y:.1f} × {z:.1f} mm"),
                            ("Volume", f"{part.volume_mm3 / 1000:.2f} cm³"),
                            ("Solid mass, PLA", f"{part.volume_mm3 * DENSITIES['PLA'] / 1000:.1f} g")])
        note = " ".join(part.warnings)
        return Model(True, card("Part built", note or "Rotate the part, then choose formats and export.", "ok"),
                     str(part.preview), views, stats)

    # ---- export ---------------------------------------------------------------------------------------------
    def export(self, sid: str, export: ExportSettings, mesh: MeshSettings, printing: PrintSettings) -> Exported:
        session = self.store.get(sid)
        if session.part is None:
            return Exported(card("Nothing to export", "Build the part first.", "check"))
        sweep(self.root)
        res = export_part(session.part, export.formats, mesh, printing, self.pipe.slicer, self.pipe.profile)
        session.exported = res
        settings = {"mesh": mesh.model_dump(), "printing": printing.model_dump(), "ai": session.ai.model_dump(),
                    "geometry": session.geometry.model_dump(), "formats": export.formats}
        zip_path = bundle(session.part, res, settings)
        x, y, z = session.part.bbox_mm
        items = [("Size", f"{x:.1f} × {y:.1f} × {z:.1f} mm"), ("Files", str(len(res.files))),
                 ("Download", f"{zip_path.stat().st_size / 1024:.0f} KB")]
        if res.print_time_s:
            grams = res.filament_g or 0
            items += [("Print time", duration_text(res.print_time_s)),
                      ("Filament", f"{grams:.1f} g · {filament_metres(grams, printing.material):.2f} m")]
        tone = "check" if res.warnings else "ok"
        body = " · ".join(res.warnings) or "Every file is in the zip, with a manifest of the values and their sources."
        return Exported(card("Files ready", body, tone), [str(p) for p in res.files.values()], str(zip_path),
                        stats_html(items))
