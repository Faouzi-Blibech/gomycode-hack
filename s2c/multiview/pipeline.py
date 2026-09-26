"""The multi-view path end to end. Model calls happen in observe(); fuse() calls TripoSR at most once per
request and caches the mesh; build() never calls a model. Spec sections 6 and 7."""
from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from pydantic import ValidationError

from s2c.multiview import spec as S
from s2c.multiview.build import BuildError, export
from s2c.multiview.build import build as build_solid
from s2c.multiview.complete import MeshProvider, complete
from s2c.multiview.depth import DepthProvider, apply_depth, solaria_depth
from s2c.multiview.fuse import Observation, assemble, attach_label, canonical_outlines, features_from, fuse_envelope
from s2c.multiview.label import Chat, MvLabel, env_chat, hint_label, label_image
from s2c.multiview.merge_views import merge_same_face
from s2c.multiview.ocr import BatchReader, Reader, link, read_values
from s2c.multiview.outline import PixelOutline, extract, resize_long_side
from s2c.multiview.qwen_faces import RESCUE_PENALTY, SEED, TRIES, rescue_sketch
from s2c.multiview.qwen_image import MAX_REFS, ImageGen, default_gen
from s2c.multiview.qwen_reader import qwen_batch_reader
from s2c.multiview.raster import Mesh, face_mask, iou, normalize_mask, polygon_mask, solid_mesh
from s2c.multiview.reference import find_reference
from s2c.multiview.settings import AiSettings, GeometrySettings
from s2c.multiview.slice import slice_solid

log = logging.getLogger(__name__)
IOU_GREEN = 0.85


@dataclass
class ImageInput:
    data: bytes
    face: str | None = None
    kind: str | None = None


@dataclass
class Observed:
    """Everything that needed a model call, cached per request so merge never repeats it."""
    observations: list[Observation]
    images: list[np.ndarray]
    masks: dict[str, np.ndarray]
    labels: list[MvLabel]
    warnings: list[str] = field(default_factory=list)
    mesh: Mesh | None = None
    qwen_cache: dict = field(default_factory=dict)            # (face, seed) -> drawn image, or None after a failure
    filled_by: dict[str, str] = field(default_factory=dict)   # canonical face -> who filled it


@dataclass
class BuildResult:
    step: Path
    stl: Path
    print_stl: Path | None
    gcode: Path | None
    print_time_s: float | None
    filament_g: float | None
    views: dict[str, np.ndarray]
    iou: dict[str, float]
    warnings: list[str]


def input_mask(outline: PixelOutline) -> np.ndarray:
    """The input silhouette of one image: outer outline filled, openings and circles cut out, normalised."""
    holes = list(outline.inner)
    holes += [cv2.ellipse2Poly((round(c.cx), round(c.cy)), (round(c.d / 2), round(c.d / 2)), 0, 0, 360, 5)
              for c in outline.circles]
    return normalize_mask(polygon_mask(outline.outer, holes, outline.shape))


class MvPipeline:
    def __init__(self, chat: Chat | None = None, reader: Reader | None = None,
                 mesh_provider: MeshProvider | None = None, slicer: Path | None = None, profile: Path | None = None,
                 batch_reader: BatchReader | None = None, image_gen: ImageGen | None = None,
                 depth: DepthProvider | None = None):
        self.chat, self.reader, self.mesh_provider = chat, reader, mesh_provider
        self.slicer, self.profile = slicer, profile
        self.batch_reader = batch_reader
        self.image_gen = image_gen
        self.depth = depth
        self.seed, self.attempts = SEED, TRIES
        self.draw_faces = self.rescue_enabled = True

    def configured(self, ai: AiSettings) -> MvPipeline:
        """A copy for one request with the user's AI switches, seed and attempts; the shared pipeline never changes."""
        pipe = copy.copy(self)
        if not ai.use_reader:
            pipe.batch_reader = None
        if not ai.use_triposr:
            pipe.mesh_provider = None
        if not ai.use_solaria:
            pipe.depth = None
        pipe.draw_faces, pipe.rescue_enabled = ai.use_qwen_image, ai.use_rescue
        pipe.seed, pipe.attempts = ai.seed, ai.attempts
        return pipe

    def _label(self, item: ImageInput) -> MvLabel | S.MvAbstain:
        if self.chat is not None:
            return label_image(item.data, self.chat, item.face, item.kind)
        if item.face:
            return hint_label(item.face, item.kind or "sketch")
        return S.MvAbstain(stage="label", reason="face_unknown", remedy="Tell us which face this photo shows.")

    def observe(self, images: list[ImageInput], reference: str | None = None) -> Observed | S.MvAbstain:
        observed = Observed([], [], {}, [])
        reads = self.reader is not None or self.batch_reader is not None
        if not reads:
            observed.warnings.append("OCR unavailable: enter the dimensions by hand")
        excluded: list[tuple] = []
        for item in images:
            bgr = cv2.imdecode(np.frombuffer(item.data, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                return S.MvAbstain(stage="outline", reason="bad_image",
                                   remedy="The file is not an image. Upload a JPEG or PNG.")
            bgr = resize_long_side(bgr)
            label = self._label(item)
            if isinstance(label, S.MvAbstain):
                return label
            mm_per_px, mask_out = None, ()
            if reference and label.input_kind == "photo":
                ref = find_reference(bgr, reference)
                if isinstance(ref, S.MvAbstain):
                    return ref
                bgr, mm_per_px = ref.image, ref.mm_per_px
                mask_out = (ref.bbox,) if ref.bbox else ()
            outline, rescued = self._outline(bgr, mask_out, label.input_kind)
            if isinstance(outline, S.MvAbstain):
                return outline
            values = []
            if reads and label.input_kind != "photo":
                values = link(read_values(bgr, outline, self.reader, self.batch_reader), outline)
            obs = Observation(face=label.face, kind=label.input_kind, outline=outline, values=values,
                              mm_per_px=mm_per_px, confidence=label.confidence * (RESCUE_PENALTY if rescued else 1.0))
            attach_label(obs, label)
            if rescued:
                observed.warnings.append(f"{label.face}: sketch cleaned by Qwen-Image, check it")
            observed.observations.append(obs)
            observed.images.append(bgr)
            observed.labels.append(label)
            excluded.append(mask_out)
        if self.depth is not None:
            observed.warnings += self._depths(observed, excluded)
        return self._merge(observed)

    def _outline(self, bgr: np.ndarray, mask_out, kind: str) -> tuple[PixelOutline | S.MvAbstain, bool]:
        """The outline, and whether Qwen-Image had to redraw the sketch (spec 2026-09-23 section 8)."""
        outline = extract(bgr, mask_out)
        if (isinstance(outline, S.MvAbstain) and outline.reason == "no_outline" and kind != "photo"
                and self.rescue_enabled and self.image_gen is not None):
            fixed = rescue_sketch(bgr, self.image_gen, self.seed)
            if fixed is not None:
                return fixed, True
        return outline, False

    def _depths(self, observed: Observed, excluded: list[tuple]) -> list[str]:
        """Solaria once per face, on its most confident photo that shows a hole (spec 2026-09-23 section 9)."""
        best: dict[str, int] = {}
        for k, o in enumerate(observed.observations):
            if o.kind == "photo" and o.outline.circles and (
                    o.face not in best or o.confidence > observed.observations[best[o.face]].confidence):
                best[o.face] = k
        warnings = []
        for face, k in best.items():
            try:
                depth = self.depth(observed.images[k])
                warnings += apply_depth(observed.observations[k], depth, excluded[k])
            except Exception as e:
                log.warning("Solaria failed on %s: %s", face, e)
                warnings.append(f"{face}: depth unavailable")
        return warnings

    @staticmethod
    def _merge(observed: Observed) -> Observed:
        """One observation per face: several photos of a face are merged (spec 2026-09-23 section 5)."""
        merged, images, warnings = merge_same_face(observed.observations, observed.images)
        observed.observations, observed.images = merged, images
        observed.warnings += warnings
        observed.masks = {o.face: input_mask(o.outline) for o in merged}
        return observed

    def fuse(self, observed: Observed, user_values: dict | None = None, accepted=(), rejected=(),
             geometry: GeometrySettings | None = None) -> S.MultiViewSpec | S.MvAbstain:
        geometry = geometry or GeometrySettings()
        env_result = fuse_envelope(observed.observations, user_values)
        if isinstance(env_result, S.MvAbstain):
            return env_result
        env, env_prov, warnings = env_result
        outlines, more = canonical_outlines(observed.observations, env)
        warnings = observed.warnings + warnings + more
        best = max(range(len(observed.observations)), key=lambda i: observed.observations[i].confidence)
        target = observed.observations[best]
        image = observed.images[best] if best < len(observed.images) else None  # None once routes dropped them
        pairs = sorted(zip(observed.observations, observed.images), key=lambda p: -p[0].confidence)
        refs = [(o.face, img) for o, img in pairs][:MAX_REFS]
        full, more, observed.mesh = complete({f: ol for f, (ol, _) in outlines.items()}, env, target.face,
                                             observed.masks[target.face], image, self.mesh_provider,
                                             observed.mesh, tuple(rejected),
                                             gen=self.image_gen if self.draw_faces else None, refs=refs,
                                             qwen_cache=observed.qwen_cache, filled_by=observed.filled_by,
                                             seed=self.seed, attempts=self.attempts)
        warnings += more
        with_prov = {f: (ol, outlines[f][1] if f in outlines else ("inferred" if ol.source == "inferred" else "default"))
                     for f, ol in full.items()}
        feats, feat_prov = features_from(observed.observations, env)
        try:
            return assemble(env, env_prov, with_prov, feats, feat_prov, warnings, user_values, accepted,
                            snap_values=geometry.snap, clearance=geometry.clearance)
        except ValidationError as e:
            log.warning("spec rejected: %s", e)
            return S.MvAbstain(stage="dimensions", reason="invalid_value",
                               remedy="A value is out of range. Check the numbers you entered.")

    def build(self, spec: S.MultiViewSpec, out_dir: Path, masks: dict | None = None) -> BuildResult | S.MvAbstain:
        try:
            solid = build_solid(spec)
        except BuildError as e:
            return S.MvAbstain(stage="build", reason=e.reason, remedy=e.remedy)
        step, stl = export(solid, out_dir)
        sliced = slice_solid(solid, out_dir, self.profile, self.slicer)
        if isinstance(sliced, S.MvAbstain):
            return sliced
        mesh = solid_mesh(solid)
        views = {f: normalize_mask(face_mask(mesh, f, spec.envelope)[0]) for f in S.FACES}
        scores = {f: round(iou(views[f], m), 3) for f, m in (masks or {}).items()}
        warnings = list(spec.warnings) + sliced.warnings
        warnings += [f"Low confidence on {f}, check the dimensions." for f, s in scores.items() if s < IOU_GREEN]
        return BuildResult(step, stl, sliced.print_stl, sliced.gcode, sliced.print_time_s, sliced.filament_g,
                           views, scores, warnings)


def default_pipeline() -> MvPipeline:
    """Qwen-VL, Qwen-Image and Solaria from the environment; TrOCR and TripoSR when the ai extra is installed."""
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
    space = os.environ.get("SOLARIA_SPACE")
    image_gen = default_gen()
    if image_gen is None:
        log.warning("Qwen-Image unavailable: set QWEN_IMAGE_SPACE, or QWEN_IMAGE_BACKEND=dashscope with its "
                    "settings; missing faces fall back to TripoSR or an assumed rectangle")
    if not space:
        log.warning("Solaria unavailable: set SOLARIA_SPACE; hole depth stays with the vision model's labels")
    return MvPipeline(chat=env_chat(), reader=reader, mesh_provider=provider,
                      batch_reader=qwen_batch_reader(read_chat) if read_chat else None, image_gen=image_gen,
                      depth=solaria_depth(space, os.environ.get("HF_TOKEN")) if space else None)
