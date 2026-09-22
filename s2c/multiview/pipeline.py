"""The multi-view path end to end. Model calls happen in observe(); fuse() calls TripoSR at most once per
request and caches the mesh; build() never calls a model. Spec sections 6 and 7."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from pydantic import ValidationError

from s2c.multiview import spec as S
from s2c.multiview.build import BuildError, export
from s2c.multiview.build import build as build_solid
from s2c.multiview.complete import MeshProvider, complete
from s2c.multiview.fuse import Observation, assemble, attach_label, canonical_outlines, features_from, fuse_envelope
from s2c.multiview.label import Chat, MvLabel, env_chat, hint_label, label_image
from s2c.multiview.ocr import Reader, link, read_values
from s2c.multiview.outline import PixelOutline, extract, resize_long_side
from s2c.multiview.raster import Mesh, face_mask, iou, normalize_mask, polygon_mask, solid_mesh
from s2c.multiview.reference import find_reference
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
                 mesh_provider: MeshProvider | None = None, slicer: Path | None = None, profile: Path | None = None):
        self.chat, self.reader, self.mesh_provider = chat, reader, mesh_provider
        self.slicer, self.profile = slicer, profile

    def _label(self, item: ImageInput) -> MvLabel | S.MvAbstain:
        if self.chat is not None:
            return label_image(item.data, self.chat, item.face, item.kind)
        if item.face:
            return hint_label(item.face, item.kind or "sketch")
        return S.MvAbstain(stage="label", reason="face_unknown", remedy="Tell us which face this photo shows.")

    def observe(self, images: list[ImageInput], reference: str | None = None) -> Observed | S.MvAbstain:
        observed = Observed([], [], {}, [])
        if self.reader is None:
            observed.warnings.append("OCR unavailable: enter the dimensions by hand")
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
            outline = extract(bgr, mask_out)
            if isinstance(outline, S.MvAbstain):
                return outline
            values = []
            if self.reader is not None and label.input_kind != "photo":
                values = link(read_values(bgr, outline, self.reader), outline)
            obs = Observation(face=label.face, kind=label.input_kind, outline=outline, values=values,
                              mm_per_px=mm_per_px, confidence=label.confidence)
            attach_label(obs, label)
            observed.observations.append(obs)
            observed.images.append(bgr)
            observed.labels.append(label)
            observed.masks[label.face] = input_mask(outline)
        return observed

    def fuse(self, observed: Observed, user_values: dict | None = None, accepted=(),
             rejected=()) -> S.MultiViewSpec | S.MvAbstain:
        env_result = fuse_envelope(observed.observations, user_values)
        if isinstance(env_result, S.MvAbstain):
            return env_result
        env, env_prov, warnings = env_result
        outlines, more = canonical_outlines(observed.observations, env)
        warnings = observed.warnings + warnings + more
        best = max(range(len(observed.observations)), key=lambda i: observed.observations[i].confidence)
        target = observed.observations[best]
        full, more, observed.mesh = complete({f: ol for f, (ol, _) in outlines.items()}, env, target.face,
                                             observed.masks[target.face], observed.images[best],
                                             self.mesh_provider, observed.mesh, tuple(rejected))
        warnings += more
        with_prov = {f: (ol, outlines[f][1] if f in outlines else ("inferred" if ol.source == "inferred" else "default"))
                     for f, ol in full.items()}
        feats, feat_prov = features_from(observed.observations, env)
        try:
            return assemble(env, env_prov, with_prov, feats, feat_prov, warnings, user_values, accepted)
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
    """Vision model from the environment, TrOCR and TripoSR when the ai extra is installed."""
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
    return MvPipeline(chat=env_chat(), reader=reader, mesh_provider=provider)
