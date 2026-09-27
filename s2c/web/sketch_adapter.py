"""One sheet, several views -> the per-face `Observation` list `MvPipeline.fuse()` consumes.

Sheet mode reads one photo of a hand-drawn sheet with `s2c.sketch.read_sketch`, which returns a
`SketchReading` (frozen contract, `s2c/sketch/models.py`). This module turns that reading into the
same shapes the per-face photo mode builds in `MvPipeline.observe()`, so the analyze job can hand
the result straight to `MvPipeline.fuse()` and everything after (Review, Model, Export) is unchanged.

Trust rule (spec section 4/5, mirrored here): a number only reaches `fuse()` as a trusted, "green"
value (`confirmed=True` on a `Reading`, or a `Feature` written straight to a hole) when the sketch
reading itself marked it confirmed -- a `Dimension`/`Feature` badge of "written" or "edited", meaning
the readers agreed or the user fixed it by hand. Anything else (`derived`, `uncertain`, `predicted`,
`conflict`) is never silently trusted: it is dropped from the written path and left to the sheet's own
page-scale calibration (a "measured" provenance, the same tier a coin gives a photo) or to the
existing suggested/typed-value path in Review.

Assumption (undocumented in `SketchReading`, no `read_sketch` in this tree yet to check against):
`View.size_mm` is the physical size of that view's own region on the page (from the sheet's own scale
calibration), in the same (a, b) axis order as `s2c.multiview.spec.FACE_AXES` for that face -- i.e. it
plays the role a coin or card plays for a photo, not a user-written dimension. `Feature.position_mm`
is in the same global x/y/z frame `s2c.multiview` builds parts in. Raw `Entity` line/arc geometry
(`px`/`mm` dicts) is intentionally not used here: their key shape is still being decided by the
sketch pipeline task (Task 12) running in parallel, so the outer profile of each face falls back to
its view's bounding box (a rectangle) and only `Feature` holes/slots -- already solved to millimetres
by the sketch package -- add detail. This is a deliberately conservative v1; a later pass can vectorize
the true outline once `s2c/sketch/pipeline.py` and its `Entity` shape are settled.
"""
from __future__ import annotations

import logging

import numpy as np

from s2c.multiview import spec as S
from s2c.multiview.fuse import Observation
from s2c.multiview.ocr import Linked, Reading
from s2c.multiview.outline import PixelCircle, PixelOutline
from s2c.multiview.pipeline import Observed, input_mask
from s2c.sketch.models import Dimension, Feature, SketchReading, View

log = logging.getLogger(__name__)

CONFIRMED_BADGES = frozenset({"written", "edited"})  # readers agreed, or the user fixed it by hand
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
# The face that looks along each global axis, restricted to the three canonical faces a sheet draws.
_LOOK_TO_FACE = {info[2]: face for face, info in S.FACE_AXES.items() if face in S.CANONICAL_FACES}


def _confidence(readings: list) -> float:
    return max((r.confidence for r in readings), default=0.6)


def _rect_outline(view: View, circles: list[PixelCircle]) -> PixelOutline:
    x, y, w, h = (float(v) for v in view.bbox_px)
    outer = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float64)
    return PixelOutline(outer=outer, inner=[], circles=circles, bbox=(x, y, w, h),
                        shape=(round(y + h), round(x + w)))


def _view_values(view: View, dims: list[Dimension]) -> list[Linked]:
    """Linear a/b dimensions drawn on this view -> the envelope candidates `fuse_envelope` reads."""
    out = []
    for d in dims:
        if d.view != view.name or d.kind != "linear" or d.axis is None or d.value is None:
            continue
        confirmed = d.badge in CONFIRMED_BADGES
        reading = Reading(value_mm=d.value, kind="linear", bbox=d.bbox_px, confidence=_confidence(d.readings),
                          text=d.text_raw, confirmed=confirmed)
        out.append(Linked(reading=reading, axis=d.axis, hole_index=None))
    return out


def _feature_circle(feature: Feature, view: View) -> tuple[PixelCircle, Linked | None] | None:
    """One hole's pixel circle on its view, plus a written value when the sketch confirmed it."""
    if feature.diameter is None:
        return None
    face = _LOOK_TO_FACE.get(feature.axis)
    if face is None or face != view.name:
        return None
    a_axis, b_axis, _ = S.FACE_AXES[face]
    a_mm = feature.position_mm[_AXIS_INDEX[a_axis]]
    b_mm = feature.position_mm[_AXIS_INDEX[b_axis]]
    sa_mm, sb_mm = view.size_mm
    x, y, w, h = view.bbox_px
    if sa_mm <= 0 or sb_mm <= 0:
        return None
    cx = x + (a_mm / sa_mm) * w
    cy = y + (1 - b_mm / sb_mm) * h
    d_px = (feature.diameter / sa_mm) * w
    circle = PixelCircle(cx=cx, cy=cy, d=d_px)
    if feature.badge not in CONFIRMED_BADGES:
        return circle, None
    reading = Reading(value_mm=feature.diameter, kind="diameter", bbox=view.bbox_px, confidence=1.0,
                      text=f"⌀{feature.diameter:g}", confirmed=True)
    return circle, Linked(reading=reading, axis=None, hole_index=None)  # hole_index filled once the index is known


def observations_from_sketch(reading: SketchReading) -> list[Observation]:
    """One `Observation` per drawn view, ready for `MvPipeline.fuse()`. Never raises on a reading with
    no views or no confirmed numbers -- an empty or low-confidence result still reaches Review, where
    the existing missing/suggested-value flow takes over exactly as it does for a photo with no OCR."""
    observations = []
    for view in reading.views:
        circles: list[PixelCircle] = []
        written: list[Linked] = []
        for feature in reading.features:
            got = _feature_circle(feature, view)
            if got is None:
                continue
            circle, link = got
            if link is not None:
                link = Linked(reading=link.reading, axis=None, hole_index=len(circles))
                written.append(link)
            circles.append(circle)
        outline = _rect_outline(view, circles)
        values = _view_values(view, reading.dimensions) + written
        sa_mm, _sb_mm = view.size_mm
        _, _, w, _h = view.bbox_px
        mm_per_px = sa_mm / w if w else None
        confidence = _confidence([r for d in reading.dimensions if d.view == view.name for r in d.readings])
        observations.append(Observation(face=view.name, kind="sketch", outline=outline, values=values,
                                        mm_per_px=mm_per_px, confidence=confidence))
    return observations


def observed_from_sketch(reading: SketchReading) -> Observed:
    """The full `Observed` `MvPipeline.fuse()` needs: the per-view observations plus their input masks
    (spec section 5's `input_mask`, the same one `MvPipeline.observe()` builds after merging photos)."""
    observations = observations_from_sketch(reading)
    masks = {o.face: input_mask(o.outline) for o in observations}
    return Observed(observations=observations, images=[None] * len(observations), masks=masks, labels=[])
