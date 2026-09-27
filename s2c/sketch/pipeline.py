"""read_sketch: one photo of a hand-drawn sheet -> SketchReading. Stages 1 to 9 of the spec."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from s2c.reading import Reader, readers_from_env
from s2c.sketch.capture import Captured, capture, sheet_to_original
from s2c.sketch.classify import Classified, classify
from s2c.sketch.features import find_holes
from s2c.sketch.link import Link, Ref, link_view
from s2c.sketch.models import Dimension, Entity, Issue, SketchAbstain, SketchReading, View, ViewName
from s2c.sketch.solve import Solved, px_to_mm, solve
from s2c.sketch.text import TextItem, detect_text_boxes, read_texts
from s2c.sketch.vectorize import Prim, vectorize
from s2c.sketch.views import ViewRegion, split_views


@dataclass
class Trace:
    captured: Captured | None = None
    texts: list[TextItem] | None = None
    views: list[ViewRegion] | None = None
    classified: dict[ViewName, Classified] | None = None
    refs: dict[ViewName, dict[str, list[Ref]]] | None = None
    links: list[Link] | None = None
    solved: Solved | None = None


def read_sketch(image_bytes: bytes, readers: list[Reader] | None = None) -> SketchReading:
    return analyse(image_bytes, readers)[0]


def _box_to_photo(box, H) -> tuple[float, float, float, float]:
    x, y, w, h = box
    pts = sheet_to_original(np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]), H)
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    return float(x0), float(y0), float(x1 - x0), float(y1 - y0)


def _pt_to_photo(p, H) -> list[float]:
    return [float(v) for v in sheet_to_original(np.float32([p]), H)[0]]


def _empty(size, abstain: SketchAbstain, timings, dims=None) -> SketchReading:
    return SketchReading(image_size_px=size, views=[], entities=[], dimensions=dims or [], features=[],
                         envelope={}, issues=[], abstain=abstain, timings_ms=timings)


def _unplaced(texts: list[TextItem], H) -> list[Dimension]:
    return [Dimension(id=t.id, view=None, kind=t.parsed.kind, value=t.parsed.value,
                      text_raw=t.readings[0].text if t.readings else "", readings=t.readings,
                      candidates=t.candidates, measures=[], badge=t.badge or "uncertain",
                      evidence="reader", bbox_px=_box_to_photo(t.box, H))
            for t in texts if t.role == "dimension"]


def _unit_issue(texts: list[TextItem]) -> list[Issue]:
    """Spec 3: values that look like inches (all under 10, mostly written with a leading dot)."""
    dims = [t for t in texts if t.role == "dimension" and t.parsed and t.parsed.kind == "linear"]
    if len(dims) < 3 or not all(t.parsed.value < 10 for t in dims):
        return []
    dotted = sum(any(r.text.strip().startswith(".") for r in t.readings) for t in dims)
    if dotted < len(dims) / 2:
        return []
    return [Issue(severity="red", kind="unit", targets=[t.id for t in dims],
                  message="The values look like inches. Are they mm or inches?")]


def _entity(p: Prim, view: str, line_type: str, refs, solved: Solved, H) -> Entity:
    def mm(pt):
        return [px_to_mm(solved, refs[view]["a"], view, "a", pt[0]),
                px_to_mm(solved, refs[view]["b"], view, "b", -pt[1])]

    s = solved.scale.get((view, "a"), 0.25)
    if p.kind == "line":
        mmd = {"p0": mm(p.p0), "p1": mm(p.p1)}
        pxd = {"p0": _pt_to_photo(p.p0, H), "p1": _pt_to_photo(p.p1, H)}
    elif p.kind in ("circle", "arc"):
        radius = solved.circle_d.get(p.id, (2 * p.radius * s, "predicted"))[0] / 2
        mmd = {"centre": mm(p.center), "radius": radius}
        edge = _pt_to_photo((p.center[0] + p.radius, p.center[1]), H)
        centre_px = _pt_to_photo(p.center, H)
        pxd = {"centre": centre_px, "radius": float(np.hypot(edge[0] - centre_px[0], edge[1] - centre_px[1]))}
        if p.kind == "arc":
            ang = np.degrees(np.unwrap(np.arctan2(-(p.pts[:, 1] - p.center[1]), p.pts[:, 0] - p.center[0])))
            mmd["start_deg"], mmd["end_deg"] = float(ang.min()), float(ang.max())
            pxd["start_deg"], pxd["end_deg"] = mmd["start_deg"], mmd["end_deg"]
    else:
        mmd = {"points": [mm(q) for q in p.pts]}
        pxd = {"points": [_pt_to_photo(q, H) for q in p.pts]}
    return Entity(id=p.id, view=view, type=p.kind, line_type=line_type, mm=mmd, px=pxd,
                  confidence=0.9 if line_type == "visible" else 0.8)


def analyse(image_bytes: bytes, readers: list[Reader] | None = None) -> tuple[SketchReading, Trace]:
    t0 = time.perf_counter()
    timings: dict[str, float] = {}
    trace = Trace()

    def lap(name):
        timings[name] = round((time.perf_counter() - t0) * 1000 - sum(timings.values()), 1)

    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return _empty((0, 0), SketchAbstain(stage="capture", reason="unreadable_image",
                                            remedy="Send a JPEG or PNG photo of the sheet."), timings), trace
    size = (int(img.shape[1]), int(img.shape[0]))
    cap = capture(img)
    lap("capture")
    if isinstance(cap, SketchAbstain):
        return _empty(size, cap, timings), trace
    trace.captured = cap
    H = cap.to_original

    readers = readers_from_env() if readers is None else readers
    texts = read_texts(cap.sheet, detect_text_boxes(cap.sheet, cap.ink, cap.stroke_px), readers)
    lap("text")
    if isinstance(texts, SketchAbstain):
        return _empty(size, texts, timings), trace
    trace.texts = texts

    split = split_views(cap.ink, texts, cap.stroke_px)
    lap("views")
    if isinstance(split, SketchAbstain):
        return _empty(size, split, timings, _unplaced(texts, H)), trace
    views, issues = split
    issues += _unit_issue(texts)
    trace.views = views

    classified, refs, links = {}, {}, []
    for v in views:
        c = classify(vectorize(v.ink, cap.stroke_px, v.name), v.ink, cap.stroke_px, v.texts)
        r, lk, iss = link_view(v.name, c, v.texts, cap.stroke_px)
        classified[v.name], refs[v.name] = c, r
        links += lk
        issues += iss
    trace.classified, trace.refs, trace.links = classified, refs, links
    lap("geometry")

    circles = {p.id: p for c in classified.values() for p in c.visible + c.hidden if p.kind == "circle"}
    solved = solve(refs, links, circles)
    trace.solved = solved
    issues += solved.issues
    features, fi = find_holes(classified, refs, solved)
    issues += fi
    lap("solve")

    out_views = []
    for v in views:
        ext = [solved.mm[refs[v.name][ax][-1].id] if refs[v.name][ax] else 0.0 for ax in ("a", "b")]
        out_views.append(View(name=v.name, label_text=v.label_text, bbox_px=_box_to_photo(v.bbox, H),
                              size_mm=(ext[0], ext[1])))
    entities = [_entity(p, v, lt, refs, solved, H) for v, c in classified.items()
                for lt, group in (("visible", c.visible), ("hidden", c.hidden), ("centre", c.centre))
                for p in group]
    dimensions = []
    for L in links:
        d = solved.dims.get(L.text.id)
        dimensions.append(Dimension(
            id=L.text.id, view=None if L.how == "unplaced" else L.view, kind=L.kind,
            value=d.value if d else L.text.parsed.value,
            text_raw=L.text.readings[0].text if L.text.readings else "", readings=L.text.readings,
            candidates=d.candidates if d else L.text.candidates, implied=d.implied if d else None,
            tolerance=L.text.parsed.tolerance, measures=L.measures, axis=L.axis,
            badge=d.badge if d else (L.text.badge or "uncertain"), evidence=d.evidence if d else "reader",
            bbox_px=_box_to_photo(L.text.box, H)))
    ref_px = {r.id: r for axes in refs.values() for rs in axes.values() for r in rs}
    for k, g in enumerate(solved.gaps):
        a, b = ref_px[g.refs[0]], ref_px[g.refs[1]]
        vb = next(v.bbox for v in views if v.name == g.view)
        if g.axis == "a":
            box = (a.px, vb[1] + vb[3], b.px - a.px, 1)
        else:
            box = (vb[0], -b.px, 1, b.px - a.px)
        dimensions.append(Dimension(id=f"pred-{k}", view=g.view, kind="linear", value=g.value, text_raw="",
                                    readings=[], measures=list(g.refs), axis=g.axis, badge="predicted",
                                    evidence="proportion", bbox_px=_box_to_photo(box, H)))
    lap("assemble")
    timings["total"] = round((time.perf_counter() - t0) * 1000, 1)
    reading = SketchReading(image_size_px=size, views=out_views, entities=entities, dimensions=dimensions,
                            features=features, envelope=solved.envelope, issues=issues, timings_ms=timings)
    debug_dir = os.environ.get("SKETCH_DEBUG_DIR")
    if debug_dir:
        from s2c.sketch.debug import write_overlays
        write_overlays(trace, Path(debug_dir))
    return reading, trace
