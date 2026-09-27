"""Stage 3: split the sheet into views and name them from their labels or the third-angle layout."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from scipy import ndimage

from s2c.sketch.models import Issue, SketchAbstain, ViewName
from s2c.sketch.text import TextItem, erase_mask

REACH = 0.10          # of the sheet's long side: how far a view's annotations may sit
MIN_HOLE = 0.03       # side of the smallest enclosed area of an outline, of the long side
THIRD_ANGLE = {"above": "top", "below": "bottom", "right": "right", "left": "left"}


@dataclass
class ViewRegion:
    name: ViewName
    label_text: str | None
    bbox: tuple[int, int, int, int]
    ink: np.ndarray
    texts: list[TextItem]
    named_by: Literal["label", "layout"]


def _seeds(geo: np.ndarray) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    """Closed outlines: components that enclose an area. Overlapping ones are merged."""
    long_side = max(geo.shape)
    closed = cv2.morphologyEx(geo, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    min_hole = (MIN_HOLE * long_side) ** 2
    boxes = []
    for i, c in enumerate(contours):
        child = hierarchy[0][i][2]
        if hierarchy[0][i][3] != -1 or child == -1:
            continue
        holes = []
        while child != -1:
            holes.append(cv2.contourArea(contours[child]))
            child = hierarchy[0][child][0]
        if max(holes) >= min_hole:
            boxes.append(list(cv2.boundingRect(c)))
    merged = True
    while merged:  # merge overlapping boxes: inner outlines belong to the outer one
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]:
                    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
                    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
                    boxes[i] = [x0, y0, x1 - x0, y1 - y0]
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    out = []
    for x, y, w, h in boxes:
        mask = np.zeros(geo.shape, bool)
        mask[y:y + h, x:x + w] = geo[y:y + h, x:x + w] > 0
        out.append((mask, (x, y, w, h)))
    return out


def _relation(a, b) -> str | None:
    """Where box b sits relative to box a."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x_overlap = min(ax + aw, bx + bw) - max(ax, bx) > 0.3 * min(aw, bw)
    y_overlap = min(ay + ah, by + bh) - max(ay, by) > 0.3 * min(ah, bh)
    if x_overlap and by + bh <= ay + 0.2 * ah:
        return "above"
    if x_overlap and by >= ay + 0.8 * ah:
        return "below"
    if y_overlap and bx >= ax + 0.8 * aw:
        return "right"
    if y_overlap and bx + bw <= ax + 0.2 * aw:
        return "left"
    return None


def _layout_names(boxes, known: dict[int, ViewName]) -> dict[int, ViewName]:
    names = dict(known)
    front = next((i for i, n in names.items() if n == "front"), None)
    if front is None:
        def score(i):
            rel = [_relation(boxes[i], boxes[j]) for j in range(len(boxes)) if j != i]
            x, y, _w, h = boxes[i]
            return (sum(r is not None for r in rel), y + h, -x)
        front = max((i for i in range(len(boxes)) if i not in names), key=score, default=None)
        if front is None:
            return names
        names[front] = "front"
    taken = set(names.values())
    rights = sorted((i for i in range(len(boxes)) if i not in names
                     and _relation(boxes[front], boxes[i]) == "right"), key=lambda i: boxes[i][0])
    for k, i in enumerate(rights):
        name = "right" if k == 0 and "right" not in taken else "back"
        if name not in taken:
            names[i] = name
            taken.add(name)
    for i in range(len(boxes)):
        if i in names:
            continue
        rel = _relation(boxes[front], boxes[i])
        name = THIRD_ANGLE.get(rel) if rel else None
        if name and name not in taken:
            names[i] = name
            taken.add(name)
    return names


def _distance_to_box(point, box) -> float:
    x, y, w, h = box
    dx = max(x - point[0], 0, point[0] - (x + w))
    dy = max(y - point[1], 0, point[1] - (y + h))
    return float(np.hypot(dx, dy))


def split_views(ink, texts: list[TextItem], stroke_px: float):
    geo = cv2.bitwise_and(ink, cv2.bitwise_not(erase_mask(texts, ink.shape)))
    seeds = _seeds(geo)
    if not seeds:
        return SketchAbstain(stage="views", reason="no_views_found",
                             remedy="Draw the views with a dark pen and retake.")
    markers = np.zeros(geo.shape, np.int32)
    for k, (mask, _) in enumerate(seeds):
        markers[mask] = k + 1
    dist, (iy, ix) = ndimage.distance_transform_edt(markers == 0, return_indices=True)
    nearest = markers[iy, ix]
    reach = REACH * max(geo.shape)
    boxes = [box for _, box in seeds]

    labels = [t for t in texts if t.role == "label"]
    known: dict[int, ViewName] = {}
    label_text: dict[int, str] = {}
    for t in sorted(labels, key=lambda t: t.confidence, reverse=True):
        centre = (t.box[0] + t.box[2] / 2, t.box[1] + t.box[3] / 2)
        free = [k for k in range(len(boxes)) if k not in known]
        if not free:
            break
        k = min(free, key=lambda k: _distance_to_box(centre, boxes[k]))
        if t.label not in known.values():
            known[k] = t.label
            label_text[k] = t.readings[0].text if t.readings else t.label
    names = _layout_names(boxes, known)

    issues: list[Issue] = []
    views: list[ViewRegion] = []
    for k in range(len(boxes)):
        if k not in names:
            issues.append(Issue(severity="amber", kind="label", targets=[],
                                message="A drawing on the sheet could not be named as a view."))
            continue
        mine = (geo > 0) & (nearest == k + 1) & (dist <= reach)
        ys, xs = np.nonzero(mine)
        bbox = (int(xs.min()), int(ys.min()), int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1)
        by_label = k in known
        if not by_label:
            issues.append(Issue(severity="amber", kind="label", targets=[names[k]],
                                message=f"View named {names[k]} from its position; check it."))
        views.append(ViewRegion(names[k], label_text.get(k), bbox,
                                mine.astype(np.uint8) * 255, [], "label" if by_label else "layout"))
    for t in texts:
        if t.role == "label" or not views:
            continue
        centre = (t.box[0] + t.box[2] / 2, t.box[1] + t.box[3] / 2)
        min(views, key=lambda v: _distance_to_box(centre, v.bbox)).texts.append(t)
    return views, issues
