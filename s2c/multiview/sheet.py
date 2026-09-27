"""One image holding several orthographic views -> drawings and views (spec 3.1)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

import cv2
import numpy as np

Box = tuple[int, int, int, int]  # x, y, w, h in image pixels

INK_DELTA = 40          # grey levels away from the median border grey
BORDER_NEAR = 0.05      # a frame "nearly touches" a side: within this share of the image size
BORDER_FILL = 0.05      # ... and inks under this share of its box
BORDER_SIDE = 0.9       # ... and inks this share of each side of its box, which a round or L-shaped view does not
BORDER_BAND = 0.03
SEPARATOR_INK = 0.9     # a row or column this full of ink is a separator band
DILATE = 0.008
SPECK = 0.0002
LABEL_HEIGHT = 0.06
LABEL_ASPECT = 2.5
LABEL_REACH = 1.5
TEXT_GAP = 1.0          # words and letters of one label line join across gaps up to this many heights
DRAWING_GAP = 3.0       # views farther apart than this many median view sizes are separate drawings
ALIGN = 0.10            # centres of two views in one row (column) within this share of their height (width)
EXTENT = 0.25           # ... and their heights (widths) within this share: views in a row share the front's height
MIN_VIEW = 0.15         # a view shorter than this share of the drawing's largest cannot make a sheet (dimension text)
LINE_ART = 0.35         # a view is line art when its ink fills under this share of its filled outline (spec 3.3)


@dataclass
class View:
    box: Box
    label_box: Box | None = None
    line_art: bool = True


@dataclass
class Drawing:
    views: list[View]


@dataclass
class Sheet:
    drawings: list[Drawing]
    shape: tuple[int, int]
    warnings: list[str] = field(default_factory=list)


def split_sheet(image_bgr: np.ndarray) -> Sheet:
    """The drawings and views of one image. An image with no drawing of 2 or more views is not a sheet."""
    ink = ink_mask(image_bgr)
    h, w = ink.shape
    _remove_border(ink)
    cuts = _cut_separators(ink)
    k = max(3, round(DILATE * max(h, w)))
    grown = cv2.dilate(ink, np.ones((k, k), np.uint8))
    for axis, a, b in cuts:
        if axis == "row":
            grown[a: b + 1] = 0
        else:
            grown[:, a: b + 1] = 0
    n, comp = cv2.connectedComponents(grown, connectivity=8)
    boxes, count = _ink_boxes(ink, comp, n)
    boxes = {i: b for i, b in boxes.items() if count[i] >= SPECK * h * w}
    if not boxes:
        return Sheet([], (h, w), ["No drawing found in the image."])

    groups, labels = _sort(boxes, h, k)
    views = []
    for ids in groups:
        box = _union(boxes[i] for i in ids)
        label = [labels[i] for i in ids if i in labels]
        art = _line_art(comp, ids, box, k, sum(int(count[i]) for i in ids))
        views.append(View(box, _union(b for bs in label for b in bs) if label else None, art))
    return Sheet(_drawings(views, cuts), (h, w), [])


def is_sheet(sheet: Sheet) -> bool:
    """True when some drawing holds two line-drawn views that line up in a row or a column the way orthographic
    views do: centres within 10 % and the shared extent within 25 %. Photos (filled blobs), a part beside its shadow
    or a coin, and dimension text beside a sketch do not (Review Focus 1)."""
    return any(_aligned_pair(d.views) for d in sheet.drawings)


def ink_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Pixels that differ from the median border grey by more than INK_DELTA are 255."""
    if image_bgr.ndim == 2:
        gray = image_bgr
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2GRAY if image_bgr.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    edge = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    diff = np.abs(gray.astype(np.int16) - int(np.median(edge)))
    return np.where(diff > INK_DELTA, 255, 0).astype(np.uint8)


def _remove_border(ink: np.ndarray) -> None:
    """Erase a sheet frame: near all four sides, thin, and inked along every side of its box."""
    h, w = ink.shape
    n, comp, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if max(x, w - x - bw) > BORDER_NEAR * w or max(y, h - y - bh) > BORDER_NEAR * h:
            continue
        if area >= BORDER_FILL * bw * bh:
            continue
        own = comp[y: y + bh, x: x + bw] == i
        band = max(3, round(BORDER_BAND * min(bw, bh)))
        sides = (own[:band].any(0), own[-band:].any(0), own[:, :band].any(1), own[:, -band:].any(1))
        if min(float(s.mean()) for s in sides) >= BORDER_SIDE:
            ink[y: y + bh, x: x + bw][own] = 0


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate([[0], flags.astype(np.int8), [0]]))
    return list(zip(np.nonzero(edges == 1)[0].tolist(), (np.nonzero(edges == -1)[0] - 1).tolist()))


def _cut_separators(ink: np.ndarray) -> list[tuple[str, int, int]]:
    """Erase separator bands and return them. A band at the edge of the ink separates nothing and is kept, so the
    top edge of a view that spans the image is never cut away."""
    cuts = []
    for axis in ("row", "col"):
        full = (ink > 0).mean(axis=1 if axis == "row" else 0) > SEPARATOR_INK
        for a, b in _runs(full):
            before, after = (ink[:a], ink[b + 1:]) if axis == "row" else (ink[:, :a], ink[:, b + 1:])
            if before.any() and after.any():
                cuts.append((axis, a, b))
                if axis == "row":
                    ink[a: b + 1] = 0
                else:
                    ink[:, a: b + 1] = 0
    return cuts


def _ink_boxes(ink: np.ndarray, comp: np.ndarray, n: int) -> tuple[dict[int, Box], np.ndarray]:
    """Box of the ink (not of the dilated blob) of each component, and its ink pixel count."""
    ys, xs = np.nonzero(ink)
    ids = comp[ys, xs]
    count = np.bincount(ids, minlength=n)
    lo_x, lo_y = np.full(n, ink.shape[1]), np.full(n, ink.shape[0])
    hi_x, hi_y = np.full(n, -1), np.full(n, -1)
    np.minimum.at(lo_x, ids, xs)
    np.minimum.at(lo_y, ids, ys)
    np.maximum.at(hi_x, ids, xs)
    np.maximum.at(hi_y, ids, ys)
    boxes = {i: (int(lo_x[i]), int(lo_y[i]), int(hi_x[i] - lo_x[i] + 1), int(hi_y[i] - lo_y[i] + 1))
             for i in range(1, n) if count[i] > 0}
    return boxes, count


def _union(boxes) -> Box:
    boxes = list(boxes)
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
    return x0, y0, x1 - x0, y1 - y0


def _inside(inner: Box, outer: Box, tol: int = 0) -> bool:
    return (inner[0] >= outer[0] - tol and inner[1] >= outer[1] - tol
            and inner[0] + inner[2] <= outer[0] + outer[2] + tol and inner[1] + inner[3] <= outer[1] + outer[3] + tol)


class _Sets:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def join(self, a, b):
        self.parent[self.find(a)] = self.find(b)

    def groups(self) -> list[list]:
        out: dict = {}
        for i in self.parent:
            out.setdefault(self.find(i), []).append(i)
        return list(out.values())


def _text_lines(small: list[int], boxes: dict[int, Box]) -> list[list[int]]:
    """Small components on one text line, close together: the letters and words of one label."""
    sets = _Sets(small)
    for i, j in combinations(small, 2):
        a, b = boxes[i], boxes[j]
        tall = max(a[3], b[3])
        if min(a[3], b[3]) < 0.5 * tall or abs((a[1] + a[3] / 2) - (b[1] + b[3] / 2)) > 0.5 * tall:
            continue
        if max(b[0] - a[0] - a[2], a[0] - b[0] - b[2]) <= TEXT_GAP * tall:
            sets.join(i, j)
    return sets.groups()


def _label_owner(box: Box, pool: dict[int, Box]) -> int | None:
    """The nearest view that `box` sits directly above or below, within LABEL_REACH of its height."""
    x, y, w, h = box
    best = None
    for vid, (vx, vy, vw, vh) in pool.items():
        if _inside(box, (vx, vy, vw, vh)):
            return None
        if x + w <= vx or vx + vw <= x:
            continue
        if y >= vy + vh:
            d = y - (vy + vh)
        elif y + h <= vy:
            d = vy - (y + h)
        else:
            continue
        if d <= LABEL_REACH * h and (best is None or d < best[0]):
            best = (d, vid)
    return None if best is None else best[1]


def _sort(boxes: dict[int, Box], image_h: int, k: int) -> tuple[list[list[int]], dict[int, list[Box]]]:
    """Components -> views (component ids, largest first) and the label boxes of each view's main component.

    Labels are short, wide text lines directly above or below a view. A text line with no view next to it stays a
    view. A second pass lets a label sit under a thin view that is itself text-shaped (the top view of a plate)."""
    small = [i for i, b in boxes.items() if b[3] < LABEL_HEIGHT * image_h]
    lines = [g for g in _text_lines(small, boxes)
             if (u := _union(boxes[i] for i in g))[3] < LABEL_HEIGHT * image_h and u[2] >= LABEL_ASPECT * u[3]]
    in_text = {i for g in lines for i in g}
    pool = {i: b for i, b in boxes.items() if i not in in_text}
    labels: dict[int, list[Box]] = {}
    pending = []
    for g in lines:
        box = _union(boxes[i] for i in g)
        owner = _label_owner(box, pool)
        if owner is None:
            pending.append((g, box))
        else:
            labels.setdefault(owner, []).append(box)
    for g, box in pending:
        pool.update({i: boxes[i] for i in g})
    for g, box in sorted(pending, key=lambda p: p[1][2]):  # the narrower of a label and a flat view is the label
        others = {i: b for i, b in pool.items() if i not in g}
        owner = _label_owner(box, others)
        if owner is not None:
            labels.setdefault(owner, []).append(box)
            for i in g:
                del pool[i]

    views: list[list[int]] = []
    for i in sorted(pool, key=lambda i: -pool[i][2] * pool[i][3]):
        home = next((v for v in views if _inside(pool[i], pool[v[0]], k // 2)), None)
        if home is None:
            views.append([i])
        else:
            home.append(i)
    owner_of = {i: v[0] for v in views for i in v}
    joined: dict[int, list[Box]] = {}
    for owner, bs in labels.items():
        if owner in owner_of:
            joined.setdefault(owner_of[owner], []).extend(bs)
    return views, joined


def _line_art(comp: np.ndarray, ids: list[int], box: Box, k: int, ink_count: int) -> bool:
    """Ink fills under LINE_ART of the view's filled outline (dilated, so small gaps in a stroke do not matter)."""
    x, y, w, h = box
    y0, x0 = max(0, y - k), max(0, x - k)
    own = np.isin(comp[y0: y + h + k, x0: x + w + k], ids).astype(np.uint8)
    contours, _ = cv2.findContours(own, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(own)
    cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
    return ink_count < LINE_ART * int(filled.sum())


def _gap(a: Box, b: Box) -> float:
    dx = max(0, b[0] - (a[0] + a[2]), a[0] - (b[0] + b[2]))
    dy = max(0, b[1] - (a[1] + a[3]), a[1] - (b[1] + b[3]))
    return math.hypot(dx, dy)


def _separated(a: Box, b: Box, cuts) -> bool:
    for axis, lo, hi in cuts:
        p, s = (1, 3) if axis == "row" else (0, 2)
        if (a[p] + a[s] <= lo and b[p] > hi) or (b[p] + b[s] <= lo and a[p] > hi):
            return True
    return False


def _drawings(views: list[View], cuts) -> list[Drawing]:
    """Views closer than DRAWING_GAP median view sizes, with no separator band between them, form one drawing."""
    limit = DRAWING_GAP * float(np.median([max(v.box[2], v.box[3]) for v in views]))
    sets = _Sets(range(len(views)))
    for i, j in combinations(range(len(views)), 2):
        a, b = views[i].box, views[j].box
        if _gap(a, b) <= limit and not _separated(a, b, cuts):
            sets.join(i, j)
    drawings = []
    for group in sets.groups():
        members = sorted((views[i] for i in group), key=lambda v: (v.box[1] + v.box[3] / 2, v.box[0]))
        drawings.append(Drawing(members))
    return sorted(drawings, key=lambda d: (min(v.box[1] for v in d.views), min(v.box[0] for v in d.views)))


def _in_line(a: Box, b: Box, row: bool) -> bool:
    p, s = (1, 3) if row else (0, 2)
    size = max(a[s], b[s])
    return (abs((a[p] + a[s] / 2) - (b[p] + b[s] / 2)) <= ALIGN * size
            and abs(a[s] - b[s]) <= EXTENT * size)


def _aligned_pair(views: list[View]) -> bool:
    art = [v.box for v in views if v.line_art]
    if len(art) < 2:
        return False
    biggest = max(max(b[2], b[3]) for b in art)
    art = [b for b in art if max(b[2], b[3]) >= MIN_VIEW * biggest]
    return any(_in_line(a, b, True) or _in_line(a, b, False) for a, b in combinations(art, 2))
