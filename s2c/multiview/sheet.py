"""One image holding several orthographic views -> drawings and views (spec 3.1), and the face each view shows
(spec 3.2)."""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from s2c.multiview.ocr import Reader

log = logging.getLogger(__name__)

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
SCALE = 0.08            # a name's implied size and the view's size agree within this share (spec 3.2)
SLACK = 0.004           # ... or within this share of the long side, for thin views and stroke widths
SPUR = 0.006            # filled rows or columns this thin (share of the long side) are chain lines, not the view
MARK_AREA = 0.15        # an unnamed view under this share of the largest view's area is a mark: text, balloon, note
CROP_MARGIN = 0.04
AUTO, SKIP = "auto", "skip"


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


@dataclass
class Naming:
    drawing: int                # index into sheet.drawings of the part drawing; -1 when the sheet has none
    faces: list[str]            # per view of that drawing: a face, "auto" (the user picks) or "skip" (a mark)
    projection: str             # "first" | "third"
    projection_source: str      # "symbol" | "setting"
    warnings: list[str] = field(default_factory=list)


def named_count(naming: Naming) -> int:
    """Views given a face, each of which passed the scale check. The Studio treats an upload as a sheet only from 2
    on: `is_sheet` alone accepts two unrelated sketches that happen to line up."""
    return sum(f not in (AUTO, SKIP) for f in naming.faces)


def name_views(sheet: Sheet, image_bgr: np.ndarray, projection: str = "first",
               reader: Reader | None = None) -> Naming:
    """The face each view of the part drawing shows (spec 3.2).

    A projection symbol overrides `projection`, and is not the part while another drawing of 2 or more views
    exists. A read label names its view when the size it implies fits the sheet's shared scale; otherwise the
    layout does, under the same check. A view neither names stays "auto", never a guess; an unnamed view under
    MARK_AREA of the largest is a mark (dimension text, a balloon, a note) and is "skip"."""
    if projection not in ("first", "third"):
        raise ValueError(f"projection {projection!r}")
    if not sheet.drawings:
        return Naming(-1, [], projection, "setting", ["No drawing found in the image."])
    image = _bgr(image_bgr)
    ink = ink_mask(image)
    long = max(sheet.shape)
    source, warnings = "setting", []
    symbol = find_symbol(sheet, image)
    pool = list(range(len(sheet.drawings)))
    if symbol is not None:
        projection, source = symbol[1], "symbol"
        rest = [i for i in pool if i != symbol[0]]
        if any(len(sheet.drawings[i].views) >= 2 for i in rest):
            pool = rest
    part = max(pool, key=lambda i: (len(sheet.drawings[i].views), sum(_area(v.box) for v in sheet.drawings[i].views)))
    views = sheet.drawings[part].views
    if len(pool) > 1:
        warnings.append(f"The sheet holds {len(pool)} drawings; the views were read from the one with "
                        f"{len(views)} views.")

    texts = [_read_label(image, v.label_box, reader) for v in views]
    boxes = [_body(ink, v.box, long) for v in views]
    faces, aligned, checked, notes = _name(boxes, texts, projection, _slack(long))
    warnings += notes
    largest = max(_area(v.box) for v in views)
    marks = [i for i, f in enumerate(faces)
             if f == AUTO and _area(views[i].box) < MARK_AREA * largest and not (aligned[i] and checked[i])]
    for i in marks:
        faces[i] = SKIP
    unnamed = faces.count(AUTO)
    if unnamed:
        warnings.append(f"{unnamed} view{'s' if unnamed > 1 else ''} could not be named; pick the face by hand.")
    if marks:
        warnings.append(f"Skipped {len(marks)} small mark{'s' if len(marks) > 1 else ''} "
                        "(dimension text, balloons or notes).")
    return Naming(part, faces, projection, source, warnings)


def crop_views(sheet: Sheet, image_bgr: np.ndarray, naming: Naming) -> list[tuple[bytes, str]]:
    """PNG crop and face of each view of the part drawing, in view order, marks left out. Each crop is the view box
    plus a CROP_MARGIN margin, on white, at the sheet's scale so the views keep their shared scale (spec 3.2)."""
    if naming.drawing < 0:
        return []
    image = _bgr(image_bgr)
    paper = ink_mask(image) == 0
    out = []
    for view, face in zip(sheet.drawings[naming.drawing].views, naming.faces):
        if face == SKIP:
            continue
        x, y, w, h = view.box
        m = max(2, round(CROP_MARGIN * max(w, h)))
        crop = np.full((h + 2 * m, w + 2 * m, 3), 255, np.uint8)
        body = image[y: y + h, x: x + w].copy()
        body[paper[y: y + h, x: x + w]] = 255
        crop[m: m + h, m: m + w] = body
        out.append((cv2.imencode(".png", crop)[1].tobytes(), face))
    return out


def find_symbol(sheet: Sheet, image_bgr: np.ndarray) -> tuple[int, str] | None:
    """The drawing that is the ISO 5456-2 projection symbol, and the projection it sets (spec 2, 3.2).

    Exactly 2 views: a truncated cone seen side-on (a 4-vertex trapezoid, parallel sides upright) and end-on (a
    circle with one concentric inner circle), on one centre line, the tall and short sides matching the outer and
    inner diameters within SCALE. Circles beside the large end mean first-angle, beside the small end third-angle."""
    ink = ink_mask(_bgr(image_bgr))
    slack = _slack(max(sheet.shape))
    k = max(3, round(DILATE * max(sheet.shape)))
    for d, drawing in enumerate(sheet.drawings):
        if len(drawing.views) != 2:
            continue
        for a, b in (drawing.views, drawing.views[::-1]):
            cone, end = _trapezoid(ink, a.box, k), _rings(ink, b.box)
            if cone is None or end is None:
                continue
            tall, short, tall_x, short_x, axis_y = cone
            outer, inner, cx, cy = end
            if (_close(tall, outer, slack) and _close(short, inner, slack)
                    and abs(axis_y - cy) <= max(SCALE * outer, slack)):
                big_end_right, circles_right = tall_x > short_x, cx > (tall_x + short_x) / 2
                return d, "first" if big_end_right == circles_right else "third"
    return None


_KEYWORDS = {"front": "front", "rear": "back", "back": "back", "top": "top", "plan": "top", "bottom": "bottom",
             "left": "left", "right": "right", "side": "side", "end": "side", "elevation": "elevation"}


def label_face(text: str) -> str | None:
    """The face a view label names, or None. "Side" or "end" alone ("End Elevation") names a side view without
    saying which, so it is None here and the layout picks left or right."""
    hint = _label_hint(text)
    return None if hint == "side" else hint


def _label_hint(text: str) -> str | None:
    """A face, "side" (a side view, left or right unsaid) or None. Keywords match case-insensitively with one wrong
    character; two different faces in one label name nothing; "elevation" alone is the front."""
    words = re.findall(r"[a-z0-9]+", (text or "").lower().replace("view", " "))
    said = {_KEYWORDS[k] for w in words for k in _KEYWORDS if _is_word(w, k)}
    faces = said - {"side", "elevation"}
    if faces:
        return faces.pop() if len(faces) == 1 else None
    if "side" in said:
        return "side"
    return "front" if "elevation" in said else None


def _is_word(word: str, key: str) -> bool:
    """Equal, or one character wrong: substituted, missing or extra. A three-letter key forgives only a non-letter
    (a 0 read for an O), so "and" is not "end" and "tip" is not "top"."""
    if word == key:
        return True
    if len(key) < 4:
        return len(word) == len(key) and sum(a != b for a, b in zip(word, key)) == 1 and not word.isalpha()
    if len(word) == len(key):
        return sum(a != b for a, b in zip(word, key)) == 1
    if abs(len(word) - len(key)) != 1:
        return False
    short, longer = sorted((word, key), key=len)
    return any(longer[:i] + longer[i + 1:] == short for i in range(len(longer)))


def _read_label(image: np.ndarray, box: Box | None, reader: Reader | None) -> str:
    if reader is None or box is None:
        return ""
    x, y, w, h = box
    m = max(2, h // 4)
    crop = image[max(0, y - m): y + h + m, max(0, x - m): x + w + m]
    try:
        text, _ = reader(crop)
    except Exception:  # an OCR failure must not stall the sheet: the layout still names the view
        log.warning("label reader failed", exc_info=True)
        return ""
    return str(text or "").strip()


def _name(boxes: list[Box], texts: list[str], projection: str, slack: int):
    """Faces, whether each view lines up with the front, whether some name for it passed the scale check, and the
    warnings. Every view is tried as the front; the naming that agrees with the most labels, then names the most
    view area, wins, and a tie goes to the layout's front (most aligned neighbours, spec 3.2)."""
    hints = [_label_hint(t) for t in texts]
    usual = _layout_front(boxes, slack)
    best = None
    for f in range(len(boxes)):
        faces, aligned, checked, notes, agreed = _assign(boxes, hints, texts, f, projection, slack)
        score = (agreed, sum(_area(boxes[i]) for i, x in enumerate(faces) if x != AUTO), f == usual)
        if best is None or score > best[0]:
            best = (score, faces, aligned, checked, notes)
    return best[1:]


def _assign(boxes, hints, texts, f, projection, slack):
    layout, aligned = _layout(boxes, f, projection, slack)
    width, height = boxes[f][2], boxes[f][3]
    depth = _depth(boxes, layout, hints, width, height, slack)
    placed = [bool(n) and i != f and _fits(n, boxes[i], width, height, depth, slack) for i, n in enumerate(layout)]
    faces, notes, agreed = [], [], 0
    for i, box in enumerate(boxes):
        hint = hints[i]
        said = None if hint in (None, "side") else hint
        if i == f:
            face = "front"
        elif said not in (None, "front") and _fits(said, box, width, height, depth, slack):
            face = said
        else:
            face = layout[i] if placed[i] else AUTO
        faces.append(face)
        if said == face or (hint == "side" and face in ("left", "right")):
            agreed += 1
        elif said:
            notes.append(f"{texts[i]} does not match its size; "
                         + (f"used as {face}" if face != AUTO else "left for you to name"))
    checked = [x != AUTO for x in faces]
    for face in sorted({x for x in faces if x != AUTO}):
        same = [i for i, x in enumerate(faces) if x == face]
        if len(same) < 2:
            continue
        keep = [i for i in same if layout[i] == face]
        for i in same:
            if keep == [i]:
                continue
            what = texts[i] if hints[i] == face else f"The view placed as {face}"
            agreed -= hints[i] == face
            faces[i] = layout[i] if placed[i] and layout[i] not in faces else AUTO
            notes.append(f"{what} is also another view's name; "
                         + (f"used as {faces[i]}" if faces[i] != AUTO else "left for you to name"))
    return faces, aligned, checked, notes, agreed


def _layout_front(boxes: list[Box], slack: int) -> int:
    """With two views the left (or upper) one; otherwise the view with the most neighbours in line with it and
    sharing its extent, then the largest, then the top-left."""
    if len(boxes) == 2:
        return min(range(2), key=lambda i: sum(_centre(boxes[i])))

    def neighbours(i):
        return sum(_in_line_with(boxes[i], b, row, slack) and _close(boxes[i][3 if row else 2], b[3 if row else 2],
                                                                     slack)
                   for j, b in enumerate(boxes) if j != i for row in (True, False))

    return min(range(len(boxes)), key=lambda i: (-neighbours(i), -_area(boxes[i]), boxes[i][0] + boxes[i][1]))


_STEPS = {  # steps from the front along its row, and down its column -> face (spec 2)
    "first": ({1: "left", -1: "right", 2: "back", -2: "back"}, {1: "top", -1: "bottom"}),
    "third": ({1: "right", -1: "left", 2: "back", -2: "back"}, {-1: "top", 1: "bottom"}),
}


def _layout(boxes: list[Box], f: int, projection: str, slack: int) -> tuple[list[str | None], list[bool]]:
    """The face each view's place around the front f implies (spec 2), or None; and whether it is in line with f.
    Only views sharing the front's height (row) or width (column) take a place, so a dimension number between two
    views does not push the side view one step out."""
    row, col, aligned = [f], [f], [False] * len(boxes)
    for i, b in enumerate(boxes):
        if i == f:
            continue
        in_row, in_col = _in_line_with(boxes[f], b, True, slack), _in_line_with(boxes[f], b, False, slack)
        aligned[i] = in_row or in_col
        if in_row and _close(b[3], boxes[f][3], slack):
            row.append(i)
        elif in_col and _close(b[2], boxes[f][2], slack):
            col.append(i)
    names: list[str | None] = [None] * len(boxes)
    for line, axis, steps in ((row, 0, _STEPS[projection][0]), (col, 1, _STEPS[projection][1])):
        line.sort(key=lambda i: _centre(boxes[i])[axis])
        at = line.index(f)
        for k, i in enumerate(line):
            names[i] = steps.get(k - at)
    names[f] = "front"
    return names, aligned


def _depth(boxes, layout, hints, width, height, slack) -> float | None:
    """The part's depth in px, from the views that show it (top and bottom as height, sides as width): the value
    most of them agree on. Layout names first; labels only when no placed view shows it."""
    def values(names):
        return [boxes[i][3] if n in ("top", "bottom") else boxes[i][2] for i, n in enumerate(names)
                if n in ("top", "bottom", "left", "right") and _fits(n, boxes[i], width, height, None, slack)]

    found = values(layout) or values(hints)
    if not found:
        return None
    return max(found, key=lambda v: sum(_close(v, u, slack) for u in found))


def _fits(face: str, box: Box, width: int, height: int, depth: float | None, slack: int) -> bool:
    """The scale check: front and rear are W x H, top and bottom W x D, the sides D x H (spec 2)."""
    w, h = box[2], box[3]
    if face in ("front", "back"):
        return _close(w, width, slack) and _close(h, height, slack)
    if face in ("top", "bottom"):
        return _close(w, width, slack) and (depth is None or _close(h, depth, slack))
    return _close(h, height, slack) and (depth is None or _close(w, depth, slack))


def _close(a: float, b: float, slack: int) -> bool:
    return abs(a - b) <= max(SCALE * max(a, b), slack)


def _in_line_with(front: Box, b: Box, row: bool, slack: int) -> bool:
    """Same row (centres within ALIGN of the front's height) or same column (of its width)."""
    p, s = (1, 3) if row else (0, 2)
    return abs((front[p] + front[s] / 2) - (b[p] + b[s] / 2)) <= max(ALIGN * front[s], slack)


def _centre(box: Box) -> tuple[float, float]:
    return box[0] + box[2] / 2, box[1] + box[3] / 2


def _area(box: Box) -> int:
    return box[2] * box[3]


def _slack(long: int) -> int:
    return max(3, round(SLACK * long))


def _bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR) if image.shape[2] == 4 else image


def _filled(ink: np.ndarray, box: Box, k: int) -> np.ndarray:
    """The box's ink closed over gaps of k px (dashes, chain lines) and filled inside its outer outlines, 0/1."""
    x, y, w, h = box
    sub = cv2.copyMakeBorder(ink[y: y + h, x: x + w], k, k, k, k, cv2.BORDER_CONSTANT, value=0)
    closed = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(closed)
    cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
    return filled[k: k + h, k: k + w]


def _body(ink: np.ndarray, box: Box, long: int) -> Box:
    """The view's box without the chain lines and leaders that run past its outline (spec 2: never part of it):
    rows and columns of the filled view thinner than SPUR of the long side are trimmed from its ends. A view that
    thin all through (a sheet-metal edge) keeps its box."""
    filled = _filled(ink, box, max(3, round(DILATE * long)))
    thin = max(5, round(SPUR * long))
    cols, rows = np.nonzero(filled.sum(0) > thin)[0], np.nonzero(filled.sum(1) > thin)[0]
    x0, x1 = (int(cols[0]), int(cols[-1])) if len(cols) else (0, box[2] - 1)
    y0, y1 = (int(rows[0]), int(rows[-1])) if len(rows) else (0, box[3] - 1)
    return box[0] + x0, box[1] + y0, x1 - x0 + 1, y1 - y0 + 1


def _trapezoid(ink: np.ndarray, box: Box, k: int) -> tuple[float, float, float, float, float] | None:
    """A cone seen side-on: a convex 4-vertex outline with two upright parallel sides. Returns the tall and short
    side lengths, their x (image px) and the axis height."""
    filled = _filled(ink, box, k).astype(np.uint8)
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    poly = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True).reshape(-1, 2)
    if len(poly) != 4 or not cv2.isContourConvex(poly):
        return None
    ends = [(p, q) for p, q in zip(poly, np.roll(poly, -1, axis=0)) if abs(p[0] - q[0]) <= 0.1 * abs(p[1] - q[1])]
    if len(ends) != 2:
        return None
    (tall, tall_x, axis_y), (short, short_x, _) = sorted(
        ((abs(float(p[1] - q[1])), (p[0] + q[0]) / 2 + box[0], (p[1] + q[1]) / 2 + box[1]) for p, q in ends),
        reverse=True)
    return tall, short, tall_x, short_x, axis_y


def _rings(ink: np.ndarray, box: Box) -> tuple[float, float, float, float] | None:
    """A cone seen end-on: exactly two concentric circles and nothing else. Returns the outer and inner diameters
    (to the outside of each stroke) and the centre (image px)."""
    x, y = box[0], box[1]
    contours, _ = cv2.findContours(ink[y: y + box[3], x: x + box[2]].copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    circles = []
    for c in contours:
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if r < 3:
            continue
        if cv2.contourArea(c) < 0.8 * math.pi * r * r:
            return None
        circles.append((r, cx, cy))
    if not circles:
        return None
    circles.sort(reverse=True)
    big, ox, oy = circles[0]
    if any(math.hypot(cx - ox, cy - oy) > SCALE * big + 1 for _, cx, cy in circles):
        return None
    rings = [big]
    for (r, _, _), (prev, _, _) in zip(circles[1:], circles):
        if prev - r > max(4.0, 0.15 * big):
            rings.append(r)
    if len(rings) != 2:
        return None
    return 2 * rings[0], 2 * rings[1], ox + x, oy + y
