"""Stage 5: what each stroke is. Pure geometry rules; no public model exists for hand-drawn
line types (see the research report)."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from s2c.sketch.text import TextItem
from s2c.sketch.vectorize import Prim, circle_fit

CHAIN_MIN = 3
ANGLE_TOL = np.radians(20)


@dataclass
class Arrow:
    tip: np.ndarray
    direction: np.ndarray
    prim_id: str


@dataclass
class DimLine:
    id: str
    p0: np.ndarray
    p1: np.ndarray
    arrows: list[Arrow]
    prim_ids: list[str]

    @property
    def horizontal(self) -> bool:
        d = self.p1 - self.p0
        return abs(d[0]) >= abs(d[1])


@dataclass
class Leader:
    id: str
    tip: np.ndarray
    tail: np.ndarray
    prim_ids: list[str]


@dataclass
class Classified:
    visible: list[Prim] = field(default_factory=list)
    hidden: list[Prim] = field(default_factory=list)
    centre: list[Prim] = field(default_factory=list)
    dimlines: list[DimLine] = field(default_factory=list)
    extensions: dict[str, Prim] = field(default_factory=dict)
    leaders: list[Leader] = field(default_factory=list)


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def _angle(u, v) -> float:
    """Unsigned angle between two directions, ignoring orientation (0..pi/2)."""
    c = abs(float(np.dot(_unit(u), _unit(v))))
    return float(np.arccos(min(1.0, c)))


def _point_line_dist(p, a, b) -> float:
    d = _unit(b - a)
    v = p - a
    return float(abs(v[0] * d[1] - v[1] * d[0]))


def _ends(lines: list[Prim]):
    """(prim, end index, end point, outward direction) for every line end."""
    for p in lines:
        yield p, 0, p.p0, _unit(p.p0 - p.p1)
        yield p, 1, p.p1, _unit(p.p1 - p.p0)


def _axis_widths(ink, pts, e, d, ts) -> np.ndarray:
    """Width of the ink run through the axis e + t*d at each t, across a window as wide as the blob
    `pts`: other strokes inside the window do not count."""
    n = np.array([-d[1], d[0]])
    half = float(np.abs((pts - e) @ n).max()) + 2
    us = np.arange(-half, half + 1)
    h, w = ink.shape
    mid = len(us) // 2
    widths = []
    for t in ts:
        q = np.round(e + t * d + us[:, None] * n).astype(int)
        q[:, 0] = np.clip(q[:, 0], 0, w - 1)
        q[:, 1] = np.clip(q[:, 1], 0, h - 1)
        on = ink[q[:, 1], q[:, 0]] > 0
        seed = next((mid + o for o in (0, -1, 1) if on[mid + o]), None)
        if seed is None:
            widths.append(0)
            continue
        lo, hi = seed, seed
        while lo > 0 and on[lo - 1]:
            lo -= 1
        while hi < len(on) - 1 and on[hi + 1]:
            hi += 1
        widths.append(hi - lo + 1)
    return np.array(widths)


def _tip(ink, pts, e, d, reach) -> np.ndarray:
    """Walk the original ink along the carrier axis through the head: the tip is the narrowest
    cross-section past the widest one (the head's base). The opening erased the thin tip itself;
    an extension line crossing at the tip reads as a full cross-section and never wins; two heads
    meeting tip to tip narrow to the same point."""
    proj = (pts - e) @ d
    ts = np.arange(np.floor(proj.min()) - 1, proj.max() + reach + 1)
    widths = _axis_widths(ink, pts, e, d, ts)
    # the base is the widest cross-section of the head itself (the opened blob), not a line
    # crossing near the tip
    base = int(np.argmax(np.where(ts <= proj.max(), widths, -1)))
    # the head ends where the ink ends or where a stroke wider than the head crosses it (an
    # extension line, the outline or circle the tip touches)
    end = len(widths)
    stop = np.nonzero((widths[base + 1:] == 0) | (widths[base + 1:] > widths[base]))[0]
    if len(stop):
        end = base + 1 + int(stop[0])
    seg = widths[base:end]
    # the far end of the first narrowest run: a chained head beyond the tip narrows to the same
    # width and must not pull the tip past the crossing
    k = int(np.argmin(seg))
    while k + 1 < len(seg) and seg[k + 1] == seg[k]:
        k += 1
    return e + d * (ts[base + k] + 1)  # the run's last pixel is still inside the head


def _filled_arrows(lines, ink, stroke) -> tuple[dict[tuple[str, int], Arrow], set[str]]:
    """Filled heads and the skeleton pieces inside them (spurs, the stretch through the head)."""
    k = max(4, round(1.8 * stroke) + 1)
    core = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    dist = cv2.distanceTransform((ink > 0).astype(np.uint8), cv2.DIST_L2, 3)
    _, strokes = cv2.connectedComponents((ink > 0).astype(np.uint8), connectivity=8)
    h, w = ink.shape
    near_tol = 3 * stroke + 2

    def stroke_at(e) -> int:
        x, y = np.clip(np.round(e).astype(int), 0, [w - 1, h - 1])
        return int(strokes[y, x])

    def width_near(p, pts) -> float:
        """The carrier's width, or its ink width just outside the blob when that is larger: the
        mean width is diluted when a stretch of the line elsewhere was partly erased with text."""
        seg = p.p0 + np.linspace(0, 1, max(2, int(p.length)))[:, None] * (p.p1 - p.p0)
        gap = np.min(np.linalg.norm(seg[:, None, :] - pts[None, :, :], axis=2), axis=1)
        keep = seg[(gap > near_tol) & (gap <= 2 * near_tol)]
        if not len(keep):
            return p.width
        ij = np.clip(np.round(keep).astype(int), 0, [w - 1, h - 1])
        return max(p.width, 2 * float(np.median(dist[ij[:, 1], ij[:, 0]])))

    blobs = []
    for i in range(1, n):
        if not (6 <= stats[i, 4] <= 60 * stroke * stroke and max(stats[i, 2], stats[i, 3]) <= 12 * stroke):
            continue
        mask = labels == i
        ys, xs = np.nonzero(mask)
        pts = np.stack([xs, ys], 1).astype(float)
        own = int(strokes[ys[0], xs[0]])
        near = []
        for p, end, e, d in _ends(lines):
            gap = float(np.min(np.linalg.norm(pts - e, axis=1)))
            # a line end across a gap (an extension line stopping short of an outline corner) is
            # not attached to the blob
            if gap <= near_tol and stroke_at(e) == own:
                near.append((p, end, e, d, gap))
        ends_at: dict[str, int] = {}
        for p, *_ in near:
            ends_at[p.id] = ends_at.get(p.id, 0) + 1
        blobs.append((pts, 2 * float(dist[mask].max()), near, {pid for pid, c in ends_at.items() if c == 2}))
    all_parts = set().union(*(b[3] for b in blobs)) if blobs else set()

    claims: dict[tuple[str, int], tuple[float, int, np.ndarray, np.ndarray]] = {}
    for b, (pts, thick, near, _) in enumerate(blobs):
        # a corner of the outline: two lines at least as thick as the blob end in it at an angle. A thin
        # leader crossing the corner makes it wider than the leader, not an arrowhead
        edges = [d for p, _, _, d, gap in near
                 if p.id not in all_parts and gap <= stroke and thick < 1.5 * p.width]
        if any(np.dot(u, v) > -0.9 for i, u in enumerate(edges) for v in edges[i + 1:]):
            continue
        c = pts.mean(0)
        cands = []
        for p, end, e, d, gap in near:
            # a head is clearly wider than the line carrying it; an outline corner is not
            if p.id in all_parts or thick < 1.5 * width_near(p, pts):
                continue
            off = _point_line_dist(c, p.p0, p.p1)
            if off > 1.5 * stroke:
                continue
            # The head's base (its widest cross-section) comes before its tip along the carrier. The
            # blob may lie behind the carrier's end: where text erased half a head, the skeleton runs
            # through it to the tip or into the extension-line corner. A head whose base lies beyond
            # the blob's middle points back at the carrier: it belongs to the next dimension in a chain.
            proj = (pts - e) @ d
            ts = np.arange(np.floor(proj.min()), np.ceil(proj.max()) + 1)
            if ts[int(np.argmax(_axis_widths(ink, pts, e, d, ts)))] > (proj.min() + proj.max()) / 2:
                continue
            cands.append((off, p, end, e, d, gap))
        taken: list[np.ndarray] = []
        for _, p, end, e, d, gap in sorted(cands, key=lambda t: t[0]):
            # one carrier per blob, or two opposite ones where two chained heads merged
            if any(np.dot(d, d2) > -0.9 for d2 in taken):
                continue
            taken.append(d)
            # a line end belongs to the head it touches, not to a remnant further along it (a
            # thick bit of the circle or outline the tip lands on)
            key = (p.id, end)
            if key not in claims or gap < claims[key][0]:
                claims[key] = (gap, b, e, d)
    out = {key: Arrow(tip=_tip(ink, blobs[b][0], e, d, near_tol), direction=d, prim_id=key[0])
           for key, (_, b, e, d) in claims.items()}
    consumed = set().union(*(blobs[b][3] for _, b, _, _ in claims.values())) if claims else set()
    return out, consumed


def _open_arrows(lines, stroke) -> tuple[dict[tuple[str, int], Arrow], set[str]]:
    out, barbs = {}, set()
    short = [p for p in lines if p.length <= 8 * stroke]
    for p, end, e, d in _ends(lines):
        if p.length <= 8 * stroke:
            continue
        sides = []
        for q in short:
            for qe, qo in ((q.p0, q.p1), (q.p1, q.p0)):
                if np.linalg.norm(qe - e) <= 2 * stroke + 1:
                    v = _unit(qo - qe)
                    ang = np.degrees(np.arccos(np.clip(np.dot(v, -d), -1, 1)))
                    if 15 <= ang <= 60:
                        sides.append((np.sign(d[0] * v[1] - d[1] * v[0]), q.id))
        signs = {s for s, _ in sides}
        if {1.0, -1.0} <= signs:
            out[(p.id, end)] = Arrow(tip=e.copy(), direction=d, prim_id=p.id)
            barbs |= {qid for _, qid in sides}
    return out, barbs


def _chains(pieces: list[Prim], stroke: float) -> list[list[Prim]]:
    if not pieces:
        return []
    med = float(np.median([p.length for p in pieces]))
    gap_max = max(3 * stroke, 2.5 * med)
    parent = list(range(len(pieces)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(pieces):
        for j in range(i + 1, len(pieces)):
            b = pieces[j]
            if _angle(a.direction(), b.direction()) > ANGLE_TOL:
                continue
            ends = [(ea, eb) for ea in (a.p0, a.p1) for eb in (b.p0, b.p1)]
            ea, eb = min(ends, key=lambda t: np.linalg.norm(t[0] - t[1]))
            gap = np.linalg.norm(ea - eb)
            if gap > gap_max or gap < 1e-6:
                continue
            if _angle(eb - ea, a.direction()) > ANGLE_TOL:
                continue
            parent[find(i)] = find(j)
    groups: dict[int, list[Prim]] = {}
    for i, p in enumerate(pieces):
        groups.setdefault(find(i), []).append(p)
    return [g for g in groups.values() if len(g) >= CHAIN_MIN]


def _merge_chain(chain: list[Prim], kind_id: str) -> Prim:
    pts = np.vstack([p.pts for p in chain])
    centre = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - centre)
    d = vt[0]
    t = (pts - centre) @ d
    resid = np.abs((pts - centre) @ np.array([-d[1], d[0]]))
    width = float(np.mean([p.width for p in chain]))
    if resid.max() <= 4:
        return Prim(kind_id, "line", np.array([centre + d * t.min(), centre + d * t.max()]), width)
    c, r, _ = circle_fit(pts)
    order = np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
    return Prim(kind_id, "arc", pts[order], width, (float(c[0]), float(c[1])), r)


def classify(prims: list[Prim], ink: np.ndarray, stroke_px: float, texts: list[TextItem]) -> Classified:
    out = Classified()
    s = stroke_px
    lines = [p for p in prims if p.kind == "line"]
    others = [p for p in prims if p.kind != "line"]

    arrows, head_parts = _filled_arrows(lines, ink, s)
    lines = [p for p in lines if p.id not in head_parts]
    open_arrows, barbs = _open_arrows(lines, s)
    for key, a in open_arrows.items():
        arrows.setdefault(key, a)
    lines = [p for p in lines if p.id not in barbs]

    dash_max = 12 * s
    candidates = [p for p in lines if p.length <= dash_max
                  and (p.id, 0) not in arrows and (p.id, 1) not in arrows]
    in_chain: set[str] = set()
    for k, chain in enumerate(_chains(candidates, s)):
        lengths = np.array([p.length for p in chain])
        merged = _merge_chain(chain, f"{chain[0].id}-h{k}")
        (out.centre if lengths.std() / lengths.mean() > 0.6 else out.hidden).append(merged)
        in_chain |= {p.id for p in chain}
    lines = [p for p in lines if p.id not in in_chain]

    dim_texts = [t for t in texts if t.role == "dimension"]
    carriers = [p for p in lines if (p.id, 0) in arrows or (p.id, 1) in arrows]
    used = {p.id for p in carriers}
    for p in carriers:
        ends = [e for e in (0, 1) if (p.id, e) in arrows]
        if len(ends) == 1:
            tail = p.p1 if ends[0] == 0 else p.p0
            near_text = any(np.hypot(tail[0] - (t.box[0] + t.box[2] / 2),
                                     tail[1] - (t.box[1] + t.box[3] / 2)) <= 1.5 * max(t.box[2], t.box[3])
                            for t in dim_texts)
            if near_text:
                out.leaders.append(Leader(f"{p.id}-L", arrows[(p.id, ends[0])].tip, tail, [p.id]))
                continue
        out.dimlines.append(DimLine(f"{p.id}-D", p.p0, p.p1, [arrows[(p.id, e)] for e in ends], [p.id]))

    rest = [p for p in lines if p.id not in used]
    ext_ids: set[str] = set()
    for d in out.dimlines:
        axis = d.p1 - d.p0
        for k, end in enumerate((d.p0, d.p1)):
            tip = next((a.tip for a in d.arrows if np.linalg.norm(a.tip - end) < 6 * s), end)
            best = None
            for q in rest:
                if _angle(q.direction(), axis) < np.radians(70):
                    continue
                dist = _point_line_dist(tip, q.p0, q.p1)
                t = float(np.dot(tip - q.p0, _unit(q.p1 - q.p0)))
                # an extension line ends at the dimension line; of the pieces there, the long one
                # runs to the object, the short one is the overshoot past the arrow
                ends_here = min(np.linalg.norm(q.p0 - tip), np.linalg.norm(q.p1 - tip)) <= 4 * s
                fits = dist <= 2.5 * s and -3 * s <= t <= q.length + 3 * s and ends_here
                if fits and (best is None or q.length > best[0]):
                    best = (q.length, q)
            if best:
                out.extensions[f"{d.id}:{k}"] = best[1]
                ext_ids.add(best[1].id)

    # stubs: skeleton spurs of arrowheads, and the bit of an extension line that pokes past its
    # dimension line; short lines ending near an arrow tip or a dimension-line end are not edges
    anchors = [a.tip for d in out.dimlines for a in d.arrows] + [lead.tip for lead in out.leaders]
    anchors += [e for d in out.dimlines for e in (d.p0, d.p1)]

    def is_stub(p: Prim) -> bool:
        return p.kind == "line" and p.length <= 4 * s and any(
            min(np.linalg.norm(p.p0 - q), np.linalg.norm(p.p1 - q)) <= 4 * s for q in anchors)

    for p in rest + others:
        if p.id in ext_ids or p.length < 2 * s or is_stub(p):
            continue
        out.visible.append(p)
    return out
