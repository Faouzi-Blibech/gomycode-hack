"""Stage 1: find the sheet, rectify it, flatten shadows, binarise. Pure OpenCV and scikit-image."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_sauvola
from skimage.morphology import skeletonize

from s2c.sketch.models import SketchAbstain

LONG_SIDE = 1600
MIN_SHEET_AREA = 0.2      # of the photo
FULL_FRAME_AREA = 0.95    # a "sheet" this big is the whole photo
BRIGHT_PAPER = 150
TOO_DARK = 50
BLUR_VAR = 8.0            # variance of the Laplacian; set on synthetic tests, re-check on golden photos
CORE_WIN = 7              # px: the window where a stroke edge finds its dark core
PAPER_SAT = 26            # HSV saturation below which a pixel may be paper (a wooden desk is far above)
EDGE_BAND = 15            # px: ink reaching this close to the paper's edge is not the drawing
MIN_SKELETON = 10         # px of skeleton for a piece of ink to count as a stroke (not a speck)

RETAKE_FRAME = "Put the whole sheet in the frame on a darker surface and retake."
RETAKE_LIGHT = "Retake in good light and hold the phone steady."


@dataclass
class Captured:
    sheet: np.ndarray
    gray: np.ndarray
    ink: np.ndarray
    to_original: np.ndarray
    stroke_px: float


def sheet_to_original(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def _order(pts: np.ndarray) -> np.ndarray:
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.float32([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])


def _find_sheet(gray: np.ndarray) -> np.ndarray | None:
    """Corners (tl, tr, br, bl) of the paper in `gray` pixels, or None."""
    f = 800 / max(gray.shape)
    small = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c) / th.size
    if area < MIN_SHEET_AREA or area > FULL_FRAME_AREA:
        return None
    approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True).reshape(-1, 2)
    if len(approx) != 4:
        rect = cv2.boxPoints(cv2.minAreaRect(c))
        if cv2.contourArea(c) / max(cv2.contourArea(rect), 1) < 0.85:
            return None
        approx = rect
    if not _edges_are_sharp(blur, approx.astype(np.float32)):
        return None  # a shadow boundary, not a paper edge
    return _order(approx.astype(np.float32) / f)


def _paper_outline(image: np.ndarray) -> np.ndarray | None:
    """Convex outline (photo pixels) of a sheet that has no clean four-corner edge: it runs off the photo, or
    has a spiral binding on one side. The paper is bright and nearly grey; a desk is saturated. A binding is
    a column of similar dark holes along one side and is cut off with everything beyond it."""
    f = 512 / max(image.shape[:2])
    small = cv2.resize(image, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(cv2.medianBlur(small, 5), cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    p95 = float(np.percentile(v, 95))
    paper = ((s < PAPER_SAT) & (v > 0.35 * p95)).astype(np.uint8)
    dark = (v < 0.6 * p95).astype(np.uint8)
    for flip in range(4):  # right, left, bottom, top binding
        d = _turn(dark, flip)
        cut = _binding(d)
        if cut is not None:
            p = _turn(paper, flip)
            p[:, max(cut - 4, 0):] = 0
            paper = _turn(p, flip, back=True)
    paper = cv2.morphologyEx(paper, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(paper, connectivity=4)
    if n < 2:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    sheet = ndimage.binary_fill_holes(lab == k).astype(np.uint8)
    cs, _ = cv2.findContours(sheet, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(cs, key=cv2.contourArea))
    if not MIN_SHEET_AREA <= cv2.contourArea(hull) / sheet.size <= FULL_FRAME_AREA:
        return None  # no sheet, or the whole photo is paper: nothing to crop
    return hull.reshape(-1, 2).astype(np.float32) / f


def _turn(a: np.ndarray, k: int, back: bool = False) -> np.ndarray:
    """Rotate so side k (0 right, 1 left, 2 bottom, 3 top) becomes the right side, or back."""
    ops = [lambda x: x, lambda x: x[:, ::-1], lambda x: x.T, lambda x: x.T[:, ::-1]]
    inv = [lambda x: x, lambda x: x[:, ::-1], lambda x: x.T, lambda x: x[:, ::-1].T]
    return np.ascontiguousarray((inv if back else ops)[k](a))


def _binding(dark: np.ndarray) -> int | None:
    """Left edge x of a column of at least 10 evenly spaced small dark holes in the outer tenth on the right."""
    h, w = dark.shape
    n, _, st, cen = cv2.connectedComponentsWithStats(dark, connectivity=8)
    ok = [i for i in range(1, n) if 3 <= st[i, cv2.CC_STAT_AREA] <= 0.002 * h * w and st[i, 3] < 0.05 * h
          and cen[i][0] > 0.9 * w]
    if len(ok) < 10:
        return None
    xs = np.array([cen[i][0] for i in ok])
    col = [i for i, x in zip(ok, xs) if abs(x - np.median(xs)) < 0.03 * w]
    ys = np.sort([cen[i][1] for i in col])
    gaps = np.diff(ys)
    if len(col) < 10 or np.ptp(ys) < 0.4 * h or np.std(gaps) > 0.35 * np.mean(gaps):
        return None  # a binding's holes are evenly spaced; dashes of a drawn line are not this regular
    return int(min(st[i, 0] for i in col))


def _crop_to_outline(image: np.ndarray, outline: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The outline's bounding box at LONG_SIDE, the outline mask in it and the homography back to the photo."""
    x, y, w, h = cv2.boundingRect(outline.astype(np.int32))
    k = LONG_SIDE / max(w, h)
    sheet = cv2.resize(image[y:y + h, x:x + w], None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    mask = np.zeros(sheet.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, ((outline - [x, y]) * k).astype(np.int32), 255)
    to_original = np.array([[1 / k, 0, x], [0, 1 / k, y], [0, 0, 1]], np.float64)
    return sheet, mask, to_original


def _keep_inside(ink: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Drop ink off the paper and every piece of ink reaching the paper's edge (binding, desk, other covers)."""
    ink = ink.copy()
    ink[mask == 0] = 0
    inner = cv2.erode(mask, np.ones((EDGE_BAND, EDGE_BAND), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    _, labels = cv2.connectedComponents(ink, connectivity=8)
    touch = np.unique(labels[(ink > 0) & (inner == 0)])
    ink[np.isin(labels, touch[touch > 0])] = 0
    return ink


def _edges_are_sharp(img: np.ndarray, quad: np.ndarray, min_step: float = 30.0) -> bool:
    """A paper edge is a brightness step; a shadow boundary is a slow ramp. Sides lying on the
    photo border are skipped."""
    h, w = img.shape
    centre = quad.mean(0)
    for a, b in zip(quad, np.roll(quad, -1, 0)):
        on_same_border = (
            (a[0] < 3 and b[0] < 3) or (w - 1 - a[0] < 3 and w - 1 - b[0] < 3)
            or (a[1] < 3 and b[1] < 3) or (h - 1 - a[1] < 3 and h - 1 - b[1] < 3)
        )
        if on_same_border:
            continue
        d = (b - a) / (np.linalg.norm(b - a) + 1e-9)
        n = np.array([-d[1], d[0]])
        if np.dot(n, (a + b) / 2 - centre) < 0:
            n = -n
        steps = []
        for t in np.linspace(0.1, 0.9, 15):
            p = a + t * (b - a)
            q_in, q_out = p - 4 * n, p + 4 * n
            if all(0 <= q[0] < w and 0 <= q[1] < h for q in (q_in, q_out)):
                steps.append(float(img[int(q_in[1]), int(q_in[0])])
                             - float(img[int(q_out[1]), int(q_out[0])]))
        if steps and np.median(steps) < min_step:
            return False
    return True


def _rectify(image: np.ndarray, corners: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Rectified sheet at LONG_SIDE and the homography from sheet pixels back to the photo."""
    h, w = image.shape[:2]
    if corners is None:
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    tl, tr, br, bl = corners
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    k = LONG_SIDE / max(width, height)
    W, Hh = round(width * k), round(height * k)
    dst = np.float32([[0, 0], [W, 0], [W, Hh], [0, Hh]])
    M = cv2.getPerspectiveTransform(corners, dst)
    sheet = cv2.warpPerspective(image, M, (W, Hh), flags=cv2.INTER_AREA)
    return sheet, np.linalg.inv(M)


def _flatten(gray: np.ndarray) -> np.ndarray:
    """Divide by the estimated paper brightness: shadows and uneven light disappear."""
    bg = cv2.medianBlur(cv2.dilate(gray, np.ones((7, 7), np.uint8)), 41)
    return cv2.divide(gray, bg, scale=255)


def _binarise(flat: np.ndarray) -> np.ndarray:
    thresh = threshold_sauvola(flat, window_size=31, k=0.2)
    # a stroke's edge lies half-way between its dark core and the paper; Sauvola alone also takes the
    # soft fringe of an anti-aliased or slightly blurred stroke, a pixel wider on each side
    core = cv2.erode(flat, np.ones((CORE_WIN, CORE_WIN), np.uint8)).astype(np.float32)
    ink = ((flat < thresh) & (flat < 225) & (flat < (core + 255) / 2)).astype(np.uint8) * 255
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    small = np.where(stats[:, cv2.CC_STAT_AREA] < 12)[0]
    ink[np.isin(labels, small[small > 0])] = 0
    return ink


def _stroke_px(ink: np.ndarray) -> float:
    """The pen width: the median over strokes (connected pieces of ink) of each stroke's own median
    width, so every glyph, dash and line counts once and a few long thick outlines or thin hairlines
    do not swing it. A skeleton pixel at distance d from the paper sits in a stroke 2d - 1 wide. Ink
    touching the border is the sheet's own edge after rectification, not a pen stroke."""
    if not ink.any():
        return 3.0
    skel = skeletonize(ink > 0)
    widths = 2 * cv2.distanceTransform(ink, cv2.DIST_L2, 5) - 1
    n, labels, stats, _ = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8), connectivity=8)
    h, w = ink.shape
    inside = [i for i in range(1, n) if stats[i, 0] > 0 and stats[i, 1] > 0
              and stats[i, 0] + stats[i, 2] < w and stats[i, 1] + stats[i, 3] < h]
    per_stroke = [float(np.median(ws)) for ws in (widths[skel & (labels == i)] for i in inside)
                  if ws.size >= MIN_SKELETON]
    if per_stroke:
        return float(np.median(per_stroke))
    return float(np.median(widths[skel])) if skel.any() else 3.0


def capture(image_bgr: np.ndarray) -> Captured | SketchAbstain:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    corners = _find_sheet(gray)
    if corners is None:
        mean = float(gray.mean())
        if mean < TOO_DARK:
            return SketchAbstain(stage="capture", reason="image_quality", remedy=RETAKE_LIGHT)
        if mean < BRIGHT_PAPER:
            return SketchAbstain(stage="capture", reason="sheet_not_found", remedy=RETAKE_FRAME)
    mask = None
    outline = _paper_outline(image_bgr) if corners is None else None
    if outline is not None:
        sheet, mask, to_original = _crop_to_outline(image_bgr, outline)
    else:
        sheet, to_original = _rectify(image_bgr, corners)
    sheet_gray = cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY)
    # a blank page has no contrast at all and must reach the views stage (no_views_found);
    # a blurred drawing still has some contrast but no sharp edges
    contrast = float(np.percentile(sheet_gray, 99.5) - np.percentile(sheet_gray, 0.5))
    blurry = contrast >= 10 and cv2.Laplacian(sheet_gray, cv2.CV_64F).var() < BLUR_VAR
    if float(sheet_gray.mean()) < TOO_DARK or blurry:
        return SketchAbstain(stage="capture", reason="image_quality", remedy=RETAKE_LIGHT)
    flat = _flatten(sheet_gray)
    ink = _binarise(flat)
    if mask is not None:
        ink = _keep_inside(ink, mask)
    return Captured(sheet=sheet, gray=flat, ink=ink, to_original=to_original, stroke_px=_stroke_px(ink))
