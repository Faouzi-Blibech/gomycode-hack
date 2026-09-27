"""Stage 1: find the sheet, rectify it, flatten shadows, binarise. Pure OpenCV and scikit-image."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
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
    return Captured(sheet=sheet, gray=flat, ink=ink, to_original=to_original, stroke_px=_stroke_px(ink))
