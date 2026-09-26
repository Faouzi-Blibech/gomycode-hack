"""Image -> outer outline, openings and circles, in pixels. Spec section 6.2.
One rule covers pen sketches (a drawn ring) and photos (a filled part)."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from s2c.multiview.spec import MvAbstain

LONG_SIDE = 1600
MIN_OUTLINE_FRACTION = 0.02
MIN_OPENING_FRACTION = 0.0003
CIRCULARITY = 0.85
EDGE_BAND_PX = 31  # the inside of a drawn outline touches this band next to the edge; a real opening does not
MIN_THIN_FRACTION = 0.002  # a thin part's edge view can be this small and still be a real outline
MIN_THIN_SPAN = 0.10       # ...provided it is long relative to the page...
MIN_THIN_FILL = 0.5        # ...and filled, not a hollow or broken stroke...
MIN_THIN_MARGIN = 0.01     # ...and fully in frame, not a table edge or ruler crossing the border


@dataclass
class PixelCircle:
    cx: float
    cy: float
    d: float


@dataclass
class PixelOutline:
    outer: np.ndarray                                    # (n, 2) image pixels
    inner: list[np.ndarray] = field(default_factory=list)
    circles: list[PixelCircle] = field(default_factory=list)
    bbox: tuple[int, int, int, int] = (0, 0, 1, 1)        # x, y, w, h of the outer outline
    circular: bool = False                               # the outer outline is itself a circle
    shape: tuple[int, int] = (1, 1)                      # image height, width


def resize_long_side(image: np.ndarray, long_side: int = LONG_SIDE) -> np.ndarray:
    h, w = image.shape[:2]
    s = long_side / max(h, w)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, (round(w * s), round(h * s)), interpolation=interp)


def circularity(contour) -> float:
    perimeter = cv2.arcLength(contour, True)
    return 0.0 if perimeter == 0 else float(4 * np.pi * cv2.contourArea(contour) / perimeter ** 2)


def foreground(image_bgr: np.ndarray, mask_out=()) -> np.ndarray:
    """Ink or part pixels are 255. The polarity is chosen so the image border is background."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    background = int(np.median(np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])))
    for x, y, w, h in mask_out:
        gray[max(y, 0): y + h, max(x, 0): x + w] = background
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.concatenate([th[0], th[-1], th[:, 0], th[:, -1]]).mean() > 127:
        th = 255 - th
    return cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))


def is_thin_edge_view(contour, area: float, h: int, w: int) -> bool:
    """Small but long, filled and fully in frame: a thin part's edge-on silhouette, not a speck,
    a broken stroke, or a table edge / ruler / shadow crossing the image border."""
    if area < MIN_THIN_FRACTION * h * w:
        return False
    x, y, bw, bh = cv2.boundingRect(contour)
    if max(bw, bh) < MIN_THIN_SPAN * max(h, w):
        return False
    if area < MIN_THIN_FILL * bw * bh:
        return False
    margin = MIN_THIN_MARGIN * max(h, w)
    return x >= margin and y >= margin and (w - (x + bw)) >= margin and (h - (y + bh)) >= margin


def extract(image_bgr: np.ndarray, mask_out=(), band: int = EDGE_BAND_PX) -> PixelOutline | MvAbstain:
    fg = foreground(image_bgr, mask_out)
    h, w = fg.shape
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outer = max(contours, key=cv2.contourArea) if contours else None
    outer_area = cv2.contourArea(outer) if outer is not None else 0.0
    if outer is None or (outer_area < MIN_OUTLINE_FRACTION * h * w
                          and not is_thin_edge_view(outer, outer_area, h, w)):
        return MvAbstain(stage="outline", reason="no_outline",
                         remedy="Retake on a plain background with the whole part in frame.")
    filled = np.zeros_like(fg)
    cv2.drawContours(filled, [outer], -1, 255, -1)
    edge_band = cv2.subtract(filled, cv2.erode(filled, np.ones((band, band), np.uint8)))
    gaps = cv2.morphologyEx(cv2.bitwise_and(filled, cv2.bitwise_not(fg)), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(gaps, connectivity=4)
    inner, circles = [], []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_OPENING_FRACTION * h * w:
            continue
        comp = np.where(labels == i, 255, 0).astype(np.uint8)
        if cv2.countNonZero(cv2.bitwise_and(comp, edge_band)):
            continue  # the inside of a drawn outline, not an opening
        c = max(cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
        if circularity(c) >= CIRCULARITY:
            m = cv2.moments(comp, binaryImage=True)
            circles.append(PixelCircle(m["m10"] / m["m00"], m["m01"] / m["m00"], float(2 * np.sqrt(area / np.pi))))
        else:
            inner.append(cv2.approxPolyDP(c, 2.0, True).reshape(-1, 2))
    return PixelOutline(outer=cv2.approxPolyDP(outer, 2.0, True).reshape(-1, 2), inner=inner, circles=circles,
                        bbox=tuple(int(v) for v in cv2.boundingRect(outer)),
                        circular=circularity(outer) >= CIRCULARITY, shape=(h, w))


def to_face_mm(points_px, bbox, sa: float, sb: float) -> list[tuple[float, float]]:
    """Image pixels -> face-frame millimetres: a from the left of the bbox, b up from its bottom row."""
    x, y, _, h = bbox
    pts = np.asarray(points_px, np.float64).reshape(-1, 2)
    return [(float((u - x) * sa), float((y + h - 1 - v) * sb)) for u, v in pts]
