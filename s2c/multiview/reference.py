"""Reference objects in a photo: a coin, a card or an A4 sheet gives millimetres per pixel. Spec section 4.3.
Coins are the numbers owner's domain; this detector stands in until metrology.py lands."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from s2c.multiview.outline import foreground
from s2c.multiview.spec import MvAbstain

REFERENCES = {
    "1 TND": ("coin", 25.0), "1 EUR": ("coin", 23.25), "2 EUR": ("coin", 25.75),
    "card": ("rect", (85.60, 53.98)), "a4": ("rect", (297.0, 210.0)),
}
MAX_ECCENTRICITY = 0.30
ASPECT_TOLERANCE = 0.03
A4_PX_PER_MM = 5.0
COIN_REMEDY = "Lay the coin flat, shoot top-down, retake."


@dataclass
class RefScale:
    name: str
    mm_per_px: float
    bbox: tuple[int, int, int, int] | None  # region to mask out before outline extraction
    image: np.ndarray                       # the image to use from now on (rectified for A4)
    confidence: float


def find_reference(image_bgr: np.ndarray, name: str) -> RefScale | MvAbstain:
    if name not in REFERENCES:
        raise ValueError(f"unknown reference object {name!r}; choose one of {sorted(REFERENCES)}")
    kind, size = REFERENCES[name]
    if kind == "coin":
        return _coin(image_bgr, name, size)
    return _a4(image_bgr) if name == "a4" else _card(image_bgr, size)


def _padded(x, y, w, h, pad=6):
    return x - pad, y - pad, w + 2 * pad, h + 2 * pad


def _ellipse(contour) -> tuple[float, float, float]:
    """Fitted ellipse axes (major, minor) and the worst relative distance of a contour point from it."""
    (cx, cy), (d1, d2), angle = cv2.fitEllipse(contour)
    t = np.deg2rad(angle)
    pts = contour.reshape(-1, 2).astype(np.float64) - (cx, cy)
    u = (pts[:, 0] * np.cos(t) + pts[:, 1] * np.sin(t)) / (d1 / 2)
    v = (-pts[:, 0] * np.sin(t) + pts[:, 1] * np.cos(t)) / (d2 / 2)
    return max(d1, d2), min(d1, d2), float(np.abs(np.hypot(u, v) - 1).max())


def _coin(img: np.ndarray, name: str, diameter: float) -> RefScale | MvAbstain:
    """The smallest blob that is an ellipse is the coin; the part is assumed larger than the coin.
    Candidates are found by ellipse fit, not by circularity, so a tilted coin reaches the tilt gate."""
    contours, _ = cv2.findContours(foreground(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    min_area = 0.0005 * img.shape[0] * img.shape[1]
    candidates = []
    for c in contours:
        if len(c) >= 20 and cv2.contourArea(c) > min_area:
            major, minor, deviation = _ellipse(c)
            if deviation <= 0.08:
                candidates.append((cv2.contourArea(c), c, major, minor))
    if not candidates:
        return MvAbstain(stage="dimensions", reason="coin_not_found", remedy=COIN_REMEDY)
    _, coin, major, minor = min(candidates, key=lambda e: e[0])
    eccentricity = float(np.sqrt(1 - (minor / major) ** 2))
    if eccentricity > MAX_ECCENTRICITY:
        return MvAbstain(stage="dimensions", reason="coin_tilted", remedy=COIN_REMEDY)
    return RefScale(name, diameter / major, _padded(*cv2.boundingRect(coin)), img, 1.0 - eccentricity)


def _quads(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(contours, key=cv2.contourArea):
        if cv2.contourArea(c) < 0.002 * mask.size:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            yield approx.reshape(4, 2).astype(np.float32)


def _order(q: np.ndarray) -> np.ndarray:
    """Corners as top-left, top-right, bottom-right, bottom-left."""
    s, d = q.sum(axis=1), np.diff(q, axis=1).ravel()
    return np.array([q[np.argmin(s)], q[np.argmin(d)], q[np.argmax(s)], q[np.argmax(d)]], np.float32)


def _sides(q: np.ndarray) -> tuple[float, float]:
    tl, tr, br, bl = _order(q)
    horizontal = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    vertical = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    return max(horizontal, vertical), min(horizontal, vertical)


def _aspect_ok(q: np.ndarray, long_mm: float, short_mm: float) -> bool:
    long_px, short_px = _sides(q)
    want = long_mm / short_mm
    return abs(long_px / short_px - want) <= ASPECT_TOLERANCE * want


def _card(img: np.ndarray, size) -> RefScale | MvAbstain:
    """Smallest quadrilateral with the ID-1 aspect ratio."""
    long_mm, short_mm = size
    for q in _quads(foreground(img)):
        if _aspect_ok(q, long_mm, short_mm):
            long_px, short_px = _sides(q)
            x, y, w, h = cv2.boundingRect(q.astype(np.int32))
            return RefScale("card", (long_mm / long_px + short_mm / short_px) / 2, _padded(x, y, w, h), img, 0.9)
    return MvAbstain(stage="dimensions", reason="card_not_found",
                     remedy="Place the card flat next to the part, fully in frame, and retake.")


def _a4(img: np.ndarray) -> RefScale | MvAbstain:
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    for q in sorted(_quads(bright), key=lambda q: -cv2.contourArea(q)):
        if _aspect_ok(q, 297.0, 210.0):
            return _rectify(img, q)
    return MvAbstain(stage="dimensions", reason="sheet_not_found",
                     remedy="Put the part on an A4 sheet with all four corners in frame, and retake.")


def _rectify(img: np.ndarray, q: np.ndarray) -> RefScale:
    """Warp the sheet to 5 px/mm, which also removes perspective, then drop its border."""
    tl, tr, br, bl = _order(q)
    landscape = np.linalg.norm(tr - tl) >= np.linalg.norm(bl - tl)
    w_mm, h_mm = (297.0, 210.0) if landscape else (210.0, 297.0)
    w, h = round(w_mm * A4_PX_PER_MM), round(h_mm * A4_PX_PER_MM)
    m = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl]),
                                    np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32))
    inset = int(3 * A4_PX_PER_MM)
    warped = cv2.warpPerspective(img, m, (w, h))[inset:-inset, inset:-inset].copy()
    return RefScale("a4", 1 / A4_PX_PER_MM, None, warped, 0.95)
