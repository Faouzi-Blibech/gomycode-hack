"""Input image to a normalised binary silhouette. Shared by the API and views."""
import cv2
import numpy as np


def normalize_mask(mask: np.ndarray, px: int = 512) -> np.ndarray:
    """Crop to the bounding box of nonzero pixels, pad to a square, resize to px."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((px, px), np.uint8)
    crop = mask[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    h, w = crop.shape
    side = max(h, w)
    square = np.zeros((side, side), np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    square[y0: y0 + h, x0: x0 + w] = crop
    out = cv2.resize(square, (px, px), interpolation=cv2.INTER_NEAREST)
    return np.where(out > 127, 255, 0).astype(np.uint8)


def input_silhouette(image_bgr: np.ndarray, px: int = 512) -> np.ndarray:
    """Largest closed outline in the image, filled. Works for pen sketches and top-down photos."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("no outline found")
    outline = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.fillPoly(mask, [outline], 255)
    return normalize_mask(mask, px)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a_on, b_on = a > 127, b > 127
    union = np.logical_or(a_on, b_on).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a_on, b_on).sum() / union)
