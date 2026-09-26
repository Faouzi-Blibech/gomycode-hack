"""Several photos of one face -> one observation: aligned masks, a per-pixel vote, median holes, values and scale.
Spec 2026-09-23 section 5. Only our own image processing; no model is called here."""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import replace

import cv2
import numpy as np

from s2c.multiview.fuse import Observation
from s2c.multiview.ocr import Linked
from s2c.multiview.outline import EDGE_BAND_PX, LONG_SIDE, PixelCircle, extract
from s2c.multiview.raster import iou, polygon_mask
from s2c.multiview.spec import MvAbstain

log = logging.getLogger(__name__)
GRID = 512
PAD = 16
EDGE_BAND_MIN_PX = 3
OUTLIER_IOU = 0.7
ASPECT_TOL = 0.10
SCALE_SPREAD = 0.03
VALUE_TOL = 0.05
CIRCLE_TOL = 0.05


def _solid_mask(o: Observation) -> np.ndarray:
    """Outer outline minus non-circular openings. Circles stay filled: they are merged separately."""
    return polygon_mask(o.outline.outer, o.outline.inner, o.outline.shape)


def _to_grid(o: Observation, gw: int, gh: int) -> np.ndarray:
    """3 x 3 affine: image pixels -> grid pixels, stretching the outline's bounding box onto the grid."""
    x, y, w, h = o.outline.bbox
    sx, sy = (gw - 1) / max(w - 1, 1), (gh - 1) / max(h - 1, 1)
    return np.array([[sx, 0.0, PAD - x * sx], [0.0, sy, PAD - y * sy], [0.0, 0.0, 1.0]])


def _warp(mask: np.ndarray, t: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.warpAffine(mask, t[:2].astype(np.float64), size, flags=cv2.INTER_NEAREST)


def _blur(mask: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 8)


def _align(ref: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """3 x 3 affine moving `mask` onto `ref`; identity when ECC does not converge."""
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(_blur(ref), _blur(mask), warp, cv2.MOTION_AFFINE,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), None, 5)
    except cv2.error as e:
        log.info("ECC did not converge: %s", e)
        return np.eye(3)
    # ECC maps ref coordinates into mask coordinates; the inverse moves mask points onto ref
    return np.vstack([cv2.invertAffineTransform(warp), [0.0, 0.0, 1.0]])


def _apply(t: np.ndarray, pts) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    return pts @ t[:2, :2].T + t[:2, 2]


def _map_box(t: np.ndarray, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x, y, w, h = box
    pts = _apply(t, [[x, y], [x + w, y + h]])
    (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
    return int(x0), int(y0), max(int(x1 - x0), 1), max(int(y1 - y0), 1)


def _linear_scale(t: np.ndarray) -> float:
    return float(np.sqrt(abs(np.linalg.det(t[:2, :2]))))


def _vote(masks: list[np.ndarray]) -> np.ndarray:
    stack = np.stack([m > 127 for m in masks])
    return np.where(stack.sum(axis=0) * 2 >= len(masks), 255, 0).astype(np.uint8)


def _half_turn(size: tuple[int, int]) -> np.ndarray:
    """3 x 3 turn by 180 degrees about the centre of a grid of `size` (width, height)."""
    w, h = size
    return np.array([[-1.0, 0.0, w - 1.0], [0.0, -1.0, h - 1.0], [0.0, 0.0, 1.0]])


def _fit(o: Observation, solid: np.ndarray, to_grid: np.ndarray, ref_grid: np.ndarray, ref_circles: np.ndarray,
         size: tuple[int, int], diag: float) -> np.ndarray:
    """Image -> reference grid, trying the photo as taken and turned upside down. Symmetric outlines look the same
    both ways, so the score adds the share of this photo's holes that land on a reference hole; ties keep as taken."""
    best, best_score = to_grid, -1.0
    for turn in (np.eye(3), _half_turn(size)):
        start = turn @ to_grid
        t = _align(ref_grid, _warp(solid, start, size)) @ start
        centres = _apply(t, [[c.cx, c.cy] for c in o.outline.circles])
        hits = sum(1 for p in centres if len(ref_circles) and
                   np.min(np.hypot(*(ref_circles - p).T)) < CIRCLE_TOL * diag)
        score = iou(_warp(solid, t, size), ref_grid) + (hits / len(centres) if len(centres) else 0.0)
        if score > best_score + 1e-6:
            best, best_score = t, score
    return best


def _merge_circles(face: str, group, transforms,
                    diag: float) -> tuple[list[PixelCircle], dict[tuple[int, int], int], list[str]]:
    """Clusters of circle centres; a cluster seen on at least half of the photos becomes one median circle."""
    clusters: list[list[tuple[int, int, float, float, float]]] = []
    for k, (o, t) in enumerate(zip(group, transforms)):
        s = _linear_scale(t)
        for i, c in enumerate(o.outline.circles):
            (cx, cy), = _apply(t, [[c.cx, c.cy]])
            member = (k, i, float(cx), float(cy), c.d * s)
            home = next((cl for cl in clusters if np.hypot(cl[0][2] - cx, cl[0][3] - cy) < CIRCLE_TOL * diag), None)
            if home is None:
                clusters.append([member])
            else:
                home.append(member)
    circles, index, warnings = [], {}, []
    for cl in clusters:
        support = len({m[0] for m in cl})
        if 2 * support < len(group):
            d = float(np.median([m[4] for m in cl]))
            warnings.append(f"{face}: a {d:.0f} px hole was seen on only {support} of {len(group)} photos, dropped")
            continue
        for m in cl:
            index[(m[0], m[1])] = len(circles)
        circles.append(PixelCircle(*(float(np.median([m[j] for m in cl])) for j in (2, 3, 4))))
    return circles, index, warnings


def _clusters(values: list[float]) -> list[list[int]]:
    groups: list[list[int]] = []
    for i, v in enumerate(values):
        home = next((g for g in groups if abs(v - values[g[0]]) <= VALUE_TOL * max(abs(values[g[0]]), 1e-9)), None)
        if home is None:
            groups.append([i])
        else:
            home.append(i)
    return groups


def _merge_values(face: str, group, transforms, circle_index) -> tuple[list[Linked], list[str]]:
    """Per axis or merged hole: keep every value read on at least half of the photos that read one there."""
    buckets: dict[tuple, list[tuple[int, Linked]]] = defaultdict(list)
    out: list[Linked] = []
    warnings = []
    for k, (o, t) in enumerate(zip(group, transforms)):
        for lv in o.values:
            reading = replace(lv.reading, bbox=_map_box(t, lv.reading.bbox))
            if lv.hole_index is not None:
                hole = circle_index.get((k, lv.hole_index))
                if hole is not None:
                    buckets[("hole", hole)].append((k, Linked(reading, None, hole)))
                else:
                    warnings.append(f"{face}: {reading.value_mm:g} mm was linked to a dropped hole, ignored")
            elif lv.axis is not None:
                buckets[("axis", lv.axis)].append((k, Linked(reading, lv.axis, None)))
            else:
                out.append(Linked(reading, None, None))
    for items in buckets.values():
        photos = len({k for k, _ in items})
        for g in _clusters([lv.reading.value_mm for _, lv in items]):
            members = [items[i][1] for i in g]
            support = len({items[i][0] for i in g})
            value = float(np.median([lv.reading.value_mm for lv in members]))
            if 2 * support < photos:
                warnings.append(f"{face}: {value:g} mm was read on only {support} of {photos} photos, ignored")
                continue
            best = max(members, key=lambda lv: lv.reading.confidence)
            out.append(Linked(replace(best.reading, value_mm=value), best.axis, best.hole_index))
    return out, warnings


def _merge_labels(group, circle_index, n_circles: int):
    """Blind flags and depths per merged circle. Rule 2: only a Solaria result may mark a hole blind;
    a vision-model label's blind flag, voted or not, never does."""
    blind, ratio, from_image = {}, {}, set()
    for j in range(n_circles):
        members = [(k, i) for (k, i), c in circle_index.items() if c == j]
        solaria = [(k, i) for k, i in members if i in group[k].depth_from_image]
        if solaria:
            k, i = solaria[0]
            blind[j] = group[k].blind.get(i, False)
            from_image.add(j)
            if i in group[k].depth_ratio:
                ratio[j] = group[k].depth_ratio[i]
            continue
        blind[j] = False
    return blind, ratio, from_image


def _merged_scale(face: str, group, transforms) -> tuple[float | None, list[str]]:
    scales = [o.mm_per_px / _linear_scale(t) for o, t in zip(group, transforms) if o.mm_per_px]
    if not scales:
        return None, []
    median = float(np.median(scales))
    if (max(scales) - min(scales)) / median > SCALE_SPREAD:
        return median, [f"{face}: scale varies between photos, check the reference object"]
    return median, []


def _merge_group(face: str, group: list[Observation], images: list[np.ndarray], idx: list[int] | None = None):
    idx = list(range(len(group))) if idx is None else idx
    ratios = [o.outline.bbox[2] / o.outline.bbox[3] for o in group]
    median = float(np.median(ratios))
    gw, gh = (GRID, max(round(GRID / median), 8)) if median >= 1 else (max(round(GRID * median), 8), GRID)
    size = (gw + 2 * PAD, gh + 2 * PAD)
    ref = max(range(len(group)), key=lambda k: group[k].confidence)
    to_grid = [_to_grid(o, gw, gh) for o in group]
    solids = [_solid_mask(o) for o in group]
    ref_grid = _warp(solids[ref], to_grid[ref], size)
    ref_circles = _apply(to_grid[ref], [[c.cx, c.cy] for c in group[ref].outline.circles])
    diag = float(np.hypot(gw, gh))
    transforms = [g if k == ref else _fit(group[k], solids[k], g, ref_grid, ref_circles, size, diag)
                  for k, g in enumerate(to_grid)]
    masks = [_warp(s, t, size) for s, t in zip(solids, transforms)]
    first = _vote(masks)
    keep = [k for k in range(len(group))
            if abs(ratios[k] / median - 1) <= ASPECT_TOL and iou(masks[k], first) >= OUTLIER_IOU]
    warnings = [f"{face}: photo {idx[k] + 1} disagrees with the others, ignored"
                for k in range(len(group)) if k not in keep]
    if len(keep) < 2:
        best = max(keep or [ref], key=lambda k: group[k].confidence)
        return group[best], images[best], warnings
    if ref not in keep:
        merged, image, more = _merge_group(face, [group[k] for k in keep], [images[k] for k in keep],
                                           [idx[k] for k in keep])
        return merged, image, warnings + more
    kept, kept_t = [group[k] for k in keep], [transforms[k] for k in keep]
    vote = _vote([masks[k] for k in keep])
    agreement = float(np.mean([iou(masks[k], vote) for k in keep]))
    band = max(round(EDGE_BAND_PX * GRID / LONG_SIDE), EDGE_BAND_MIN_PX)
    outline = extract(cv2.cvtColor(255 - vote, cv2.COLOR_GRAY2BGR), band=band)
    if isinstance(outline, MvAbstain):
        return group[ref], images[ref], warnings + [f"{face}: photos could not be merged, using the clearest one"]
    circles, circle_index, circle_warnings = _merge_circles(face, kept, kept_t, diag)
    voted_only = [c for c in outline.circles
                  if not any(np.hypot(c.cx - e.cx, c.cy - e.cy) < CIRCLE_TOL * diag for e in circles)]
    outline = replace(outline, circles=circles + voted_only)
    circle_warnings += [f"{face}: an opening became round after merging {len(kept)} photos, check it"
                        for _ in voted_only]
    values, more = _merge_values(face, kept, kept_t, circle_index)
    blind, ratio, from_image = _merge_labels(kept, circle_index, len(outline.circles))
    scale, scale_warnings = _merged_scale(face, kept, kept_t)
    merged = Observation(face=face, kind=group[ref].kind, outline=outline, values=values, mm_per_px=scale,
                         blind=blind, depth_ratio=ratio, depth_from_image=from_image,
                         confidence=round(max(o.confidence for o in kept) * agreement, 3))
    warnings += circle_warnings + more + scale_warnings + [f"{face}: merged {len(kept)} photos, agreement {agreement:.2f}"]
    return merged, images[ref], warnings


def merge_same_face(observations: list[Observation], images: list[np.ndarray]):
    """One observation per face tag, with the reference photo of each, and the warnings."""
    by_face: dict[str, list[int]] = defaultdict(list)
    for k, o in enumerate(observations):
        by_face[o.face].append(k)
    out_obs, out_images, warnings = [], [], []
    for face, idx in by_face.items():
        if len(idx) == 1:
            out_obs.append(observations[idx[0]])
            out_images.append(images[idx[0]])
            continue
        merged, image, more = _merge_group(face, [observations[k] for k in idx], [images[k] for k in idx])
        out_obs.append(merged)
        out_images.append(image)
        warnings += more
    return out_obs, out_images, warnings
