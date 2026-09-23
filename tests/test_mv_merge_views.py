import cv2
import numpy as np
import pytest

from s2c.multiview.fuse import Observation
from s2c.multiview.merge_views import PAD, merge_same_face
from s2c.multiview.ocr import Linked, Reading
from s2c.multiview.outline import extract
from s2c.multiview.raster import iou, polygon_mask

HOLES = [(100, 100), (450, 300)]  # centres in plate pixels, radius 30; not symmetric under a half turn
SX, SY = 511 / 600, 340 / 400     # plate bbox (601 x 401 px) onto the 512 x 341 grid


def photo(shift=(0, 0), angle=0.0, size=(600, 400), holes=HOLES, notch=False, seed=0):
    """A dark plate with light holes on a light table, like a top-down photo."""
    img = np.full((1200, 1600, 3), 200, np.uint8)
    x0, y0 = 500 + shift[0], 400 + shift[1]
    cv2.rectangle(img, (x0, y0), (x0 + size[0], y0 + size[1]), (40, 40, 40), -1)
    if notch:
        cv2.rectangle(img, (x0 + 150, y0), (x0 + size[0], y0 + 300), (200, 200, 200), -1)
    for cx, cy in holes:
        cv2.circle(img, (x0 + cx, y0 + cy), 30, (200, 200, 200), -1)
    if angle:
        m = cv2.getRotationMatrix2D((800, 600), angle, 1.0)
        img = cv2.warpAffine(img, m, (1600, 1200), borderValue=(200, 200, 200))
    noise = np.random.default_rng(seed).normal(0, 6, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def obs(img, confidence=0.9, values=(), mm_per_px=None):
    return Observation(face="top", kind="photo", outline=extract(img), values=list(values), mm_per_px=mm_per_px,
                       confidence=confidence)


def linear(value, axis="a"):
    return Linked(Reading(float(value), "linear", (800, 850, 40, 30), 0.9, str(value)), axis, None)


def test_a_single_photo_passes_through_unchanged():
    img = photo()
    o = obs(img)
    merged, images, warnings = merge_same_face([o], [img])
    assert merged[0] is o and images[0] is img and warnings == []


def test_three_photos_vote_one_clean_outline_and_drop_the_outlier():
    imgs = [photo(), photo(shift=(20, -15), seed=1), photo(angle=3, seed=2), photo(notch=True, holes=[], seed=3)]
    observations = [obs(imgs[0], 0.95), obs(imgs[1]), obs(imgs[2]), obs(imgs[3])]
    (merged,), (image,), warnings = merge_same_face(observations, imgs)
    assert image is imgs[0]
    assert "top: photo 4 disagrees with the others, ignored" in warnings
    assert any(w.startswith("top: merged 3 photos") for w in warnings)
    shape = merged.outline.shape
    truth = polygon_mask([(PAD, PAD), (PAD + 511, PAD), (PAD + 511, PAD + 340), (PAD, PAD + 340)], (), shape)
    assert iou(polygon_mask(merged.outline.outer, merged.outline.inner, shape), truth) >= 0.97
    want = [(PAD + cx * SX, PAD + cy * SY) for cx, cy in HOLES]
    got = [(c.cx, c.cy) for c in merged.outline.circles]
    diag = float(np.hypot(512, 341))
    assert len(got) == 2
    assert all(np.hypot(g[0] - w[0], g[1] - w[1]) < 0.01 * diag for g, w in zip(got, want))
    ref_d = observations[0].outline.circles[0].d * np.sqrt(SX * SY)
    assert merged.outline.circles[0].d == pytest.approx(ref_d, rel=0.03)


def test_a_photo_of_the_wrong_proportions_is_dropped():
    imgs = [photo(), photo(shift=(20, 0), seed=1), photo(size=(600, 150), holes=[], seed=2)]
    (merged,), _, warnings = merge_same_face([obs(imgs[0], 0.95), obs(imgs[1]), obs(imgs[2])], imgs)
    assert "top: photo 3 disagrees with the others, ignored" in warnings
    assert len(merged.outline.circles) == 2


def test_a_photo_turned_upside_down_does_not_add_holes():
    imgs = [photo(), photo(shift=(20, 0), seed=1), photo(angle=180, seed=2)]
    (merged,), _, _ = merge_same_face([obs(imgs[0], 0.95), obs(imgs[1]), obs(imgs[2])], imgs)
    assert len(merged.outline.circles) == 2


def test_values_keep_what_most_photos_read_and_warn_about_the_rest():
    imgs = [photo(), photo(shift=(20, 0), seed=1), photo(shift=(-20, 0), seed=2)]
    observations = [obs(imgs[0], 0.95, [linear(60), linear(40, "b")]), obs(imgs[1], values=[linear(60)]),
                    obs(imgs[2], values=[linear(66)])]
    (merged,), _, warnings = merge_same_face(observations, imgs)
    assert [lv.reading.value_mm for lv in merged.values if lv.axis == "a"] == [60.0]
    assert [lv.reading.value_mm for lv in merged.values if lv.axis == "b"] == [40.0]
    assert any("66" in w for w in warnings)


def test_scale_is_the_median_and_a_spread_warns():
    imgs = [photo(), photo(shift=(20, 0), seed=1), photo(shift=(-20, 0), seed=2)]
    observations = [obs(imgs[0], 0.95, mm_per_px=0.1), obs(imgs[1], mm_per_px=0.1), obs(imgs[2], mm_per_px=0.104)]
    (merged,), _, warnings = merge_same_face(observations, imgs)
    _, _, w, _ = merged.outline.bbox
    assert (w - 1) * merged.mm_per_px == pytest.approx(60.0, rel=0.01)
    assert "top: scale varies between photos, check the reference object" in warnings


def test_a_solaria_result_wins_over_the_labels():
    imgs = [photo(), photo(shift=(20, 0), seed=1), photo(shift=(-20, 0), seed=2)]
    observations = [obs(imgs[0], 0.95), obs(imgs[1]), obs(imgs[2])]
    for o in observations[1:]:
        o.blind = {0: False, 1: False}
    observations[0].blind, observations[0].depth_ratio, observations[0].depth_from_image = {0: True}, {0: 0.4}, {0}
    (merged,), _, _ = merge_same_face(observations, imgs)
    upper = min(range(2), key=lambda k: merged.outline.circles[k].cy)
    assert merged.blind[upper] and merged.depth_ratio[upper] == 0.4 and upper in merged.depth_from_image


@pytest.mark.parametrize("angles", [(0, 180), (0, 0, 180, 180)], ids=["two photos", "four photos"])
def test_a_half_turned_photo_is_turned_back_before_voting(angles):
    imgs = [photo(angle=a, seed=k) for k, a in enumerate(angles)]
    observations = [obs(img, 0.95 if k == 0 else 0.9) for k, img in enumerate(imgs)]
    (merged,), _, _ = merge_same_face(observations, imgs)
    want = [(PAD + cx * SX, PAD + cy * SY) for cx, cy in HOLES]
    got = sorted((c.cx, c.cy) for c in merged.outline.circles)
    assert len(got) == 2
    assert all(np.hypot(g[0] - w[0], g[1] - w[1]) < 0.01 * np.hypot(512, 341) for g, w in zip(got, sorted(want)))
