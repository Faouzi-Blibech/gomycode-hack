import cv2
import numpy as np

from s2c.silhouette import input_silhouette, iou, normalize_mask


def sketch_of_rect(w=300, h=200, thickness=3):
    img = np.full((600, 800, 3), 255, np.uint8)
    cv2.rectangle(img, (250, 200), (250 + w, 200 + h), (0, 0, 0), thickness)
    return img


def test_input_silhouette_fills_a_drawn_rectangle():
    mask = input_silhouette(sketch_of_rect(), px=256)
    assert mask.shape == (256, 256)
    assert set(np.unique(mask)) <= {0, 255}
    # a 3:2 rectangle padded to square fills 2/3 of the rows fully
    assert 0.6 < (mask == 255).mean() < 0.7


def test_iou_is_scale_invariant():
    small = input_silhouette(sketch_of_rect(150, 100), px=256)
    big = input_silhouette(sketch_of_rect(450, 300), px=256)
    assert iou(small, big) > 0.95


def test_normalisation_makes_disjoint_bands_match_but_raw_masks_score_zero():
    a = np.zeros((256, 256), np.uint8); a[:128] = 255
    b = np.zeros((256, 256), np.uint8); b[128:] = 255
    # normalize_mask crops each band to its own bounding box and rescales it
    # to fill the full px-by-px square, so both bands become an identical
    # full square regardless of where they sat originally.
    assert iou(normalize_mask(a), normalize_mask(b)) > 0.9  # both normalise to a full band
    assert iou(a, b) == 0.0
