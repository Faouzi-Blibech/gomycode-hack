"""Stand-in for the geometry owner's view renderer (s2c/views.py's
`silhouettes`). Draws a rectangle with the spec's own aspect ratio and punches
its holes out, instead of rendering real geometry. Delete once the real views
module lands.
"""
import numpy as np

VIEWS = ("front", "back", "left", "right", "top", "bottom")


def silhouettes(solid, px: int = 512) -> dict[str, np.ndarray]:
    """A filled rectangle with the aspect ratio of the front view, holes drawn if the spec has them."""
    import cv2

    from s2c.silhouette import normalize_mask
    spec = getattr(solid, "spec", None)
    w, h = 60.0, 40.0
    if spec is not None and spec.part.type == "plate":
        w, h = spec.part.width_mm, spec.part.height_mm
    scale = 400 / max(w, h)
    mask = np.zeros((int(h * scale) + 2, int(w * scale) + 2), np.uint8)
    mask[1:-1, 1:-1] = 255
    if spec is not None:
        for f in spec.features:
            if f.type == "hole":
                cv2.circle(mask, (int(f.x_mm * scale), int((h - f.y_mm) * scale)), int(f.diameter_mm * scale / 2), 0, -1)
    front = normalize_mask(mask, px)
    return {v: front for v in VIEWS}
