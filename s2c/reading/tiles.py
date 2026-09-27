"""One tiled image holding every crop, so a batch reader sends one image instead of one per crop."""
from __future__ import annotations

import cv2
import numpy as np


def tile_grid(images: list[np.ndarray], cols: int = 4, cell_h: int = 80, max_w: int = 320) -> bytes:
    """One white image, one numbered cell per crop (BGR arrays), PNG-encoded.
    The number `#k` (1-based) sits in a header strip above the ink, never on it."""
    if not images:
        raise ValueError("tile_grid needs at least one image")
    header = 26
    tiles = []
    for image in images:
        h, w = image.shape[:2]
        k = cell_h / max(h, 1)
        tiles.append(cv2.resize(image, (min(max_w, max(1, round(w * k))), cell_h)))
    cell_w = max_w + 16
    rows = (len(tiles) + cols - 1) // cols
    grid = np.full((rows * (cell_h + header + 12), cols * cell_w, 3), 255, np.uint8)
    for i, tile in enumerate(tiles):
        r, col = divmod(i, cols)
        y, x = r * (cell_h + header + 12), col * cell_w
        cv2.putText(grid, f"#{i + 1}", (x + 4, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 0, 0), 2)
        grid[y + header: y + header + cell_h, x + 8: x + 8 + tile.shape[1]] = tile
        cv2.rectangle(grid, (x + 4, y + header - 2), (x + cell_w - 4, y + header + cell_h + 2),
                      (180, 180, 180), 1)
    _, buf = cv2.imencode(".png", grid)
    return buf.tobytes()
