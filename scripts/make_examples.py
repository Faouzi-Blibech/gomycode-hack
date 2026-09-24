"""Draw the demo sketches for the Studio's "Try an example": the L-bracket of examples/mv/l_bracket.json, front and
top views, in pen, with the dimensions written the way a user would. Run: uv run python scripts/make_examples.py"""
import json
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "examples" / "mv" / "sketches"
PX = 12  # pixels per millimetre
INK = (40, 40, 40)


def canvas():
    """Off-white paper, a little darker towards the corners like a phone photo of a sheet. Smooth on purpose: the
    per-pixel noise this used to add made each PNG about 1.9 MB; this keeps them near 40 KB."""
    h, w = 1000, 1400
    yy, xx = np.mgrid[0:h, 0:w]
    shade = 250 - 12 * (((xx - w / 2) / w) ** 2 + ((yy - h / 2) / h) ** 2)
    return np.repeat(np.round(shade).astype(np.uint8)[:, :, None], 3, axis=2)


def to_px(points, x0, y0, height_mm):
    return np.array([(x0 + a * PX, y0 + (height_mm - b) * PX) for a, b in points], np.int32)


def write(img, text, at):
    cv2.putText(img, text, at, cv2.FONT_HERSHEY_SIMPLEX, 1.6, INK, 3, cv2.LINE_AA)


def front():
    img = canvas()
    x0, y0 = 400, 250
    pts = to_px([(0, 0), (50, 0), (50, 30), (47, 30), (47, 3), (0, 3)], x0, y0, 30)
    cv2.polylines(img, [pts], True, INK, 4, cv2.LINE_AA)
    write(img, "50", (x0 + 25 * PX - 30, y0 + 30 * PX + 90))
    write(img, "30", (x0 + 50 * PX + 60, y0 + 15 * PX + 15))
    return img


def top():
    img = canvas()
    x0, y0 = 400, 300
    cv2.polylines(img, [to_px([(0, 0), (50, 0), (50, 20), (0, 20)], x0, y0, 20)], True, INK, 4, cv2.LINE_AA)
    cv2.circle(img, (x0 + 20 * PX, y0 + (20 - 10) * PX), round(2.75 * PX), INK, 3, cv2.LINE_AA)
    write(img, "20", (x0 - 110, y0 + 10 * PX + 15))
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "front.png"), front())
    cv2.imwrite(str(OUT / "top.png"), top())
    (OUT / "examples.json").write_text(json.dumps([
        {"file": "front.png", "face": "front", "kind": "sketch"},
        {"file": "top.png", "face": "top", "kind": "sketch"}], indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
