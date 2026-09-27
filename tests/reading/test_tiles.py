import cv2
import numpy as np

from s2c.reading.tiles import tile_grid

CROP = np.full((40, 60, 3), 255, np.uint8)


def test_the_png_decodes_to_a_bgr_image():
    png = tile_grid([CROP, CROP, CROP])
    grid = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    assert grid is not None
    assert grid.ndim == 3 and grid.shape[2] == 3


def test_a_single_crop_makes_a_one_row_one_cell_grid():
    png = tile_grid([CROP])
    grid = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    header, cell_h, cols, max_w = 26, 80, 4, 320
    cell_w = max_w + 16
    assert grid.shape[0] == 1 * (cell_h + header + 12)
    assert grid.shape[1] == cols * cell_w


def test_seventeen_crops_make_five_rows():
    png = tile_grid([CROP] * 17)
    grid = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    header, cell_h = 26, 80
    rows = -(-17 // 4)
    assert grid.shape[0] == rows * (cell_h + header + 12)
