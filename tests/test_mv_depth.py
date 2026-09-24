import time
import tracemalloc

import cv2
import numpy as np
import pytest

from s2c.multiview.depth import apply_depth, hole_depths, read_depth, solaria_depth
from s2c.multiview.fuse import Observation, features_from
from s2c.multiview.outline import PixelCircle, PixelOutline, extract
from s2c.multiview.spec import Envelope
from tests.test_mv_qwen_image import FakeJob


def solaria_ply(path, depth, stride=4):
    """A point cloud written exactly like Solaria's pointcloud.py: ASCII, normalised x and y, z scaled to 0..2."""
    h, w = depth.shape
    y, x = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[::stride, ::stride]
    z = (z - z.min()) / max(float(z.max() - z.min()), 1e-8) * 2.0
    xs, ys = (x - w / 2) / w, -(y - h / 2) / h
    header = ("ply\nformat ascii 1.0\nelement vertex {n}\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write(header.format(n=z.size))
        f.writelines(f"{a:.6f} {b:.6f} {c:.6f} 128 128 128\n" for a, b, c in zip(xs.ravel(), ys.ravel(), z.ravel()))


def plate_photo():
    img = np.full((1200, 1600, 3), 200, np.uint8)
    cv2.rectangle(img, (500, 400), (1100, 800), (40, 40, 40), -1)
    for cx in (650, 950):
        cv2.circle(img, (cx, 600), 40, (200, 200, 200), -1)
    return img


def scene(blind_ratio=0.4, table=10.0, face=0.0):
    """Depth of plate_photo: table, part face, a through hole at x 650 and a blind hole at x 950."""
    depth = np.full((1200, 1600), table, np.float32)
    depth[400:801, 500:1101] = face
    yy, xx = np.mgrid[0:1200, 0:1600]
    depth[(xx - 650) ** 2 + (yy - 600) ** 2 <= 40 ** 2] = table
    depth[(xx - 950) ** 2 + (yy - 600) ** 2 <= 40 ** 2] = face + blind_ratio * (table - face)
    return depth


def plate(**kw):
    return Observation(face="top", kind="photo", outline=extract(plate_photo()), **kw)


def _peak_bytes(depth, outline):
    tracemalloc.start()
    try:
        hole_depths(depth, outline)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_depth_masks_are_small():
    """Per-circle stats must come from a bounding-box crop, not a full-frame array kept per circle."""
    depth = np.zeros((1200, 1600), np.float32)

    def outline_with(n):
        return PixelOutline(outer=[(0, 0), (1599, 0), (1599, 1199), (0, 1199)], bbox=(0, 0, 1600, 1200),
                            shape=(1200, 1600), circles=[PixelCircle(80 + 140 * i, 300, 40) for i in range(n)])

    one, ten = _peak_bytes(depth, outline_with(1)), _peak_bytes(depth, outline_with(10))
    assert ten - one < 20 * 1024 * 1024


def test_the_point_cloud_turns_back_into_a_depth_map(tmp_path):
    depth = np.random.default_rng(0).random((120, 160)).astype(np.float32)
    solaria_ply(tmp_path / "c.ply", depth)
    got = read_depth(tmp_path / "c.ply", 160, 120)
    want = (depth[::4, ::4] - depth[::4, ::4].min()) / (depth[::4, ::4].max() - depth[::4, ::4].min()) * 2
    assert got.shape == (30, 40) and np.allclose(got, want, atol=1e-5)


def test_through_and_blind_holes_from_a_depth_map():
    o = plate()
    assert apply_depth(o, scene(0.4)) == []
    assert o.blind == {0: False, 1: True} and o.depth_ratio[1] == pytest.approx(0.4, abs=0.02)
    assert o.depth_from_image == {0, 1}


def test_ratios_do_not_depend_on_scale_shift_or_direction():
    o = plate()
    apply_depth(o, 3.0 - 0.25 * scene(0.4))
    assert o.depth_ratio[1] == pytest.approx(0.4, abs=0.02)


def test_a_flat_scene_is_not_measured():
    o = plate(blind={1: True})
    warnings = apply_depth(o, scene(0.4, table=0.0))
    assert o.depth_from_image == set() and o.blind == {1: True}
    assert warnings == ["top: depth: part too flat to measure, check hole depths"]


def test_a_shallow_circle_is_a_mark():
    o = plate()
    warnings = apply_depth(o, scene(0.05))
    assert 1 not in o.depth_from_image and "top: hole 2 looks like a mark, not a hole" in warnings


def test_fuse_turns_the_ratio_into_millimetres():
    o = Observation(face="front", kind="photo", outline=extract(plate_photo()), blind={1: True},
                    depth_ratio={1: 0.4})
    feats, prov = features_from([o], Envelope(x_mm=60.0, y_mm=40.0, z_mm=10.0))
    assert feats[0]["depth_mm"] is None and feats[1]["depth_mm"] == pytest.approx(4.0)
    assert prov["features[1].depth_mm"] == "estimated"


def test_the_solaria_provider_sends_a_white_mask_and_returns_depth_at_image_size(tmp_path):
    ply = tmp_path / "cloud.ply"

    class Fake:
        def __init__(self, src, token=None, **kwargs):
            pass

        def submit(self, image, mask, api_name):
            m = cv2.imread(mask["path"], cv2.IMREAD_GRAYSCALE)
            assert api_name == "/gerar_3d" and m.min() == 255
            h, w = m.shape
            solaria_ply(ply, np.tile(np.linspace(0, 1, w, dtype=np.float32), (h, 1)))
            return FakeJob((str(tmp_path / "d.png"), str(ply), "ok"))

    depth = solaria_depth("CronosSa/Solaria1.0", None, Fake, log_path=tmp_path / "l")(np.zeros((1200, 1600, 3), np.uint8))
    assert depth.shape == (1200, 1600) and np.nanmax(depth) == pytest.approx(2.0, abs=0.01)


def test_a_failing_space_raises(tmp_path):
    class Down:
        def __init__(self, *a, **k):
            pass

        def submit(self, *a, **k):
            return FakeJob((None, None, "Erro durante a geração 3D"))

    with pytest.raises(RuntimeError):
        solaria_depth("s", None, Down, log_path=tmp_path / "l")(np.zeros((100, 100, 3), np.uint8))


def test_a_slow_solaria_times_out(tmp_path):
    jobs = []

    class Slow:
        def __init__(self, *a, **k):
            pass

        def submit(self, *a, **k):
            jobs.append(FakeJob(delay=5))
            return jobs[-1]

    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        solaria_depth("s", None, Slow, timeout_s=0.2, log_path=tmp_path / "l")(np.zeros((100, 100, 3), np.uint8))
    assert time.monotonic() - t0 < 1.5
    assert jobs[0].cancelled
