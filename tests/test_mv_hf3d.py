import sys

import cv2
import numpy as np
import pytest

from s2c.multiview import hf3d
from s2c.multiview.raster import Mesh

IMAGE = np.zeros((8, 8, 3), np.uint8)


def test_the_provider_tries_local_then_the_space(monkeypatch):
    calls = []

    def local(img):
        calls.append("local")
        raise RuntimeError("CUDA out of memory")

    def space(img):
        calls.append("space")
        return Mesh(np.zeros((3, 3)), np.array([[0, 1, 2]]))

    monkeypatch.setattr(hf3d, "local_triposr", local)
    monkeypatch.setattr(hf3d, "space_triposr", space)
    assert hf3d.default_provider()(IMAGE).faces.shape == (1, 3)
    assert calls == ["local", "space"]


def test_the_provider_raises_when_both_fail(monkeypatch):
    def down(img):
        raise RuntimeError("down")

    monkeypatch.setattr(hf3d, "local_triposr", down)
    monkeypatch.setattr(hf3d, "space_triposr", down)
    with pytest.raises(RuntimeError, match="down"):
        hf3d.default_provider()(IMAGE)


def test_the_marching_cubes_shim_returns_torchmcubes_axis_order():
    torch = pytest.importorskip("torch")
    pytest.importorskip("skimage")
    hf3d._install_mcubes_shim(force=True)
    n = 32
    zz, yy, xx = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    dist2 = ((xx - 20.0) ** 2 + (yy - 16.0) ** 2 + (zz - 12.0) ** 2).astype(np.float32)
    v, f = sys.modules["torchmcubes"].marching_cubes(torch.from_numpy(25.0 - dist2), 0.0)
    assert np.allclose(v.numpy().mean(axis=0), [20, 16, 12], atol=0.5)  # (x, y, z) for a volume indexed [z][y][x]
    assert f.shape[1] == 3


@pytest.mark.gpu
def test_real_triposr_on_the_gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not hf3d.REPO_DIR.exists():
        pytest.skip("needs CUDA and vendor/TripoSR")
    img = np.full((512, 512, 3), 255, np.uint8)
    cv2.rectangle(img, (150, 200), (360, 320), (40, 40, 40), -1)
    assert len(hf3d.local_triposr(img).faces) > 100
