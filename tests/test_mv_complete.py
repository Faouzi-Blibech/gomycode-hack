import numpy as np

from s2c.multiview.build import build
from s2c.multiview.complete import complete, rotations
from s2c.multiview.raster import Mesh, face_mask, iou, outline_mask, solid_mesh
from s2c.multiview.spec import Envelope, Outline
from tests.mv_helpers import box_mesh, make_spec, outline

ENV = Envelope(x_mm=50.0, y_mm=30.0, z_mm=20.0)
FRONT_L = [(0, 0), (50, 0), (50, 30), (44, 30), (44, 6), (0, 6)]
TOP_TAPER = [(0, 5), (50, 0), (50, 20), (0, 15)]
IMAGE = np.zeros((4, 4, 3), np.uint8)


def true_mesh():
    return solid_mesh(build(make_spec((50.0, 30.0, 20.0), front=outline(FRONT_L), top=outline(TOP_TAPER))))


def front_only():
    return {"front": Outline.model_validate(outline(FRONT_L))}


def test_24_rotations():
    rs = rotations()
    assert len(rs) == 24 and len({r.tobytes() for r in rs}) == 24
    assert all(round(np.linalg.det(r)) == 1 for r in rs)


def test_the_predicted_top_matches_the_true_part_in_any_orientation():
    mesh = true_mesh()
    target, _ = face_mask(mesh, "front", ENV)
    turned = Mesh(mesh.vertices @ rotations()[7].T, mesh.faces)
    result, _, _ = complete(front_only(), ENV, "front", target, IMAGE, lambda img: turned)
    top = result["top"]
    assert top.source == "inferred" and result["right"].source == "inferred"
    assert iou(outline_mask(top.outer, top.inner, 50, 20), outline_mask(TOP_TAPER, [], 50, 20)) > 0.95


def test_a_failing_provider_falls_back_to_rectangles():
    target, _ = face_mask(true_mesh(), "front", ENV)

    def broken(img):
        raise RuntimeError("no GPU")

    result, warnings, _ = complete(front_only(), ENV, "front", target, IMAGE, broken)
    assert result["top"].source == "assumed" and result["right"].source == "assumed"
    assert "3D predictor unavailable" in warnings


def test_an_unlike_mesh_is_unreliable():
    target, _ = face_mask(true_mesh(), "front", ENV)
    result, warnings, _ = complete(front_only(), ENV, "front", target, IMAGE, lambda img: box_mesh(50, 30, 20))
    assert result["top"].source == "assumed" and "predicted view unreliable" in warnings


def test_a_rejected_face_is_assumed_and_the_mesh_is_reused():
    mesh = true_mesh()
    target, _ = face_mask(mesh, "front", ENV)
    calls = []

    def provider(img):
        calls.append(1)
        return mesh

    result, _, cached = complete(front_only(), ENV, "front", target, IMAGE, provider, rejected=("right",))
    assert result["right"].source == "assumed" and result["top"].source == "inferred"
    complete(front_only(), ENV, "front", target, IMAGE, provider, mesh=cached)
    assert calls == [1]
