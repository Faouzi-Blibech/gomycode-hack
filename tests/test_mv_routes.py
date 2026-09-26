from fastapi import FastAPI
from fastapi.testclient import TestClient

from s2c.multiview import routes
from s2c.multiview.pipeline import MvPipeline
from tests.test_mv_pipeline import sketch


def client(tmp_path, monkeypatch):
    monkeypatch.setattr(routes, "STORE_ROOT", tmp_path)
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[routes.get_pipeline] = lambda: MvPipeline()
    return TestClient(app)


def test_analyze_merge_build_and_download(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch)
    files = [("files", ("front.png", sketch(600, 400), "image/png")), ("files", ("top.png", sketch(600, 100), "image/png"))]
    body = c.post("/mv/analyze", files=files, data={"faces": '["front", "top"]', "kinds": '["sketch", "sketch"]'}).json()
    assert body["abstain"]["reason"] == "missing_x"  # no OCR in this pipeline, so the gate asks
    values = {"envelope.x_mm": 60, "envelope.y_mm": 40, "envelope.z_mm": 10}
    assert len(routes._requests[body["request_id"]][1].images) == 2  # still needed: no spec yet
    merged = c.post("/mv/merge", json={"request_id": body["request_id"], "user_values": values}).json()
    assert merged["filled_by"] == {"front": "observed", "top": "observed", "right": "assumed"}
    spec = merged["spec"]
    assert routes._requests[body["request_id"]][1].images == []  # dropped once the spec exists
    out = c.post("/mv/build", json={"spec": spec, "request_id": body["request_id"]}).json()
    assert out["iou"]["front"] > 0.85
    assert c.get(out["stl_url"]).status_code == 200
    assert c.get(out["step_url"]).status_code == 200
    assert c.get(out["views"]["front"]).status_code == 200


def test_unknown_requests_and_bad_file_names(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch)
    assert c.post("/mv/merge", json={"request_id": "nope"}).status_code == 404
    assert c.get("/mv/files/abc/part.stl").status_code == 404
    assert c.get("/mv/files/" + "0" * 32 + "/..%5Csecret").status_code == 404
