# tests/test_web_api.py
import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from s2c.multiview.pipeline import MvPipeline
from s2c.web.api import get_pipeline
from s2c.web.server import app

SK = Path(__file__).resolve().parents[1] / "examples" / "mv" / "sketches"
app.dependency_overrides[get_pipeline] = lambda: MvPipeline()
c = TestClient(app, raise_server_exceptions=False)


def analyze(values=True):
    files = [("files", ("front.png", (SK / "front.png").read_bytes(), "image/png")),
             ("files", ("top.png", (SK / "top.png").read_bytes(), "image/png"))]
    r = c.post("/api/analyze", files=files, data={"faces": json.dumps(["front", "top"]),
                                                  "kinds": json.dumps(["sketch", "sketch"]), "reference": ""})
    assert r.status_code == 202, r.text
    jid = r.json()["job_id"]
    for _ in range(200):
        job = c.get(f"/api/jobs/{jid}").json()
        if job["status"] != "running":
            return job
        time.sleep(0.05)
    raise AssertionError("job never finished")


def test_status_and_examples():
    assert "providers" in c.get("/api/status").json()
    ex = c.get("/api/examples").json()
    assert {e["face"] for e in ex} == {"front", "top"}
    assert c.get(ex[0]["url"]).status_code == 200
    assert c.get("/api/examples/..%2Fpyproject.toml").status_code == 404


def test_analyze_job_reports_real_stages_and_images():
    job = analyze()
    assert job["status"] == "done", job
    assert [s["key"] for s in job["stages"]] == ["label", "outline", "read", "draw", "fuse"]
    assert job["images"][0]["outline"] and job["images"][0]["width"] > 0
    assert job["coverage"]["front"] == "observed"
    res = job["result"]
    assert res["request_id"] == job["job_id"]
    assert res["spec"] or res["abstain"]


def test_merge_with_sizes_gives_a_spec_then_model_and_export():
    job = analyze()
    m = c.post("/api/merge", json={"request_id": job["job_id"], "accepted": [], "rejected": [],
                                    "user_values": {"envelope.x_mm": 50, "envelope.y_mm": 30, "envelope.z_mm": 20}}).json()
    assert m["spec"], m
    assert m["spec"]["provenance"]["envelope.x_mm"] in ("user_written", "user_edited")
    geo = {"snap": True, "clearance": "medium", "finish": "none", "finish_mm": 1.0, "finish_edges": "all_vertical"}
    model = c.post("/api/model", json={"request_id": job["job_id"], "spec": m["spec"], "geometry": geo}).json()
    assert model["glb_url"] and c.get(model["glb_url"]).status_code == 200
    assert model["volume_cm3"] > 0 and len(model["bbox_mm"]) == 3
    ex = c.post("/api/export", json={"spec": m["spec"], "settings": {"export": {"formats": ["stl", "step"]}}}).json()
    assert set(ex["files"]) >= {"stl", "step"} and ex["files"]["stl"]["size_bytes"] > 0
    assert c.get(ex["files"]["stl"]["url"]).status_code == 200


def test_unknown_job_and_request_are_plain_404s():
    r = c.get("/api/jobs/" + "0" * 32)
    assert r.status_code == 404 and "error" in r.json()
    r = c.post("/api/merge", json={"request_id": "0" * 32, "user_values": {}, "accepted": [], "rejected": []})
    assert r.status_code == 404 and "Traceback" not in r.text


def test_upload_limits():
    files = [("files", ("x.png", b"#!/bin/sh\n", "image/png"))]
    assert c.post("/api/analyze", files=files, data={"faces": "[]", "kinds": "[]"}).status_code == 415
    assert c.post("/api/analyze", files=[], data={"faces": "[]", "kinds": "[]"}).status_code in (400, 422)


def test_artifact_traversal_is_404():
    assert c.get("/api/artifacts/" + "a" * 20 + "/..%2F..%2Fpyproject.toml").status_code == 404


def _flat_abstain(ab):
    assert ab["partial"] is None or all(isinstance(v, (int, float)) for v in ab["partial"].values()), ab
    assert isinstance(ab["missing"], list) and isinstance(ab["suggested"], dict)
    assert all(isinstance(v, (int, float)) for v in ab["suggested"].values())


def test_missing_x_abstain_has_a_flat_partial_and_the_missing_paths():
    job = analyze()
    ab = job["result"]["abstain"]
    assert ab and ab["reason"] == "missing_x"
    _flat_abstain(ab)
    assert "envelope.x_mm" in ab["missing"]
    m = c.post("/api/merge", json={"request_id": job["job_id"], "user_values": {}, "accepted": [], "rejected": []})
    ab = m.json()["abstain"]
    _flat_abstain(ab)
    assert "envelope.x_mm" in ab["missing"]


def test_read_stage_names_the_reader_that_runs():
    from s2c.web import jobs
    job = jobs.new_job(1, MvPipeline(reader=lambda *a, **k: None))
    read = job.stage("read")
    assert (read["tool"], read["ai"], read["state"]) == ("TrOCR", True, "pending")
    job = jobs.new_job(1, MvPipeline(batch_reader=lambda *a, **k: None))
    assert job.stage("read")["tool"] == "Qwen-VL"
    assert jobs.new_job(1, MvPipeline()).stage("read")["state"] == "skipped"


def _rotated_jpeg() -> bytes:
    """A 400 x 200 JPEG whose EXIF says 'rotate 90 degrees clockwise to display' (orientation 6)."""
    import io

    from PIL import Image, ImageDraw

    im = Image.new("RGB", (400, 200), "white")
    ImageDraw.Draw(im).rectangle([60, 40, 340, 160], outline="black", width=6)
    exif = Image.Exif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=exif)
    return buf.getvalue()


def test_upright_applies_the_exif_orientation():
    import io

    from PIL import Image

    from s2c.web.api import upright

    out = upright(_rotated_jpeg())
    im = Image.open(io.BytesIO(out))
    assert im.size == (200, 400)
    assert im.getexif().get(0x0112, 1) == 1
    plain = (SK / "front.png").read_bytes()
    assert upright(plain) is plain


def test_a_rotated_phone_jpeg_is_measured_the_way_the_browser_shows_it():
    files = [("files", ("phone.jpg", _rotated_jpeg(), "image/jpeg"))]
    r = c.post("/api/analyze", files=files, data={"faces": json.dumps(["front"]), "kinds": json.dumps(["sketch"])})
    assert r.status_code == 202, r.text
    jid = r.json()["job_id"]
    for _ in range(200):
        job = c.get(f"/api/jobs/{jid}").json()
        if job["status"] != "running":
            break
        time.sleep(0.05)
    img = job["images"][0]
    assert img["height"] == 2 * img["width"] > 0, img  # portrait, as the browser shows it
