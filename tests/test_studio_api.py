import json
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from s2c.multiview import artifacts, routes

SPEC = json.loads((Path(__file__).parents[1] / "examples" / "mv" / "l_bracket.json").read_text())


def client(tmp_path, monkeypatch):
    artifacts.clear_cache()
    monkeypatch.setattr(routes, "ARTIFACT_ROOT", tmp_path)
    monkeypatch.setattr("s2c.multiview.slice.find_slicer", lambda: None)
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def test_export_returns_downloadable_files_and_a_zip(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch)
    body = c.post("/mv/export", json={"spec": SPEC, "settings": {"export": {"formats": ["stl", "dxf"]}}}).json()
    assert set(body["files"]) == {"stl", "dxf"}
    assert c.get(body["files"]["stl"]).status_code == 200 and c.get(body["zip_url"]).status_code == 200


def test_bad_settings_and_paths_are_refused(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch)
    assert c.post("/mv/export", json={"spec": SPEC, "settings": {"printing": {"layer_mm": 0.9}}}).status_code == 422
    assert c.get("/mv/artifacts/" + "0" * 20 + "/..%2F..%2Fsecret").status_code == 404
    assert c.get("/mv/artifacts/nothex/part.stl").status_code == 404


def test_the_cli_exports_formats(tmp_path, monkeypatch):
    from scripts import mv_export
    monkeypatch.setattr("s2c.multiview.slice.find_slicer", lambda: None)
    spec_path = Path(__file__).parents[1] / "examples" / "mv" / "l_bracket.json"
    monkeypatch.setattr(sys, "argv", ["mv_export", str(spec_path),
                                      "--format", "stl", "--format", "step", "--out", str(tmp_path)])
    mv_export.main()
    assert any(p.name == "part.stl" for p in tmp_path.rglob("*")) and list(tmp_path.glob("*.zip"))
