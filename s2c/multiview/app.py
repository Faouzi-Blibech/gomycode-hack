"""Stand-alone server for the multi-view path until the integrator mounts the router in s2c/api.py.
Run: uv run uvicorn s2c.multiview.app:app --host 0.0.0.0 --port 8001"""
from dotenv import load_dotenv
from fastapi import FastAPI

from s2c.multiview.routes import router

load_dotenv()
app = FastAPI(title="Sketch-to-CAD multi-view")
app.include_router(router)
