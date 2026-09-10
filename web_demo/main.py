from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from web_demo.routers.demo_router import router as demo_router
from web_demo.routers.video_demo_router import router as video_demo_router
from web_demo.routers.live_demo_router import router as live_demo_router
from web_demo.services.settings import RUNS_BASE_DIR, VIDEO_RUNS_BASE_DIR

app = FastAPI(title="Graduation Robot Vision Web Demo", version="0.1.0")

static_dir = Path(__file__).resolve().parent / "static"
RUNS_BASE_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_RUNS_BASE_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
app.mount("/demo-data", StaticFiles(directory=str(RUNS_BASE_DIR.parent)), name="demo-data")
app.mount("/demo-outputs", StaticFiles(directory=str(VIDEO_RUNS_BASE_DIR.parent)), name="demo-outputs")

app.include_router(demo_router)
app.include_router(video_demo_router)
app.include_router(live_demo_router)
