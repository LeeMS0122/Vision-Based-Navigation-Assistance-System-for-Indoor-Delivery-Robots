from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from web_demo.services.settings import (
    RUNS_BASE_DIR,
    VIDEO_RUNS_BASE_DIR,
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
)
from web_demo.services.pipeline_service import PipelineService
from web_demo.services.video_service import video_service

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parents[1] / "templates"))
pipeline_service = PipelineService()


@router.get("/")
def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request},
    )


@router.post("/api/pipeline/run")
async def run_pipeline(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported image format: {ext}")

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    run_dir = RUNS_BASE_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    input_path = run_dir / f"original{ext}"

    try:
        with input_path.open("wb") as f:
            f.write(await file.read())

        result = pipeline_service.run(input_path, run_dir)
        payload = {
            "run_id": run_id,
            **result,
        }
        return JSONResponse(content=payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}")


@router.post("/api/video/run")
async def run_video_pipeline(file: UploadFile = File(...), sample_fps: float = Form(5.0)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported video format: {ext}")

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    run_dir = VIDEO_RUNS_BASE_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    input_path = run_dir / f"input_video{ext}"

    try:
        with input_path.open("wb") as f:
            f.write(await file.read())

        result = video_service.run(input_video_path=input_path, run_dir=run_dir, sample_fps=sample_fps)
        payload = {
            "run_id": run_id,
            **result,
        }
        return JSONResponse(content=payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Video pipeline failed: {exc}")
