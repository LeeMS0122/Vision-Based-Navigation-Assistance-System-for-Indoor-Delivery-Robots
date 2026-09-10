from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from web_demo.services.settings import ALLOWED_VIDEO_EXTENSIONS, VIDEO_RUNS_BASE_DIR
from web_demo.services.video_pipeline_service_v1 import video_pipeline_service_v1

router = APIRouter(prefix="/video-demo", tags=["video-demo"])
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parents[1] / "templates"))


@router.get("")
def video_demo_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="video_demo.html",
        context={"request": request},
    )


@router.post("/api/run")
async def run_video_demo(
    file: UploadFile = File(...),
    sample_fps: float = Form(5.0),
    goal_click_x: float = Form(...),
    goal_click_y: float = Form(...),
    display_w: int = Form(...),
    display_h: int = Form(...),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported video format: {ext}")

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    run_dir = VIDEO_RUNS_BASE_DIR / f"video_demo_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    input_path = run_dir / f"input_video{ext}"
    with input_path.open("wb") as f:
        f.write(await file.read())

    goal_click = {
        "click_x": goal_click_x,
        "click_y": goal_click_y,
        "display_w": display_w,
        "display_h": display_h,
    }

    try:
        out = video_pipeline_service_v1.run_video(input_path, run_dir, sample_fps, goal_click)
        return JSONResponse({"run_id": run_id, **out})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Video demo failed: {exc}")
