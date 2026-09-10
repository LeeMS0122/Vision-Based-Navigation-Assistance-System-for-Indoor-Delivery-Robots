import base64
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from web_demo.services.live_pipeline_service_v1 import live_pipeline_service_v1

router = APIRouter(prefix="/live-demo", tags=["live-demo"])
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parents[1] / "templates"))


class GoalBody(BaseModel):
    session_id: str
    click_x: float
    click_y: float
    display_w: int
    display_h: int


class FrameBody(BaseModel):
    session_id: str
    image_base64: str


@router.get("")
def live_demo_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="live_demo.html",
        context={"request": request},
    )


@router.post("/api/session")
def create_live_session():
    st = live_pipeline_service_v1.create_session()
    return JSONResponse({"status": "ok", "session_id": st.session_id})


@router.post("/api/goal")
def set_live_goal(body: GoalBody):
    st = live_pipeline_service_v1.set_goal(
        body.session_id,
        {
            "click_x": body.click_x,
            "click_y": body.click_y,
            "display_w": body.display_w,
            "display_h": body.display_h,
        },
    )
    if st is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return JSONResponse({"status": "ok", "session_id": st.session_id})


@router.post("/api/frame")
def process_live_frame(body: FrameBody):
    b64 = body.image_base64
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image payload: {exc}")

    if frame is None:
        raise HTTPException(status_code=400, detail="Failed to decode frame")

    out = live_pipeline_service_v1.process_frame(body.session_id, frame)
    if out is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return JSONResponse(out)
