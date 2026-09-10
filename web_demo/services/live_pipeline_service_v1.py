from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import cv2

from web_demo.services.pipeline_service_v1 import pipeline_service_v1
from web_demo.services.settings import VIDEO_RUNS_BASE_DIR


@dataclass
class LiveSessionState:
    session_id: str
    run_dir: Path
    goal_click: Optional[Dict[str, float]] = None
    previous_status: Optional[Dict[str, Any]] = None
    frame_count: int = 0


class LivePipelineServiceV1:
    def __init__(self) -> None:
        self._sessions: Dict[str, LiveSessionState] = {}

    @staticmethod
    def _to_web_path(path: Path) -> str:
        rel = path.relative_to(VIDEO_RUNS_BASE_DIR.parent)
        return f"/demo-outputs/{rel.as_posix()}"

    def create_session(self) -> LiveSessionState:
        sid = uuid4().hex[:12]
        run_dir = VIDEO_RUNS_BASE_DIR / f"live_{sid}"
        run_dir.mkdir(parents=True, exist_ok=True)
        st = LiveSessionState(session_id=sid, run_dir=run_dir)
        self._sessions[sid] = st
        return st

    def get_session(self, session_id: str) -> Optional[LiveSessionState]:
        return self._sessions.get(session_id)

    def set_goal(self, session_id: str, goal_click: Dict[str, float]) -> Optional[LiveSessionState]:
        st = self.get_session(session_id)
        if st is None:
            return None
        st.goal_click = goal_click
        return st

    def process_frame(self, session_id: str, frame_bgr) -> Optional[Dict[str, Any]]:
        st = self.get_session(session_id)
        if st is None:
            return None

        frame_dir = st.run_dir / f"frame_{st.frame_count:06d}"
        out = pipeline_service_v1.run_frame(frame_bgr, frame_dir, goal_click=st.goal_click, previous_status=st.previous_status)
        status = out["status"]
        st.previous_status = {
            "path_found": status["path_found"],
            "risk": status["current_risk"],
            "obstacle_ratio": status["obstacle_ratio"],
        }
        st.frame_count += 1

        return {
            "status": "ok",
            "session_id": st.session_id,
            "frame_index": st.frame_count - 1,
            "goal_refined_map_yx": out["goal_refined_map_yx"],
            "goal_refined_image_xy": out["goal_refined_image_xy"],
            "analysis_image_path": self._to_web_path(out["artifacts"]["astar_on_rgb"]),
            "status_json": status,
            "images": {k: self._to_web_path(v) for k, v in out["artifacts"].items()},
        }


live_pipeline_service_v1 = LivePipelineServiceV1()
