import json
from pathlib import Path
from typing import Any, Dict, Optional

import cv2

from web_demo.services.pipeline_service_v1 import pipeline_service_v1
from web_demo.services.settings import VIDEO_RUNS_BASE_DIR
from web_demo.utils.video_io import compose_video_from_frames, extract_sampled_frames


class VideoPipelineServiceV1:
    @staticmethod
    def _to_web_path(path: Path) -> str:
        rel = path.relative_to(VIDEO_RUNS_BASE_DIR.parent)
        return f"/demo-outputs/{rel.as_posix()}"

    def run_video(self, input_video_path: Path, run_dir: Path, sample_fps: float, goal_click: Optional[Dict[str, float]]) -> Dict[str, Any]:
        sampled_frames_dir = run_dir / "sampled_frames"
        frame_runs_dir = run_dir / "frame_runs"
        overlay_frames_dir = run_dir / "overlay_frames"
        overlay_frames_dir.mkdir(parents=True, exist_ok=True)

        sampled_frames, video_info = extract_sampled_frames(input_video_path, sampled_frames_dir, sample_fps)
        if not sampled_frames:
            raise RuntimeError("No sampled frames extracted")

        statuses = []
        overlay_paths = []
        prev = None

        for fm in sampled_frames:
            idx = int(fm["sampled_index"])
            frame_path: Path = fm["frame_path"]
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            frame_dir = frame_runs_dir / f"frame_{idx:06d}"
            out = pipeline_service_v1.run_frame(frame, frame_dir, goal_click=goal_click, previous_status=prev)
            st = out["status"]
            prev = {
                "path_found": st["path_found"],
                "risk": st["current_risk"],
                "obstacle_ratio": st["obstacle_ratio"],
            }
            overlay_paths.append(out["artifacts"]["astar_on_rgb"])
            statuses.append(
                {
                    "frame_index": idx,
                    "timestamp_sec": float(fm["timestamp_sec"]),
                    "status": st,
                    "goal_refined_map_yx": out["goal_refined_map_yx"],
                    "goal_refined_image_xy": out["goal_refined_image_xy"],
                    "images": {k: self._to_web_path(v) for k, v in out["artifacts"].items()},
                }
            )

        result_video_path = run_dir / "result_video.mp4"
        compose_video_from_frames(overlay_paths, result_video_path, fps=video_info["sample_fps_actual"])

        statuses_path = run_dir / "frame_statuses.json"
        with statuses_path.open("w", encoding="utf-8") as f:
            json.dump({"video_info": video_info, "frames": statuses}, f, ensure_ascii=False, indent=2)

        return {
            "status": "ok",
            "video_info": video_info,
            "result_video_path": self._to_web_path(result_video_path),
            "frame_statuses_json_path": self._to_web_path(statuses_path),
            "frame_statuses": {"video_info": video_info, "frames": statuses},
        }


video_pipeline_service_v1 = VideoPipelineServiceV1()
