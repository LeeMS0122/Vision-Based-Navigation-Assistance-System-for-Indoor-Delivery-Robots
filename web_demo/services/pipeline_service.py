from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np

from .settings import RUNS_BASE_DIR, VIDEO_RUNS_BASE_DIR
from .yolo_service import YOLOService
from .scdepth_service import SCDepthService
from .costmap_service import CostMapService
from .planner_service import PlannerService


yolo_service = YOLOService()
scdepth_service = SCDepthService()
costmap_service = CostMapService()
planner_service = PlannerService()


class PipelineService:
    @staticmethod
    def _to_web_path(file_path: Path) -> str:
        try:
            rel_data = file_path.relative_to(RUNS_BASE_DIR.parent)
            return f"/demo-data/{rel_data.as_posix()}"
        except ValueError:
            rel_out = file_path.relative_to(VIDEO_RUNS_BASE_DIR.parent)
            return f"/demo-outputs/{rel_out.as_posix()}"

    @staticmethod
    def _build_risk_overlay(original_path: Path, obs_mask: np.ndarray, cost_map: np.ndarray, save_path: Path) -> None:
        rgb = cv2.imread(str(original_path))
        if rgb is None:
            raise RuntimeError(f"Failed to read image: {original_path}")

        h, w = rgb.shape[:2]
        obs_resized = cv2.resize(obs_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        cost_big = cv2.resize(cost_map.astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
        danger = (cost_big >= 180).astype(np.uint8)

        overlay = rgb.copy()
        overlay[obs_resized == 1] = (0, 0, 255)
        overlay[danger == 1] = (0, 165, 255)

        blended = cv2.addWeighted(rgb, 0.6, overlay, 0.4, 0)
        cv2.imwrite(str(save_path), blended)

    @staticmethod
    def _nearest_obstacle_warning(obs_mask: np.ndarray) -> str:
        h, w = obs_mask.shape
        bottom_region = obs_mask[int(h * 0.70):, int(w * 0.35):int(w * 0.65)]
        ratio = float(np.mean(bottom_region)) if bottom_region.size > 0 else 0.0

        if ratio > 0.20:
            return "경고: 가까운 전방 장애물 가능성 높음"
        if ratio > 0.07:
            return "주의: 전방 장애물 존재 가능"
        return "정상: 가까운 전방 장애물 위험 낮음"

    @staticmethod
    def _risk_level(cost_map: np.ndarray, obs_mask: np.ndarray) -> str:
        obs_ratio = float(np.mean(obs_mask))
        high_cost_ratio = float(np.mean(cost_map >= 180))

        score = obs_ratio * 0.7 + high_cost_ratio * 0.3
        if score >= 0.18:
            return "HIGH"
        if score >= 0.08:
            return "MEDIUM"
        return "LOW"

    def _run_core(self, original_image_path: Path, run_dir: Path) -> Dict[str, Any]:
        yolo_out = yolo_service.run(original_image_path, run_dir)
        scdepth_out = scdepth_service.run(original_image_path, run_dir)
        costmap_out = costmap_service.run(
            drive_mask=yolo_out["drive_mask"],
            obs_mask=yolo_out["obs_mask"],
            depth_map=scdepth_out["depth_map"],
            run_dir=run_dir,
        )
        planner_out = planner_service.run(
            cost_map=costmap_out["cost_map"],
            rgb_path=original_image_path,
            run_dir=run_dir,
            drive_mask=costmap_out["drive_mask_small"],
        )

        risk_overlay_path = run_dir / "risk_overlay.png"
        self._build_risk_overlay(
            original_path=original_image_path,
            obs_mask=yolo_out["obs_mask"],
            cost_map=costmap_out["cost_map"],
            save_path=risk_overlay_path,
        )

        risk_level = self._risk_level(costmap_out["cost_map"], yolo_out["obs_mask"])
        warning = self._nearest_obstacle_warning(yolo_out["obs_mask"])
        path_recalc = (not planner_out["path_found"]) or (risk_level == "HIGH")

        files = {
            "original_image": original_image_path,
            "drive_mask": yolo_out["files"]["drive_mask"],
            "obstacle_mask": yolo_out["files"]["obstacle_mask"],
            "detection_overlay": yolo_out["files"]["detection_overlay"],
            "risk_overlay": risk_overlay_path,
            "depth_visualization": scdepth_out["files"]["depth_visualization"],
            "cost_map": costmap_out["files"]["cost_map"],
            "astar_on_rgb": planner_out["files"]["astar_on_rgb"],
            "astar_on_costmap": planner_out["files"]["astar_on_costmap"],
        }

        summary = {
            "detected_objects": yolo_out["object_counts"],
            "detected_object_details": yolo_out["objects"],
            "nearest_obstacle_warning": warning,
            "current_risk": risk_level,
            "recommended_direction": planner_out["recommended_direction"],
            "path_recalculation_required": path_recalc,
            "path_found": planner_out["path_found"],
            "path_length": planner_out["path_length"],
            "start": planner_out["start"],
            "goal": planner_out["goal"],
        }

        internals = {
            "cost_map": costmap_out["cost_map"],
            "obs_mask": yolo_out["obs_mask"],
            "drive_mask": yolo_out["drive_mask"],
            "path": planner_out["path"],
            "start": planner_out["start"],
            "goal": planner_out["goal"],
            "path_found": planner_out["path_found"],
            "path_length": planner_out["path_length"],
        }

        return {
            "summary": summary,
            "files": files,
            "internals": internals,
        }

    def run(self, original_image_path: Path, run_dir: Path) -> Dict[str, Any]:
        core = self._run_core(original_image_path, run_dir)
        files_web = {k: self._to_web_path(v) for k, v in core["files"].items()}

        return {
            "status": "ok",
            "summary": core["summary"],
            "images": files_web,
        }

    def run_for_video_frame(self, frame_path: Path, frame_dir: Path) -> Dict[str, Any]:
        core = self._run_core(frame_path, frame_dir)
        files_web = {k: self._to_web_path(v) for k, v in core["files"].items()}

        return {
            "summary": core["summary"],
            "files": core["files"],
            "files_web": files_web,
            "internals": core["internals"],
        }


pipeline_service = PipelineService()
