from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from web_demo.services.costmap_service_v3 import costmap_service_v3
from web_demo.services.planner_service_v3 import planner_service_v3
from web_demo.services.scdepth_service_v3 import scdepth_service_v3
from web_demo.services.status_service_v3 import status_service_v3
from web_demo.services.yolo_service_v3 import yolo_service_v3
from web_demo.utils.coord_utils import display_to_image_xy, image_xy_to_map_yx, map_yx_to_image_xy


class PipelineServiceV1:
    @staticmethod
    def _build_risk_overlay(frame_bgr: np.ndarray, obs_mask: np.ndarray, cost_map: np.ndarray, save_path: Path) -> None:
        h, w = frame_bgr.shape[:2]
        obs_resized = cv2.resize(obs_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        cost_big = cv2.resize(cost_map.astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
        danger = (cost_big >= 180).astype(np.uint8)

        overlay = frame_bgr.copy()
        overlay[obs_resized == 1] = (0, 0, 255)
        overlay[danger == 1] = (0, 165, 255)
        blended = cv2.addWeighted(frame_bgr, 0.6, overlay, 0.4, 0)
        cv2.imwrite(str(save_path), blended)

    def _nearest_ratio(self, obs_mask: np.ndarray) -> float:
        h, w = obs_mask.shape
        region = obs_mask[int(h * 0.70):, int(w * 0.30): int(w * 0.70)]
        return float(np.mean(region > 0)) if region.size > 0 else 0.0

    def run_frame(
        self,
        frame_bgr: np.ndarray,
        frame_dir: Path,
        goal_click: Optional[Dict[str, float]] = None,
        previous_status: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_path = frame_dir / "frame_input.jpg"
        cv2.imwrite(str(frame_path), frame_bgr)

        yolo_out = yolo_service_v3.run(frame_path, frame_dir)
        depth_out = scdepth_service_v3.run(frame_path, frame_dir)
        cost_out = costmap_service_v3.run(
            drive_mask=yolo_out["drive_mask"],
            obs_mask=yolo_out["obs_mask"],
            depth_map=depth_out["depth_map"],
            run_dir=frame_dir,
        )

        user_goal_map = None
        goal_refined_image_xy: Optional[Tuple[int, int]] = None
        if goal_click is not None:
            img_h, img_w = frame_bgr.shape[:2]
            ix, iy = display_to_image_xy(
                click_x=float(goal_click["click_x"]),
                click_y=float(goal_click["click_y"]),
                display_w=int(goal_click["display_w"]),
                display_h=int(goal_click["display_h"]),
                image_w=img_w,
                image_h=img_h,
            )
            map_h, map_w = cost_out["cost_map"].shape
            user_goal_map = image_xy_to_map_yx(ix, iy, img_w, img_h, map_w, map_h)

        planner_out = planner_service_v3.run_with_user_goal(
            cost_map=cost_out["cost_map"],
            rgb_path=frame_path,
            run_dir=frame_dir,
            drive_mask=cost_out["drive_mask_small"],
            user_goal_map_yx=user_goal_map,
        )

        if planner_out["goal"] is not None:
            map_h, map_w = cost_out["cost_map"].shape
            img_h, img_w = frame_bgr.shape[:2]
            gy, gx = planner_out["goal"]
            goal_refined_image_xy = map_yx_to_image_xy(gy, gx, map_w, map_h, img_w, img_h)

        obs_ratio = float(np.mean(yolo_out["obs_mask"] > 0))
        near_ratio = self._nearest_ratio(yolo_out["obs_mask"])
        status = status_service_v3.build_status_v3(
            path_found=planner_out["path_found"],
            direction=planner_out["recommended_direction"],
            goal_active=planner_out["goal"] is not None,
            goal_reached=planner_out["goal_reached"],
            obs_ratio=obs_ratio,
            near_obs_ratio=near_ratio,
            detected_objects=yolo_out["object_counts"],
            previous_status=previous_status,
        )

        risk_overlay_path = frame_dir / "risk_overlay.png"
        self._build_risk_overlay(
            frame_bgr=frame_bgr,
            obs_mask=yolo_out["obs_mask"],
            cost_map=cost_out["cost_map"],
            save_path=risk_overlay_path,
        )

        return {
            "overlay_image_path": planner_out["files"]["astar_on_rgb"],
            "drive_mask": yolo_out["drive_mask"],
            "obstacle_mask": yolo_out["obs_mask"],
            "depth": depth_out["depth_map"],
            "cost_map": cost_out["cost_map"],
            "path": planner_out["path"],
            "status": status,
            "detected_object_details": yolo_out["objects"],
            "goal_refined_map_yx": planner_out["goal"],
            "goal_refined_image_xy": goal_refined_image_xy,
            "artifacts": {
                "frame_input": frame_path,
                "astar_on_rgb": planner_out["files"]["astar_on_rgb"],
                "astar_on_costmap": planner_out["files"]["astar_on_costmap"],
                "cost_map": cost_out["files"]["cost_map"],
                "depth_visualization": depth_out["files"]["depth_visualization"],
                "detection_overlay": yolo_out["files"]["detection_overlay"],
                "risk_overlay": risk_overlay_path,
                "drive_mask": yolo_out["files"]["drive_mask"],
                "obstacle_mask": yolo_out["files"]["obstacle_mask"],
            },
        }


pipeline_service_v1 = PipelineServiceV1()
