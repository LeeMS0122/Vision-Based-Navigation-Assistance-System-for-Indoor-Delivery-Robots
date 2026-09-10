from typing import Any, Dict, List, Optional

import numpy as np


class StatusService:
    RISK_LEVEL_SCORE = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

    @staticmethod
    def direction_from_path(path: Optional[List[List[int]]]) -> str:
        if not path or len(path) < 2:
            return "STOP"

        y0, x0 = path[0]
        y1, x1 = path[min(3, len(path) - 1)]
        dx = x1 - x0

        if abs(dx) <= 1:
            return "GO"
        return "RIGHT" if dx > 0 else "LEFT"

    @staticmethod
    def obstacle_ratio(obs_mask: np.ndarray) -> float:
        return float(np.mean(obs_mask > 0))

    @staticmethod
    def near_obstacle_ratio(obs_mask: np.ndarray) -> float:
        h, w = obs_mask.shape
        region = obs_mask[int(h * 0.70):, int(w * 0.30):int(w * 0.70)]
        if region.size == 0:
            return 0.0
        return float(np.mean(region > 0))

    def risk_level(self, path_found: bool, start: Any, near_obstacle_ratio: float, obs_ratio: float) -> str:
        if not path_found or start is None:
            return "HIGH"
        if near_obstacle_ratio > 0.06 or obs_ratio > 0.12:
            return "MEDIUM"
        return "LOW"

    def replanning_required(self, current: Dict[str, Any], previous: Optional[Dict[str, Any]]) -> bool:
        if previous is None:
            return False

        if bool(current["path_found"]) != bool(previous["path_found"]):
            return True

        cur_r = self.RISK_LEVEL_SCORE.get(str(current["risk"]).upper(), 0)
        prev_r = self.RISK_LEVEL_SCORE.get(str(previous["risk"]).upper(), 0)
        if abs(cur_r - prev_r) >= 1:
            return True

        if abs(float(current["obstacle_ratio"]) - float(previous["obstacle_ratio"])) > 0.03:
            return True

        return False

    @staticmethod
    def warning_text(near_obs_ratio: float) -> str:
        if near_obs_ratio > 0.20:
            return "경고: 가까운 전방 장애물 가능성 높음"
        if near_obs_ratio > 0.07:
            return "주의: 전방 장애물 존재 가능"
        return "정상: 가까운 전방 장애물 위험 낮음"

    def build_frame_status(
        self,
        path: Optional[List[List[int]]],
        path_found: bool,
        start: Any,
        obs_mask: np.ndarray,
        object_counts: Dict[str, int],
        previous_status: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        obs_ratio = self.obstacle_ratio(obs_mask)
        near_obs_ratio = self.near_obstacle_ratio(obs_mask)
        risk = self.risk_level(path_found=path_found, start=start, near_obstacle_ratio=near_obs_ratio, obs_ratio=obs_ratio)
        direction = self.direction_from_path(path)

        current = {
            "path_found": bool(path_found),
            "risk": risk,
            "obstacle_ratio": round(obs_ratio, 4),
        }
        replanning = self.replanning_required(current, previous_status)

        return {
            "detected_objects": object_counts,
            "nearest_obstacle_warning": self.warning_text(near_obs_ratio),
            "current_risk": risk,
            "recommended_direction": direction,
            "path_recalculation_required": replanning,
            "path_found": bool(path_found),
            "obstacle_ratio": round(obs_ratio, 4),
            "near_obstacle_ratio": round(near_obs_ratio, 4),
        }


status_service = StatusService()
