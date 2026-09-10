from typing import Any, Dict, Optional

from web_demo.services.status_service import StatusService


class StatusServiceV3(StatusService):
    def build_status_v3(
        self,
        path_found: bool,
        direction: str,
        goal_active: bool,
        goal_reached: bool,
        obs_ratio: float,
        near_obs_ratio: float,
        detected_objects: Dict[str, int],
        previous_status: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        risk = self.risk_level(path_found=path_found, start=(0, 0), near_obstacle_ratio=near_obs_ratio, obs_ratio=obs_ratio)
        cur = {"path_found": bool(path_found), "risk": risk, "obstacle_ratio": obs_ratio}
        replanning = self.replanning_required(cur, previous_status)
        return {
            "path_found": bool(path_found),
            "goal_active": bool(goal_active),
            "goal_reached": bool(goal_reached),
            "recommended_direction": direction if path_found else "STOP",
            "current_risk": risk,
            "replanning_required": replanning,
            "detected_objects": detected_objects,
            "near_obstacle_warning": self.warning_text(near_obs_ratio),
            "obstacle_ratio": round(obs_ratio, 4),
            "near_obstacle_ratio": round(near_obs_ratio, 4),
        }


status_service_v3 = StatusServiceV3()
