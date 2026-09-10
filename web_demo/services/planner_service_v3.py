from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from web_demo.services.planner_service import PlannerService
from web_demo.services.settings import OBSTACLE_THRESHOLD
from web_demo.utils.coord_utils import refine_goal_on_drivable

Point = Tuple[int, int]


class PlannerServiceV3(PlannerService):
    def run_with_user_goal(
        self,
        cost_map: np.ndarray,
        rgb_path: Path,
        run_dir: Path,
        drive_mask: Optional[np.ndarray],
        user_goal_map_yx: Optional[Point],
    ) -> Dict[str, Any]:
        start = self._find_start(cost_map)

        refined_goal = None
        if user_goal_map_yx is not None and drive_mask is not None:
            refined_goal = refine_goal_on_drivable(
                goal_yx=user_goal_map_yx,
                drive_mask=drive_mask,
                cost_map=cost_map,
                obstacle_threshold=OBSTACLE_THRESHOLD,
                max_radius=24,
            )

        goal = refined_goal
        path: Optional[List[Point]] = None
        if start is not None and goal is not None:
            path = self._astar(cost_map, start, goal)

        on_cost_path = run_dir / "astar_on_costmap.png"
        on_rgb_path = run_dir / "astar_on_rgb.png"
        self._draw_on_costmap(cost_map, path, start, goal, on_cost_path)
        self._draw_on_rgb(rgb_path, path, start, goal, cost_map.shape, on_rgb_path)

        direction = self._recommended_direction(path, cost_map.shape[1])

        goal_reached = False
        if path and len(path) > 0 and goal is not None:
            gy, gx = goal
            py, px = path[-1]
            goal_reached = (abs(gy - py) + abs(gx - px)) <= 2

        return {
            "start": start,
            "goal": goal,
            "path": path,
            "path_found": path is not None,
            "path_length": len(path) if path else 0,
            "recommended_direction": direction,
            "goal_refined": refined_goal,
            "goal_reached": goal_reached,
            "files": {
                "astar_on_costmap": on_cost_path,
                "astar_on_rgb": on_rgb_path,
            },
        }


planner_service_v3 = PlannerServiceV3()
