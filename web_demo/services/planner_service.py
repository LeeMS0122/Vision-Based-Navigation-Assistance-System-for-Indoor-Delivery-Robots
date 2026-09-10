from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import heapq

import cv2
import numpy as np

from .settings import (
    OBSTACLE_THRESHOLD,
    START_MARGIN_BOTTOM,
    GOAL_CENTER_BIAS,  # kept for compatibility
)

Point = Tuple[int, int]  # (y, x)

NEIGHBORS_8 = [
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, 1.414),
    (-1, 1, 1.414),
    (1, -1, 1.414),
    (1, 1, 1.414),
]


class PlannerService:
    @staticmethod
    def _heuristic(a: Point, b: Point) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    @staticmethod
    def _neighbors(y: int, x: int, h: int, w: int):
        for dy, dx, move_cost in NEIGHBORS_8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                yield ny, nx, move_cost

    def _astar(self, cost_map: np.ndarray, start: Point, goal: Point) -> Optional[List[Point]]:
        h, w = cost_map.shape
        open_heap = [(0.0, start)]
        came_from = {}
        g_score = {start: 0.0}
        visited = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            cy, cx = current
            for ny, nx, move_cost in self._neighbors(cy, cx, h, w):
                if cost_map[ny, nx] >= OBSTACLE_THRESHOLD:
                    continue
                neighbor = (ny, nx)
                tentative_g = g_score[current] + float(cost_map[ny, nx]) + move_cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (f, neighbor))

        return None

    @staticmethod
    def _find_start(cost_map: np.ndarray) -> Optional[Point]:
        h, w = cost_map.shape
        y = h - 1 - START_MARGIN_BOTTOM
        x_center = w // 2

        for offset in range(w // 2):
            for x in (x_center - offset, x_center + offset):
                if 0 <= x < w and cost_map[y, x] < OBSTACLE_THRESHOLD:
                    return (y, x)
        return None

    @staticmethod
    def _find_goal(cost_map: np.ndarray, drive_mask: Optional[np.ndarray] = None) -> Optional[Point]:
        h, w = cost_map.shape
        x_center = w // 2
        search_bands = [
            (0.35, 0.65),
            (0.30, 0.70),
            (0.20, 0.80),
        ]

        if drive_mask is None:
            drive_mask = np.ones_like(cost_map, dtype=np.uint8)

        for left_ratio, right_ratio in search_bands:
            x_min = int(w * left_ratio)
            x_max = int(w * right_ratio)
            candidates = []

            for y in range(h):
                for x in range(x_min, x_max):
                    if cost_map[y, x] >= OBSTACLE_THRESHOLD:
                        continue
                    if drive_mask[y, x] == 0:
                        continue
                    # v2 heuristic: upper region + center proximity + lower local cost
                    score = y * 3.0 + abs(x - x_center) * 2.0 + float(cost_map[y, x])
                    candidates.append((score, (y, x)))

            if candidates:
                candidates.sort(key=lambda t: t[0])
                return candidates[0][1]

        return None

    @staticmethod
    def _draw_on_costmap(cost_map: np.ndarray, path: Optional[List[Point]], start: Optional[Point], goal: Optional[Point], save_path: Path):
        vis = cv2.applyColorMap(np.uint8(255 - cost_map), cv2.COLORMAP_JET)
        vis[cost_map >= OBSTACLE_THRESHOLD] = (0, 0, 0)

        if path and len(path) >= 2:
            pts = np.array([[x, y] for (y, x) in path], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], isClosed=False, color=(255, 255, 255), thickness=2)

        if start is not None:
            cv2.circle(vis, (start[1], start[0]), 3, (0, 255, 0), -1)
        if goal is not None:
            cv2.circle(vis, (goal[1], goal[0]), 3, (0, 0, 255), -1)

        cv2.imwrite(str(save_path), vis)

    @staticmethod
    def _draw_on_rgb(rgb_path: Path, path: Optional[List[Point]], start: Optional[Point], goal: Optional[Point], map_shape, save_path: Path):
        rgb = cv2.imread(str(rgb_path))
        h_img, w_img = rgb.shape[:2]
        h_map, w_map = map_shape

        def scale_point(pt: Point):
            y, x = pt
            sx = int(x * w_img / w_map)
            sy = int(y * h_img / h_map)
            return sx, sy

        if path and len(path) >= 2:
            pts = np.array([scale_point(p) for p in path], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(rgb, [pts], isClosed=False, color=(0, 255, 255), thickness=4)

        if start is not None:
            x, y = scale_point(start)
            cv2.circle(rgb, (x, y), 6, (0, 255, 0), -1)

        if goal is not None:
            x, y = scale_point(goal)
            cv2.circle(rgb, (x, y), 6, (0, 0, 255), -1)

        cv2.imwrite(str(save_path), rgb)

    @staticmethod
    def _recommended_direction(path: Optional[List[Point]], map_width: int) -> str:
        if not path or len(path) < 2:
            return "STOP"

        goal_x = path[-1][1]
        center_x = map_width // 2
        diff = goal_x - center_x

        if abs(diff) <= max(2, int(map_width * 0.03)):
            return "STRAIGHT"
        return "RIGHT" if diff > 0 else "LEFT"

    def run(self, cost_map: np.ndarray, rgb_path: Path, run_dir: Path, drive_mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        start = self._find_start(cost_map)
        goal = self._find_goal(cost_map, drive_mask=drive_mask)

        path = None
        if start is not None and goal is not None:
            path = self._astar(cost_map, start, goal)

        on_cost_path = run_dir / "astar_on_costmap.png"
        on_rgb_path = run_dir / "astar_on_rgb.png"

        self._draw_on_costmap(cost_map, path, start, goal, on_cost_path)
        self._draw_on_rgb(rgb_path, path, start, goal, cost_map.shape, on_rgb_path)

        direction = self._recommended_direction(path, cost_map.shape[1])

        return {
            "start": start,
            "goal": goal,
            "path": path,
            "path_found": path is not None,
            "path_length": len(path) if path else 0,
            "recommended_direction": direction,
            "files": {
                "astar_on_costmap": on_cost_path,
                "astar_on_rgb": on_rgb_path,
            },
        }
