from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np

from .settings import (
    TARGET_SIZE,
    DRIVABLE_BASE_COST,
    NON_DRIVABLE_BASE_COST,
    OBSTACLE_COST,
    BUFFER_ZONE_COST,
    DEPTH_PENALTY_SCALE,
    BOTTOM_RISK_SCALE,
    CENTER_BIAS_SCALE,
    EDGE_MARGIN_X_RATIO,
    EDGE_MARGIN_Y_RATIO,
    EDGE_MARGIN_COST,
    OBSTACLE_DILATION_KERNEL,
    OBSTACLE_DILATION_ITER,
)


class CostMapService:
    @staticmethod
    def _resize_mask(mask: np.ndarray, size):
        resized = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
        return (resized > 0).astype(np.uint8)

    @staticmethod
    def _resize_depth(depth: np.ndarray, size):
        return cv2.resize(depth.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _normalize_depth(depth: np.ndarray):
        p2 = np.percentile(depth, 2)
        p98 = np.percentile(depth, 98)
        if p98 - p2 < 1e-8:
            return np.zeros_like(depth, dtype=np.float32)
        d = np.clip(depth, p2, p98)
        return ((d - p2) / (p98 - p2)).astype(np.float32)

    @staticmethod
    def _save_costmap_visualization(cost_map: np.ndarray, save_path: Path):
        color = cv2.applyColorMap(np.uint8(255 - cost_map), cv2.COLORMAP_JET)
        color[cost_map >= OBSTACLE_COST] = (0, 0, 0)
        cv2.imwrite(str(save_path), color)

    @staticmethod
    def _dilate_obstacle_mask(obs_mask: np.ndarray):
        kernel = np.ones((OBSTACLE_DILATION_KERNEL, OBSTACLE_DILATION_KERNEL), np.uint8)
        dilated = cv2.dilate(obs_mask.astype(np.uint8), kernel, iterations=OBSTACLE_DILATION_ITER)
        return (dilated > 0).astype(np.uint8)

    @staticmethod
    def _make_center_bias_map(h: int, w: int):
        x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
        center_penalty = np.abs(x)
        center_penalty = np.tile(center_penalty.reshape(1, -1), (h, 1))
        return center_penalty * CENTER_BIAS_SCALE

    @staticmethod
    def _make_bottom_risk_map(h: int, w: int):
        y = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(-1, 1)
        y_penalty = np.tile(y, (1, w))
        return y_penalty * BOTTOM_RISK_SCALE

    @staticmethod
    def _apply_edge_margin_penalty(cost_map: np.ndarray):
        h, w = cost_map.shape
        edge_margin_x = int(w * EDGE_MARGIN_X_RATIO)
        edge_margin_y = int(h * EDGE_MARGIN_Y_RATIO)

        cost_map[:, :edge_margin_x] = np.maximum(cost_map[:, :edge_margin_x], EDGE_MARGIN_COST)
        cost_map[:, w - edge_margin_x:] = np.maximum(cost_map[:, w - edge_margin_x:], EDGE_MARGIN_COST)
        cost_map[:edge_margin_y, :] = np.maximum(cost_map[:edge_margin_y, :], EDGE_MARGIN_COST)
        return cost_map

    def run(self, drive_mask: np.ndarray, obs_mask: np.ndarray, depth_map: np.ndarray, run_dir: Path) -> Dict[str, Any]:
        drive_small = self._resize_mask(drive_mask, TARGET_SIZE)
        obs_small = self._resize_mask(obs_mask, TARGET_SIZE)
        depth_small = self._resize_depth(depth_map, TARGET_SIZE)

        obs_dilated = self._dilate_obstacle_mask(obs_small)
        buffer_zone = np.logical_and(obs_dilated == 1, obs_small == 0).astype(np.uint8)

        depth_norm = self._normalize_depth(depth_small)
        depth_penalty = (1.0 - depth_norm) * DEPTH_PENALTY_SCALE

        h, w = drive_small.shape
        center_bias = self._make_center_bias_map(h, w)
        bottom_risk = self._make_bottom_risk_map(h, w)

        cost_map = np.where(drive_small == 1, DRIVABLE_BASE_COST, NON_DRIVABLE_BASE_COST).astype(np.float32)
        cost_map += depth_penalty
        cost_map += center_bias
        cost_map += bottom_risk

        cost_map[buffer_zone == 1] = np.maximum(cost_map[buffer_zone == 1], BUFFER_ZONE_COST)
        cost_map[obs_small == 1] = OBSTACLE_COST
        cost_map = self._apply_edge_margin_penalty(cost_map)
        cost_map = np.clip(cost_map, 0, 255).astype(np.uint8)

        cost_npy_path = run_dir / "cost_map.npy"
        cost_png_path = run_dir / "cost_map.png"
        obs_dilated_path = run_dir / "obs_dilated.png"
        buffer_zone_path = run_dir / "buffer_zone.png"

        np.save(cost_npy_path, cost_map)
        self._save_costmap_visualization(cost_map, cost_png_path)
        cv2.imwrite(str(obs_dilated_path), np.uint8(obs_dilated * 255))
        cv2.imwrite(str(buffer_zone_path), np.uint8(buffer_zone * 255))

        return {
            "cost_map": cost_map,
            "drive_mask_small": drive_small,
            "obs_mask_small": obs_small,
            "obs_dilated_small": obs_dilated,
            "buffer_zone_small": buffer_zone,
            "depth_norm_small": depth_norm,
            "files": {
                "cost_map_npy": cost_npy_path,
                "cost_map": cost_png_path,
                "obs_dilated": obs_dilated_path,
                "buffer_zone": buffer_zone_path,
            },
        }
