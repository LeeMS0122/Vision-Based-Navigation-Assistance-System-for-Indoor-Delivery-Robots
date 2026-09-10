from typing import Optional, Tuple

import numpy as np

Point = Tuple[int, int]  # (y, x)


def display_to_image_xy(
    click_x: float,
    click_y: float,
    display_w: int,
    display_h: int,
    image_w: int,
    image_h: int,
) -> Tuple[int, int]:
    if display_w <= 0 or display_h <= 0:
        return image_w // 2, image_h // 2
    x = int(np.clip(click_x * image_w / float(display_w), 0, image_w - 1))
    y = int(np.clip(click_y * image_h / float(display_h), 0, image_h - 1))
    return x, y


def image_xy_to_map_yx(x: int, y: int, image_w: int, image_h: int, map_w: int, map_h: int) -> Point:
    mx = int(np.clip(x * map_w / float(max(1, image_w)), 0, map_w - 1))
    my = int(np.clip(y * map_h / float(max(1, image_h)), 0, map_h - 1))
    return my, mx


def refine_goal_on_drivable(
    goal_yx: Point,
    drive_mask: np.ndarray,
    cost_map: np.ndarray,
    obstacle_threshold: int = 255,
    max_radius: int = 20,
) -> Optional[Point]:
    h, w = drive_mask.shape
    gy, gx = goal_yx
    gy = int(np.clip(gy, 0, h - 1))
    gx = int(np.clip(gx, 0, w - 1))

    if drive_mask[gy, gx] > 0 and cost_map[gy, gx] < obstacle_threshold:
        return gy, gx

    best = None
    best_score = 1e18
    for r in range(1, max_radius + 1):
        y0, y1 = max(0, gy - r), min(h, gy + r + 1)
        x0, x1 = max(0, gx - r), min(w, gx + r + 1)
        ys, xs = np.where((drive_mask[y0:y1, x0:x1] > 0) & (cost_map[y0:y1, x0:x1] < obstacle_threshold))
        if ys.size == 0:
            continue
        ys = ys + y0
        xs = xs + x0
        d2 = (ys - gy) ** 2 + (xs - gx) ** 2
        c = cost_map[ys, xs].astype(np.float32)
        score = d2.astype(np.float32) + c * 0.1
        i = int(np.argmin(score))
        if float(score[i]) < best_score:
            best_score = float(score[i])
            best = (int(ys[i]), int(xs[i]))
    return best


def map_yx_to_image_xy(y: int, x: int, map_w: int, map_h: int, image_w: int, image_h: int) -> Tuple[int, int]:
    ix = int(np.clip(x * image_w / float(max(1, map_w)), 0, image_w - 1))
    iy = int(np.clip(y * image_h / float(max(1, map_h)), 0, image_h - 1))
    return ix, iy
