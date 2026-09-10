from pathlib import Path
import heapq
import numpy as np
import cv2

# =========================
# 경로 설정
# =========================
INPUT_DIR = Path("/home/minsu/disk_a/miniconda3/graduation_work_robot_vision/cost_map_real_outputs_v2")
COST_MAP_PATH = INPUT_DIR / "cost_map.npy"
RGB_PATH = INPUT_DIR / "input_rgb.png"
DRIVE_MASK_PATH = INPUT_DIR / "drive_mask_small.png"

OUTPUT_DIR = Path("/home/minsu/disk_a/miniconda3/graduation_work_robot_vision/astar_outputs_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# =========================
# 파라미터
# =========================
OBSTACLE_THRESHOLD = 255
START_MARGIN_BOTTOM = 5
GOAL_SEARCH_ROWS = 15
GOAL_CENTER_BIAS = 8.0  # 중앙 선호 강화
GOAL_ROW_RATIO = 0.12         # 목표점 y 위치: 화면 위에서 12% 지점
GOAL_SEARCH_RADIUS_X = 20     # 중앙에서 좌우로 탐색할 반경
GOAL_SEARCH_RADIUS_Y = 8      # 목표 y 주변 상하 탐색 반경

NEIGHBORS_8 = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)
]


def load_drive_mask(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return (img > 127).astype(np.uint8)


def heuristic(a, b):
    return np.hypot(a[0] - b[0], a[1] - b[1])


def get_neighbors(y, x, h, w):
    for dy, dx, move_cost in NEIGHBORS_8:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w:
            yield ny, nx, move_cost


def astar(cost_map, start, goal):
    h, w = cost_map.shape
    open_heap = []
    heapq.heappush(open_heap, (0.0, start))

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
        for ny, nx, move_cost in get_neighbors(cy, cx, h, w):
            if cost_map[ny, nx] >= OBSTACLE_THRESHOLD:
                continue

            neighbor = (ny, nx)
            tentative_g = g_score[current] + float(cost_map[ny, nx]) + move_cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f, neighbor))

    return None


def find_start(cost_map):
    h, w = cost_map.shape
    y = h - 1 - START_MARGIN_BOTTOM
    x_center = w // 2

    for offset in range(w // 2):
        for x in [x_center - offset, x_center + offset]:
            if 0 <= x < w and cost_map[y, x] < OBSTACLE_THRESHOLD:
                return (y, x)

    return None


def find_goal(cost_map, drive_mask):
    h, w = cost_map.shape
    x_center = w // 2

    search_bands = [
        (0.35, 0.65),
        (0.30, 0.70),
        (0.20, 0.80),
    ]

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

                score = y * 3.0 + abs(x - x_center) * 2.0 + float(cost_map[y, x])
                candidates.append((score, (y, x)))
        
        print(f"band {left_ratio:.2f}~{right_ratio:.2f}, goal candidates:", len(candidates))

        if candidates:
            candidates.sort(key=lambda t: t[0])
            return candidates[0][1]
            

    return None


def draw_path_on_costmap(cost_map, path, start, goal, save_path):
    vis = cv2.applyColorMap(np.uint8(255 - cost_map), cv2.COLORMAP_JET)
    vis[cost_map >= OBSTACLE_THRESHOLD] = (0, 0, 0)

    if path is not None and len(path) >= 2:
        pts = np.array([[x, y] for (y, x) in path], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], isClosed=False, color=(255, 255, 255), thickness=3)

    if start is not None:
        cv2.circle(vis, (start[1], start[0]), 3, (0, 255, 0), -1)
    if goal is not None:
        cv2.circle(vis, (goal[1], goal[0]), 3, (0, 0, 255), -1)

    cv2.imwrite(str(save_path), vis)


def draw_path_on_rgb(rgb_path, path, start, goal, cost_map_shape, save_path):
    rgb = cv2.imread(str(rgb_path))
    h_img, w_img = rgb.shape[:2]
    h_map, w_map = cost_map_shape

    def scale_point(pt):
        y, x = pt
        sx = int(x * w_img / w_map)
        sy = int(y * h_img / h_map)
        return sx, sy

    if path is not None and len(path) >= 2:
        pts = np.array([scale_point(pt) for pt in path], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(rgb, [pts], isClosed=False, color=(0, 255, 255), thickness=4)

    if start is not None:
        x, y = scale_point(start)
        cv2.circle(rgb, (x, y), 6, (0, 255, 0), -1)

    if goal is not None:
        x, y = scale_point(goal)
        cv2.circle(rgb, (x, y), 6, (0, 0, 255), -1)

    cv2.imwrite(str(save_path), rgb)


def main():
    cost_map = np.load(COST_MAP_PATH)
    drive_mask = load_drive_mask(DRIVE_MASK_PATH)

    print("drive_mask drivable pixels:", int(drive_mask.sum()))

    start = find_start(cost_map)
    goal = find_goal(cost_map, drive_mask)

    print("start:", start)
    print("goal:", goal)

    if start is None or goal is None:
        print("[오류] 시작점 또는 목표점을 찾지 못했습니다.")
        return

    path = astar(cost_map, start, goal)

    if path is None:
        print("[경고] 경로를 찾지 못했습니다.")
    else:
        print(f"경로 길이: {len(path)}")

    draw_path_on_costmap(
        cost_map=cost_map,
        path=path,
        start=start,
        goal=goal,
        save_path=OUTPUT_DIR / "astar_on_costmap.png"
    )

    draw_path_on_rgb(
        rgb_path=RGB_PATH,
        path=path,
        start=start,
        goal=goal,
        cost_map_shape=cost_map.shape,
        save_path=OUTPUT_DIR / "astar_on_rgb.png"
    )

    print("저장 완료:")
    print(OUTPUT_DIR / "astar_on_costmap.png")
    print(OUTPUT_DIR / "astar_on_rgb.png")


if __name__ == "__main__":
    main()