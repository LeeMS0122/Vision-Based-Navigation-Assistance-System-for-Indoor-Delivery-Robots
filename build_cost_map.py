from pathlib import Path
import numpy as np
import cv2


# =========================
# 설정
# =========================
OUTPUT_DIR = Path("/home/minsu/disk_a/miniconda3/graduation_work_robot_vision/cost_map_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = (160, 96)   # (width, height) for planning

# cost parameters
DRIVABLE_BASE_COST = 1
NON_DRIVABLE_BASE_COST = 80
OBSTACLE_COST = 255
DEPTH_PENALTY_SCALE = 50


def resize_mask(mask: np.ndarray, size):
    """0/1 mask resize"""
    resized = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    return (resized > 0).astype(np.uint8)


def resize_depth(depth: np.ndarray, size):
    return cv2.resize(depth.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)


def normalize_depth(depth: np.ndarray):
    """
    depth를 0~1로 정규화
    이상치 영향을 줄이기 위해 percentile 사용
    """
    d = depth.astype(np.float32).copy()
    p2 = np.percentile(d, 2)
    p98 = np.percentile(d, 98)

    if p98 - p2 < 1e-8:
        return np.zeros_like(d, dtype=np.float32)

    d = np.clip(d, p2, p98)
    d = (d - p2) / (p98 - p2)
    return d


def build_cost_map(drive_mask: np.ndarray, obs_mask: np.ndarray, depth_map: np.ndarray):
    """
    drive_mask: 0/1
    obs_mask:   0/1
    depth_map:  float array
    """

    # 1. planning size로 통일
    drive_mask = resize_mask(drive_mask, TARGET_SIZE)
    obs_mask = resize_mask(obs_mask, TARGET_SIZE)
    depth_map = resize_depth(depth_map, TARGET_SIZE)

    # 2. depth normalize
    depth_norm = normalize_depth(depth_map)

    # 주의:
    # 현재는 "depth_norm이 클수록 멀다"고 가정
    # 만약 실제 결과가 반대면 여기만 뒤집으면 됨.
    depth_penalty = (1.0 - depth_norm) * DEPTH_PENALTY_SCALE

    # 3. base cost
    cost_map = np.where(
        drive_mask == 1,
        DRIVABLE_BASE_COST,
        NON_DRIVABLE_BASE_COST
    ).astype(np.float32)

    # 4. depth penalty 더하기
    cost_map += depth_penalty

    # 5. obstacle은 통과 불가 수준으로 덮기
    cost_map[obs_mask == 1] = OBSTACLE_COST

    # 6. clip
    cost_map = np.clip(cost_map, 0, 255).astype(np.uint8)

    return cost_map, drive_mask, obs_mask, depth_norm


def save_costmap_visualization(cost_map: np.ndarray, save_path: Path):
    """
    낮은 cost=초록, 높은 cost=빨강 느낌의 컬러맵 저장
    obstacle(255)은 검정으로 표시
    """
    vis = cost_map.copy()

    # obstacle은 따로 표시
    obstacle_region = (vis == 255)

    color = cv2.applyColorMap(
        np.uint8(255 - vis),  # 낮은 cost가 더 밝게 보이도록 반전
        cv2.COLORMAP_JET
    )

    color[obstacle_region] = (0, 0, 0)
    cv2.imwrite(str(save_path), color)


def save_binary_mask(mask: np.ndarray, save_path: Path):
    img = (mask * 255).astype(np.uint8)
    cv2.imwrite(str(save_path), img)


def save_depth_norm(depth_norm: np.ndarray, save_path: Path):
    img = np.clip(depth_norm * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(str(save_path), img)


def main():
    """
    지금은 예시용 더미 입력으로 구조만 확인.
    나중에 여기 부분을 YOLO/SC-Depth 출력으로 교체하면 됨.
    """
    H, W = 384, 640

    # -------------------------
    # 예시용 가짜 마스크/깊이
    # -------------------------
    drive_mask = np.zeros((H, W), dtype=np.uint8)
    drive_mask[180:, :] = 1   # 아래쪽 바닥이 주행 가능하다고 가정

    obs_mask = np.zeros((H, W), dtype=np.uint8)
    obs_mask[220:320, 260:360] = 1  # 중앙 장애물 하나 가정

    # 가까운 쪽이 아래, 먼 쪽이 위라고 가정한 depth
    y = np.linspace(0, 1, H).reshape(H, 1)
    depth_map = np.repeat(y, W, axis=1).astype(np.float32)

    # -------------------------
    # cost map 생성
    # -------------------------
    cost_map, drive_r, obs_r, depth_norm = build_cost_map(
        drive_mask=drive_mask,
        obs_mask=obs_mask,
        depth_map=depth_map
    )

    # -------------------------
    # 저장
    # -------------------------
    save_binary_mask(drive_r, OUTPUT_DIR / "drive_mask.png")
    save_binary_mask(obs_r, OUTPUT_DIR / "obs_mask.png")
    save_depth_norm(depth_norm, OUTPUT_DIR / "depth_norm.png")
    save_costmap_visualization(cost_map, OUTPUT_DIR / "cost_map.png")

    print("저장 완료:")
    print(OUTPUT_DIR / "drive_mask.png")
    print(OUTPUT_DIR / "obs_mask.png")
    print(OUTPUT_DIR / "depth_norm.png")
    print(OUTPUT_DIR / "cost_map.png")


if __name__ == "__main__":
    main()