# 샘플용 몇개 확인 해보는 용도
import random
from pathlib import Path
from PIL import Image, ImageDraw

# =========================
# 경로 설정
# =========================
DATASET_ROOT = Path("/home/minsu/disk_a/robot_dataset_yolo_details_v2")
OUTPUT_ROOT = Path("/home/minsu/disk_a/miniconda3/graduation_work_robot_vision/label_vis")

# split 선택: "train" 또는 "val"
# SPLIT = "train"
SPLIT = "val"

# 몇 장 저장할지
NUM_SAMPLES = 10

# 클래스 이름
CLASS_NAMES = {
    0: "drivable_area",
    1: "obstacle",
}

# 색상 (RGBA)
CLASS_COLORS = {
    0: (0, 255, 0, 90),     # drivable_area: 초록
    1: (255, 0, 0, 90),     # obstacle: 빨강
}

# 외곽선 색
OUTLINE_COLORS = {
    0: (0, 180, 0, 255),
    1: (200, 0, 0, 255),
}


def load_yolo_seg_label(label_path, img_w, img_h):
    polygons = []

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) < 7:
            continue

        class_id = int(parts[0])
        coords = list(map(float, parts[1:]))

        if len(coords) % 2 != 0:
            continue

        points = []
        for i in range(0, len(coords), 2):
            x = coords[i] * img_w
            y = coords[i + 1] * img_h
            points.append((x, y))

        if len(points) >= 3:
            polygons.append((class_id, points))

    return polygons


def draw_polygons(image_path, label_path, save_path):
    image = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    img_w, img_h = image.size
    polygons = load_yolo_seg_label(label_path, img_w, img_h)

    for class_id, points in polygons:
        fill_color = CLASS_COLORS.get(class_id, (255, 255, 0, 90))
        outline_color = OUTLINE_COLORS.get(class_id, (255, 255, 0, 255))

        draw.polygon(points, fill=fill_color, outline=outline_color)

    blended = Image.alpha_composite(image, overlay).convert("RGB")
    blended.save(save_path)


def main():
    image_dir = DATASET_ROOT / "images" / SPLIT
    label_dir = DATASET_ROOT / "labels" / SPLIT

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_split_dir = OUTPUT_ROOT / SPLIT
    out_split_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(list(image_dir.glob("*.png")))
    if len(image_paths) == 0:
        print(f"[오류] 이미지가 없습니다: {image_dir}")
        return

    sample_paths = random.sample(image_paths, min(NUM_SAMPLES, len(image_paths)))

    print(f"[{SPLIT}] 샘플 {len(sample_paths)}장 시각화 시작")

    for idx, image_path in enumerate(sample_paths, 1):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            print(f"[경고] 라벨 없음: {label_path}")
            continue

        save_path = out_split_dir / f"{image_path.stem}_vis.jpg"
        draw_polygons(image_path, label_path, save_path)
        print(f"[{idx}/{len(sample_paths)}] 저장 완료: {save_path.name}")

    print("\n완료")
    print(f"저장 경로: {out_split_dir}")


if __name__ == "__main__":
    main()