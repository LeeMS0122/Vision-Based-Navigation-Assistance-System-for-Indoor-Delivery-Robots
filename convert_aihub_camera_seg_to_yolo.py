import json
import shutil
from pathlib import Path
from collections import Counter

# =========================
# 원본 데이터 경로
# =========================
SRC_ROOT = Path("/home/minsu/disk_a/robot_dataset/182.대형시설_실내·인접_자율_배송_데이터/01-1.정식개방데이터")

TRAIN_IMG_ROOT = SRC_ROOT / "Training" / "01.원천데이터" / "extracted"
TRAIN_LBL_ROOT = SRC_ROOT / "Training" / "02.라벨링데이터" / "extracted"

VAL_IMG_ROOT = SRC_ROOT / "Validation" / "01.원천데이터" / "extracted"
VAL_LBL_ROOT = SRC_ROOT / "Validation" / "02.라벨링데이터" / "extracted"

# =========================
# 출력 경로
# =========================
DST_ROOT = Path("/home/minsu/disk_a/robot_dataset_yolo_details_v2")

# =========================
# 클래스 매핑
# =========================
# CLASS_MAP = {
#     "OUT_Road": "drivable_area",

#     "Vehicle": "obstacle",
#     "Pedestrian": "obstacle",
#     "OUT_Tree": "obstacle",
#     "OUT_Sign": "obstacle",
#     "OUT_Structure": "obstacle",
#     "OUT_Fence": "obstacle",
#     "OUT_Pole": "obstacle",
#     "OUT_Curb": "obstacle",
# }
CLASS_MAP = {
    # drivable_area
    "OUT_Road": "drivable_area",
    "OUT_Pavement": "drivable_area",
    "IN_Hall Way": "drivable_area",
    "IN_Open Space": "drivable_area",

    # obstacle
    "Vehicle": "obstacle",
    "Two-wheeled Vehicle": "obstacle",
    "Human": "obstacle",
    "Animal": "obstacle",
    "Stroller": "obstacle",
    "Wheelchair": "obstacle",
    "Pedestrian": "obstacle",

    "OUT_Tree": "obstacle",
    "OUT_Sign": "obstacle",
    "OUT_Structure": "obstacle",
    "OUT_Fence": "obstacle",
    "OUT_Pole": "obstacle",
    "OUT_Curbstone": "obstacle",
    "OUT_Bench": "obstacle",
    "OUT_Gate": "obstacle",
    "OUT_Traffic Safety Stuff": "obstacle",
    "OUT_Sculpture": "obstacle",

    "IN_Wall": "obstacle",
    "IN_Obstruction": "obstacle",
    "IN_Gate": "obstacle",
    "IN_Pillar": "obstacle",
    "IN_Sign": "obstacle",
    "IN_Fence": "obstacle",
    "IN_Bench": "obstacle",
    "IN_Emergency Stuff": "obstacle",
    "IN_Elevator": "obstacle",
}

FINAL_CLASSES = ["drivable_area", "obstacle"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(FINAL_CLASSES)}


def ensure_dirs():
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (DST_ROOT / sub).mkdir(parents=True, exist_ok=True)


def write_classes_txt():
    with open(DST_ROOT / "classes.txt", "w", encoding="utf-8") as f:
        for c in FINAL_CLASSES:
            f.write(c + "\n")


def write_data_yaml():
    yaml_text = f"""path: {DST_ROOT}
train: images/train
val: images/val

names:
  0: drivable_area
  1: obstacle
"""
    with open(DST_ROOT / "data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_text)


def normalize_point(x, y, width, height):
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    return x / width, y / height


def flatten_polygon(annotation):
    polygons = []

    def recurse(item):
        if isinstance(item, list):
            if len(item) > 0 and all(isinstance(p, dict) and "x" in p and "y" in p for p in item):
                polygons.append(item)
            else:
                for sub in item:
                    recurse(sub)

    recurse(annotation)
    return polygons


def get_camera_dirs(root: Path):
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.endswith(".camera")])


def get_seg_dirs(root: Path):
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.endswith(".segmentation")])


def camera_key(name: str):
    return (
        name.replace("TS_", "")
            .replace("VS_", "")
            .replace(".camera", "")
    )

def seg_key(name: str):
    return (
        name.replace("TL_", "")
            .replace("VL_", "")
            .replace(".segmentation", "")
    )


def build_matched_dir_pairs(img_root: Path, lbl_root: Path):
    camera_dirs = {camera_key(p.name): p for p in get_camera_dirs(img_root)}
    seg_dirs = {seg_key(p.name): p for p in get_seg_dirs(lbl_root)}

    common_keys = sorted(set(camera_dirs.keys()) & set(seg_dirs.keys()))
    return [(camera_dirs[k], seg_dirs[k], k) for k in common_keys]


def convert_one_json(json_path, image_path, out_label_path, stats_raw, stats_final):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    width = data["meta"]["size"]["width"]
    height = data["meta"]["size"]["height"]
    objects = data.get("objects", [])

    yolo_lines = []

    for obj in objects:
        raw_class = obj.get("class_name", "").strip()
        stats_raw[raw_class] += 1

        if raw_class not in CLASS_MAP:
            continue

        final_class = CLASS_MAP[raw_class]
        class_id = CLASS_TO_ID[final_class]
        stats_final[final_class] += 1

        annotation = obj.get("annotation", [])
        polygons = flatten_polygon(annotation)

        for poly in polygons:
            if len(poly) < 3:
                continue

            coords = []
            for pt in poly:
                nx, ny = normalize_point(pt["x"], pt["y"], width, height)
                coords.extend([f"{nx:.6f}", f"{ny:.6f}"])

            if len(coords) >= 6:
                yolo_lines.append(f"{class_id} " + " ".join(coords))

    if not yolo_lines:
        return False

    with open(out_label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines) + "\n")

    split = out_label_path.parent.name
    out_image_path = DST_ROOT / "images" / split / image_path.name
    shutil.copy2(image_path, out_image_path)
    return True


def process_split(split_name, img_root, lbl_root):
    stats_raw = Counter()
    stats_final = Counter()

    matched_pairs = build_matched_dir_pairs(img_root, lbl_root)

    print(f"[{split_name}] 매칭된 camera/segmentation 폴더 수: {len(matched_pairs)}")

    converted = 0
    skipped_no_label = 0
    missing_images = 0
    missing_jsons = 0

    out_label_dir = DST_ROOT / "labels" / split_name

    for idx, (camera_dir, seg_dir, key) in enumerate(matched_pairs, 1):
        if idx % 20 == 0:
            print(f"[{split_name}] 폴더 진행중... {idx}/{len(matched_pairs)}")

        image_map = {p.stem: p for p in camera_dir.glob("*.png")}
        json_map = {p.stem: p for p in seg_dir.glob("*.json")}

        common_stems = sorted(set(image_map.keys()) & set(json_map.keys()))
        only_imgs = set(image_map.keys()) - set(json_map.keys())
        only_jsons = set(json_map.keys()) - set(image_map.keys())

        missing_jsons += len(only_imgs)
        missing_images += len(only_jsons)

        for stem in common_stems:
            image_path = image_map[stem]
            json_path = json_map[stem]
            out_label_path = out_label_dir / f"{stem}.txt"

            ok = convert_one_json(
                json_path=json_path,
                image_path=image_path,
                out_label_path=out_label_path,
                stats_raw=stats_raw,
                stats_final=stats_final,
            )

            if ok:
                converted += 1
            else:
                skipped_no_label += 1

    print(f"\n[{split_name}] 완료")
    print(f"- 변환 성공: {converted}")
    print(f"- 라벨 없음 스킵: {skipped_no_label}")
    print(f"- 이미지 없음: {missing_images}")
    print(f"- JSON 없음: {missing_jsons}")

    print(f"\n[{split_name}] 원본 클래스 통계")
    for k, v in stats_raw.most_common():
        print(f"  {k}: {v}")

    print(f"\n[{split_name}] 최종 클래스 통계")
    for k, v in stats_final.items():
        print(f"  {k}: {v}")


def main():
    ensure_dirs()
    write_classes_txt()
    write_data_yaml()

    process_split("train", TRAIN_IMG_ROOT, TRAIN_LBL_ROOT)
    process_split("val", VAL_IMG_ROOT, VAL_LBL_ROOT)

    print("\n완료")
    print(f"출력 경로: {DST_ROOT}")


if __name__ == "__main__":
    main()