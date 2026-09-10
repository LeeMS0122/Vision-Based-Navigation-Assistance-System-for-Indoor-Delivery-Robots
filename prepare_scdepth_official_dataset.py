import csv
import re
import shutil
from pathlib import Path

# =========================
# 입력 경로
# =========================
SEQ_ROOT = Path("/home/minsu/disk_a/robot_dataset_scdepth_v3")
META_CSV = SEQ_ROOT / "sequence_meta.csv"

CALIB_ROOT = Path(
    "/home/minsu/disk_a/robot_dataset/182.대형시설_실내·인접_자율_배송_데이터/01-1.정식개방데이터/Other/extracted"
)

# =========================
# 출력 경로
# =========================
OUT_ROOT = Path("/home/minsu/disk_a/robot_dataset_scdepth_v3_official")
TRAINING_ROOT = OUT_ROOT / "Training"
TESTING_ROOT = OUT_ROOT / "Testing"

TRAIN_TXT = TRAINING_ROOT / "train.txt"
VAL_TXT = TRAINING_ROOT / "val.txt"

SCENE_FMT = "Scene{:04d}"


def ensure_dirs():
    TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
    TESTING_ROOT.mkdir(parents=True, exist_ok=True)


def load_meta(meta_csv_path: Path):
    rows = []
    with open(meta_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def group_sequences(meta_rows):
    grouped = {}
    for row in meta_rows:
        key = (row["split"], row["seq_name"])
        if key not in grouped:
            grouped[key] = {
                "split": row["split"],
                "seq_name": row["seq_name"],
                "original_folder_name": row["original_folder_name"],
            }
    return list(grouped.values())


def normalize_camera_folder_name(name: str):
    """
    예:
    TS_객체인식(1Hz)_05.대구엑스코_시나리오B_sunny_20220828_09-11_01.camera
    -> 객체인식(1Hz)_05.대구엑스코_시나리오B_sunny_20220828_09-11
    """
    s = name.strip()

    if s.startswith("TS_"):
        s = s[3:]
    elif s.startswith("VS_"):
        s = s[3:]

    if s.endswith(".camera"):
        s = s[:-7]

    s = re.sub(r"_\d+$", "", s)
    return s


def normalize_calibration_folder_name(name: str):
    """
    예:
    01.메타데이터_객체인식(1Hz)_05.대구엑스코_시나리오B_sunny_20220828_09-11_03.calibration
    -> 객체인식(1Hz)_05.대구엑스코_시나리오B_sunny_20220828_09-11
    """
    s = name.strip()

    s = re.sub(r"^\d+\.메타데이터_", "", s)

    if s.endswith(".calibration"):
        s = s[:-12]

    s = re.sub(r"_\d+$", "", s)
    return s


def find_intrinsic_file(calib_dir: Path):
    """
    지원 패턴:
    1) cam_intrinsic_calibration_0719.txt
    2) camera.txt
    3) 2022_07_24_camera.txt
    """
    files = sorted(calib_dir.glob("cam_intrinsic_calibration_*.txt"))
    if files:
        return files[0]

    fallback = calib_dir / "camera.txt"
    if fallback.exists():
        return fallback

    files = sorted(calib_dir.glob("*_camera.txt"))
    if files:
        return files[0]

    return None


def parse_intrinsic_file(txt_path: Path):
    """
    width / height / camera matrix 기반 파싱
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    width = None
    height = None
    K = None

    for i, line in enumerate(lines):
        lower = line.lower()

        if lower == "width" and i + 1 < len(lines):
            try:
                width = int(float(lines[i + 1]))
            except:
                pass

        elif lower == "height" and i + 1 < len(lines):
            try:
                height = int(float(lines[i + 1]))
            except:
                pass

        elif lower == "camera matrix" and i + 3 < len(lines):
            try:
                row1 = list(map(float, lines[i + 1].split()))
                row2 = list(map(float, lines[i + 2].split()))
                row3 = list(map(float, lines[i + 3].split()))
                if len(row1) == 3 and len(row2) == 3 and len(row3) == 3:
                    K = [row1, row2, row3]
            except:
                pass

    if K is None:
        raise ValueError(f"camera matrix를 찾지 못했습니다: {txt_path}")

    return {
        "width": width,
        "height": height,
        "K": K,
        "source_file": str(txt_path),
    }


def collect_calibrations(calib_root: Path):
    """
    calibration 폴더명을 정규화한 key -> intrinsic 정보
    """
    calib_map = {}
    duplicate_keys = []
    parse_failures = []

    calib_dirs = sorted([p for p in calib_root.iterdir() if p.is_dir()])
    print(f"calibration 폴더 수: {len(calib_dirs)}")

    for idx, calib_dir in enumerate(calib_dirs, 1):
        if idx % 20 == 0:
            print(f"[calib] 진행중... {idx}/{len(calib_dirs)}")

        norm_key = normalize_calibration_folder_name(calib_dir.name)
        intrinsic_file = find_intrinsic_file(calib_dir)

        if intrinsic_file is None:
            continue

        try:
            intrinsic_info = parse_intrinsic_file(intrinsic_file)
        except Exception as e:
            parse_failures.append((str(intrinsic_file), str(e)))
            continue

        if norm_key in calib_map:
            duplicate_keys.append((norm_key, str(calib_dir), calib_map[norm_key]["source_file"]))
            continue

        calib_map[norm_key] = intrinsic_info

    if parse_failures:
        fail_csv = OUT_ROOT / "calibration_parse_failures.csv"
        fail_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(fail_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["intrinsic_file", "error"])
            writer.writerows(parse_failures)
        print(f"[참고] intrinsic 파싱 실패 목록 저장: {fail_csv}")

    if duplicate_keys:
        dup_csv = OUT_ROOT / "duplicate_calibration_keys.csv"
        with open(dup_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["normalized_key", "duplicate_calibration_dir", "kept_intrinsic_source_file"])
            writer.writerows(duplicate_keys)
        print(f"[참고] 중복 calibration key 목록 저장: {dup_csv}")

    return calib_map


def write_cam_txt(cam_txt_path: Path, K):
    with open(cam_txt_path, "w", encoding="utf-8") as f:
        for row in K:
            f.write(" ".join(f"{v:.6f}" for v in row) + "\n")


def convert_split_dir(split_name: str):
    if split_name == "train":
        return SEQ_ROOT / "train"
    elif split_name == "val":
        return SEQ_ROOT / "val"
    else:
        raise ValueError(split_name)


def main():
    ensure_dirs()

    meta_rows = load_meta(META_CSV)
    seq_infos = group_sequences(meta_rows)
    calib_map = collect_calibrations(CALIB_ROOT)

    print(f"시퀀스 수: {len(seq_infos)}")
    print(f"수집된 calibration key 수: {len(calib_map)}")

    train_scene_names = []
    val_scene_names = []

    scene_idx = 0
    missing_calib = []

    for seq_info in seq_infos:
        split = seq_info["split"]
        seq_name = seq_info["seq_name"]
        original_folder_name = seq_info["original_folder_name"]

        src_seq_dir = convert_split_dir(split) / seq_name
        if not src_seq_dir.exists():
            print(f"[경고] 시퀀스 폴더 없음: {src_seq_dir}")
            continue

        norm_key = normalize_camera_folder_name(original_folder_name)
        intrinsic_info = calib_map.get(norm_key)

        scene_name = SCENE_FMT.format(scene_idx)
        dst_scene_dir = TRAINING_ROOT / scene_name
        dst_scene_dir.mkdir(parents=True, exist_ok=True)

        # 이미지 복사
        for img_path in sorted(src_seq_dir.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                shutil.copy2(img_path, dst_scene_dir / img_path.name)

        # cam.txt 생성
        if intrinsic_info is None:
            missing_calib.append((scene_name, split, seq_name, original_folder_name, norm_key))
            print(f"[경고] calibration 없음: {scene_name} | {original_folder_name}")
        else:
            write_cam_txt(dst_scene_dir / "cam.txt", intrinsic_info["K"])

        if split == "train":
            train_scene_names.append(scene_name)
        elif split == "val":
            val_scene_names.append(scene_name)

        scene_idx += 1

    with open(TRAIN_TXT, "w", encoding="utf-8") as f:
        for name in train_scene_names:
            f.write(name + "\n")

    with open(VAL_TXT, "w", encoding="utf-8") as f:
        for name in val_scene_names:
            f.write(name + "\n")

    print("\n완료")
    print(f"총 scene 수: {scene_idx}")
    print(f"train scene 수: {len(train_scene_names)}")
    print(f"val scene 수: {len(val_scene_names)}")
    print(f"출력 경로: {OUT_ROOT}")
    print(f"train.txt: {TRAIN_TXT}")
    print(f"val.txt: {VAL_TXT}")

    if missing_calib:
        missing_path = OUT_ROOT / "missing_calibration.csv"
        with open(missing_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["scene_name", "split", "seq_name", "original_folder_name", "normalized_key"])
            writer.writerows(missing_calib)

        print(f"\ncalibration 누락 scene 수: {len(missing_calib)}")
        print(f"누락 목록 저장: {missing_path}")
    else:
        print("\n모든 scene에 calibration 매칭 완료")


if __name__ == "__main__":
    main()