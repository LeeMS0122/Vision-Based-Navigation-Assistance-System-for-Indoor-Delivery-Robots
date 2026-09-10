import csv
import shutil
from pathlib import Path

# =========================
# 원본 데이터 경로
# =========================
SRC_ROOT = Path("/home/minsu/disk_a/robot_dataset/182.대형시설_실내·인접_자율_배송_데이터/01-1.정식개방데이터")

TRAIN_IMG_ROOT = SRC_ROOT / "Training" / "01.원천데이터" / "extracted"
VAL_IMG_ROOT = SRC_ROOT / "Validation" / "01.원천데이터" / "extracted"

# =========================
# 출력 경로
# =========================
DST_ROOT = Path("/home/minsu/disk_a/robot_dataset_scdepth_v3")

TRAIN_DST = DST_ROOT / "train"
VAL_DST = DST_ROOT / "val"

TRAIN_LIST = DST_ROOT / "train_files.txt"
VAL_LIST = DST_ROOT / "val_files.txt"
META_CSV = DST_ROOT / "sequence_meta.csv"

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def ensure_dirs():
    TRAIN_DST.mkdir(parents=True, exist_ok=True)
    VAL_DST.mkdir(parents=True, exist_ok=True)


def get_camera_dirs(root: Path):
    """
    extracted 바로 아래의 *.camera 폴더만 수집
    """
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.endswith(".camera")])


def get_sorted_images(camera_dir: Path):
    """
    파일명 기준 정렬
    예: 20220719_174205_71.png, 20220719_174206_31.png ...
    """
    return sorted([p for p in camera_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def process_split(split_name: str, src_root: Path, dst_root: Path, start_seq_idx: int, meta_rows: list, list_lines: list):
    camera_dirs = get_camera_dirs(src_root)
    print(f"[{split_name}] camera 폴더 수: {len(camera_dirs)}")

    seq_idx = start_seq_idx
    total_frames = 0
    skipped_empty_dirs = 0

    for folder_idx, camera_dir in enumerate(camera_dirs, 1):
        if folder_idx % 20 == 0:
            print(f"[{split_name}] 진행중... {folder_idx}/{len(camera_dirs)}")

        images = get_sorted_images(camera_dir)
        if len(images) == 0:
            skipped_empty_dirs += 1
            continue

        seq_name = f"seq_{seq_idx:05d}"
        seq_dir = dst_root / seq_name
        seq_dir.mkdir(parents=True, exist_ok=True)

        for frame_idx, img_path in enumerate(images):
            new_name = f"{frame_idx:06d}{img_path.suffix.lower()}"
            dst_img_path = seq_dir / new_name

            shutil.copy2(img_path, dst_img_path)

            stem_no_ext = Path(new_name).stem
            list_lines.append(f"{seq_name} {stem_no_ext}")

            meta_rows.append({
                "split": split_name,
                "seq_name": seq_name,
                "frame_idx": frame_idx,
                "new_file_name": new_name,
                "original_folder_name": camera_dir.name,
                "original_file_name": img_path.name,
                "original_full_path": str(img_path),
            })

            total_frames += 1

        seq_idx += 1

    print(f"\n[{split_name}] 완료")
    print(f"- 생성된 시퀀스 수: {seq_idx - start_seq_idx}")
    print(f"- 총 프레임 수: {total_frames}")
    print(f"- 빈 폴더 스킵: {skipped_empty_dirs}")

    return seq_idx


def write_list_file(path: Path, lines: list):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def write_meta_csv(path: Path, rows: list):
    fieldnames = [
        "split",
        "seq_name",
        "frame_idx",
        "new_file_name",
        "original_folder_name",
        "original_file_name",
        "original_full_path",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ensure_dirs()

    meta_rows = []
    train_lines = []
    val_lines = []

    next_seq_idx = 1
    next_seq_idx = process_split(
        split_name="train",
        src_root=TRAIN_IMG_ROOT,
        dst_root=TRAIN_DST,
        start_seq_idx=next_seq_idx,
        meta_rows=meta_rows,
        list_lines=train_lines,
    )

    next_seq_idx = process_split(
        split_name="val",
        src_root=VAL_IMG_ROOT,
        dst_root=VAL_DST,
        start_seq_idx=1,
        meta_rows=meta_rows,
        list_lines=val_lines,
    )

    write_list_file(TRAIN_LIST, train_lines)
    write_list_file(VAL_LIST, val_lines)
    write_meta_csv(META_CSV, meta_rows)

    print("\n모든 작업 완료")
    print(f"출력 경로: {DST_ROOT}")
    print(f"train list: {TRAIN_LIST}")
    print(f"val list: {VAL_LIST}")
    print(f"meta csv: {META_CSV}")


if __name__ == "__main__":
    main()