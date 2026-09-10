from pathlib import Path
import zipfile

ZIP_DIR = Path("/home/minsu/disk_a/robot_dataset/182.대형시설_실내·인접_자율_배송_데이터/01-1.정식개방데이터/Other/zipped")
EXTRACT_DIR = Path("/home/minsu/disk_a/robot_dataset/182.대형시설_실내·인접_자율_배송_데이터/01-1.정식개방데이터/Other/extracted")

EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

zip_files = sorted(ZIP_DIR.glob("*.zip"))
print(f"zip 파일 수: {len(zip_files)}")

for idx, zip_path in enumerate(zip_files, 1):
    out_dir = EXTRACT_DIR / zip_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{idx}/{len(zip_files)}] 압축 해제 중: {zip_path.name}")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
    except Exception as e:
        print(f"  오류: {e}")

print("완료")