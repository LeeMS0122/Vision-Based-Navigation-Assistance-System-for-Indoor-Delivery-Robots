from pathlib import Path

ROOT = Path("/home/minsu/disk_a/robot_dataset_scdepth_v3_official/training")

def to_scene3(name: str):
    if not name.startswith("Scene"):
        return name
    num = int(name.replace("Scene", ""))
    return f"Scene{num:03d}"

def rewrite_list_file(path: Path):
    if not path.exists():
        return
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    new_lines = [to_scene3(x) for x in lines]
    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

def main():
    scene_dirs = sorted([p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("Scene")], reverse=True)

    for p in scene_dirs:
        new_name = to_scene3(p.name)
        if new_name != p.name:
            p.rename(p.parent / new_name)
            print(f"{p.name} -> {new_name}")

    for fname in ["train.txt", "val.txt", "train_small.txt", "val_small.txt",
                  "train_full_backup.txt", "val_full_backup.txt"]:
        rewrite_list_file(ROOT / fname)

    print("done")

if __name__ == "__main__":
    main()