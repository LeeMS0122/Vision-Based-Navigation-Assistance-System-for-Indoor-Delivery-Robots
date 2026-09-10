from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn

# =========================
# 경로 설정
# =========================
DATA_ROOT = Path("/home/minsu/disk_a/miniconda3/graduation_work_robot_vision/planner_dataset/val")
CKPT_PATH = Path("/home/minsu/disk_a/miniconda3/graduation_work_robot_vision/planner_ckpts/best.pt")
OUTPUT_DIR = Path("/home/minsu/disk_a/miniconda3/graduation_work_robot_vision/neural_astar_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 시각화할 샘플 번호
SAMPLE_INDEX = 0


class SmallUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.enc1 = block(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = block(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = block(64, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = block(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = block(64, 32)

        self.out_conv = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)


def save_binary(mask: np.ndarray, path: Path):
    img = (mask * 255).astype(np.uint8)
    cv2.imwrite(str(path), img)


def save_costmap(cost_map: np.ndarray, path: Path):
    vis = cv2.applyColorMap(np.uint8(255 - cost_map), cv2.COLORMAP_JET)
    vis[cost_map >= 255] = (0, 0, 0)
    cv2.imwrite(str(path), vis)


def overlay_path(cost_map: np.ndarray, gt_path: np.ndarray, pred_path: np.ndarray,
                 start: np.ndarray, goal: np.ndarray, path: Path):
    vis = cv2.applyColorMap(np.uint8(255 - cost_map), cv2.COLORMAP_JET)
    vis[cost_map >= 255] = (0, 0, 0)

    # GT path = white
    vis[gt_path > 0] = (255, 255, 255)

    # Pred path = yellow
    vis[pred_path > 0] = (0, 255, 255)

    # Start = green
    ys, xs = np.where(start > 0)
    for y, x in zip(ys, xs):
        cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)

    # Goal = red
    ys, xs = np.where(goal > 0)
    for y, x in zip(ys, xs):
        cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)

    cv2.imwrite(str(path), vis)


def main():
    cost_files = sorted(DATA_ROOT.glob("*_cost.npy"))
    if len(cost_files) == 0:
        print("[오류] planner val 데이터가 없습니다.")
        return

    cost_path = cost_files[SAMPLE_INDEX]
    prefix = cost_path.stem.replace("_cost", "")

    cost = np.load(DATA_ROOT / f"{prefix}_cost.npy").astype(np.float32)
    start = np.load(DATA_ROOT / f"{prefix}_start.npy").astype(np.float32)
    goal = np.load(DATA_ROOT / f"{prefix}_goal.npy").astype(np.float32)
    gt_path = np.load(DATA_ROOT / f"{prefix}_path.npy").astype(np.float32)

    x = np.stack([cost / 255.0, start, goal], axis=0)
    x_tensor = torch.tensor(x).unsqueeze(0).to(DEVICE)

    model = SmallUNet().to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        logits = model(x_tensor)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()

    pred_path = (prob > 0.5).astype(np.uint8)

    # 저장
    save_costmap(cost.astype(np.uint8), OUTPUT_DIR / f"{prefix}_cost.png")
    save_binary(start, OUTPUT_DIR / f"{prefix}_start.png")
    save_binary(goal, OUTPUT_DIR / f"{prefix}_goal.png")
    save_binary(gt_path, OUTPUT_DIR / f"{prefix}_gt_path.png")
    save_binary(pred_path, OUTPUT_DIR / f"{prefix}_pred_path.png")

    overlay_path(
        cost_map=cost.astype(np.uint8),
        gt_path=gt_path,
        pred_path=pred_path,
        start=start,
        goal=goal,
        path=OUTPUT_DIR / f"{prefix}_overlay.png"
    )

    print("저장 완료:")
    print(OUTPUT_DIR / f"{prefix}_cost.png")
    print(OUTPUT_DIR / f"{prefix}_start.png")
    print(OUTPUT_DIR / f"{prefix}_goal.png")
    print(OUTPUT_DIR / f"{prefix}_gt_path.png")
    print(OUTPUT_DIR / f"{prefix}_pred_path.png")
    print(OUTPUT_DIR / f"{prefix}_overlay.png")


if __name__ == "__main__":
    main()