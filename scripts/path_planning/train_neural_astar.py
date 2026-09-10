from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

DATA_ROOT = Path("/home/minsu/disk_a/miniconda3/graduation_work_robot_vision/planner_dataset")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-3

CKPT_DIR = Path("/home/minsu/disk_a/miniconda3/graduation_work_robot_vision/planner_ckpts")
CKPT_DIR.mkdir(parents=True, exist_ok=True)


class PlannerDataset(Dataset):
    def __init__(self, split="train"):
        self.root = DATA_ROOT / split
        self.cost_files = sorted(self.root.glob("*_cost.npy"))

    def __len__(self):
        return len(self.cost_files)

    def __getitem__(self, idx):
        cost_path = self.cost_files[idx]
        prefix = cost_path.stem.replace("_cost", "")

        cost = np.load(self.root / f"{prefix}_cost.npy").astype(np.float32) / 255.0
        start = np.load(self.root / f"{prefix}_start.npy").astype(np.float32)
        goal = np.load(self.root / f"{prefix}_goal.npy").astype(np.float32)
        path = np.load(self.root / f"{prefix}_path.npy").astype(np.float32)

        x = np.stack([cost, start, goal], axis=0)   # (3,H,W)
        y = np.expand_dims(path, axis=0)            # (1,H,W)

        return torch.tensor(x), torch.tensor(y)


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


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0

    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0

    for x, y in tqdm(loader, desc="val", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def main():
    train_ds = PlannerDataset("train")
    val_ds = PlannerDataset("val")

    print("train samples:", len(train_ds))
    print("val samples:", len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = SmallUNet().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = float("inf")

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = validate(model, val_loader, criterion)

        print(f"Epoch {epoch+1}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        torch.save(model.state_dict(), CKPT_DIR / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), CKPT_DIR / "best.pt")

    print("완료")
    print("best val:", best_val)
    print("ckpt dir:", CKPT_DIR)


if __name__ == "__main__":
    main()