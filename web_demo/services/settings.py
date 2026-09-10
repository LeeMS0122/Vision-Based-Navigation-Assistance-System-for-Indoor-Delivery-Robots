from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_DEMO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCDEPTH_ROOT = PROJECT_ROOT.parent / "sc_depth_pl"

YOLO_MODEL_PATH = Path(
    os.getenv(
        "YOLO_MODEL_PATH",
        str(PROJECT_ROOT / "runs" / "v8m_seg_b16_e500" / "weights" / "best.pt"),
    )
)
SCDEPTH_CKPT_PATH = Path(
    os.getenv(
        "SCDEPTH_CKPT_PATH",
        str(
            DEFAULT_SCDEPTH_ROOT
            / "ckpts"
            / "robot_scv3_full"
            / "version_0"
            / "epoch=16-val_loss=0.3801.ckpt"
        ),
    )
)
SCDEPTH_PROJECT_ROOT = Path(os.getenv("SCDEPTH_PROJECT_ROOT", str(DEFAULT_SCDEPTH_ROOT)))

RUNS_BASE_DIR = WEB_DEMO_ROOT / "data" / "runs"
VIDEO_RUNS_BASE_DIR = WEB_DEMO_ROOT / "outputs" / "video_runs"

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

SCDEPTH_INFER_SIZE = (640, 384)  # (width, height)
TARGET_SIZE = (160, 96)  # (width, height)

DRIVE_CLASS_ID = 0
OBS_CLASS_ID = 1

DRIVABLE_BASE_COST = 1
NON_DRIVABLE_BASE_COST = 160
OBSTACLE_COST = 255
BUFFER_ZONE_COST = 220
DEPTH_PENALTY_SCALE = 80
BOTTOM_RISK_SCALE = 18
CENTER_BIAS_SCALE = 20
EDGE_MARGIN_X_RATIO = 0.12
EDGE_MARGIN_Y_RATIO = 0.05
EDGE_MARGIN_COST = 230

OBSTACLE_DILATION_KERNEL = 9
OBSTACLE_DILATION_ITER = 1

OBSTACLE_THRESHOLD = 255
START_MARGIN_BOTTOM = 5
GOAL_SEARCH_ROWS = 15
GOAL_CENTER_BIAS = 8.0
