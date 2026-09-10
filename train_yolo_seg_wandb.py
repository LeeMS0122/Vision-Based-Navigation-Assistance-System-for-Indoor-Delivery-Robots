import wandb
from ultralytics import YOLO

DATA_YAML = "/home/minsu/disk_a/robot_dataset_yolo_details_v2/data.yaml"
MODEL_NAME = "yolov8m-seg.pt"

run = wandb.init(
    project="robot-vision-seg",
    name="v8m_seg_b16_e500",
    config={
        "model": MODEL_NAME,
        "data": DATA_YAML,
        "epochs": 500,
        "patience": 50,
        "imgsz": 640,
        "batch": 16,
        "device": 0,
        "workers": 16,
    },
)

model = YOLO(MODEL_NAME)

results = model.train(
    data=DATA_YAML,
    epochs=500,
    patience=50,
    imgsz=640,
    batch=16,
    device=0,
    workers=16,
    project="/home/minsu/disk_a/miniconda3/graduation_work_robot_vision/runs",
    name="v8m_seg_b16_e500",
    plots=True,
    save=True,
)

wandb.finish()