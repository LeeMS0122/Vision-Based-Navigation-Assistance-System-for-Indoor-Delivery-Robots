# 주행 영역·장애물 분할

YOLO Segmentation을 학습하고 추론해 주행 가능 영역과 장애물 마스크를 생성합니다.

- `train_yolo_seg_wandb.py`: YOLO-seg 학습과 W&B 실험 로깅
- `run_yolo_infer.py`: 이미지 추론 후 `drivable_area`·`obstacle` 마스크 생성

학습 데이터셋과 모델 가중치는 코드에 포함하지 않으며, 스크립트 상단의 경로를 실행 환경에 맞게 설정해야 합니다.
