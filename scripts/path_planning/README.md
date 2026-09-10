# 경로계획

Cost Map을 입력으로 받아 안전 경로를 탐색하고, 별도 학습 실험을 위한 경로 데이터셋을 생성합니다.

- `run_astar.py`, `run_astar_v2.py`: 8방향 고전 A* 기반 안전 경로 탐색
- `generate_planner_dataset.py`: Cost Map·시작점·목표점·A* 경로 학습 샘플 생성
- `train_neural_astar.py`, `infer_neural_astar.py`: Small U-Net 기반 경로 마스크 예측 실험

웹 데모에 통합된 경로 탐색은 고전 A*이며, Small U-Net은 별도 실험으로 구분합니다.
