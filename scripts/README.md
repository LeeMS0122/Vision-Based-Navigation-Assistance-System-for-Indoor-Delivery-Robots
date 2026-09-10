# 실험 스크립트 구성

루트에 나열되어 있던 실험 코드를 실행 단계에 따라 다섯 영역으로 구분했습니다. 각 스크립트의 경로 상수는 자신의 데이터셋·모델 환경에 맞게 설정해야 합니다.

## 권장 실행 순서

```text
data_preparation
       ↓
segmentation + depth_estimation
       ↓
costmap
       ↓
path_planning
       ↓
web_demo
```

## `data_preparation/` — 데이터 준비

| 파일 | 설명 |
|---|---|
| `extract_other_zips.py` | AI Hub 부가 데이터 ZIP 일괄 해제 |
| `convert_aihub_camera_seg_to_yolo.py` | AI Hub Polygon 라벨을 YOLO Segmentation 형식으로 변환 |
| `prepare_scdepth_dataset.py` | 카메라 프레임을 SC-Depth 시퀀스 데이터셋으로 구성 |
| `prepare_scdepth_official_dataset.py` | 시퀀스·보정 정보를 SC-Depth V3 학습 구조로 재구성 |
| `rename_scene_to_3digits.py` | Scene 폴더와 목록의 번호 표기를 일괄 변경 |
| `visualize_yolo_seg_labels.py` | YOLO Segmentation 라벨 샘플 시각화 |

## `segmentation/` — 주행 영역·장애물 분할

| 파일 | 설명 |
|---|---|
| `train_yolo_seg_wandb.py` | YOLO-seg 학습과 W&B 실험 로깅 |
| `run_yolo_infer.py` | 학습된 모델로 주행 영역·장애물 마스크 생성 |

## `depth_estimation/` — 상대 심도

| 파일 | 설명 |
|---|---|
| `generate_pseudo_depth_small.py` | 소규모 학습 목록에 대한 가상 심도 생성 |
| `generate_pseudo_depth_full.py` | 전체 학습·검증 장면의 가상 심도 생성 |
| `infer_scdepth_sample.py` | SC-Depth V3 대표 프레임 추론 |
| `evaluate_depth_samples.py` | 샘플별 상대 심도 분포와 시각화 결과 저장 |

## `costmap/` — 위험도 맵 생성

| 파일 | 설명 |
|---|---|
| `build_cost_map.py` | Road·Obstacle·Depth 합성 로직 기본 검증 |
| `build_cost_map_real.py` | YOLO-seg·SC-Depth V3 추론 결과로 Cost Map 생성 |
| `build_cost_map_real_v2.py` | 장애물 버퍼·중앙 편향·가장자리 비용을 반영한 개선형 Cost Map 생성 |

## `path_planning/` — 경로계획

| 파일 | 설명 |
|---|---|
| `run_astar.py` | Cost Map 기반 8방향 고전 A* 경로 탐색 |
| `run_astar_v2.py` | 주행 가능 마스크와 목표점 탐색을 보강한 A* 실험 |
| `generate_planner_dataset.py` | Cost Map·시작점·목표점·A* 경로 학습 샘플 생성 |
| `train_neural_astar.py` | Small U-Net 기반 경로 마스크 예측 모델 학습 |
| `infer_neural_astar.py` | Small U-Net 추론 결과와 정답 경로 비교 |
