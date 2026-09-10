# 실내 배송 로봇 비전 웹 데모

이미지 1장, 영상 업로드, 실시간 카메라 입력을 처리하는 FastAPI 기반 데모입니다. YOLO 분할, SC-Depth V3 상대 심도, Cost Map, 고전 A*를 하나의 파이프라인으로 연결합니다.

## 1. 구조

```text
web_demo/
  main.py
  routers/
    demo_router.py
    live_demo_router.py
    video_demo_router.py
  services/
    yolo_service.py
    scdepth_service.py
    costmap_service.py
    planner_service.py
    pipeline_service.py
    status_service.py
    video_service.py
    settings.py
  utils/
    coord_utils.py
    video_io.py
  templates/
    index.html
    live_demo.html
    video_demo.html
  static/
    css/style.css
    js/app.js
  data/runs/
  outputs/video_runs/
```

## 2. 의존성 설치

프로젝트 루트에서:

```bash
pip install -r web_demo/requirements.txt
```

또는 conda 재현 스크립트 사용:

```bash
bash web_demo/setup_env.sh
```

커스텀 환경명/파이썬 버전:

```bash
bash web_demo/setup_env.sh robot_vision_demo 3.10
```

주의:
- `torch`, `ultralytics`, `opencv-python`는 환경에 따라 설치 시간이 길 수 있습니다.
- SC-Depth V3 코드와 학습 가중치는 별도로 준비해야 합니다.
- `setup_env.sh`는 SC-Depth 호환을 위해 `pip<24.1`, `setuptools<81`, `pytorch-lightning==1.7.3`, `torchmetrics==0.9.3`를 자동 적용합니다.
- `ffmpeg`가 설치되어 있으면 결과 영상을 자동으로 H.264(`libx264`, `yuv420p`)로 변환해 브라우저 호환성을 높입니다.

## 3. 환경 변수

자신의 모델과 SC-Depth V3 설치 경로를 설정합니다. `.env.example`에는 비밀값 없이 필요한 항목만 정리해 두었습니다.

```bash
export YOLO_MODEL_PATH=/absolute/path/to/yolo-seg-best.pt
export SCDEPTH_CKPT_PATH=/absolute/path/to/sc-depth-v3.ckpt
export SCDEPTH_PROJECT_ROOT=/absolute/path/to/sc_depth_pl
```

환경 변수를 생략하면 YOLO 가중치는 프로젝트 내 `runs/v8m_seg_b16_e500/weights/best.pt`, SC-Depth V3는 프로젝트의 상위 경로에 있는 `sc_depth_pl`을 기준으로 찾습니다.

## 4. 실행

```bash
cd Vision-Based-Navigation-Assistance-System-for-Indoor-Delivery-Robots
uvicorn web_demo.main:app --host 0.0.0.0 --port 8000 --reload
```

브라우저 접속:

```text
http://<server-ip>:8000
```

## 5. API

- `GET /` : 업로드/결과 표시 페이지
- `GET /live-demo` : 실시간 카메라 데모 페이지
- `GET /video-demo` : 영상 업로드 데모 페이지
- `POST /api/pipeline/run` : 이미지 업로드 후 파이프라인 실행
- `POST /api/video/run` : 영상 업로드 후 샘플링 프레임 단위 분석 실행

`POST /api/pipeline/run` 응답(JSON) 예시:

```json
{
  "run_id": "20260331_203000_ab12cd34",
  "status": "ok",
  "summary": {
    "detected_objects": {"obstacle": 2},
    "nearest_obstacle_warning": "주의: 전방 장애물 존재 가능",
    "current_risk": "MEDIUM",
    "recommended_direction": "LEFT",
    "path_recalculation_required": false,
    "path_found": true,
    "path_length": 84
  },
  "images": {
    "original_image": "/demo-data/runs/.../original.png",
    "drive_mask": "/demo-data/runs/.../drive_mask.png",
    "obstacle_mask": "/demo-data/runs/.../obstacle_mask.png",
    "depth_visualization": "/demo-data/runs/.../depth_visualization.png",
    "cost_map": "/demo-data/runs/.../cost_map.png",
    "astar_on_rgb": "/demo-data/runs/.../astar_on_rgb.png"
  }
}
```

`POST /api/video/run` 요청 파라미터:
- `file` : 업로드 영상(`.mp4`, `.mov`, `.avi`, `.mkv`)
- `sample_fps` : 샘플링 fps (기본 `5`)

`POST /api/video/run` 응답(JSON) 주요 항목:

```json
{
  "run_id": "20260427_120000_ab12cd34",
  "status": "ok",
  "result_video_path": "/demo-outputs/video_runs/.../result_video.mp4",
  "frames_metadata_json_path": "/demo-outputs/video_runs/.../frames_metadata.json",
  "summary_json_path": "/demo-outputs/video_runs/.../summary.json",
  "summary": {
    "total_processed_frames": 120,
    "risk_counts": { "LOW": 80, "MEDIUM": 30, "HIGH": 10 }
  }
}
```

프레임 메타데이터(`frames_metadata.json`) 구조 예시:

```json
{
  "video_info": {
    "native_fps": 30.0,
    "sample_fps_actual": 5.0
  },
  "frames": [
    {
      "frame_index": 0,
      "source_frame_index": 0,
      "timestamp_sec": 0.0,
      "status": {
        "current_risk": "MEDIUM",
        "recommended_direction": "LEFT",
        "path_recalculation_required": false
      },
      "summary": {
        "detected_objects": { "obstacle": 2 },
        "path_length": 84
      }
    }
  ]
}
```

## 6. 출력 구조

- 기존 알고리즘을 최대한 유지하고 서비스 레이어에서 순차 연결
- `cost_map`은 `scripts/costmap/build_cost_map_real_v2.py` 튜닝 규칙을 서비스로 이관해 사용
  - drivable/non-drivable 기본 cost
  - obstacle=255
  - obstacle dilation + buffer zone cost
  - depth percentile clipping + normalize penalty
  - center bias + bottom risk + edge penalty
  - A* goal: 중앙 근처 + drivable 영역 제약
- 모델 로딩은 서비스 단에서 lazy load(첫 요청 시 1회 로딩)
- 이미지 업로드 결과: `web_demo/data/runs/<run_id>/`
- 영상 업로드 결과: `web_demo/outputs/video_runs/<run_id>/`
  - `sampled_frames/`
  - `frame_runs/frame_xxxxxx/` (프레임별 중간 산출물)
  - `overlay_frames/`
  - `result_video.mp4`
  - `frames_metadata.json`
  - `summary.json`
