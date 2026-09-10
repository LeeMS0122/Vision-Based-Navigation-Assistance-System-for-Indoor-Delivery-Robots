<div align="center">

# 🤖 실내 배송 로봇용 경로 주행 보조 비전 시스템

### 시각 정보로 주행 가능 영역과 장애물을 판단하고 안전 경로계획 입력맵 생성

<p>
  <img src="https://img.shields.io/badge/Project-졸업작품-6C5CE7?style=for-the-badge" alt="졸업작품" />
  <img src="https://img.shields.io/badge/Status-Code_Released-00A86B?style=for-the-badge" alt="코드 공개" />
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/YOLO_Segmentation-111F68?style=for-the-badge" alt="YOLO Segmentation" />
  <img src="https://img.shields.io/badge/Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="Computer Vision" />
  <img src="https://img.shields.io/badge/Path_Planning-00A86B?style=for-the-badge" alt="Path Planning" />
</p>

</div>

---

## 📌 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 프로젝트 | 전자컴퓨터공학 졸업작품 |
| 기간 | 2026.04 ~ 진행 중 |
| 목표 | 실내 배송 로봇용 비전 기반 주행 보조 입력맵 생성 |
| 담당 | 데이터 검증·전처리, YOLO-seg 변환, 모델 학습, 경로계획 구조 정리 |
| 데이터 | AI Hub 이미지·세그멘테이션 데이터 |

## 🎯 문제 정의

- 실내 환경의 주행 가능 영역과 장애물 구분
- 객체 종류와 거리를 함께 반영한 위험도 계산
- 비전 결과를 경로 탐색용 Cost Map으로 변환
- 고정 지도만으로 대응하기 어려운 동적 장애물 반영

## 🔄 시스템 파이프라인

```mermaid
flowchart LR
    A["이미지·JSON 데이터"] --> B["파일 매칭·분포 검증"]
    B --> C["YOLO-seg 형식 변환"]
    C --> D["주행 영역·장애물 분할"]
    D --> E["SC-Depth V3 상대 심도 추정"]
    E --> F["Road·Obstacle·Depth 합성"]
    F --> G["Cost Map 생성"]
    G --> H["고전 A* 안전 경로 탐색"]
    H --> I["안전 경로 시각화"]
```

## 🧩 핵심 구현 내용

### 1. 데이터 품질 검증

- 이미지와 JSON 라벨 매칭
- 누락 파일 탐지
- 클래스 목록과 클래스별 샘플 수 집계
- 학습 전 데이터 상태 확인

### 2. YOLO Segmentation 데이터 변환

- JSON Polygon 라벨의 YOLO Segmentation TXT 변환
- `images/train`, `images/val`, `labels/train`, `labels/val` 구조 생성
- 원본 클래스의 `drivable_area`·`obstacle` 중심 재구성

### 3. Cost Map 설계

- Road Mask 기반 주행 가능 영역 표현
- Obstacle Mask 기반 이동 제한 영역 표현
- SC-Depth V3의 상대 심도를 정규화해 근접 위험도 가중치 반영
- 세 입력값 합성을 통한 경로계획용 Cost Map 생성

### 4. 경로계획 연결

- 웹 데모에 고전 A* 기반 최단 안전 경로 탐색 통합
- Small U-Net 기반 경로 예측은 별도 실험 코드로 분리
- 객체·위험 영역·추천 경로 통합 시각화

## 📊 현재 성과

| 구분 | 결과 |
|---|---|
| Train 데이터 | 이미지·라벨 66,429건 |
| Validation 데이터 | 이미지·라벨 8,214건 |
| 핵심 클래스 | `drivable_area`, `obstacle` |
| 설계 결과 | Road·Obstacle·Depth 기반 Cost Map |
| 개발 흐름 | 데이터 검증 → 변환 → 학습 → Cost Map → 고전 A* 경로계획 |

## 🗂️ 코드 구성

| 구분 | 주요 코드 | 설명 |
|---|---|---|
| 실험 스크립트 | [`scripts/`](scripts/) | 데이터 준비·분할·상대 심도·Cost Map·경로계획 코드 |
| 웹 데모 | `web_demo/` | 이미지·영상·실시간 입력을 처리하는 FastAPI 기반 데모 |

```text
.
├── scripts/
│   ├── data_preparation/   # AI Hub·SC-Depth 데이터 준비
│   ├── segmentation/       # YOLO-seg 학습·추론
│   ├── depth_estimation/   # 가상 심도·SC-Depth V3
│   ├── costmap/            # 경로계획용 위험도 맵
│   └── path_planning/      # 고전 A*·Small U-Net 실험
├── web_demo/                # FastAPI 기반 통합 데모
├── requirements.txt
└── README.md
```

각 스크립트의 역할은 [`scripts/README.md`](scripts/README.md)에서 확인할 수 있습니다.

## ▶️ 실행 준비

```bash
git clone git@github.com:LeeMS0122/Vision-Based-Navigation-Assistance-System-for-Indoor-Delivery-Robots.git
cd Vision-Based-Navigation-Assistance-System-for-Indoor-Delivery-Robots
python -m pip install -r requirements.txt
```

실험 스크립트는 파일 상단의 경로 설정을 자신의 데이터셋·가중치 경로로 변경해 사용합니다. SC-Depth V3 소스와 학습 가중치는 별도로 준비해야 합니다.

웹 데모 실행 방법과 필요 환경 변수는 [`web_demo/README.md`](web_demo/README.md)에 정리했습니다.

## 🛠️ 기술 및 실험 환경

<p>
  <img src="https://img.shields.io/badge/Depth_Estimation-34495E?style=flat-square" alt="Depth Estimation" />
  <img src="https://img.shields.io/badge/A*-E67E22?style=flat-square" alt="A star" />
  <img src="https://img.shields.io/badge/Small_U--Net-8E44AD?style=flat-square" alt="Small U-Net" />
  <img src="https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=flat-square&logo=weightsandbiases&logoColor=black" alt="Weights and Biases" />
  <img src="https://img.shields.io/badge/Linux-FCC624?style=flat-square&logo=linux&logoColor=black" alt="Linux" />
  <img src="https://img.shields.io/badge/Notion-000000?style=flat-square&logo=notion&logoColor=white" alt="Notion" />
</p>

## 🗺️ 개발 로드맵

- [x] 이미지·라벨 매칭 및 데이터 분포 검증
- [x] JSON Polygon → YOLO Segmentation 변환
- [x] 주행 영역·장애물 중심 클래스 재구성
- [x] Road·Obstacle·Depth 기반 Cost Map 구조 설계
- [x] SC-Depth V3 상대 심도와 Cost Map 통합
- [x] 고전 A* 경로 탐색 웹 데모 통합
- [x] Small U-Net 기반 경로 예측 별도 실험
- [ ] YOLO-seg 모델 성능 고도화
- [ ] 실제 주행 환경 검증

## 📂 저장소 공개 범위

- Python·HTML·CSS·JavaScript 소스 코드와 실행 문서만 공개
- AI Hub 원본 데이터와 변환 데이터셋 미포함
- 학습 가중치, 체크포인트, W&B 로그, 실험 결과 이미지·영상 미포함
- 웹 데모 모델 경로는 환경 변수로 분리하고, 생성 산출물과 비밀값은 `.gitignore`로 제외

---

<div align="center">

[![Profile](https://img.shields.io/badge/Developer-LeeMS0122-181717?style=flat-square&logo=github)](https://github.com/LeeMS0122)
[![Portfolio](https://img.shields.io/badge/Portfolio-Notion-000000?style=flat-square&logo=notion&logoColor=white)](https://cake-oviraptor-b43.notion.site/314eefba1b6d81538fe2f56c4adb52b9?source=copy_link)

</div>
