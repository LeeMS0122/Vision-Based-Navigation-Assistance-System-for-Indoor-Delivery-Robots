# Cost Map 생성

YOLO-seg의 주행 영역·장애물 마스크와 SC-Depth V3 상대 심도를 합성해 경로계획용 위험도 맵을 생성합니다.

- `build_cost_map.py`: 기본 Cost Map 합성 로직 검증
- `build_cost_map_real.py`: 실제 모델 추론 결과 통합
- `build_cost_map_real_v2.py`: 장애물 버퍼, 화면 중앙 편향, 하단 위험도, 가장자리 비용 보강

결과 Cost Map은 0∼255 비용으로 표현되며, 장애물 영역은 고전 A* 탐색에서 통과할 수 없는 공간으로 처리됩니다.
