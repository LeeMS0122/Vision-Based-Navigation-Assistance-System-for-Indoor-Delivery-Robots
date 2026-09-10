from web_demo.services.costmap_service import CostMapService


class CostMapServiceV3(CostMapService):
    """v2 튜닝 규칙을 그대로 재사용하는 v3 래퍼."""


costmap_service_v3 = CostMapServiceV3()
