"""ROI(관심 영역) 설정 도구 모듈."""

import logging
from typing import List, Tuple, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ROIRegion:
    """단일 ROI 영역을 나타내는 클래스."""

    def __init__(self, name: str, coordinates: List[int]):
        self.name = name
        self.x1, self.y1, self.x2, self.y2 = coordinates

    def contains_point(self, x: int, y: int) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y1 : self.y2, self.x1 : self.x2]

    def draw(self, frame: np.ndarray, color=(0, 255, 0), thickness=2):
        cv2.rectangle(frame, (self.x1, self.y1), (self.x2, self.y2), color, thickness)
        cv2.putText(
            frame,
            self.name,
            (self.x1, self.y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )


class ROIManager:
    """ROI 영역을 관리하는 클래스."""

    def __init__(self, config: dict):
        roi_cfg = config.get("roi", {})
        self.enabled = roi_cfg.get("enabled", False)
        self.regions: List[ROIRegion] = []

        for region_data in roi_cfg.get("regions", []):
            region = ROIRegion(
                name=region_data["name"],
                coordinates=region_data["coordinates"],
            )
            self.regions.append(region)

    def apply_mask(self, frame: np.ndarray) -> np.ndarray:
        """ROI 영역만 남기고 나머지를 마스킹한다."""
        if not self.enabled or not self.regions:
            return frame

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for region in self.regions:
            mask[region.y1 : region.y2, region.x1 : region.x2] = 255

        result = cv2.bitwise_and(frame, frame, mask=mask)
        return result

    def get_region_at(self, x: int, y: int) -> Optional[ROIRegion]:
        """주어진 좌표가 속하는 ROI 영역을 반환한다."""
        for region in self.regions:
            if region.contains_point(x, y):
                return region
        return None

    def draw_regions(self, frame: np.ndarray) -> np.ndarray:
        """프레임에 ROI 영역을 표시한다."""
        display = frame.copy()
        if not self.enabled:
            return display
        for region in self.regions:
            region.draw(display)
        return display
