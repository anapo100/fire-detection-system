"""HSV 색상 기반 1차 화재 필터 모듈."""

import logging
from typing import Tuple, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ColorFilter:
    """HSV 색공간에서 화염 특유의 색상을 검출하는 필터.

    출력: Binary Mask + 신뢰도 0-40점
    """

    def __init__(self, config: dict):
        cf_cfg = config.get("color_filter", {})
        hsv_ranges = cf_cfg.get("hsv_ranges", {})

        self.ranges = []
        for name, vals in hsv_ranges.items():
            lower = np.array(vals["lower"], dtype=np.uint8)
            upper = np.array(vals["upper"], dtype=np.uint8)
            self.ranges.append((name, lower, upper))

        # 기본 범위 (설정이 없을 경우)
        if not self.ranges:
            self.ranges = [
                ("flame_orange", np.array([0, 100, 200]), np.array([20, 255, 255])),
                ("flame_yellow", np.array([20, 100, 200]), np.array([40, 255, 255])),
            ]

        self.min_area = cf_cfg.get("min_area", 300)
        self.max_area = cf_cfg.get("max_area", 50000)

    def analyze(self, frame: np.ndarray) -> Tuple[float, np.ndarray, list]:
        """프레임에서 화염 색상을 검출한다.

        Args:
            frame: BGR 프레임

        Returns:
            (score, mask, contours) 튜플
            - score: 색상 신뢰도 (0-40)
            - mask: 화염 영역 바이너리 마스크
            - contours: 유효한 컨투어 목록
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 모든 화염 색상 범위에 대한 마스크 결합
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for name, lower, upper in self.ranges:
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # 모폴로지 연산으로 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        # 컨투어 추출 및 필터링
        contours, _ = cv2.findContours(
            combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            # 종횡비 검사 (LED 조명 같은 가로로 긴 형태 제외)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / max(h, 1)
            if aspect_ratio > 3:
                continue

            # Saturation 검사 (반사광 제외)
            roi_hsv = hsv[y : y + h, x : x + w]
            if roi_hsv.size > 0:
                mean_sat = np.mean(roi_hsv[:, :, 1])
                if mean_sat < 120:
                    continue

            valid_contours.append(cnt)

        # 점수 계산 (0-40)
        score = self._calculate_score(valid_contours, frame.shape)
        return score, combined_mask, valid_contours

    def _calculate_score(self, contours: list, frame_shape: tuple) -> float:
        """컨투어 기반으로 색상 신뢰도 점수를 계산한다."""
        if not contours:
            return 0.0

        total_area = sum(cv2.contourArea(c) for c in contours)
        frame_area = frame_shape[0] * frame_shape[1]
        area_ratio = total_area / max(frame_area, 1)

        # 면적 비율에 기반한 점수 (최대 40점)
        score = min(area_ratio * 1000, 40.0)

        # 여러 개의 화염 영역이 있으면 가산점
        if len(contours) >= 2:
            score = min(score + 5.0, 40.0)

        return score
