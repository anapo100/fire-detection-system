"""Contour 형태 분석 3차 필터 모듈."""

import logging
from collections import deque
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ShapeFilter:
    """화염의 형태적 특성(불규칙성, 수직 확장)을 검증하는 필터.

    출력: 형태 점수 0-30점
    """

    def __init__(self, config: dict):
        sf_cfg = config.get("shape_filter", {})
        self.irregularity_threshold = sf_cfg.get("irregularity_threshold", 20)
        self.vertical_expansion_rate = sf_cfg.get("vertical_expansion_rate", 0.2)

        # 수직 확장 추적을 위한 히스토리
        self._bbox_history: deque = deque(maxlen=15)  # ~3초 (5 FPS 기준)

    def analyze(self, contours: list, frame: np.ndarray) -> float:
        """컨투어의 형태를 분석한다.

        Args:
            contours: 색상 필터에서 추출된 유효 컨투어
            frame: 원본 BGR 프레임

        Returns:
            형태 점수 (0-30)
        """
        if not contours:
            self._bbox_history.append(None)
            return 0.0

        irregularity_score = self._analyze_irregularity(contours)
        expansion_score = self._analyze_vertical_expansion(contours)
        gradient_score = self._analyze_color_gradient(contours, frame)

        total = irregularity_score + expansion_score + gradient_score
        return min(total, 30.0)

    def _analyze_irregularity(self, contours: list) -> float:
        """경계선 불규칙성을 분석한다.

        불규칙성 지수 = (Perimeter^2) / Area
        화재: > 20 (경계선 복잡)
        원형 조명: < 15 (경계선 단순)
        """
        scores = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1:
                continue
            perimeter = cv2.arcLength(cnt, True)
            irregularity = (perimeter ** 2) / area

            if irregularity > self.irregularity_threshold:
                # 불규칙성이 높을수록 화재 가능성 높음
                score = min((irregularity - self.irregularity_threshold) * 0.5, 10.0)
                scores.append(score)

        return max(scores) if scores else 0.0

    def _analyze_vertical_expansion(self, contours: list) -> float:
        """수직 확장 패턴을 추적한다.

        화염은 위로 번지는 특성이 있으므로 Bounding Box 높이 변화를 추적한다.
        """
        if not contours:
            self._bbox_history.append(None)
            return 0.0

        # 가장 큰 컨투어의 바운딩 박스 추적
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        self._bbox_history.append({"x": x, "y": y, "w": w, "h": h})

        # 최소 5프레임 이상 히스토리가 있어야 분석
        valid_history = [b for b in self._bbox_history if b is not None]
        if len(valid_history) < 5:
            return 0.0

        # 높이 변화율 계산
        heights = [b["h"] for b in valid_history]
        initial_h = heights[0]
        current_h = heights[-1]

        if initial_h < 1:
            return 0.0

        expansion_rate = (current_h - initial_h) / initial_h

        if expansion_rate >= self.vertical_expansion_rate:
            score = min(expansion_rate * 20.0, 10.0)
            return score

        return 0.0

    def _analyze_color_gradient(self, contours: list, frame: np.ndarray) -> float:
        """색상 분포 그라데이션을 분석한다.

        화재: 중심부(노랑) -> 외곽(빨강) 그라데이션
        조명: 균일한 색상 분포
        """
        if not contours:
            return 0.0

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        if w < 10 or h < 10:
            return 0.0

        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            return 0.0

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 중심부와 외곽의 Hue 차이 분석
        center_h = h // 2
        center_w = w // 2
        margin_h = max(h // 4, 1)
        margin_w = max(w // 4, 1)

        center_region = hsv_roi[
            center_h - margin_h : center_h + margin_h,
            center_w - margin_w : center_w + margin_w,
        ]
        if center_region.size == 0:
            return 0.0

        center_hue = float(np.mean(center_region[:, :, 0]))
        overall_hue = float(np.mean(hsv_roi[:, :, 0]))

        # 중심부가 외곽보다 노란색(Hue 높음)이면 화재 패턴
        hue_diff = center_hue - overall_hue
        if hue_diff > 5:
            score = min(hue_diff * 1.0, 10.0)
            return score

        return 0.0

    def reset(self):
        """상태를 초기화한다."""
        self._bbox_history.clear()
