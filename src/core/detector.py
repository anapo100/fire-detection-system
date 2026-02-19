"""화재 감지 메인 엔진 모듈."""

import time
import logging
from typing import Optional, List, Dict
from collections import deque

import numpy as np

from src.filters.color_filter import ColorFilter
from src.filters.motion_filter import MotionFilter
from src.filters.shape_filter import ShapeFilter

logger = logging.getLogger(__name__)


class DetectionResult:
    """감지 결과를 담는 데이터 클래스."""

    def __init__(
        self,
        confidence: float,
        color_score: float,
        motion_score: float,
        shape_score: float,
        contours: list,
        level: str,
    ):
        self.confidence = confidence
        self.color_score = color_score
        self.motion_score = motion_score
        self.shape_score = shape_score
        self.contours = contours
        self.level = level  # "normal", "warning", "alert", "critical"
        self.timestamp = time.time()


class FireDetector:
    """3단계 필터링 기반 화재 감지 엔진."""

    def __init__(self, config: dict):
        self.config = config

        det_cfg = config.get("detection", {})
        self.confidence_threshold = det_cfg.get("confidence_threshold", 70)
        self.debounce_seconds = det_cfg.get("debounce_seconds", 30)
        self.consecutive_frames = det_cfg.get("consecutive_frames", 3)

        self.color_filter = ColorFilter(config)
        self.motion_filter = MotionFilter(config)
        self.shape_filter = ShapeFilter(config)

        self._detection_history: deque = deque(maxlen=30)
        self._consecutive_count = 0
        self._last_alert_time: float = 0

    def detect(self, frame: np.ndarray, prev_frame: Optional[np.ndarray] = None) -> DetectionResult:
        """프레임에서 화재 감지를 수행한다.

        Args:
            frame: 현재 프레임 (BGR)
            prev_frame: 이전 프레임 (BGR), 움직임 분석용

        Returns:
            DetectionResult 객체
        """
        # 1단계: 색상 필터 (0-40점)
        color_score, color_mask, contours = self.color_filter.analyze(frame)

        # 2단계: 움직임 필터 (0-30점)
        motion_score = 0.0
        if prev_frame is not None:
            motion_score = self.motion_filter.analyze(frame, prev_frame, color_mask)

        # 3단계: 형태 필터 (0-30점)
        shape_score = self.shape_filter.analyze(contours, frame)

        # 최종 점수 계산
        total = (color_score * 0.4) + (motion_score * 0.3) + (shape_score * 0.3)
        # 0-100 범위로 정규화
        confidence = min(max(total, 0), 100)

        # 레벨 판정
        level = self._determine_level(confidence)

        # 연속 감지 카운트 업데이트
        if confidence >= 60:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 0

        result = DetectionResult(
            confidence=confidence,
            color_score=color_score,
            motion_score=motion_score,
            shape_score=shape_score,
            contours=contours,
            level=level,
        )

        self._detection_history.append(result)
        return result

    def _determine_level(self, confidence: float) -> str:
        """신뢰도에 따른 경고 레벨을 결정한다."""
        if confidence >= 85:
            return "critical"
        elif confidence >= 70:
            return "alert"
        elif confidence >= 60:
            return "warning"
        return "normal"

    def should_alert(self, result: DetectionResult) -> bool:
        """알림을 발송해야 하는지 판단한다.

        - 연속 감지 프레임 수가 임계값 이상이어야 한다
        - 마지막 알림으로부터 debounce 시간이 지나야 한다
        """
        if result.level == "normal":
            return False

        if self._consecutive_count < self.consecutive_frames:
            return False

        now = time.time()
        if now - self._last_alert_time < self.debounce_seconds:
            return False

        self._last_alert_time = now
        return True

    def reset(self):
        """감지 상태를 초기화한다."""
        self._detection_history.clear()
        self._consecutive_count = 0
        self._last_alert_time = 0
        self.motion_filter.reset()
        self.shape_filter.reset()
