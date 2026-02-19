"""화재 감지 메인 엔진 모듈.

화염과 연기를 독립적으로 감지한 뒤 복합 위험도를 산출한다.

위험도 계산 방식:
  - 화염 위험도 (0-100): 색상(40%) + 움직임(30%) + 형태(30%)
  - 연기 위험도 (0-100): 색상/텍스처/확산 복합
  - 최종 신뢰도: 화염과 연기 동시 존재 시 시너지 보너스 적용

시나리오별 위험도:
  - 연기만 존재: 초기 화재 가능성 (최대 50점)
  - 화염만 존재: 활성 화재 (최대 80점)
  - 화염 + 연기 동시: 확정 화재 (최대 100점, 시너지 보너스)
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

import numpy as np

from src.filters.color_filter import ColorFilter
from src.filters.motion_filter import MotionFilter
from src.filters.shape_filter import ShapeFilter
from src.filters.smoke_filter import SmokeFilter
from src.filters.yolo_filter import YOLOFilter

logger = logging.getLogger(__name__)

# 레벨 임계값 상수 (선제적 설정: 조기 경보 우선)
_LEVEL_CRITICAL = 65
_LEVEL_ALERT = 45
_LEVEL_WARNING = 30


@dataclass(slots=True)
class DetectionResult:
    """감지 결과를 담는 데이터 클래스."""

    confidence: float         # 최종 복합 신뢰도 (0-100)
    flame_risk: float         # 화염 위험도 (0-100)
    smoke_risk: float         # 연기 위험도 (0-100)
    color_score: float        # 색상 필터 점수
    motion_score: float       # 움직임 필터 점수
    shape_score: float        # 형태 필터 점수
    smoke_score: float        # 연기 필터 점수
    yolo_score: float         # YOLO 종합 점수 (0-100)
    yolo_fire_conf: float     # YOLO 화재 신뢰도 (0.0-1.0)
    yolo_smoke_conf: float    # YOLO 연기 신뢰도 (0.0-1.0)
    contours: list            # 화염 컨투어
    smoke_contours: list      # 연기 컨투어
    yolo_detections: list     # YOLO bbox 검출 리스트
    level: str                # "normal", "warning", "alert", "critical"
    has_flame: bool           # 화염 존재 여부
    has_smoke: bool           # 연기 존재 여부
    timestamp: float = field(default_factory=time.time)


class FireDetector:
    """화염 + 연기 복합 분석 기반 화재 감지 엔진.

    화염과 연기를 각각 독립적으로 감지한 후,
    두 요소의 존재 여부와 강도에 따라 최종 위험도를 산출한다.
    """

    # 화염 내부 가중치
    _FLAME_COLOR_W = 0.4
    _FLAME_MOTION_W = 0.3
    _FLAME_SHAPE_W = 0.3

    # 최종 신뢰도 가중치
    _FLAME_RISK_W = 0.55     # 화염 위험도 가중치
    _SMOKE_RISK_W = 0.25     # 연기 위험도 가중치
    _SYNERGY_BONUS = 20.0    # 화염+연기 동시 존재 보너스

    # 존재 판정 임계값 (선제적: 약한 징후도 조기 감지)
    _FLAME_PRESENCE_THRESHOLD = 5.0
    _SMOKE_PRESENCE_THRESHOLD = 8.0

    def __init__(self, config: dict):
        self.config = config

        det_cfg = config.get("detection", {})
        self.confidence_threshold = det_cfg.get("confidence_threshold", 70)
        self.debounce_seconds = det_cfg.get("debounce_seconds", 30)
        self.consecutive_frames = det_cfg.get("consecutive_frames", 3)

        # 5개 필터 초기화
        self.color_filter = ColorFilter(config)
        self.motion_filter = MotionFilter(config)
        self.shape_filter = ShapeFilter(config)
        self.smoke_filter = SmokeFilter(config)
        self.yolo_filter = YOLOFilter(config)

        self._detection_history: deque = deque(maxlen=30)
        self._consecutive_count = 0
        self._last_alert_time: float = 0

        # Optical Flow 2프레임 간격 실행 (성능 최적화)
        self._frame_count = 0
        self._cached_motion_score: float = 0.0

    def detect(self, frame: np.ndarray, prev_frame: Optional[np.ndarray] = None) -> DetectionResult:
        """프레임에서 화재 감지를 수행한다.

        Args:
            frame: 현재 프레임 (BGR)
            prev_frame: 이전 프레임 (BGR), 움직임 분석용

        Returns:
            DetectionResult 객체
        """
        # === 화염 분석 ===
        color_score, color_mask, flame_contours = self.color_filter.analyze(frame)

        # Optical Flow: 2프레임 간격으로 실행하여 ~20ms 비용을 절반으로 절감
        self._frame_count += 1
        if prev_frame is not None and self._frame_count % 2 == 0:
            self._cached_motion_score = self.motion_filter.analyze(
                frame, prev_frame, color_mask
            )
        motion_score = self._cached_motion_score

        shape_score = self.shape_filter.analyze(flame_contours, frame)

        # 화염 위험도 (0-100)
        flame_raw = (
            color_score * self._FLAME_COLOR_W
            + motion_score * self._FLAME_MOTION_W
            + shape_score * self._FLAME_SHAPE_W
        )
        # color_score 최대 40, motion/shape 최대 30이므로
        # 원시 최대 = 40*0.4 + 30*0.3 + 30*0.3 = 34
        # 100점 스케일로 변환
        flame_risk = max(0.0, min(flame_raw * (100.0 / 34.0), 100.0))

        # === 연기 분석 ===
        smoke_score, smoke_mask, smoke_contours = self.smoke_filter.analyze(frame, prev_frame)
        smoke_risk = max(0.0, min(smoke_score, 100.0))

        # === 존재 판정 ===
        has_flame = flame_risk >= self._FLAME_PRESENCE_THRESHOLD
        has_smoke = smoke_risk >= self._SMOKE_PRESENCE_THRESHOLD

        # === 최종 신뢰도 계산 ===
        base_confidence = self._calculate_combined_confidence(
            flame_risk, smoke_risk, has_flame, has_smoke
        )

        # === YOLO 검증 (5차 필터) ===
        yolo_result = self.yolo_filter.analyze(frame)
        confidence = self.yolo_filter.get_verification_adjustment(base_confidence)

        level = self._determine_level(confidence)

        # 연속 감지 카운트
        if confidence >= _LEVEL_WARNING:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 0

        result = DetectionResult(
            confidence=confidence,
            flame_risk=flame_risk,
            smoke_risk=smoke_risk,
            color_score=color_score,
            motion_score=motion_score,
            shape_score=shape_score,
            smoke_score=smoke_score,
            yolo_score=yolo_result["score"],
            yolo_fire_conf=yolo_result["fire_conf"],
            yolo_smoke_conf=yolo_result["smoke_conf"],
            contours=flame_contours,
            smoke_contours=smoke_contours,
            yolo_detections=yolo_result["detections"],
            level=level,
            has_flame=has_flame,
            has_smoke=has_smoke,
        )

        self._detection_history.append(result)
        return result

    def _calculate_combined_confidence(
        self,
        flame_risk: float,
        smoke_risk: float,
        has_flame: bool,
        has_smoke: bool,
    ) -> float:
        """화염과 연기의 복합 위험도를 계산한다.

        시나리오:
        1. 둘 다 없음      -> 기본 가중합 (낮은 값)
        2. 연기만 존재      -> 연기 위험도 × 0.5 (초기 화재 의심, 최대 50)
        3. 화염만 존재      -> 화염 위험도 × 0.8 (활성 화재, 최대 80)
        4. 화염 + 연기 동시 -> 가중합 + 시너지 보너스 (확정 화재, 최대 100)
        """
        if has_flame and has_smoke:
            # 화염 + 연기 동시: 가중합 + 시너지 보너스
            base = flame_risk * self._FLAME_RISK_W + smoke_risk * self._SMOKE_RISK_W
            confidence = base + self._SYNERGY_BONUS
        elif has_flame:
            # 화염만: 최대 80점
            confidence = flame_risk * 0.8
        elif has_smoke:
            # 연기만: 초기 화재 의심, 최대 50점
            confidence = smoke_risk * 0.5
        else:
            # 둘 다 미감지
            confidence = (flame_risk * 0.3 + smoke_risk * 0.1)

        return max(0.0, min(confidence, 100.0))

    def _determine_level(self, confidence: float) -> str:
        """신뢰도에 따른 경고 레벨을 결정한다."""
        if confidence >= _LEVEL_CRITICAL:
            return "critical"
        if confidence >= _LEVEL_ALERT:
            return "alert"
        if confidence >= _LEVEL_WARNING:
            return "warning"
        return "normal"

    def should_alert(self, result: DetectionResult) -> bool:
        """알림을 발송해야 하는지 판단한다."""
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
        self.smoke_filter.reset()
        self.yolo_filter.reset()
