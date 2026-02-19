"""Optical Flow 기반 2차 움직임 필터 모듈."""

import logging
from collections import deque
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class MotionFilter:
    """Optical Flow로 화염의 불규칙한 움직임 패턴을 감지하는 필터.

    출력: 움직임 점수 0-30점
    """

    # Optical Flow 계산용 축소 비율 (원본 대비)
    _FLOW_SCALE = 0.5

    def __init__(self, config: dict):
        mf_cfg = config.get("motion_filter", {})
        self.flow_threshold = mf_cfg.get("optical_flow_threshold", 2.5)
        self.temporal_frames = mf_cfg.get("temporal_frames", 5)
        self.pixel_change_threshold = mf_cfg.get("pixel_change_threshold", 0.15)

        self._flow_magnitudes: deque = deque(maxlen=self.temporal_frames)
        self._cached_prev_gray: Optional[np.ndarray] = None

    def analyze(
        self,
        frame: np.ndarray,
        prev_frame: np.ndarray,
        fire_mask: Optional[np.ndarray] = None,
    ) -> float:
        """프레임 간 움직임을 분석한다.

        Args:
            frame: 현재 프레임 (BGR)
            prev_frame: 이전 프레임 (BGR)
            fire_mask: 색상 필터에서 생성된 화염 영역 마스크

        Returns:
            움직임 점수 (0-30)
        """
        # 축소 해상도에서 Optical Flow 계산 (성능 핵심)
        small = cv2.resize(frame, None, fx=self._FLOW_SCALE, fy=self._FLOW_SCALE,
                           interpolation=cv2.INTER_AREA)
        curr_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # 캐싱된 이전 gray 사용, 없으면 새로 변환
        if self._cached_prev_gray is not None:
            prev_gray = self._cached_prev_gray
        else:
            small_prev = cv2.resize(prev_frame, None, fx=self._FLOW_SCALE, fy=self._FLOW_SCALE,
                                    interpolation=cv2.INTER_AREA)
            prev_gray = cv2.cvtColor(small_prev, cv2.COLOR_BGR2GRAY)

        # 현재 gray를 다음 호출을 위해 캐싱
        self._cached_prev_gray = curr_gray

        # Farneback Optical Flow (경량화 파라미터)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=2, winsize=11,
            iterations=2, poly_n=5, poly_sigma=1.1,
            flags=0,
        )

        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # 화염 영역 내의 움직임만 분석 (마스크를 축소 크기로 리사이즈)
        if fire_mask is not None:
            small_mask = cv2.resize(fire_mask, (curr_gray.shape[1], curr_gray.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
            fire_region = small_mask > 0
            if np.any(fire_region):
                region_magnitude = magnitude[fire_region]
            else:
                region_magnitude = magnitude.flatten()
        else:
            region_magnitude = magnitude.flatten()

        mean_mag = float(np.mean(region_magnitude)) if region_magnitude.size > 0 else 0.0
        self._flow_magnitudes.append(mean_mag)

        score = self._calculate_score(mean_mag)
        return score

    def _calculate_score(self, mean_magnitude: float) -> float:
        """움직임 점수를 계산한다."""
        score = 0.0

        # 1. 움직임 강도 기반 점수 (0-15)
        if mean_magnitude > self.flow_threshold:
            intensity_score = min(
                (mean_magnitude - self.flow_threshold) * 5.0, 15.0
            )
            score += intensity_score

        # 2. 시간적 변동성 기반 점수 (0-15)
        if len(self._flow_magnitudes) >= 3:
            magnitudes = list(self._flow_magnitudes)
            std_dev = float(np.std(magnitudes))
            # 불규칙한 움직임일수록 높은 점수
            variability_score = min(std_dev * 3.0, 15.0)
            score += variability_score

        return min(score, 30.0)

    def reset(self):
        """상태를 초기화한다."""
        self._flow_magnitudes.clear()
        self._cached_prev_gray = None
