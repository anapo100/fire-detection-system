"""연기 감지 필터 모듈.

회색조 분석, 텍스처 분석, 확산 패턴을 통해 연기를 검출한다.
출력: 연기 점수 0-100
"""

import logging
from collections import deque
from typing import Tuple, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SmokeFilter:
    """연기의 시각적 특성을 기반으로 연기를 검출하는 필터.

    연기 특징:
    - 색상: 회색~흰색 (낮은 채도, 중간~높은 밝기)
    - 텍스처: 부드럽고 흐릿한 경계 (낮은 에지 밀도)
    - 움직임: 위로 천천히 확산하는 패턴
    - 투명도: 배경이 반투명하게 비침
    """

    def __init__(self, config: dict):
        sf_cfg = config.get("smoke_filter", {})

        # 연기 색상 범위 (HSV)
        self.sat_max = sf_cfg.get("saturation_max", 60)
        self.val_min = sf_cfg.get("value_min", 80)
        self.val_max = sf_cfg.get("value_max", 220)

        # 면적 필터
        self.min_area = sf_cfg.get("min_area", 500)
        self.max_area = sf_cfg.get("max_area", 100000)

        # 텍스처 분석
        self.edge_density_max = sf_cfg.get("edge_density_max", 0.15)

        # 확산 추적
        self.expansion_threshold = sf_cfg.get("expansion_threshold", 0.1)

        # 모폴로지 커널 캐싱
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        # 확산 추적용 히스토리
        self._area_history: deque = deque(maxlen=15)

    def analyze(self, frame: np.ndarray, prev_frame: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray, list]:
        """프레임에서 연기를 검출한다.

        Args:
            frame: BGR 프레임
            prev_frame: 이전 BGR 프레임 (확산 분석용)

        Returns:
            (score, mask, contours) 튜플
            - score: 연기 신뢰도 (0-100)
            - mask: 연기 영역 바이너리 마스크
            - contours: 유효한 연기 컨투어 목록
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 1. 연기 색상 마스크: 낮은 채도 + 중간 밝기
        sat_mask = hsv[:, :, 1] < self.sat_max
        val_mask = (hsv[:, :, 2] >= self.val_min) & (hsv[:, :, 2] <= self.val_max)
        smoke_mask = (sat_mask & val_mask).astype(np.uint8) * 255

        # 모폴로지 연산으로 노이즈 제거
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, self._morph_kernel)
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, self._morph_kernel)

        # 컨투어 추출
        contours, _ = cv2.findContours(
            smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 면적 필터링
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area <= area <= self.max_area:
                valid_contours.append(cnt)

        if not valid_contours:
            self._area_history.append(0)
            return 0.0, smoke_mask, []

        # 점수 산출
        color_score = self._score_color(valid_contours, frame.shape)
        texture_score = self._score_texture(valid_contours, frame)
        diffusion_score = self._score_diffusion(valid_contours)

        total = color_score + texture_score + diffusion_score
        total = max(0.0, min(total, 100.0))

        return total, smoke_mask, valid_contours

    def _score_color(self, contours: list, frame_shape: tuple) -> float:
        """연기 색상 영역 비율 기반 점수 (0-40)."""
        total_area = sum(cv2.contourArea(c) for c in contours)
        frame_area = frame_shape[0] * frame_shape[1]
        ratio = total_area / max(frame_area, 1)

        # 연기는 넓은 영역에 걸쳐 나타남
        score = min(ratio * 500, 40.0)
        return score

    def _score_texture(self, contours: list, frame: np.ndarray) -> float:
        """텍스처 분석 점수 (0-30).

        연기는 경계가 부드러워 라플라시안 분산이 낮다.
        Canny 대비 ~3배 빠른 Laplacian variance 방식 사용.
        """
        if not contours:
            return 0.0

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        if w < 20 or h < 20:
            return 0.0

        roi = frame[y: y + h, x: x + w]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 라플라시안 분산: 값이 낮을수록 흐릿함 (연기 특성)
        lap_var = cv2.Laplacian(gray_roi, cv2.CV_64F).var()

        # 분산 임계값 (경험적: 연기 < 500, 선명한 물체 > 1000)
        blur_threshold = 500.0
        if lap_var < blur_threshold:
            score = (1.0 - lap_var / blur_threshold) * 30.0
            return score

        return 0.0

    def _score_diffusion(self, contours: list) -> float:
        """확산 패턴 점수 (0-30).

        연기는 시간에 따라 면적이 점진적으로 확대된다.
        """
        total_area = sum(cv2.contourArea(c) for c in contours)
        self._area_history.append(total_area)

        valid = [a for a in self._area_history if a > 0]
        if len(valid) < 5:
            return 0.0

        initial = valid[0]
        current = valid[-1]

        if initial < 1:
            return 0.0

        growth_rate = (current - initial) / initial

        if growth_rate >= self.expansion_threshold:
            score = min(growth_rate * 30.0, 30.0)
            return score

        return 0.0

    def reset(self):
        """상태를 초기화한다."""
        self._area_history.clear()
