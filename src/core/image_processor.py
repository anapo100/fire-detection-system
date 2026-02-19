"""영상 전처리 모듈 (노이즈 제거, 히스토그램 균일화 등)."""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImageProcessor:
    """영상 전처리를 담당하는 클래스."""

    def __init__(self, config: dict):
        preproc = config.get("preprocessing", {})
        self.clahe_clip_limit = preproc.get("clahe_clip_limit", 2.0)
        self.blur_size = preproc.get("gaussian_blur_size", 5)
        self.denoise_strength = preproc.get("denoise_strength", 10)

        self.clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit, tileGridSize=(8, 8)
        )

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """전처리 파이프라인을 실행한다.

        1. 가우시안 블러 (노이즈 제거)
        2. CLAHE (대비 강화)
        """
        processed = self._apply_gaussian_blur(frame)
        processed = self._apply_clahe(processed)
        return processed

    def _apply_gaussian_blur(self, frame: np.ndarray) -> np.ndarray:
        """가우시안 블러를 적용하여 노이즈를 제거한다."""
        k = self.blur_size
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(frame, (k, k), 0)

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """CLAHE를 적용하여 대비를 강화한다."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        channels = list(cv2.split(lab))
        channels[0] = self.clahe.apply(channels[0])
        lab = cv2.merge(channels)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def denoise(self, frame: np.ndarray) -> np.ndarray:
        """Non-local Means 디노이징을 적용한다."""
        return cv2.fastNlMeansDenoisingColored(
            frame, None, self.denoise_strength, self.denoise_strength, 7, 21
        )

    def to_hsv(self, frame: np.ndarray) -> np.ndarray:
        """BGR 프레임을 HSV 색공간으로 변환한다."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    def to_gray(self, frame: np.ndarray) -> np.ndarray:
        """BGR 프레임을 그레이스케일로 변환한다."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
