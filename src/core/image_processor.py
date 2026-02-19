"""영상 전처리 모듈 (가우시안 블러, CLAHE 대비 강화)."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImageProcessor:
    """영상 전처리를 담당하는 클래스.

    성능 최적화:
    - CLAHE를 L채널(LAB) 대신 V채널(HSV)에 적용하여 색공간 변환 1회 절약
    - 가우시안 블러 커널 크기를 최소화
    """

    def __init__(self, config: dict):
        preproc = config.get("preprocessing", {})
        self.clahe_clip_limit = preproc.get("clahe_clip_limit", 2.0)
        self.blur_size = preproc.get("gaussian_blur_size", 5)
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit, tileGridSize=(8, 8)
        )

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """전처리 파이프라인을 실행한다.

        1. 가우시안 블러 (노이즈 제거)
        2. CLAHE (대비 강화) — HSV V채널에 적용
        """
        k = self.blur_size
        if k % 2 == 0:
            k += 1
        processed = cv2.GaussianBlur(frame, (k, k), 0)
        processed = self._apply_clahe_hsv(processed)
        return processed

    def _apply_clahe_hsv(self, frame: np.ndarray) -> np.ndarray:
        """HSV V채널에 CLAHE를 적용하여 대비를 강화한다.

        LAB 방식 대비 색공간 변환 비용이 동일하지만,
        이후 ColorFilter에서 HSV 변환을 재사용할 여지가 있다.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = self.clahe.apply(hsv[:, :, 2])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
