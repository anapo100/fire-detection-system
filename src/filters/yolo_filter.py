"""YOLOv8 기반 화재 검증 필터 모듈.

사전 학습된 YOLOv8 모델로 화염/연기를 객체 수준에서 검출하여
기존 필터의 결과를 검증(확인 또는 감쇠)한다.

출력: yolo_result dict (score, fire_conf, smoke_conf, detections, ...)
"""

import logging
from pathlib import Path
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


class YOLOFilter:
    """YOLOv8 기반 화재/연기 검증 필터.

    기존 4개 필터의 결과를 독립적으로 검증하는 역할을 한다.
    GPU가 있으면 CUDA, 없으면 CPU로 자동 폴백한다.
    N프레임 간격으로 실행하여 성능 영향을 최소화한다.
    """

    _FIRE_CLASSES = {"fire", "flame"}
    _SMOKE_CLASSES = {"smoke"}

    def __init__(self, config: dict):
        yolo_cfg = config.get("yolo_filter", {})

        self.enabled = yolo_cfg.get("enabled", True)
        self.model_path = yolo_cfg.get("model_path", "models/yolov8n-fire.pt")
        self.confidence_threshold = yolo_cfg.get("confidence_threshold", 0.35)
        self.frame_interval = yolo_cfg.get("frame_interval", 3)
        self.input_size = yolo_cfg.get("input_size", 320)

        # 검증 가중치
        self.fire_boost_max = yolo_cfg.get("fire_boost_max", 25.0)
        self.smoke_boost_max = yolo_cfg.get("smoke_boost_max", 10.0)
        self.no_detect_penalty = yolo_cfg.get("no_detect_penalty", 0.75)

        # 내부 상태
        self._model = None
        self._device = "cpu"
        self._frame_count = 0
        self._model_loaded = False
        self._cached_result: Dict = self._empty_result()

        if self.enabled:
            self._load_model()

    @staticmethod
    def _empty_result() -> Dict:
        return {
            "score": 0.0,
            "fire_conf": 0.0,
            "smoke_conf": 0.0,
            "detections": [],
            "has_fire": False,
            "has_smoke": False,
        }

    def _load_model(self):
        """YOLO 모델을 로드한다. 실패 시 필터를 비활성화한다."""
        try:
            import torch
            from ultralytics import YOLO

            # GPU/CPU 결정
            if torch.cuda.is_available():
                self._device = "0"
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"YOLO 필터: CUDA GPU 사용 ({gpu_name})")
            else:
                self._device = "cpu"
                logger.warning("YOLO 필터: CUDA 불가, CPU 모드로 동작")

            # 모델 파일 확인
            model_file = Path(self.model_path)
            if not model_file.exists():
                # exe 실행 시 기준 경로에서도 확인
                import sys
                if getattr(sys, "frozen", False):
                    alt_path = Path(sys._MEIPASS) / self.model_path
                    if alt_path.exists():
                        model_file = alt_path
                    else:
                        self._disable("모델 파일 없음: " + self.model_path)
                        return
                else:
                    self._disable("모델 파일 없음: " + self.model_path)
                    return

            self._model = YOLO(str(model_file))

            # Warm-up 추론 (첫 CUDA 추론의 지연 방지)
            dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            self._model.predict(
                dummy, device=self._device, imgsz=self.input_size,
                conf=self.confidence_threshold, verbose=False,
            )
            self._model_loaded = True
            logger.info(
                f"YOLO 모델 로드 완료: {model_file} "
                f"(device={self._device}, input={self.input_size}px)"
            )

        except ImportError:
            self._disable("ultralytics 또는 torch 패키지 없음")
        except Exception as e:
            self._disable(f"모델 로드 실패: {e}")

    def _disable(self, reason: str):
        """필터를 비활성화하고 사유를 로깅한다."""
        logger.warning(f"YOLO 필터 비활성화 — {reason} (기존 필터만 사용)")
        self.enabled = False

    def analyze(self, frame: np.ndarray) -> Dict:
        """프레임에서 YOLO 화재/연기 검출을 수행한다.

        N프레임 간격으로 실제 추론을 실행하고,
        중간 프레임에서는 캐시된 결과를 반환한다.

        Args:
            frame: BGR 프레임

        Returns:
            dict: score, fire_conf, smoke_conf, detections, has_fire, has_smoke
        """
        if not self.enabled or not self._model_loaded:
            return self._cached_result

        self._frame_count += 1
        if self._frame_count % self.frame_interval != 0:
            return self._cached_result

        try:
            results = self._model.predict(
                frame,
                device=self._device,
                imgsz=self.input_size,
                conf=self.confidence_threshold,
                verbose=False,
            )

            fire_conf = 0.0
            smoke_conf = 0.0
            detections = []

            if results and len(results) > 0:
                result = results[0]
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = result.names.get(cls_id, "").lower()
                    xyxy = box.xyxy[0].cpu().numpy().tolist()

                    detections.append({
                        "class": cls_name,
                        "confidence": conf,
                        "bbox": xyxy,
                    })

                    if cls_name in self._FIRE_CLASSES:
                        fire_conf = max(fire_conf, conf)
                    elif cls_name in self._SMOKE_CLASSES:
                        smoke_conf = max(smoke_conf, conf)

            has_fire = fire_conf >= self.confidence_threshold
            has_smoke = smoke_conf >= self.confidence_threshold

            score = min(fire_conf * 70.0 + smoke_conf * 30.0, 100.0)

            self._cached_result = {
                "score": score,
                "fire_conf": fire_conf,
                "smoke_conf": smoke_conf,
                "detections": detections,
                "has_fire": has_fire,
                "has_smoke": has_smoke,
            }

        except Exception as e:
            logger.warning(f"YOLO 추론 실패: {e}")

        return self._cached_result

    def get_verification_adjustment(self, base_confidence: float) -> float:
        """기존 파이프라인 신뢰도를 YOLO 결과로 보정한다.

        Args:
            base_confidence: 기존 4개 필터의 최종 신뢰도 (0-100)

        Returns:
            보정된 최종 신뢰도 (0-100)
        """
        if not self.enabled or not self._model_loaded:
            return base_confidence

        r = self._cached_result

        if r["has_fire"]:
            boost = min(r["fire_conf"] * 40.0, self.fire_boost_max)
            return min(base_confidence + boost, 100.0)

        if r["has_smoke"]:
            boost = min(r["smoke_conf"] * 15.0, self.smoke_boost_max)
            return min(base_confidence + boost, 100.0)

        if base_confidence > 15.0:
            return base_confidence * self.no_detect_penalty

        return base_confidence

    def reset(self):
        """상태를 초기화한다."""
        self._frame_count = 0
        self._cached_result = self._empty_result()
