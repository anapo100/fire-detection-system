"""디버깅 화면 오버레이 모듈."""

import logging
from typing import Optional, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Visualizer:
    """실시간 모니터링 화면에 감지 정보를 오버레이하는 클래스."""

    COLORS = {
        "normal": (0, 255, 0),      # 초록
        "warning": (0, 255, 255),    # 노랑
        "alert": (0, 165, 255),      # 주황
        "critical": (0, 0, 255),     # 빨강
    }

    def __init__(self):
        self.show_contours = True
        self.show_info = True

    def draw_detection(
        self,
        frame: np.ndarray,
        contours: list,
        level: str,
        confidence: float,
        yolo_detections: list = None,
        copy: bool = True,
    ) -> np.ndarray:
        """감지 결과를 프레임에 표시한다.

        Args:
            yolo_detections: YOLO 검출 결과 리스트 [{class, confidence, bbox}, ...]
            copy: True면 복사본에 그림, False면 원본에 직접 그림
        """
        display = frame.copy() if copy else frame
        color = self.COLORS.get(level, (255, 255, 255))

        # 감지된 영역에 컨투어 그리기
        if self.show_contours and contours:
            cv2.drawContours(display, contours, -1, color, 2)

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)

        # YOLO 검출 박스 그리기
        if yolo_detections:
            for det in yolo_detections:
                x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                cls_name = det["class"]
                conf = det["confidence"]
                # 시안(화재), 마젠타(연기)
                det_color = (255, 255, 0) if cls_name in ("fire", "flame") else (255, 0, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), det_color, 2)
                label = f"YOLO:{cls_name} {conf:.0%}"
                cv2.putText(
                    display, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, det_color, 1,
                )

        return display

    def draw_info_overlay(
        self,
        frame: np.ndarray,
        confidence: float,
        level: str,
        fps: float,
        latency: float,
    ) -> np.ndarray:
        """상태 정보 오버레이를 그린다."""
        if not self.show_info:
            return frame

        display = frame  # 복사 없이 직접 수정 (draw_detection에서 이미 복사됨)
        h, w = display.shape[:2]
        color = self.COLORS.get(level, (0, 255, 0))

        # 상단 정보 바 (오버레이 영역만 부분 복사)
        roi_overlay = display[0:80, 0:w].copy()
        cv2.rectangle(roi_overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.addWeighted(roi_overlay, 0.6, display[0:80, 0:w], 0.4, 0, display[0:80, 0:w])

        # 상태 텍스트
        status_text = self._get_status_text(level)
        cv2.putText(
            display, status_text, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )

        # 신뢰도
        conf_text = f"Confidence: {confidence:.0f}%"
        cv2.putText(
            display, conf_text, (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

        # 성능 정보
        perf_text = f"FPS: {fps:.0f}  |  Latency: {latency:.0f}ms"
        cv2.putText(
            display, perf_text, (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
        )

        # 화재 감지 시 경고 테두리
        if level in ("alert", "critical"):
            border_color = self.COLORS[level]
            cv2.rectangle(display, (0, 0), (w - 1, h - 1), border_color, 4)

        return display

    def _get_status_text(self, level: str) -> str:
        status_map = {
            "normal": "Status: Normal Monitoring",
            "warning": "Status: WARNING",
            "alert": "Status: FIRE SUSPECTED!",
            "critical": "Status: FIRE DETECTED!",
        }
        return status_map.get(level, "Status: Unknown")
