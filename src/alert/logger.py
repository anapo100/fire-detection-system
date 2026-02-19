"""로그 및 스냅샷 저장 모듈."""

import os
import logging
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class EventLogger:
    """감지 이벤트의 로그와 스냅샷/비디오를 저장하는 클래스."""

    def __init__(self, config: dict):
        log_cfg = config.get("alert", {}).get("logging", {})
        self.save_snapshots = log_cfg.get("save_snapshots", True)
        self.save_videos = log_cfg.get("save_videos", True)
        self.video_duration = log_cfg.get("video_duration", 20)
        self.log_directory = log_cfg.get("log_directory", "logs/")

        self._video_writer: Optional[cv2.VideoWriter] = None
        self._video_frames: list = []
        self._ensure_log_directory()

    def _ensure_log_directory(self):
        """로그 디렉토리를 생성한다."""
        os.makedirs(self.log_directory, exist_ok=True)

    def _get_date_directory(self) -> str:
        """날짜별 디렉토리 경로를 반환한다."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        path = os.path.join(self.log_directory, date_str)
        os.makedirs(path, exist_ok=True)
        return path

    def save_snapshot(self, frame: np.ndarray, confidence: float, level: str) -> str:
        """감지 시점의 스냅샷을 저장한다."""
        if not self.save_snapshots:
            return ""

        date_dir = self._get_date_directory()
        timestamp = datetime.now().strftime("%H-%M-%S")
        filename = f"{timestamp}_fire_detected_{level}_{confidence:.0f}.jpg"
        filepath = os.path.join(date_dir, filename)

        cv2.imwrite(filepath, frame)
        logger.info(f"스냅샷 저장: {filepath}")
        return filepath

    def buffer_frame(self, frame: np.ndarray):
        """비디오 녹화를 위해 프레임을 버퍼에 추가한다."""
        if not self.save_videos:
            return

        max_buffer = self.video_duration * 15  # 15 FPS 기준
        self._video_frames.append(frame.copy())
        if len(self._video_frames) > max_buffer:
            self._video_frames.pop(0)

    def save_video(self, fps: int = 15) -> str:
        """버퍼에 저장된 프레임을 비디오로 저장한다."""
        if not self.save_videos or not self._video_frames:
            return ""

        date_dir = self._get_date_directory()
        timestamp = datetime.now().strftime("%H-%M-%S")
        filename = f"{timestamp}_fire_event.avi"
        filepath = os.path.join(date_dir, filename)

        h, w = self._video_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(filepath, fourcc, fps, (w, h))

        for f in self._video_frames:
            writer.write(f)
        writer.release()

        logger.info(f"비디오 저장: {filepath} ({len(self._video_frames)} frames)")
        self._video_frames.clear()
        return filepath

    def log_event(self, confidence: float, level: str, details: str = ""):
        """감지 이벤트를 로그에 기록한다."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if level == "normal":
            logger.info(f"[{timestamp}] 화재 미감지 (신뢰도: {confidence:.0f}%)")
        elif level == "warning":
            logger.warning(
                f"[{timestamp}] 화재 경계 (신뢰도: {confidence:.0f}%) {details}"
            )
        elif level == "alert":
            logger.warning(
                f"[{timestamp}] 화재 의심 감지! (신뢰도: {confidence:.0f}%) {details}"
            )
        elif level == "critical":
            logger.critical(
                f"[{timestamp}] 화재 감지 확정! (신뢰도: {confidence:.0f}%) {details}"
            )
