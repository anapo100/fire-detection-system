"""로그 및 스냅샷 저장 모듈."""

import os
import logging
from collections import deque
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
        # deque로 변경: maxlen 자동 관리로 pop(0)의 O(n) 비용 제거
        max_buffer = self.video_duration * 15  # 15 FPS 기준
        self._video_frames: deque = deque(maxlen=max_buffer)
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
        # deque의 maxlen이 자동으로 오래된 프레임을 제거
        self._video_frames.append(frame.copy())

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

    def log_event(
        self,
        confidence: float,
        level: str,
        details: str = "",
        flame_risk: float = 0.0,
        smoke_risk: float = 0.0,
        has_flame: bool = False,
        has_smoke: bool = False,
    ):
        """감지 이벤트를 로그에 기록한다."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 화염/연기 상태 태그
        tags = []
        if has_flame:
            tags.append(f"flame:{flame_risk:.0f}")
        if has_smoke:
            tags.append(f"smoke:{smoke_risk:.0f}")
        tag_str = f" [{'+'.join(tags)}]" if tags else ""

        if level == "warning":
            logger.warning(
                f"[{timestamp}] 화재 경계 (신뢰도: {confidence:.0f}%){tag_str} {details}"
            )
        elif level == "alert":
            logger.warning(
                f"[{timestamp}] 화재 의심 감지! (신뢰도: {confidence:.0f}%){tag_str} {details}"
            )
        elif level == "critical":
            logger.critical(
                f"[{timestamp}] 화재 감지 확정! (신뢰도: {confidence:.0f}%){tag_str} {details}"
            )
