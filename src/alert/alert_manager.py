"""알림 통합 관리 모듈."""

import logging
from datetime import datetime
from typing import Optional

import numpy as np

from src.alert.logger import EventLogger
from src.alert.notifier import Notifier

logger = logging.getLogger(__name__)


class AlertManager:
    """감지 결과에 따라 알림 액션을 관리하는 클래스."""

    def __init__(self, config: dict):
        self.config = config
        self.event_logger = EventLogger(config)
        self.notifier = Notifier(config)

        levels_cfg = config.get("alert", {}).get("levels", {})
        self.level_actions = {
            "warning": levels_cfg.get("warning", {}).get("actions", ["log"]),
            "alert": levels_cfg.get("alert", {}).get("actions", ["log", "slack", "snapshot"]),
            "critical": levels_cfg.get("critical", {}).get(
                "actions", ["log", "slack", "email", "sound", "video"]
            ),
        }

    def handle_detection(
        self,
        confidence: float,
        level: str,
        frame: Optional[np.ndarray] = None,
        roi_name: str = "",
    ):
        """감지 결과에 따른 알림 액션을 실행한다."""
        if level == "normal":
            self.event_logger.log_event(confidence, level)
            return

        actions = self.level_actions.get(level, ["log"])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        location_info = f" - 위치: {roi_name}" if roi_name else ""
        snapshot_path = ""

        for action in actions:
            if action == "log":
                self.event_logger.log_event(
                    confidence, level, location_info
                )

            elif action == "snapshot" and frame is not None:
                snapshot_path = self.event_logger.save_snapshot(
                    frame, confidence, level
                )

            elif action == "slack":
                message = (
                    f"🔥 *화재 {'감지 확정' if level == 'critical' else '의심 감지'}!*\n"
                    f"• 신뢰도: {confidence:.0f}%\n"
                    f"• 시간: {timestamp}\n"
                    f"{f'• 위치: {roi_name}' if roi_name else ''}"
                )
                self.notifier.send_slack(message, snapshot_path)

            elif action == "email":
                subject = f"[긴급] 화재 감지 - 신뢰도 {confidence:.0f}%"
                body = (
                    f"<h2>화재 감지 알림</h2>"
                    f"<p><b>신뢰도:</b> {confidence:.0f}%</p>"
                    f"<p><b>레벨:</b> {level.upper()}</p>"
                    f"<p><b>시간:</b> {timestamp}</p>"
                    f"{'<p><b>위치:</b> ' + roi_name + '</p>' if roi_name else ''}"
                )
                self.notifier.send_email(subject, body, snapshot_path)

            elif action == "sound":
                self.notifier.play_sound()

            elif action == "video":
                video_path = self.event_logger.save_video()
                if video_path:
                    logger.info(f"이벤트 비디오 저장: {video_path}")

    def buffer_frame(self, frame: np.ndarray):
        """비디오 녹화를 위해 프레임을 버퍼에 추가한다."""
        self.event_logger.buffer_frame(frame)
