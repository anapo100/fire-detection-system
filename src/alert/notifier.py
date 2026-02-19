"""Slack/Email/사운드 알림 발송 모듈."""

import os
import logging
import smtplib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from typing import Optional

logger = logging.getLogger(__name__)


class Notifier:
    """Slack, Email, 사운드 알림을 발송하는 클래스."""

    def __init__(self, config: dict):
        alert_cfg = config.get("alert", {})

        # Slack 설정
        slack_cfg = alert_cfg.get("slack", {})
        self.slack_enabled = slack_cfg.get("enabled", False)
        self.slack_webhook_url = slack_cfg.get("webhook_url", "")
        self.slack_channel = slack_cfg.get("channel", "#fire-alerts")
        self.slack_mention_users = slack_cfg.get("mention_users", [])

        # Email 설정
        email_cfg = alert_cfg.get("email", {})
        self.email_enabled = email_cfg.get("enabled", False)
        self.smtp_server = email_cfg.get("smtp_server", "")
        self.smtp_port = email_cfg.get("smtp_port", 587)
        self.email_sender = email_cfg.get("sender", "")
        self.email_password = email_cfg.get("password", "")
        self.email_recipients = email_cfg.get("recipients", [])

        # 사운드 설정
        sound_cfg = alert_cfg.get("sound", {})
        self.sound_enabled = sound_cfg.get("enabled", True)
        self.sound_file = sound_cfg.get("file_path", "assets/siren.wav")
        self.sound_volume = sound_cfg.get("volume", 0.8)

    def send_slack(self, message: str, snapshot_path: Optional[str] = None):
        """Slack 웹훅으로 알림을 보낸다."""
        if not self.slack_enabled:
            return

        try:
            import requests

            mentions = " ".join(self.slack_mention_users)
            payload = {
                "channel": self.slack_channel,
                "text": f"{mentions}\n{message}",
                "username": "Fire Detection Bot",
                "icon_emoji": ":fire:",
            }
            resp = requests.post(self.slack_webhook_url, json=payload, timeout=10)
            if resp.status_code == 200:
                logger.info("Slack 알림 발송 완료")
            else:
                logger.error(f"Slack 알림 실패: {resp.status_code}")
        except Exception as e:
            logger.error(f"Slack 알림 오류: {e}")

    def send_email(
        self, subject: str, body: str, snapshot_path: Optional[str] = None
    ):
        """Email로 알림을 보낸다."""
        if not self.email_enabled:
            return

        try:
            msg = MIMEMultipart()
            msg["From"] = self.email_sender
            msg["To"] = ", ".join(self.email_recipients)
            msg["Subject"] = subject

            msg.attach(MIMEText(body, "html"))

            if snapshot_path and os.path.exists(snapshot_path):
                with open(snapshot_path, "rb") as f:
                    img = MIMEImage(f.read())
                    img.add_header(
                        "Content-Disposition",
                        "attachment",
                        filename=os.path.basename(snapshot_path),
                    )
                    msg.attach(img)

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_sender, self.email_password)
                server.sendmail(
                    self.email_sender, self.email_recipients, msg.as_string()
                )
            logger.info("Email 알림 발송 완료")
        except Exception as e:
            logger.error(f"Email 알림 오류: {e}")

    def play_sound(self):
        """경고 사운드를 재생한다."""
        if not self.sound_enabled:
            return

        def _play():
            try:
                if not os.path.exists(self.sound_file):
                    # 사운드 파일이 없으면 시스템 비프음 사용
                    import sys
                    if sys.platform == "win32":
                        import winsound
                        for _ in range(3):
                            winsound.Beep(1000, 500)
                    else:
                        print("\a", end="", flush=True)
                    return

                # 사운드 파일이 있으면 재생 시도
                try:
                    from playsound import playsound
                    playsound(self.sound_file)
                except ImportError:
                    import sys
                    if sys.platform == "win32":
                        import winsound
                        winsound.PlaySound(
                            self.sound_file, winsound.SND_FILENAME
                        )
                    else:
                        os.system(f'aplay "{self.sound_file}" 2>/dev/null || afplay "{self.sound_file}" 2>/dev/null')
            except Exception as e:
                logger.debug(f"사운드 재생 실패: {e}")

        thread = threading.Thread(target=_play, daemon=True)
        thread.start()
        logger.info("사이렌 재생 중...")
