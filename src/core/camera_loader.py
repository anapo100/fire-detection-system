"""스마트폰 카메라 연결 및 영상 스트리밍 모듈."""

import time
import logging
import threading
from typing import Optional, Tuple

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)


class CameraLoader:
    """스마트폰(IP Webcam) 카메라 연결 및 프레임 수신을 담당하는 클래스."""

    def __init__(self, config: dict):
        self.config = config
        phone_cfg = config["camera"]["smartphone"]
        self.phone_ip = phone_cfg["phone_ip"]
        self.port = phone_cfg["port"]
        self.username = phone_cfg.get("username", "")
        self.password = phone_cfg.get("password", "")
        self.stream_type = phone_cfg.get("stream_type", "mjpeg")

        proc_cfg = config["camera"].get("processing", {})
        self.target_resolution = tuple(proc_cfg.get("target_resolution", [640, 480]))
        self.target_fps = proc_cfg.get("target_fps", 15)
        self.buffer_size = proc_cfg.get("buffer_size", 1)

        recon_cfg = config["camera"].get("reconnection", {})
        self.max_attempts = recon_cfg.get("max_attempts", 10)
        self.retry_interval = recon_cfg.get("retry_interval", 3)

        self.cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_time: float = 0
        self._connection_attempts = 0

    @property
    def base_url(self) -> str:
        auth = ""
        if self.username and self.password:
            auth = f"{self.username}:{self.password}@"
        return f"http://{auth}{self.phone_ip}:{self.port}"

    @property
    def stream_url(self) -> str:
        if self.stream_type == "mjpeg":
            return f"{self.base_url}/video"
        elif self.stream_type == "rtsp":
            auth = ""
            if self.username and self.password:
                auth = f"{self.username}:{self.password}@"
            return f"rtsp://{auth}{self.phone_ip}:{self.port}/h264_ulaw.sdp"
        elif self.stream_type == "snapshot":
            return f"{self.base_url}/shot.jpg"
        return f"{self.base_url}/video"

    def connect(self) -> bool:
        """카메라에 연결을 시도한다."""
        logger.info(f"Connecting to smartphone camera: {self.stream_url}")

        if self.stream_type == "snapshot":
            try:
                resp = requests.get(f"{self.base_url}/shot.jpg", timeout=5)
                if resp.status_code == 200:
                    logger.info("Camera connected successfully (snapshot mode)")
                    self._running = True
                    return True
            except requests.RequestException as e:
                logger.error(f"Snapshot connection failed: {e}")
                return False

        self.cap = cv2.VideoCapture(self.stream_url)
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

        if self.cap is not None and self.cap.isOpened():
            logger.info("Camera connected successfully")
            self._running = True
            self._connection_attempts = 0
            return True

        logger.error("Connection failed: Failed to open stream")
        return False

    def connect_with_retry(self) -> bool:
        """재연결 로직을 포함한 연결 시도."""
        for attempt in range(1, self.max_attempts + 1):
            if self.connect():
                return True
            logger.warning(
                f"Reconnecting... (attempt {attempt}/{self.max_attempts})"
            )
            time.sleep(self.retry_interval)

        logger.error(
            f"Failed to connect after {self.max_attempts} attempts"
        )
        return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """한 프레임을 읽어서 반환한다."""
        if self.stream_type == "snapshot":
            return self._read_snapshot()

        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Frame read error: timeout")
            return False, None

        frame = cv2.resize(frame, self.target_resolution)
        self._latest_frame = frame
        self._frame_time = time.time()
        return True, frame

    def _read_snapshot(self) -> Tuple[bool, Optional[np.ndarray]]:
        """스냅샷 모드로 프레임을 읽는다."""
        try:
            resp = requests.get(f"{self.base_url}/shot.jpg", timeout=5)
            if resp.status_code != 200:
                return False, None
            img_array = np.frombuffer(resp.content, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is None:
                return False, None
            frame = cv2.resize(frame, self.target_resolution)
            self._latest_frame = frame
            self._frame_time = time.time()
            return True, frame
        except requests.RequestException as e:
            logger.warning(f"Snapshot read error: {e}")
            return False, None

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """가장 최근에 읽은 프레임과 시간을 반환한다."""
        return self._latest_frame, self._frame_time

    def release(self):
        """카메라 리소스를 해제한다."""
        self._running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        logger.info("Camera released")

    @property
    def is_connected(self) -> bool:
        if self.stream_type == "snapshot":
            return self._running
        return self.cap is not None and self.cap.isOpened()
