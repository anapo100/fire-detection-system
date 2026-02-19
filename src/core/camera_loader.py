"""USB(ADB 포트포워딩) + IP Webcam MJPEG 스트림 영상 수신 모듈.

백그라운드 스레드에서 프레임을 연속 수신하여
메인 스레드가 항상 최신 프레임을 즉시 가져올 수 있도록 한다.
"""

import time
import logging
import subprocess
import shutil
import threading
from typing import Optional, Tuple
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraLoader:
    """ADB 포트포워딩을 통한 IP Webcam MJPEG 스트림 수신 클래스.

    USB 케이블로 연결된 스마트폰의 IP Webcam 영상을
    ADB 포트포워딩(localhost)으로 수신하여 네트워크 지연을 최소화한다.

    백그라운드 스레드가 프레임을 연속으로 읽어 최신 프레임만 보관하므로
    메인 스레드의 read_frame()은 블로킹 없이 즉시 반환된다.
    """

    def __init__(self, config: dict):
        self.config = config
        cam_cfg = config["camera"]

        stream_cfg = cam_cfg.get("stream", {})
        self.port = stream_cfg.get("port", 8080)
        self.stream_url = f"http://localhost:{self.port}/video"

        proc_cfg = cam_cfg.get("processing", {})
        self.target_resolution = tuple(proc_cfg.get("target_resolution", [640, 480]))
        self.buffer_size = proc_cfg.get("buffer_size", 1)
        self.stream_refresh_interval = proc_cfg.get("stream_refresh_interval", 300)

        recon_cfg = cam_cfg.get("reconnection", {})
        self.max_attempts = recon_cfg.get("max_attempts", 10)
        self.retry_interval = recon_cfg.get("retry_interval", 3)

        self.cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._stream_start_time: float = 0
        self._adb_path = self._find_adb()

        # 스레드 기반 프레임 수신
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_ready = False

    def __del__(self):
        """GC 시 리소스를 안전하게 해제한다."""
        self.release()

    def _find_adb(self) -> str:
        """ADB 실행 파일 경로를 찾는다."""
        # 1. 프로젝트/exe 옆의 platform-tools 폴더
        import sys
        if getattr(sys, "frozen", False):
            base = Path(sys._MEIPASS)
        else:
            base = Path(__file__).parent.parent.parent

        local_adb = base / "platform-tools" / "adb.exe"
        if local_adb.exists():
            return str(local_adb)

        # 2. 시스템 PATH에서 찾기
        system_adb = shutil.which("adb")
        if system_adb:
            return system_adb

        logger.warning("ADB를 찾을 수 없습니다. platform-tools 폴더를 확인하세요.")
        return "adb"

    def _setup_adb_forward(self) -> bool:
        """ADB 포트포워딩을 설정한다."""
        try:
            # 기존 포워딩 제거 후 재설정
            subprocess.run(
                [self._adb_path, "forward", "--remove-all"],
                capture_output=True, timeout=5
            )
            result = subprocess.run(
                [self._adb_path, "forward", f"tcp:{self.port}", f"tcp:{self.port}"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                logger.info(f"ADB 포트포워딩 설정 완료 (localhost:{self.port})")
                return True
            else:
                logger.error(f"ADB 포트포워딩 실패: {result.stderr.strip()}")
                return False
        except FileNotFoundError:
            logger.error(f"ADB 실행 파일을 찾을 수 없습니다: {self._adb_path}")
            return False
        except subprocess.TimeoutExpired:
            logger.error("ADB 명령 시간 초과")
            return False
        except Exception as e:
            logger.error(f"ADB 포트포워딩 오류: {e}")
            return False

    def connect(self) -> bool:
        """ADB 포트포워딩 설정 후 MJPEG 스트림에 연결한다."""
        logger.info("USB(ADB) 카메라 연결 시도...")

        # ADB 포트포워딩 설정
        if not self._setup_adb_forward():
            return False

        # 포트포워딩 후 스트림 안정화 대기
        time.sleep(1)

        # MJPEG 스트림 연결
        self.cap = cv2.VideoCapture(self.stream_url)
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

        if self.cap is not None and self.cap.isOpened():
            logger.info(f"카메라 연결 성공: {self.stream_url}")
            self._running = True
            self._stream_start_time = time.time()
            # 백그라운드 프레임 수신 스레드 시작
            self._start_reader_thread()
            return True

        logger.error("MJPEG 스트림 연결 실패 — IP Webcam 앱이 실행 중인지 확인하세요.")
        return False

    def connect_with_retry(self) -> bool:
        """재연결 로직을 포함한 연결 시도."""
        for attempt in range(1, self.max_attempts + 1):
            if self.connect():
                return True
            logger.warning(
                f"재연결 시도 중... ({attempt}/{self.max_attempts})"
            )
            time.sleep(self.retry_interval)

        logger.error(
            f"{self.max_attempts}회 시도 후 연결 실패"
        )
        return False

    def _start_reader_thread(self):
        """백그라운드 프레임 수신 스레드를 시작한다."""
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        logger.info("프레임 수신 스레드 시작")

    def _reader_loop(self):
        """백그라운드에서 프레임을 연속으로 읽어 최신 프레임만 보관한다.

        이 스레드가 MJPEG 디코딩을 전담하므로
        메인 스레드는 블로킹 없이 최신 프레임만 가져갈 수 있다.
        """
        while self._running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.01)
                continue

            # 장시간 사용 시 TCP 버퍼 누적 방지를 위한 주기적 재연결
            if (self.stream_refresh_interval > 0
                    and time.time() - self._stream_start_time > self.stream_refresh_interval):
                logger.info("스트림 버퍼 초기화를 위한 재연결...")
                self.cap.release()
                self.cap = cv2.VideoCapture(self.stream_url)
                if self.cap is not None:
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
                self._stream_start_time = time.time()

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.001)
                continue

            # 리사이즈도 백그라운드에서 처리
            frame = cv2.resize(frame, self.target_resolution)

            with self._lock:
                self._latest_frame = frame
                self._frame_ready = True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """최신 프레임을 즉시 반환한다.

        백그라운드 스레드가 연속으로 프레임을 수신하므로
        이 메서드는 블로킹 없이 즉시 반환된다 (~0ms).
        """
        with self._lock:
            if not self._frame_ready:
                return False, None
            frame = self._latest_frame
            self._frame_ready = False
        return True, frame

    def release(self):
        """카메라 리소스를 해제한다."""
        self._running = False

        # 스레드 종료 대기
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3)
            self._thread = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        # ADB 포워딩 정리
        try:
            subprocess.run(
                [self._adb_path, "forward", "--remove-all"],
                capture_output=True, timeout=5
            )
        except Exception:
            pass
        logger.info("Camera released")

    @property
    def is_connected(self) -> bool:
        return self._running and self.cap is not None and self.cap.isOpened()
