"""제조 현장 화재 감지 시스템 - 메인 실행 파일.

스마트폰 카메라(IP Webcam)로부터 영상을 수신하여
3단계 필터링 기반의 실시간 화재 감지를 수행한다.

사용법:
    python main.py
    python main.py --config config/camera_config.yaml
"""

import sys
import time
import signal
import logging
import argparse
from pathlib import Path

import cv2
import yaml
import numpy as np

from src.core.camera_loader import CameraLoader
from src.core.image_processor import ImageProcessor
from src.core.detector import FireDetector
from src.core.phone_monitor import PhoneMonitor
from src.filters.color_filter import ColorFilter
from src.filters.motion_filter import MotionFilter
from src.filters.shape_filter import ShapeFilter
from src.alert.alert_manager import AlertManager
from src.utils.roi_manager import ROIManager
from src.utils.performance_monitor import PerformanceMonitor
from src.utils.visualizer import Visualizer


def setup_logging():
    """로깅 설정을 초기화한다."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/system.log", encoding="utf-8"),
        ],
    )


def load_config(config_dir: str = "config") -> dict:
    """설정 파일을 로드하여 통합된 설정 딕셔너리를 반환한다."""
    config = {}

    config_files = {
        "camera_config.yaml": None,
        "detection_config.yaml": None,
        "alert_config.yaml": None,
    }

    for filename in config_files:
        filepath = Path(config_dir) / filename
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data:
                    config.update(data)

    return config


class FireDetectionSystem:
    """화재 감지 시스템의 메인 컨트롤러."""

    def __init__(self, config: dict):
        self.config = config
        self.running = False

        # 구성 요소 초기화
        self.camera = CameraLoader(config)
        self.processor = ImageProcessor(config)
        self.detector = FireDetector(config)
        self.phone_monitor = PhoneMonitor(config)
        self.alert_manager = AlertManager(config)
        self.roi_manager = ROIManager(config)
        self.perf_monitor = PerformanceMonitor()
        self.visualizer = Visualizer()

        self._prev_frame = None
        self._phone_check_interval = 30  # 30초마다 스마트폰 상태 체크
        self._last_phone_check = 0
        self._phone_status_str = ""

    def start(self):
        """시스템을 시작한다."""
        logger = logging.getLogger(__name__)
        logger.info("설정 파일 로드 완료")

        # 카메라 연결
        if not self.camera.connect_with_retry():
            logger.error("카메라 연결 실패. 시스템을 종료합니다.")
            return

        # 스마트폰 상태 확인
        self._check_phone_status()

        logger.info("화재 감지 시스템 시작...")
        self.running = True

        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("\n사용자에 의해 시스템 종료")
        finally:
            self.stop()

    def _main_loop(self):
        """메인 처리 루프."""
        logger = logging.getLogger(__name__)
        frame_interval = 1.0 / self.config.get("camera", {}).get(
            "processing", {}
        ).get("target_fps", 15)

        while self.running:
            loop_start = time.time()
            self.perf_monitor.tick()

            # 1. 프레임 읽기
            ret, frame = self.camera.read_frame()
            if not ret:
                if not self._handle_connection_loss():
                    break
                continue

            # 2. 전처리
            processed = self.processor.preprocess(frame)

            # 3. ROI 적용
            if self.roi_manager.enabled:
                processed = self.roi_manager.apply_mask(processed)

            # 4. 화재 감지
            result = self.detector.detect(processed, self._prev_frame)
            self._prev_frame = processed.copy()

            # 5. 지연시간 기록
            latency_ms = (time.time() - loop_start) * 1000
            self.perf_monitor.record_latency(latency_ms)

            # 6. 알림 처리
            if self.detector.should_alert(result):
                roi_name = ""
                if result.contours and self.roi_manager.enabled:
                    x, y, w, h = cv2.boundingRect(result.contours[0])
                    region = self.roi_manager.get_region_at(
                        x + w // 2, y + h // 2
                    )
                    if region:
                        roi_name = region.name

                self.alert_manager.handle_detection(
                    result.confidence, result.level, frame, roi_name
                )
            else:
                self.alert_manager.event_logger.log_event(
                    result.confidence, result.level
                )

            # 7. 비디오 버퍼
            self.alert_manager.buffer_frame(frame)

            # 8. 스마트폰 상태 주기적 확인
            now = time.time()
            if now - self._last_phone_check > self._phone_check_interval:
                self._check_phone_status()

            # 9. 화면 표시
            self._display_frame(frame, result)

            # FPS 제어
            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _display_frame(self, frame, result):
        """모니터링 화면을 표시한다."""
        display = self.visualizer.draw_detection(
            frame, result.contours, result.level, result.confidence
        )

        if self.roi_manager.enabled:
            display = self.roi_manager.draw_regions(display)

        display = self.visualizer.draw_info_overlay(
            display,
            result.confidence,
            result.level,
            self.perf_monitor.fps,
            self.perf_monitor.avg_latency_ms,
            self._phone_status_str,
        )

        cv2.imshow("Fire Detection System", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            self.running = False

    def _handle_connection_loss(self) -> bool:
        """연결 끊김을 처리한다."""
        logger = logging.getLogger(__name__)
        logger.warning("카메라 연결 끊김. 재연결 시도 중...")

        self.camera.release()
        if self.camera.connect_with_retry():
            logger.info("카메라 재연결 성공")
            return True

        logger.error("카메라 재연결 실패")
        return False

    def _check_phone_status(self):
        """스마트폰 상태를 확인한다."""
        self._phone_status_str = self.phone_monitor.get_status_string()
        status = self.phone_monitor.check_status()
        self._last_phone_check = time.time()

        logger = logging.getLogger(__name__)
        if status.get("battery_level") is not None:
            logger.info(
                f"스마트폰 배터리: {status['battery_level']}%"
            )

        for warning in status.get("warnings", []):
            logger.warning(warning)

    def stop(self):
        """시스템을 종료한다."""
        logger = logging.getLogger(__name__)
        self.running = False
        self.camera.release()
        cv2.destroyAllWindows()
        logger.info("화재 감지 시스템 종료")


def parse_args():
    parser = argparse.ArgumentParser(
        description="제조 현장 화재 감지 시스템"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config",
        help="설정 파일 디렉토리 경로 (기본값: config)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 로그 디렉토리 생성
    Path("logs").mkdir(exist_ok=True)

    setup_logging()
    logger = logging.getLogger(__name__)

    # 설정 로드
    config = load_config(args.config)
    if not config:
        logger.error("설정 파일을 찾을 수 없습니다. config/ 디렉토리를 확인해주세요.")
        sys.exit(1)

    # 시스템 시작
    system = FireDetectionSystem(config)

    # SIGINT 핸들링
    def signal_handler(sig, frame):
        system.running = False

    signal.signal(signal.SIGINT, signal_handler)

    system.start()


if __name__ == "__main__":
    main()
