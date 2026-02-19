"""제조 현장 화재 감지 시스템 - 메인 실행 파일.

USB(ADB 포트포워딩) + IP Webcam으로부터 영상을 수신하여
4단계 필터링 기반의 실시간 화재 감지를 수행한다.

사용법:
    python main.py
    python main.py --config config
"""

import os
import sys
import time
import signal
import logging
import argparse
from pathlib import Path

import cv2
import yaml


def get_base_dir() -> Path:
    """데이터 파일(config/models 등)의 기준 경로를 반환한다."""
    if getattr(sys, "frozen", False):
        # PyInstaller: _internal 폴더 (datas가 위치하는 곳)
        return Path(sys._MEIPASS)
    return Path(__file__).parent


def get_exe_dir() -> Path:
    """실행 파일 옆 경로를 반환한다 (logs 등 쓰기용)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent

from src.core.camera_loader import CameraLoader
from src.core.image_processor import ImageProcessor
from src.core.detector import FireDetector
from src.alert.alert_manager import AlertManager
from src.utils.roi_manager import ROIManager
from src.utils.performance_monitor import PerformanceMonitor
from src.utils.visualizer import Visualizer


def setup_logging():
    """로깅 설정을 초기화한다."""
    log_dir = get_exe_dir() / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_dir / "system.log"), encoding="utf-8"),
        ],
    )


def load_config(config_dir: str = "") -> dict:
    """설정 파일을 로드하여 통합된 설정 딕셔너리를 반환한다."""
    config = {}

    if config_dir:
        base = Path(config_dir)
    else:
        base = get_base_dir() / "config"

    config_files = [
        "camera_config.yaml",
        "detection_config.yaml",
        "alert_config.yaml",
    ]

    for filename in config_files:
        filepath = base / filename
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
        self._logger = logging.getLogger(__name__)

        # 구성 요소 초기화
        self.camera = CameraLoader(config)
        self.processor = ImageProcessor(config)
        self.detector = FireDetector(config)
        self.alert_manager = AlertManager(config)
        self.roi_manager = ROIManager(config)
        self.perf_monitor = PerformanceMonitor()
        self.visualizer = Visualizer()

        self._prev_frame = None
        self._display_counter = 0  # imshow 간격 제어용

    def start(self):
        """시스템을 시작한다."""
        self._logger.info("설정 파일 로드 완료")

        # 카메라 연결
        if not self.camera.connect_with_retry():
            self._logger.error("카메라 연결 실패. 시스템을 종료합니다.")
            return

        self._logger.info("화재 감지 시스템 시작...")
        self.running = True

        try:
            self._main_loop()
        except KeyboardInterrupt:
            self._logger.info("\n사용자에 의해 시스템 종료")
        finally:
            self.stop()

    def _main_loop(self):
        """메인 처리 루프."""
        frame_interval = 1.0 / self.config.get("camera", {}).get(
            "processing", {}
        ).get("target_fps", 15)

        while self.running:
            loop_start = time.time()
            self.perf_monitor.tick()

            # 1. 프레임 읽기 (스레드 방식: 즉시 반환)
            ret, frame = self.camera.read_frame()
            if not ret:
                # 스레드가 아직 프레임을 준비하지 못한 경우 vs 실제 연결 끊김
                if not self.camera.is_connected:
                    if not self._handle_connection_loss():
                        break
                else:
                    time.sleep(0.001)  # 프레임 대기
                continue

            # 2. 전처리
            processed = self.processor.preprocess(frame)

            # 3. ROI 적용
            if self.roi_manager.enabled:
                processed = self.roi_manager.apply_mask(processed)

            # 4. 화재 감지
            result = self.detector.detect(processed, self._prev_frame)
            self._prev_frame = processed

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

            # 7. 비디오 버퍼 (warning 이상일 때만 프레임 저장 → 메모리 절약)
            if result.level != "normal":
                self.alert_manager.buffer_frame(frame)

            # 8. 화면 표시 (2프레임에 1번만 갱신 → imshow 11ms 오버헤드 절감)
            self._display_counter += 1
            if self._display_counter % 2 == 0:
                self._display_frame(frame, result)
            else:
                # 화면 갱신 생략 시에도 키 입력은 확인
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == ord("Q"):
                    self.running = False

            # FPS 제어
            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _display_frame(self, frame, result):
        """모니터링 화면을 표시한다."""
        # draw_detection에서 copy=True로 1회만 복사, 이후는 in-place
        display = self.visualizer.draw_detection(
            frame, result.contours, result.level, result.confidence,
            yolo_detections=result.yolo_detections, copy=True,
        )

        if self.roi_manager.enabled:
            display = self.roi_manager.draw_regions(display)

        display = self.visualizer.draw_info_overlay(
            display,
            result.confidence,
            result.level,
            self.perf_monitor.fps,
            self.perf_monitor.avg_latency_ms,
        )

        cv2.imshow("Fire Detection System", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            self.running = False

    def _handle_connection_loss(self) -> bool:
        """연결 끊김을 처리한다."""
        self._logger.warning("카메라 연결 끊김. 재연결 시도 중...")

        self.camera.release()
        if self.camera.connect_with_retry():
            self._logger.info("카메라 재연결 성공")
            return True

        self._logger.error("카메라 재연결 실패")
        return False

    def stop(self):
        """시스템을 종료한다."""
        self.running = False
        self.camera.release()
        cv2.destroyAllWindows()
        self._logger.info("화재 감지 시스템 종료")


def parse_args():
    parser = argparse.ArgumentParser(
        description="제조 현장 화재 감지 시스템"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="설정 파일 디렉토리 경로 (기본값: exe와 같은 폴더의 config/)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 작업 디렉토리를 exe 기준으로 변경
    os.chdir(get_exe_dir())

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
