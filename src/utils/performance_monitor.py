"""FPS/지연시간 측정 모듈."""

import time
import logging
from collections import deque

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """실시간 FPS와 지연시간을 측정하는 클래스."""

    def __init__(self, window_size: int = 30):
        self._frame_times: deque = deque(maxlen=window_size)
        self._latencies: deque = deque(maxlen=window_size)
        self._last_time: float = 0
        self._start_time: float = time.time()
        self._total_frames: int = 0

    def tick(self):
        """프레임 처리 시작 시점을 기록한다."""
        now = time.time()
        if self._last_time > 0:
            self._frame_times.append(now - self._last_time)
        self._last_time = now
        self._total_frames += 1

    def record_latency(self, latency_ms: float):
        """프레임 처리 지연시간을 기록한다."""
        self._latencies.append(latency_ms)

    @property
    def fps(self) -> float:
        """현재 FPS를 반환한다."""
        if not self._frame_times:
            return 0.0
        avg_interval = sum(self._frame_times) / len(self._frame_times)
        if avg_interval <= 0:
            return 0.0
        return 1.0 / avg_interval

    @property
    def avg_latency_ms(self) -> float:
        """평균 지연시간(ms)을 반환한다."""
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time

    def get_status_string(self) -> str:
        """모니터링 화면에 표시할 성능 문자열을 반환한다."""
        return f"FPS: {self.fps:.0f}  |  Latency: {self.avg_latency_ms:.0f}ms"
