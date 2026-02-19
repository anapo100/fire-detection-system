"""배우 기본적인 FPS/지연시간 측정 모듈."""

import time
import logging
from collections import deque

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """실시간 FPS와 지연시간을 측정하는 클래스.

    누적합 관리로 sum()/len() 반복 호출을 제거하여 O(1) 접근을 보장한다.
    """

    def __init__(self, window_size: int = 30):
        self._window_size = window_size
        self._frame_times: deque = deque(maxlen=window_size)
        self._latencies: deque = deque(maxlen=window_size)
        self._last_time: float = 0

        # 누적합 캐싱 (O(1) 평균 계산)
        self._frame_time_sum: float = 0.0
        self._latency_sum: float = 0.0

    def tick(self):
        """프레임 처리 시작 시점을 기록한다."""
        now = time.time()
        if self._last_time > 0:
            interval = now - self._last_time
            # deque가 꼬차면 가장 오래된 값을 누적합에서 제거
            if len(self._frame_times) == self._window_size:
                self._frame_time_sum -= self._frame_times[0]
            self._frame_times.append(interval)
            self._frame_time_sum += interval
        self._last_time = now

    def record_latency(self, latency_ms: float):
        """프레임 처리 지연시간을 기록한다."""
        if len(self._latencies) == self._window_size:
            self._latency_sum -= self._latencies[0]
        self._latencies.append(latency_ms)
        self._latency_sum += latency_ms

    @property
    def fps(self) -> float:
        """현재 FPS를 반환한다."""
        n = len(self._frame_times)
        if n == 0 or self._frame_time_sum <= 0:
            return 0.0
        return n / self._frame_time_sum

    @property
    def avg_latency_ms(self) -> float:
        """평균 지연시간(ms)을 반환한다."""
        n = len(self._latencies)
        if n == 0:
            return 0.0
        return self._latency_sum / n

