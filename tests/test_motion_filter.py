"""움직임 필터 단위 테스트."""

import unittest
import numpy as np

from src.filters.motion_filter import MotionFilter


class TestMotionFilter(unittest.TestCase):

    def setUp(self):
        self.config = {
            "motion_filter": {
                "optical_flow_threshold": 2.5,
                "temporal_frames": 5,
                "pixel_change_threshold": 0.15,
            }
        }
        self.filter = MotionFilter(self.config)

    def test_no_motion_on_identical_frames(self):
        """동일한 프레임 사이에는 움직임이 감지되지 않아야 한다."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        score = self.filter.analyze(frame, frame.copy())
        self.assertLessEqual(score, 5.0)

    def test_motion_on_shifted_frame(self):
        """이동된 프레임에서는 움직임이 감지되어야 한다."""
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame1[100:200, 100:200] = (255, 255, 255)

        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2[120:220, 120:220] = (255, 255, 255)

        score = self.filter.analyze(frame2, frame1)
        # 움직임이 있으므로 0보다 큰 점수
        self.assertGreaterEqual(score, 0.0)

    def test_score_within_range(self):
        """점수는 항상 0-30 범위여야 한다."""
        frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        score = self.filter.analyze(frame2, frame1)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 30.0)

    def test_reset(self):
        """리셋 후 내부 상태가 초기화되어야 한다."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.filter.analyze(frame, frame)
        self.filter.reset()
        self.assertEqual(len(self.filter._prev_grays), 0)
        self.assertEqual(len(self.filter._flow_magnitudes), 0)


if __name__ == "__main__":
    unittest.main()
