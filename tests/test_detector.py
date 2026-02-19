"""화재 감지 엔진 단위 테스트."""

import unittest
import numpy as np

from src.core.detector import FireDetector, DetectionResult


class TestFireDetector(unittest.TestCase):

    def setUp(self):
        self.config = {
            "color_filter": {
                "hsv_ranges": {
                    "flame_orange": {"lower": [0, 100, 200], "upper": [20, 255, 255]},
                    "flame_yellow": {"lower": [20, 100, 200], "upper": [40, 255, 255]},
                },
                "min_area": 300,
                "max_area": 50000,
            },
            "motion_filter": {
                "optical_flow_threshold": 2.5,
                "temporal_frames": 5,
                "pixel_change_threshold": 0.15,
            },
            "shape_filter": {
                "irregularity_threshold": 20,
                "vertical_expansion_rate": 0.2,
            },
            "detection": {
                "confidence_threshold": 70,
                "debounce_seconds": 30,
                "consecutive_frames": 3,
            },
        }
        self.detector = FireDetector(self.config)

    def test_no_fire_on_black_frame(self):
        """검은 프레임에서는 화재가 감지되지 않아야 한다."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.detect(frame)
        self.assertEqual(result.level, "normal")
        self.assertLess(result.confidence, 60)

    def test_result_has_all_fields(self):
        """결과 객체에 모든 필드가 있어야 한다."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.detect(frame)
        self.assertIsInstance(result, DetectionResult)
        self.assertIsNotNone(result.confidence)
        self.assertIsNotNone(result.color_score)
        self.assertIsNotNone(result.motion_score)
        self.assertIsNotNone(result.shape_score)
        self.assertIsNotNone(result.level)
        self.assertIsNotNone(result.timestamp)

    def test_confidence_in_range(self):
        """신뢰도는 0-100 범위여야 한다."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = self.detector.detect(frame)
        self.assertGreaterEqual(result.confidence, 0)
        self.assertLessEqual(result.confidence, 100)

    def test_should_alert_requires_consecutive(self):
        """연속 감지 없이는 알림이 발생하지 않아야 한다."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.detect(frame)
        # 단일 프레임으로는 알림 안됨
        self.assertFalse(self.detector.should_alert(result))

    def test_level_determination(self):
        """레벨 판정이 올바르게 동작해야 한다."""
        self.assertEqual(self.detector._determine_level(50), "normal")
        self.assertEqual(self.detector._determine_level(65), "warning")
        self.assertEqual(self.detector._determine_level(75), "alert")
        self.assertEqual(self.detector._determine_level(90), "critical")

    def test_reset(self):
        """리셋 후 상태가 초기화되어야 한다."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.detector.detect(frame)
        self.detector.reset()
        self.assertEqual(self.detector._consecutive_count, 0)
        self.assertEqual(len(self.detector._detection_history), 0)


if __name__ == "__main__":
    unittest.main()
