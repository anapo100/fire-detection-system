"""색상 필터 단위 테스트."""

import unittest
import numpy as np
import cv2

from src.filters.color_filter import ColorFilter


class TestColorFilter(unittest.TestCase):

    def setUp(self):
        self.config = {
            "color_filter": {
                "hsv_ranges": {
                    "flame_orange": {
                        "lower": [0, 100, 200],
                        "upper": [20, 255, 255],
                    },
                    "flame_yellow": {
                        "lower": [20, 100, 200],
                        "upper": [40, 255, 255],
                    },
                },
                "min_area": 300,
                "max_area": 50000,
            }
        }
        self.filter = ColorFilter(self.config)

    def test_no_fire_on_black_frame(self):
        """검은 프레임에서는 화재가 감지되지 않아야 한다."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        score, mask, contours = self.filter.analyze(frame)
        self.assertEqual(score, 0.0)
        self.assertEqual(len(contours), 0)

    def test_no_fire_on_blue_frame(self):
        """파란색 프레임에서는 화재가 감지되지 않아야 한다."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :] = (255, 0, 0)  # BGR: Blue
        score, mask, contours = self.filter.analyze(frame)
        self.assertEqual(score, 0.0)

    def test_detects_orange_region(self):
        """주황색 영역이 충분히 크면 화재 후보로 감지해야 한다."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # HSV: Hue=10, Sat=200, Val=240 -> 주황색 화염 범위
        hsv_color = np.array([[[10, 200, 240]]], dtype=np.uint8)
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        # 충분히 큰 사각형 영역에 주황색 채우기
        frame[100:200, 100:200] = bgr_color
        score, mask, contours = self.filter.analyze(frame)
        self.assertGreater(score, 0.0)

    def test_filters_small_areas(self):
        """최소 면적보다 작은 영역은 필터링되어야 한다."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        hsv_color = np.array([[[10, 200, 240]]], dtype=np.uint8)
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        # 매우 작은 영역 (5x5 = 25px < min_area 300)
        frame[100:105, 100:105] = bgr_color
        score, mask, contours = self.filter.analyze(frame)
        self.assertEqual(len(contours), 0)

    def test_score_within_range(self):
        """점수는 항상 0-40 범위여야 한다."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        hsv_color = np.array([[[10, 200, 240]]], dtype=np.uint8)
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        frame[50:300, 50:300] = bgr_color
        score, _, _ = self.filter.analyze(frame)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 40.0)


if __name__ == "__main__":
    unittest.main()
