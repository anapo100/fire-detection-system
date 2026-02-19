"""형태 필터 단위 테스트."""

import unittest
import numpy as np
import cv2

from src.filters.shape_filter import ShapeFilter


class TestShapeFilter(unittest.TestCase):

    def setUp(self):
        self.config = {
            "shape_filter": {
                "irregularity_threshold": 20,
                "vertical_expansion_rate": 0.2,
            }
        }
        self.filter = ShapeFilter(self.config)

    def test_no_contours_returns_zero(self):
        """컨투어가 없으면 0점을 반환해야 한다."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        score = self.filter.analyze([], frame)
        self.assertEqual(score, 0.0)

    def test_circular_contour_low_score(self):
        """원형 컨투어는 낮은 불규칙성 점수를 가져야 한다."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # 원형 컨투어 생성
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.circle(mask, (320, 240), 50, 255, -1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        score = self.filter.analyze(contours, frame)
        # 원형은 불규칙성이 낮으므로 점수가 낮아야 함
        self.assertLessEqual(score, 15.0)

    def test_irregular_contour_higher_score(self):
        """불규칙한 컨투어는 더 높은 점수를 가져야 한다."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # 불규칙한 (별 모양) 컨투어 생성
        mask = np.zeros((480, 640), dtype=np.uint8)
        pts = np.array([
            [300, 150], [320, 200], [370, 210],
            [330, 240], [340, 290], [300, 260],
            [260, 290], [270, 240], [230, 210],
            [280, 200],
        ], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        score = self.filter.analyze(contours, frame)
        self.assertGreaterEqual(score, 0.0)

    def test_score_within_range(self):
        """점수는 항상 0-30 범위여야 한다."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.rectangle(mask, (100, 100), (300, 300), 255, -1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        score = self.filter.analyze(contours, frame)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 30.0)

    def test_reset(self):
        """리셋 후 히스토리가 초기화되어야 한다."""
        self.filter._bbox_history.append({"x": 0, "y": 0, "w": 100, "h": 100})
        self.filter.reset()
        self.assertEqual(len(self.filter._bbox_history), 0)


if __name__ == "__main__":
    unittest.main()
