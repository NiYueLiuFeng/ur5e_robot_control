import unittest

import numpy as np


class CalibrationPatternTest(unittest.TestCase):
    def test_chessboard_object_points_follow_row_major_order(self):
        from ur5e_control.calibration.patterns import (
            ChessboardPattern,
            chessboard_object_points,
        )

        pattern = ChessboardPattern(columns=3, rows=2, square_size_m=0.04)

        points = chessboard_object_points(pattern)

        self.assertEqual(points.shape, (6, 3))
        self.assertEqual(points[0].tolist(), [0.0, 0.0, 0.0])
        self.assertTrue(np.allclose(points[4], [0.04, 0.04, 0.0]))

    def test_charuco_pattern_rejects_marker_larger_than_square(self):
        from ur5e_control.calibration.patterns import CharucoPattern

        with self.assertRaisesRegex(ValueError, "marker_length_m"):
            CharucoPattern(
                squares_x=5,
                squares_y=7,
                square_length_m=0.04,
                marker_length_m=0.05,
            )


if __name__ == "__main__":
    unittest.main()
