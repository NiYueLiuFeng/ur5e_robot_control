import unittest

import numpy as np


class DepthQualityTest(unittest.TestCase):
    def test_counts_zero_and_saturated_uint16_pixels_as_invalid(self):
        from ur5e_control.data_collection.depth_quality import depth_invalid_ratio

        depth = np.array([[0, 100], [65535, 200]], dtype=np.uint16)

        self.assertEqual(depth_invalid_ratio(depth), 0.5)

    def test_uses_median_across_samples_for_gate(self):
        from ur5e_control.data_collection.depth_quality import (
            evaluate_depth_quality,
        )

        samples = [
            {"wrist_depth": np.array([[0, 0], [100, 100]], dtype=np.uint16)},
            {"wrist_depth": np.array([[0, 100], [100, 100]], dtype=np.uint16)},
            {"wrist_depth": np.array([[0, 0], [0, 100]], dtype=np.uint16)},
        ]

        result = evaluate_depth_quality(samples, {"wrist_depth": 0.6})[0]

        self.assertEqual(result.minimum_invalid_ratio, 0.25)
        self.assertEqual(result.median_invalid_ratio, 0.5)
        self.assertEqual(result.maximum_invalid_ratio, 0.75)
        self.assertTrue(result.ok)

    def test_missing_depth_stream_is_a_failure(self):
        from ur5e_control.data_collection.depth_quality import (
            evaluate_depth_quality,
        )

        with self.assertRaisesRegex(ValueError, "missing"):
            evaluate_depth_quality([{}], {"wrist_depth": 0.5})


if __name__ == "__main__":
    unittest.main()
