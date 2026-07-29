import unittest

import numpy as np


class CalibrationSolverTest(unittest.TestCase):
    def test_base_extrinsics_solver_recovers_fixed_camera_pose(self):
        from ur5e_control.calibration.solvers import solve_base_extrinsics
        from ur5e_control.calibration.transforms import (
            compose_transforms,
            invert_transform,
            matrix_from_translation_quaternion,
        )
        from ur5e_control.calibration.types import BaseExtrinsicsSample

        expected_base_T_camera = matrix_from_translation_quaternion(
            translation=(0.5, -0.25, 0.8),
            quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        )
        base_T_tools = [
            matrix_from_translation_quaternion((0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0)),
            matrix_from_translation_quaternion((0.2, 0.1, 0.3), (0.0, 0.0, 0.0, 1.0)),
            matrix_from_translation_quaternion((0.3, -0.1, 0.4), (0.0, 0.0, 0.0, 1.0)),
        ]
        samples = [
            BaseExtrinsicsSample(
                base_T_tool=base_T_tool,
                camera_T_target=compose_transforms(
                    invert_transform(expected_base_T_camera),
                    base_T_tool,
                ),
            )
            for base_T_tool in base_T_tools
        ]

        result = solve_base_extrinsics(samples)

        self.assertEqual(result.parent_frame, "base")
        self.assertEqual(result.child_frame, "camera")
        self.assertTrue(np.allclose(result.matrix, expected_base_T_camera, atol=1e-8))

    def test_stereo_extrinsics_solver_recovers_camera_to_camera_pose(self):
        from ur5e_control.calibration.solvers import solve_stereo_extrinsics
        from ur5e_control.calibration.transforms import (
            compose_transforms,
            invert_transform,
            matrix_from_translation_quaternion,
        )
        from ur5e_control.calibration.types import StereoExtrinsicsSample

        expected_reference_T_camera = matrix_from_translation_quaternion(
            translation=(0.12, 0.03, -0.05),
            quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        )
        reference_T_targets = [
            matrix_from_translation_quaternion((0.2, 0.0, 0.8), (0.0, 0.0, 0.0, 1.0)),
            matrix_from_translation_quaternion((0.3, 0.1, 0.9), (0.0, 0.0, 0.0, 1.0)),
            matrix_from_translation_quaternion((0.4, -0.1, 1.0), (0.0, 0.0, 0.0, 1.0)),
        ]
        samples = [
            StereoExtrinsicsSample(
                reference_camera_T_target=reference_T_target,
                camera_T_target=compose_transforms(
                    invert_transform(expected_reference_T_camera),
                    reference_T_target,
                ),
            )
            for reference_T_target in reference_T_targets
        ]

        result = solve_stereo_extrinsics(
            samples,
            reference_camera="wrist",
            camera="third_front",
        )

        self.assertEqual(result.parent_frame, "wrist")
        self.assertEqual(result.child_frame, "third_front")
        self.assertTrue(
            np.allclose(result.matrix, expected_reference_T_camera, atol=1e-8)
        )

    def test_hand_eye_solver_requires_multiple_samples(self):
        from ur5e_control.calibration.solvers import solve_hand_eye

        with self.assertRaisesRegex(ValueError, "at least 3"):
            solve_hand_eye([])


if __name__ == "__main__":
    unittest.main()
