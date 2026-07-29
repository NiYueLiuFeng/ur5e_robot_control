import unittest

import numpy as np


class CalibrationIOTest(unittest.TestCase):
    def test_transform_to_mapping_contains_matrix_translation_and_quaternion(self):
        from ur5e_control.calibration.io import transform_to_mapping
        from ur5e_control.calibration.transforms import matrix_from_translation_quaternion
        from ur5e_control.calibration.types import CalibrationTransform

        matrix = matrix_from_translation_quaternion(
            translation=(0.1, 0.2, 0.3),
            quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        )
        transform = CalibrationTransform(
            name="base_T_camera",
            parent_frame="base",
            child_frame="third_front",
            matrix=matrix,
            sample_count=12,
            error=0.001,
        )

        mapping = transform_to_mapping(transform)

        self.assertEqual(mapping["name"], "base_T_camera")
        self.assertEqual(mapping["parent_frame"], "base")
        self.assertEqual(mapping["child_frame"], "third_front")
        self.assertEqual(mapping["translation"], [0.1, 0.2, 0.3])
        self.assertEqual(mapping["quaternion_xyzw"], [0.0, 0.0, 0.0, 1.0])
        self.assertEqual(mapping["sample_count"], 12)
        self.assertTrue(np.allclose(mapping["matrix"], matrix.tolist()))

    def test_intrinsics_to_mapping_serializes_numeric_arrays(self):
        from ur5e_control.calibration.io import intrinsics_to_mapping
        from ur5e_control.calibration.types import CameraIntrinsics

        intrinsics = CameraIntrinsics(
            camera_name="wrist",
            serial_number="333422304095",
            image_size=(640, 480),
            camera_matrix=np.eye(3),
            distortion_coefficients=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            reprojection_error=0.25,
            sample_count=18,
            model="brown_conrady",
        )

        mapping = intrinsics_to_mapping(intrinsics)

        self.assertEqual(mapping["camera_name"], "wrist")
        self.assertEqual(mapping["image_size"], [640, 480])
        self.assertEqual(mapping["camera_matrix"], np.eye(3).tolist())
        self.assertEqual(mapping["distortion_coefficients"], [0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertEqual(mapping["sample_count"], 18)


if __name__ == "__main__":
    unittest.main()
