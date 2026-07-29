import math
import unittest

import numpy as np


class CalibrationTransformTest(unittest.TestCase):
    def test_ur_pose_vector_round_trips_through_matrix(self):
        from ur5e_control.calibration.transforms import (
            matrix_from_ur_pose_vector,
            ur_pose_vector_from_matrix,
        )

        pose = (0.1, -0.2, 0.3, 0.2, -0.1, 0.4)

        matrix = matrix_from_ur_pose_vector(pose)
        recovered = ur_pose_vector_from_matrix(matrix)

        self.assertTrue(np.allclose(recovered, pose, atol=1e-6))

    def test_transform_inverse_composes_to_identity(self):
        from ur5e_control.calibration.transforms import (
            compose_transforms,
            invert_transform,
            matrix_from_translation_quaternion,
        )

        angle = math.pi / 3.0
        transform = matrix_from_translation_quaternion(
            translation=(0.4, -0.2, 0.7),
            quaternion_xyzw=(0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)),
        )

        identity = compose_transforms(transform, invert_transform(transform))

        self.assertTrue(np.allclose(identity, np.eye(4), atol=1e-8))

    def test_average_transforms_averages_translation_and_rotation(self):
        from ur5e_control.calibration.transforms import (
            average_transforms,
            matrix_from_translation_quaternion,
            translation_quaternion_from_matrix,
        )

        first = matrix_from_translation_quaternion(
            translation=(1.0, 2.0, 3.0),
            quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        )
        second = matrix_from_translation_quaternion(
            translation=(3.0, 4.0, 5.0),
            quaternion_xyzw=(0.0, 0.0, 0.0, -1.0),
        )

        averaged = average_transforms([first, second])
        translation, quaternion = translation_quaternion_from_matrix(averaged)

        self.assertTrue(np.allclose(translation, (2.0, 3.0, 4.0), atol=1e-8))
        self.assertTrue(np.allclose(abs(quaternion[3]), 1.0, atol=1e-8))


if __name__ == "__main__":
    unittest.main()
