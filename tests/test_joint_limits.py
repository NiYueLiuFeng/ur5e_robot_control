import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class JointLimitsTest(unittest.TestCase):
    def test_joint_limits_clamp_only_arm_joints_and_preserve_gripper(self):
        from ur5e_control.runtime.safety import JointLimits

        limits = JointLimits(
            enabled=True,
            lower=(0.0, -1.0, -2.0, -3.0, -4.0, -5.0),
            upper=(1.0, 1.0, 2.0, 3.0, 4.0, 5.0),
            margin=0.1,
        )

        action, clamped = limits.clamp_action((-1.0, 0.0, 3.0, -4.0, 0.0, 6.0, 800.0))

        self.assertTrue(clamped)
        self.assertEqual(action, (0.1, 0.0, 1.9, -2.9, 0.0, 4.9, 800.0))

    def test_joint_limits_round_trip_yaml(self):
        from ur5e_control.runtime.safety import JointLimits, load_joint_limits

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "limits.yaml"
            limits = JointLimits(
                enabled=True,
                lower=(0, 1, 2, 3, 4, 5),
                upper=(6, 7, 8, 9, 10, 11),
                margin=0.02,
            )

            limits.save(path)
            loaded = load_joint_limits(path)

        self.assertEqual(loaded, limits)


if __name__ == "__main__":
    unittest.main()
