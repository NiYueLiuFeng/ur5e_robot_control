import unittest

import numpy as np


class RobotEnvCameraDepthTest(unittest.TestCase):
    def test_get_obs_skips_depth_key_when_camera_returns_none(self):
        from ur5e_control.data_collection.teleop.env import RobotEnv

        env = RobotEnv(
            robot=FakeRobot(),
            camera_name_id={"serial-wrist": "wrist"},
            cameras={"serial-wrist": FakeRGBCamera()},
        )

        obs = env.get_obs()

        self.assertEqual(obs["wrist_rgb"], "rgb-frame")
        self.assertNotIn("wrist_depth", obs)
        self.assertEqual(obs["joint_positions"].tolist(), [0, 1, 2, 3, 4, 5, 6])

    def test_get_obs_keeps_depth_key_when_camera_returns_depth(self):
        from ur5e_control.data_collection.teleop.env import RobotEnv

        env = RobotEnv(
            robot=FakeRobot(),
            camera_name_id={"serial-wrist": "wrist"},
            cameras={"serial-wrist": FakeRGBDCamera()},
        )

        obs = env.get_obs()

        self.assertEqual(obs["wrist_rgb"], "rgb-frame")
        self.assertEqual(obs["wrist_depth"], "depth-frame")


class FakeRobot:
    def num_dofs(self):
        return 7

    def command_joint_state(self, joint_state):
        self.last_command = joint_state

    def get_observations(self):
        return {
            "joint_positions": np.array([0, 1, 2, 3, 4, 5, 6]),
            "gripper_position": np.array([6]),
        }


class FakeRGBCamera:
    def read(self):
        return "rgb-frame", None


class FakeRGBDCamera:
    def read(self):
        return "rgb-frame", "depth-frame"


if __name__ == "__main__":
    unittest.main()
