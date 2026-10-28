import unittest

import numpy as np


class RobotEnvCameraDepthTest(unittest.TestCase):
    def test_ur_robot_observations_include_actual_tcp_pose(self):
        from ur5e_control.hardware.robots.ur import URRobot

        robot = object.__new__(URRobot)
        robot._use_gripper = False
        robot.r_inter = FakeRTDEReceive()

        observations = robot.get_observations()

        np.testing.assert_array_equal(
            observations["tcp_pose"],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        )

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
        self.assertEqual(obs["joint_positions"].tolist(), [0, 1, 2, 3, 4, 5, 0])
        np.testing.assert_array_equal(obs["tcp_pose"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

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

    def test_recording_binarizes_feedback_and_actions_without_changing_commands(self):
        from ur5e_control.data_collection.lerobot_v3 import build_lerobot_frame
        from ur5e_control.data_collection.observations import observation_from_env_dict
        from ur5e_control.data_collection.teleop.env import RobotEnv
        from ur5e_control.runtime.types import RobotAction, TaskInfo

        for raw_position in (0, 1, 300, 700, 701, 1000):
            for command in (0, 700, 701, 1000):
                with self.subTest(raw_position=raw_position, command=command):
                    env = RobotEnv(robot=FakeRobot(raw_position))
                    obs = env.get_obs()
                    action = RobotAction(values=(0, 1, 2, 3, 4, 5, command))
                    frame = build_lerobot_frame(
                        observation=observation_from_env_dict(obs),
                        action=action,
                        task=TaskInfo(name="pick up the block"),
                    )

                    binary_position = float(raw_position > 700)
                    self.assertEqual(obs["gripper_position_raw"][0], raw_position)
                    self.assertEqual(obs["gripper_position"][0], binary_position)
                    np.testing.assert_array_equal(
                        frame["observation.state"],
                        [0, 1, 2, 3, 4, 5, binary_position],
                    )
                    np.testing.assert_array_equal(
                        frame["action"], [0, 1, 2, 3, 4, 5, float(command > 700)]
                    )
                    self.assertEqual(action.values[-1], command)
                    self.assertNotIn("gripper_position_raw", frame)


class FakeRobot:
    def __init__(self, gripper_position=6):
        self.gripper_position = gripper_position

    def num_dofs(self):
        return 7

    def command_joint_state(self, joint_state):
        self.last_command = joint_state

    def get_observations(self):
        return {
            "joint_positions": np.array([0, 1, 2, 3, 4, 5, self.gripper_position]),
            "gripper_position": np.array([self.gripper_position]),
            "tcp_pose": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            "tcp_force": np.zeros(6),
        }


class FakeRTDEReceive:
    def getActualQ(self):
        return [0, 1, 2, 3, 4, 5]

    def getActualTCPPose(self):
        return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    def getActualTCPForce(self):
        return [0, 0, 0, 0, 0, 0]


class FakeRGBCamera:
    def read(self):
        return "rgb-frame", None


class FakeRGBDCamera:
    def read(self):
        return "rgb-frame", "depth-frame"


if __name__ == "__main__":
    unittest.main()
