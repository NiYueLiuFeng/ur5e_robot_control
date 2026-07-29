import unittest

import numpy as np


class VLARequestAdapterTest(unittest.TestCase):
    def test_maps_live_env_observation_to_cloudrobo_request(self):
        from ur5e_control.inference.request_adapter import VLARequestAdapter
        from ur5e_control.runtime.types import TaskInfo

        third = np.ones((2, 3, 3), dtype=np.uint8)
        wrist = np.zeros((2, 3, 3), dtype=np.uint8)
        env_obs = {
            "joint_positions": np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 900.0]),
            "gripper_position": np.array([900.0]),
            "third_front_rgb": third,
            "wrist_rgb": wrist,
        }
        task = TaskInfo(
            name="pick up the pen",
            task_type="long_horizon_task",
            subtasks=("approach the pen",),
        )

        adapter = VLARequestAdapter(
            repo_id="chopintelligence/ur5e-pen-pick-place",
            state_key="observation.state",
            image_key_map={
                "third_front_rgb": "observation.images.third_front_image",
                "wrist_rgb": "observation.images.wrist_image",
            },
        )
        request = adapter.from_env_obs(
            env_obs,
            task=task,
            sub_task="approach the pen",
        )

        self.assertEqual(
            request["repo_id"],
            "chopintelligence/ur5e-pen-pick-place",
        )
        self.assertEqual(request["task"], "pick up the pen")
        self.assertEqual(request["sub_task"], "approach the pen")
        np.testing.assert_allclose(
            request["observation.state"],
            np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 900.0], dtype=np.float32),
        )
        self.assertIs(request["observation.images.third_front_image"], third)
        self.assertIs(request["observation.images.wrist_image"], wrist)

    def test_appends_gripper_position_when_joint_positions_are_arm_only(self):
        from ur5e_control.inference.request_adapter import VLARequestAdapter
        from ur5e_control.runtime.types import TaskInfo

        adapter = VLARequestAdapter(
            repo_id="repo",
            image_key_map={},
        )
        request = adapter.from_env_obs(
            {
                "joint_positions": np.array([0, 1, 2, 3, 4, 5], dtype=float),
                "gripper_position": np.array([700], dtype=float),
            },
            task=TaskInfo(name="task"),
        )

        np.testing.assert_allclose(
            request["observation.state"],
            np.array([0, 1, 2, 3, 4, 5, 700], dtype=np.float32),
        )

    def test_missing_configured_image_key_is_rejected(self):
        from ur5e_control.inference.request_adapter import VLARequestAdapter
        from ur5e_control.runtime.types import TaskInfo

        adapter = VLARequestAdapter(
            repo_id="repo",
            image_key_map={"wrist_rgb": "observation.images.wrist_image"},
        )

        with self.assertRaisesRegex(KeyError, "wrist_rgb"):
            adapter.from_env_obs(
                {"joint_positions": np.zeros(7)},
                task=TaskInfo(name="task"),
            )


if __name__ == "__main__":
    unittest.main()
