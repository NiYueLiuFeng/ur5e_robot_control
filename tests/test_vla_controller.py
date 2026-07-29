import unittest

import numpy as np


class FakeEnv:
    def __init__(self, obs):
        self.obs = obs
        self.commands = []

    def get_obs(self):
        return self.obs

    def step(self, action):
        self.commands.append(np.asarray(action, dtype=float))
        self.obs = dict(self.obs)
        self.obs["joint_positions"] = np.asarray(action, dtype=float)
        return self.obs


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.requests = []
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1
        return {"ok": True}

    def infer(self, request):
        self.requests.append(request)
        return self.response


class VLAInferenceControllerTest(unittest.TestCase):
    def test_runs_one_infer_step_and_executes_robot_action(self):
        from ur5e_control.inference.action_adapter import ActionAdapter
        from ur5e_control.inference.controller import VLAInferenceController
        from ur5e_control.inference.request_adapter import VLARequestAdapter
        from ur5e_control.runtime.types import TaskInfo

        obs = {
            "joint_positions": np.array([0, 1, 2, 3, 4, 5, 800], dtype=float),
            "gripper_position": np.array([800], dtype=float),
            "third_front_rgb": np.ones((2, 2, 3), dtype=np.uint8),
            "wrist_rgb": np.zeros((2, 2, 3), dtype=np.uint8),
        }
        env = FakeEnv(obs)
        client = FakeClient(
            {
                "actions": np.array(
                    [[[0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 700.0]]],
                    dtype=np.float32,
                )
            }
        )
        controller = VLAInferenceController(
            env=env,
            client=client,
            task=TaskInfo(name="pick up the pen"),
            request_adapter=VLARequestAdapter(
                repo_id="repo",
                image_key_map={
                    "third_front_rgb": "observation.images.third_front_image",
                    "wrist_rgb": "observation.images.wrist_image",
                },
            ),
            action_adapter=ActionAdapter(expected_dim=7),
            action_space="abs",
            execute_horizon=1,
            reset_on_start=True,
        )

        executed_steps = controller.run(init_pos=[1.56, -1.57, 1.58, -1.58, -1.57, -0.0077, 1000.0], max_steps=1)

        self.assertEqual(executed_steps, 1)
        self.assertEqual(client.reset_calls, 1)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(client.requests[0]["repo_id"], "repo")
        self.assertEqual(client.requests[0]["task"], "pick up the pen")
        self.assertIn("observation.images.wrist_image", client.requests[0])

    def test_execute_horizon_runs_multiple_actions_from_one_chunk(self):
        from ur5e_control.inference.action_adapter import ActionAdapter
        from ur5e_control.inference.controller import VLAInferenceController
        from ur5e_control.inference.request_adapter import VLARequestAdapter
        from ur5e_control.runtime.types import TaskInfo

        env = FakeEnv(
            {
                "joint_positions": np.array([0, 0, 0, 0, 0, 0, 500], dtype=float),
                "gripper_position": np.array([500], dtype=float),
            }
        )
        client = FakeClient(
            {
                "actions": np.array(
                    [
                        [
                            [0.01, 0, 0, 0, 0, 0, 500],
                            [0.02, 0, 0, 0, 0, 0, 500],
                        ]
                    ],
                    dtype=np.float32,
                )
            }
        )
        controller = VLAInferenceController(
            env=env,
            client=client,
            task=TaskInfo(name="task"),
            request_adapter=VLARequestAdapter(repo_id="repo", image_key_map={}),
            action_adapter=ActionAdapter(expected_dim=7),
            action_space="abs",
            execute_horizon=2,
            reset_on_start=False,
        )

        executed_steps = controller.run(init_pos=[1.56, -1.57, 1.58, -1.58, -1.57, -0.0077, 1000.0], max_steps=2)

        self.assertEqual(executed_steps, 2)
        self.assertEqual(client.reset_calls, 0)
        self.assertEqual(len(client.requests), 1)


if __name__ == "__main__":
    unittest.main()
