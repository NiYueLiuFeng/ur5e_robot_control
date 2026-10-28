import io
import json
import unittest

import cv2
import numpy as np


class _FakeHTTPResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


class FlexPiRequestAdapterTest(unittest.TestCase):
    def test_maps_live_observation_to_flexpi_contract(self):
        from ur5e_control.inference.flexpi_http import FlexPiRequestAdapter

        high = np.zeros((480, 640, 3), dtype=np.uint8)
        wrist = np.ones((480, 640, 3), dtype=np.uint8)
        request = FlexPiRequestAdapter().from_env_obs(
            {
                "joint_positions": np.array([0, 1, 2, 3, 4, 5, 1]),
                "tcp_pose": np.arange(6),
                "third_left_top_rgb": high,
                "wrist_rgb": wrist,
            },
            instruction="pick up the block",
        )

        self.assertEqual(request["depth_mode"], "da3")
        self.assertEqual(set(request["rgb"]), {"cam_high", "cam_left_wrist"})
        np.testing.assert_array_equal(request["state"], [0, 1, 2, 3, 4, 5, 1])
        np.testing.assert_array_equal(request["tcp_pose"], np.arange(6))

    def test_rejects_raw_gripper_units(self):
        from ur5e_control.inference.flexpi_http import FlexPiRequestAdapter

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "normalized"):
            FlexPiRequestAdapter().from_env_obs(
                {
                    "joint_positions": [0, 0, 0, 0, 0, 0, 1000],
                    "tcp_pose": [0, 0, 0, 0, 0, 0],
                    "third_left_top_rgb": image,
                    "wrist_rgb": image,
                },
                instruction="task",
            )


class FlexPiHTTPClientTest(unittest.TestCase):
    def test_shared_websocket_url_is_mapped_to_http_for_flexpi(self):
        from apps.run_flexpi_inference import _flexpi_http_url

        self.assertEqual(
            _flexpi_http_url("ws://127.0.0.1:8000"),
            "http://127.0.0.1:8000",
        )
        self.assertEqual(
            _flexpi_http_url("wss://example.test/inference"),
            "https://example.test/inference",
        )

    def test_posts_png_encoded_rgb_and_validates_actions(self):
        from ur5e_control.inference.flexpi_http import FlexPiHTTPClient

        captured = {}

        def opener(request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            captured["body"] = json.loads(request.data)
            response = {
                "actions": [[0, 0, 0, 0, 0, 0, 1]],
                "action_shape": [1, 7],
            }
            return _FakeHTTPResponse(json.dumps(response).encode())

        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        rgb[:, :, 0] = 255
        result = FlexPiHTTPClient(
            "http://127.0.0.1:8000/", timeout_s=12, opener=opener
        ).infer(
            {
                "instruction": "task",
                "state": np.zeros(7),
                "tcp_pose": np.zeros(6),
                "rgb": {"cam_high": rgb},
                "depth_mode": "da3",
            }
        )

        self.assertEqual(captured["url"], "http://127.0.0.1:8000/predict")
        self.assertEqual(captured["timeout"], 12)
        encoded = captured["body"]["rgb"]["cam_high"]
        decoded = cv2.imdecode(
            np.frombuffer(__import__("base64").b64decode(encoded), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        decoded_rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        np.testing.assert_array_equal(decoded_rgb, rgb)
        self.assertEqual(result["action_shape"], [1, 7])

    def test_timeout_reports_configured_limit(self):
        from ur5e_control.inference.flexpi_http import FlexPiHTTPClient

        def opener(request, timeout):
            raise TimeoutError("timed out")

        client = FlexPiHTTPClient(
            "http://127.0.0.1:8000", timeout_s=600, opener=opener
        )
        with self.assertRaisesRegex(RuntimeError, "timed out after 600s"):
            client.health()


class _FakeEnv:
    def __init__(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.obs = {
            "joint_positions": np.zeros(7),
            "tcp_pose": np.zeros(6),
            "third_left_top_rgb": image,
            "wrist_rgb": image,
        }
        self.commands = []

    def get_obs(self):
        return self.obs

    def step(self, command):
        self.commands.append(np.asarray(command).copy())


class _FakeClient:
    def infer(self, request, **kwargs):
        return {"actions": [[0.5, -0.5, 0, 0, 0, 0, 0.8]]}


class FlexPiInferenceControllerTest(unittest.TestCase):
    def test_moves_to_configured_initial_position_before_inference(self):
        from ur5e_control.inference.flexpi_controller import (
            FlexPiInferenceController,
        )
        from ur5e_control.inference.flexpi_http import FlexPiRequestAdapter

        env = _FakeEnv()
        controller = FlexPiInferenceController(
            env=env,
            client=_FakeClient(),
            request_adapter=FlexPiRequestAdapter(),
            instruction="task",
            init_position=[0.02, -0.01, 0, 0, 0, 0, 1000],
        )

        steps = controller.move_to_init_position(max_step_rad=0.01)

        self.assertEqual(steps, 2)
        self.assertEqual(len(env.commands), 2)
        np.testing.assert_allclose(
            env.commands[-1], [0.02, -0.01, 0, 0, 0, 0, 1000]
        )

    def test_clamps_joint_delta_and_converts_gripper_to_hardware_units(self):
        from ur5e_control.inference.flexpi_controller import (
            FlexPiInferenceController,
        )
        from ur5e_control.inference.flexpi_http import FlexPiRequestAdapter

        env = _FakeEnv()
        controller = FlexPiInferenceController(
            env=env,
            client=_FakeClient(),
            request_adapter=FlexPiRequestAdapter(),
            instruction="task",
        )
        executed = controller.run(max_steps=1, max_joint_delta=0.1)

        self.assertEqual(executed, 1)
        np.testing.assert_allclose(env.commands[0][:6], [0.1, -0.1, 0, 0, 0, 0])
        self.assertEqual(env.commands[0][6], 1000)


if __name__ == "__main__":
    unittest.main()
