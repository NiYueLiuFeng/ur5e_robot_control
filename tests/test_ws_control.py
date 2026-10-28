import base64
import json
import unittest
from types import SimpleNamespace

import numpy as np

try:
    import msgpack_numpy
except ModuleNotFoundError:  # optional dependency in some test environments
    msgpack_numpy = None


class FakeConnection:
    def __init__(self, responses):
        self.responses = list(responses)
        self.sent = []
        self.closed = False

    def send(self, payload):
        self.sent.append(payload)

    def recv(self):
        return self.responses.pop(0)

    def close(self):
        self.closed = True


class FakeEnv:
    def __init__(self, obs):
        self.obs = obs
        self.commands = []

    def get_obs(self):
        return self.obs

    def step(self, action):
        self.commands.append(np.asarray(action, dtype=float))
        return self.obs


def make_vla(**overrides):
    values = {
        "repo_id": "repo",
        "execute_horizon": 2,
        "threshold": 0.7,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class WSPolicyControlTest(unittest.TestCase):
    def test_build_request_encodes_state_and_images(self):
        from apps.run_ws_control import get_model_body

        obs = {
            "joint_positions": np.array([0, 1, 2, 3, 4, 5, 1], dtype=float),
            "third_left_top_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
            "wrist_rgb": np.full((8, 8, 3), 255, dtype=np.uint8),
        }

        request = get_model_body(obs, model="json", repo_id="repo", task="pick the pen")

        self.assertEqual(request["repo_id"], "repo")
        self.assertEqual(request["task"], "pick the pen")
        self.assertEqual(request["state"], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0])
        self.assertEqual(set(request["images"]), {"third_person", "wrist"})
        # JPEG frames start with the SOI marker after base64 decoding.
        self.assertTrue(
            base64.b64decode(request["images"]["wrist"]).startswith(b"\xff\xd8")
        )

    def test_parse_actions_accepts_action_chunk(self):
        from apps.run_ws_control import _parse_actions

        actions = _parse_actions(
            {"action": [[0, 1, 2, 3, 4, 5, 1], [1, 2, 3, 4, 5, 6, 0]]},
            model="json",
        )

        self.assertEqual(actions.shape, (2, 7))

    def test_parse_actions_rejects_wrong_dimension(self):
        from apps.run_ws_control import _parse_actions

        with self.assertRaisesRegex(ValueError, "7-value actions"):
            _parse_actions({"action": [[0, 1, 2]]}, model="json")

    def test_request_sends_json_and_reads_action(self):
        from apps.run_ws_control import _request

        connection = FakeConnection([json.dumps({"action": [[0] * 7]})])

        response = _request(connection, {"state": [0] * 7}, model="json")

        self.assertEqual(json.loads(connection.sent[0]), {"state": [0] * 7})
        self.assertEqual(response, {"action": [[0] * 7]})

    def test_run_executes_returned_action_chunk(self):
        from apps.run_ws_control import _run

        connection = FakeConnection(
            [json.dumps({"action": [[0.1] * 6 + [0.9], [0.2] * 6 + [0.1]]})]
        )
        env = FakeEnv(
            {
                "joint_positions": np.zeros(7),
                "third_left_top_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
                "wrist_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
            }
        )

        executed = _run(
            env,
            connection,
            model="json",
            vla=make_vla(),
            task="pick",
            max_steps=2,
            control_hz=100.0,
        )

        self.assertEqual(executed, 2)
        self.assertEqual(len(env.commands), 2)
        self.assertEqual(env.commands[0][-1], 1000.0)
        self.assertEqual(env.commands[1][-1], 0.0)


class FastWAMProtocolTest(unittest.TestCase):
    def test_build_request_uses_fixed_state_and_images(self):
        from apps.run_ws_control import get_model_body

        obs = {
            "joint_positions": np.arange(7, dtype=float),
            "third_left_top_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
            "wrist_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        }

        frame = get_model_body(
            obs,
            model="fastwam",
            task="plug the charger",
            request_id=3,
        )

        self.assertEqual(frame["v"], 1)
        self.assertEqual(frame["type"], "infer")
        self.assertEqual(frame["id"], "req-3")
        self.assertEqual(frame["state"], [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(
            [image["key"] for image in frame["images"]], ["front", "wrist"]
        )
        self.assertTrue(frame["images"][0]["data"].startswith(b"\xff\xd8"))

    def test_parse_actions_does_not_assume_a_dimension(self):
        from apps.run_ws_control import _parse_actions

        actions = _parse_actions(
            {"type": "actions", "actions": [[0.0] * 5, [1.0] * 5]},
            model="fastwam",
        )

        self.assertEqual(actions.shape, (2, 5))

    def test_token_is_appended_to_url(self):
        from apps.run_ws_control import _url_with_token

        self.assertEqual(
            _url_with_token("ws://host:8765/v1/ws", "secret"),
            "ws://host:8765/v1/ws?token=secret",
        )
        self.assertEqual(
            _url_with_token("ws://host:8765/v1/ws?x=1", "a b"),
            "ws://host:8765/v1/ws?x=1&token=a%20b",
        )
        self.assertEqual(_url_with_token("ws://host/v1/ws", None), "ws://host/v1/ws")

    def test_first_frame_is_infer_without_handshake(self):
        from apps.run_ws_control import _pack, _request, _unpack

        connection = FakeConnection(
            [_pack({"type": "actions", "actions": [[0.1] * 5]})]
        )
        body = {
            "v": 1,
            "type": "infer",
            "id": "req-0",
            "prompt": "plug",
            "images": [],
            "state": [0.0] * 7,
        }

        response = _request(connection, body, model="fastwam")

        self.assertEqual(response["type"], "actions")
        self.assertEqual(len(connection.sent), 1)
        self.assertEqual(_unpack(connection.sent[0])["type"], "infer")

    def test_run_parses_actions_without_commanding_robot(self):
        from apps.run_ws_control import _pack, _run

        connection = FakeConnection(
            [_pack({"type": "actions", "actions": [[0.1] * 5]})]
        )
        env = FakeEnv(
            {
                "joint_positions": np.zeros(7),
                "third_left_top_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
                "wrist_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
            }
        )

        executed = _run(
            env,
            connection,
            model="fastwam",
            vla=make_vla(execute_horizon=1),
            task="plug",
            max_steps=1,
            control_hz=1000.0,
        )

        self.assertEqual(executed, 1)
        self.assertEqual(env.commands, [])


class OpenPI05ProtocolTest(unittest.TestCase):
    def test_body_matches_openpi_layout(self):
        from apps.run_ws_control import get_model_body

        obs = {
            "joint_positions": np.arange(7, dtype=float),
            "third_left_top_rgb": np.zeros((8, 6, 3), dtype=np.uint8),
            "wrist_rgb": np.full((8, 6, 3), 255, dtype=np.uint8),
        }

        body = get_model_body(obs, model="openpi05", task="pick the block")

        self.assertEqual(
            set(body["images"]),
            {"cam_high", "cam_left_wrist", "cam_right_wrist"},
        )
        for image in body["images"].values():
            self.assertEqual(image.shape, (3, 8, 6))
            self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(body["state"].shape, (7,))
        self.assertEqual(body["state"].dtype, np.float32)
        np.testing.assert_array_equal(body["state"], np.arange(7))
        self.assertEqual(body["prompt"], "pick the block")

    def test_parse_actions_reads_openpi_chunk(self):
        from apps.run_ws_control import _parse_actions

        actions = _parse_actions(
            {"actions": np.zeros((3, 7), dtype=np.float32)},
            model="openpi05",
        )

        self.assertEqual(actions.shape, (3, 7))

    @unittest.skipIf(msgpack_numpy is None, "msgpack-numpy is not installed")
    def test_request_round_trips_numpy_body(self):
        from apps.run_ws_control import _request

        body = {
            "images": {"cam_high": np.zeros((3, 4, 4), dtype=np.uint8)},
            "state": np.arange(7, dtype=np.float32),
            "prompt": "pick",
        }
        connection = FakeConnection(
            [
                msgpack_numpy.packb(
                    {
                        "actions": np.zeros((2, 7), dtype=np.float32),
                        "state": np.arange(7, dtype=np.float32),
                        "server_timing": {"infer_ms": 1.0},
                    }
                )
            ]
        )

        response = _request(connection, body, model="openpi05")

        sent = msgpack_numpy.unpackb(connection.sent[0])
        np.testing.assert_array_equal(sent["state"], body["state"])
        self.assertEqual(response["actions"].shape, (2, 7))

    @unittest.skipIf(msgpack_numpy is None, "msgpack-numpy is not installed")
    def test_run_parses_actions_without_commanding_robot(self):
        from apps.run_ws_control import _run

        connection = FakeConnection(
            [msgpack_numpy.packb({"actions": np.zeros((1, 7), dtype=np.float32)})]
        )
        env = FakeEnv(
            {
                "joint_positions": np.zeros(7),
                "third_left_top_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
                "wrist_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
            }
        )

        executed = _run(
            env,
            connection,
            model="openpi05",
            vla=make_vla(execute_horizon=1),
            task="pick",
            max_steps=1,
            control_hz=1000.0,
        )

        self.assertEqual(executed, 1)
        self.assertEqual(env.commands, [])


if __name__ == "__main__":
    unittest.main()
