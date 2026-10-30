import base64
import json
import unittest
import unittest.mock
from types import SimpleNamespace

import msgpack
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
            [image["key"] for image in frame["images"]], ["third_person", "wrist"]
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

    def test_run_executes_returned_action_chunk(self):
        from apps.run_ws_control import _pack, _run

        connection = FakeConnection(
            [
                _pack(
                    {
                        "type": "actions",
                        "actions": [[0.1] * 6 + [0.9], [0.2] * 6 + [0.1]],
                    }
                )
            ]
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
            vla=make_vla(),
            task="plug",
            max_steps=2,
            control_hz=1000.0,
        )

        self.assertEqual(executed, 2)
        self.assertEqual(len(env.commands), 2)
        # 夹爪按 threshold=0.7 二值化为 1000 / 0。
        self.assertEqual(env.commands[0][-1], 1000.0)
        self.assertEqual(env.commands[1][-1], 0.0)

    def test_run_rejects_action_without_gripper_dimension(self):
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

        with self.assertRaisesRegex(ValueError, "7-dim single-arm UR5 actions"):
            _run(
                env,
                connection,
                model="fastwam",
                vla=make_vla(execute_horizon=1),
                task="plug",
                max_steps=1,
                control_hz=1000.0,
            )
        self.assertEqual(env.commands, [])

    def test_dry_run_warns_on_non_ur5_action_width(self):
        from apps.run_ws_control import _pack, _run

        connection = FakeConnection(
            [_pack({"type": "actions", "actions": [[0.1] * 13]})]
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
            dry_run=True,
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

        body = get_model_body(obs, model="pi05", task="pick the block")

        self.assertEqual(
            set(body),
            {
                "observation.state",
                "observation.images.third_person",
                "observation.images.wrist",
                "prompt",
            },
        )
        for name in (
            "observation.images.third_person",
            "observation.images.wrist",
        ):
            self.assertEqual(body[name].shape, (8, 6, 3))
            self.assertEqual(body[name].dtype, np.uint8)
        self.assertEqual(body["observation.state"].shape, (7,))
        self.assertEqual(body["observation.state"].dtype, np.float32)
        np.testing.assert_array_equal(body["observation.state"], np.arange(7))
        self.assertEqual(body["prompt"], "pick the block")

    def test_openpi05_alias_matches_pi05(self):
        from apps.run_ws_control import _normalize_model, get_model_body

        obs = {
            "joint_positions": np.arange(7, dtype=float),
            "third_left_top_rgb": np.zeros((8, 6, 3), dtype=np.uint8),
            "wrist_rgb": np.zeros((8, 6, 3), dtype=np.uint8),
        }

        self.assertEqual(_normalize_model("openpi05"), "pi05")
        alias = get_model_body(obs, model="openpi05", task="pick")
        canonical = get_model_body(obs, model="pi05", task="pick")
        self.assertEqual(set(alias), set(canonical))

    def test_parse_actions_reads_openpi_chunk(self):
        from apps.run_ws_control import _parse_actions

        actions = _parse_actions(
            {"actions": np.zeros((3, 7), dtype=np.float32)},
            model="pi05",
        )

        self.assertEqual(actions.shape, (3, 7))

    @unittest.skipIf(msgpack_numpy is None, "msgpack-numpy is not installed")
    def test_prepare_server_consumes_metadata(self):
        from apps.run_ws_control import _prepare_server

        connection = FakeConnection(
            [msgpack_numpy.packb({"server": "openpi", "action_dim": 7})]
        )

        _prepare_server(connection, model="pi05", task="pick")

        self.assertEqual(connection.sent, [])
        self.assertEqual(connection.responses, [])

    def test_request_round_trips_ndarray_body(self):
        from apps.run_ws_control import _request

        def server_array(array):
            array = np.ascontiguousarray(array)
            return {
                b"__ndarray__": True,
                b"data": array.tobytes(),
                b"dtype": array.dtype.str,
                b"shape": list(array.shape),
            }

        body = {
            "observation.images.third_person": np.zeros((3, 4, 4), dtype=np.uint8),
            "observation.state": np.arange(7, dtype=np.float32),
            "prompt": "pick",
        }
        connection = FakeConnection(
            [
                msgpack.packb(
                    {
                        "actions": server_array(np.zeros((2, 7), dtype=np.float32)),
                        "server_timing": {"infer_ms": 1.0},
                    },
                    use_bin_type=True,
                )
            ]
        )

        response = _request(connection, body, model="pi05")

        sent = msgpack.unpackb(connection.sent[0], raw=False, strict_map_key=False)
        state = sent["observation.state"]
        self.assertTrue(state[b"__ndarray__"])
        self.assertEqual(state[b"shape"], [7])
        np.testing.assert_array_equal(
            np.frombuffer(state[b"data"], dtype=np.float32), body["observation.state"]
        )
        self.assertEqual(response["actions"].shape, (2, 7))

    @unittest.skipIf(msgpack_numpy is None, "msgpack-numpy is not installed")
    def test_dry_run_prints_actions_without_commanding_robot(self):
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
            model="pi05",
            vla=make_vla(execute_horizon=1),
            task="pick",
            max_steps=1,
            control_hz=1000.0,
            dry_run=True,
        )

        self.assertEqual(executed, 1)
        self.assertEqual(env.commands, [])

    def test_eef_model_refuses_to_drive_the_robot(self):
        from apps.run_ws_control import _ensure_executable

        with self.assertRaisesRegex(ValueError, "absolute 6-joint"):
            _ensure_executable("flexpi", dry_run=False)
        _ensure_executable("flexpi", dry_run=True)
        # pi05 交付的是绝对关节角，可以直接驱动 UR5e。
        _ensure_executable("pi05", dry_run=False)
        _ensure_executable("fastwam", dry_run=False)

    @unittest.skipIf(msgpack_numpy is None, "msgpack-numpy is not installed")
    def test_joint_actions_drive_the_robot(self):
        from apps.run_ws_control import _run

        connection = FakeConnection(
            [
                msgpack_numpy.packb(
                    {
                        "actions": np.array(
                            [[0.1] * 6 + [0.9], [0.2] * 6 + [0.1]],
                            dtype=np.float32,
                        )
                    }
                )
            ]
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
            model="pi05",
            vla=make_vla(),
            task="pick",
            max_steps=2,
            control_hz=1000.0,
        )

        self.assertEqual(executed, 2)
        self.assertEqual(len(env.commands), 2)
        self.assertEqual(env.commands[0][-1], 1000.0)
        self.assertEqual(env.commands[1][-1], 0.0)


class LingBotProtocolTest(unittest.TestCase):
    def setUp(self):
        if msgpack_numpy is None:
            self.skipTest("msgpack-numpy is not installed")
        self.obs = {
            "joint_positions": np.arange(7, dtype=float),
            "third_left_top_rgb": np.zeros((8, 6, 3), dtype=np.uint8),
            "wrist_rgb": np.zeros((8, 6, 3), dtype=np.uint8),
        }

    def test_body_uses_lingbot_camera_keys(self):
        from apps.run_ws_control import get_model_body

        body = get_model_body(self.obs, model="lingbot", task="plug the charger")

        self.assertEqual(set(body), {"obs", "prompt"})
        self.assertEqual(
            set(body["obs"]),
            {"observation.images.third_person", "observation.images.wrist"},
        )
        for image in body["obs"].values():
            self.assertEqual(image.shape, (8, 6, 3))
        self.assertEqual(body["prompt"], "plug the charger")

    def test_prepare_server_consumes_metadata_and_resets(self):
        from apps.run_ws_control import _prepare_server, _unpack_numpy

        connection = FakeConnection(
            [
                msgpack_numpy.packb({"frame_chunk_size": 2}),
                msgpack_numpy.packb({}),
            ]
        )

        _prepare_server(connection, model="lingbot", task="plug")

        self.assertEqual(len(connection.sent), 1)
        self.assertEqual(
            _unpack_numpy(connection.sent[0]),
            {"reset": True, "prompt": "plug"},
        )

    def test_lingbot_request_round_trips_ndarray_body(self):
        from apps.run_ws_control import _request

        def server_array(array):
            array = np.ascontiguousarray(array)
            return {
                b"__ndarray__": True,
                b"data": array.tobytes(),
                b"dtype": array.dtype.str,
                b"shape": list(array.shape),
            }

        body = {
            "obs": {
                "observation.images.third_person": np.zeros(
                    (3, 4, 4), dtype=np.uint8
                ),
                "observation.images.wrist": np.zeros((3, 4, 4), dtype=np.uint8),
            },
            "prompt": "pick",
        }
        raw = np.zeros((7, 2, 2), dtype=np.float32)
        raw[:6, 1, 0] = 0.1
        connection = FakeConnection(
            [
                msgpack.packb(
                    {
                        "action": server_array(raw),
                        "server_timing": {"infer_ms": 1.0},
                    },
                    use_bin_type=True,
                )
            ]
        )

        response = _request(connection, body, model="lingbot")

        sent = msgpack.unpackb(connection.sent[0], raw=False, strict_map_key=False)
        image = sent["obs"]["observation.images.third_person"]
        self.assertTrue(image[b"__ndarray__"])
        self.assertEqual(image[b"shape"], [3, 4, 4])
        self.assertEqual(image[b"dtype"], np.uint8(0).dtype.str)
        self.assertEqual(response["action"].shape, (7, 2, 2))

    def test_prepare_server_accepts_reset_server_timing(self):
        from apps.run_ws_control import _prepare_server

        connection = FakeConnection(
            [
                msgpack_numpy.packb({"frame_chunk_size": 2}),
                msgpack_numpy.packb({"server_timing": {"infer_ms": 1111.98}}),
            ]
        )

        # 该服务端 reset 会带回 infer_ms，不应被当成协议错误。
        _prepare_server(connection, model="lingbot", task="plug")

        self.assertEqual(len(connection.sent), 1)
        self.assertEqual(connection.responses, [])

    def test_prepare_server_rejects_unexpected_reset_payload(self):
        from apps.run_ws_control import _prepare_server

        connection = FakeConnection(
            [
                msgpack_numpy.packb({"frame_chunk_size": 2}),
                msgpack_numpy.packb({"action": np.zeros((7, 2, 2))}),
            ]
        )

        with self.assertRaises(RuntimeError):
            _prepare_server(connection, model="lingbot", task="plug")

    def test_parse_drops_the_conditioning_frame(self):
        from apps.run_ws_control import _parse_actions

        raw = np.zeros((7, 2, 3), dtype=np.float32)
        raw[:, 0, :] = 99.0  # context block produced from zero conditioning
        raw[:, 1, :] = np.arange(3, dtype=np.float32)

        actions = _parse_actions({"action": raw}, model="lingbot")

        self.assertEqual(actions.shape, (3, 7))
        self.assertFalse(np.any(actions == 99.0))
        np.testing.assert_array_equal(actions[0], np.zeros(7))

    def test_run_executes_lingbot_actions(self):
        from apps.run_ws_control import _run

        raw = np.zeros((7, 2, 2), dtype=np.float32)
        raw[:6, 1, 0] = 0.1
        raw[6, 1, 0] = 0.9
        raw[6, 1, 1] = 0.1
        reset_ack = msgpack_numpy.packb({"server_timing": {"infer_ms": 1.0}})
        connection = FakeConnection(
            [
                reset_ack,
                msgpack_numpy.packb({"action": raw}),
                reset_ack,
                msgpack_numpy.packb({"action": raw}),
            ]
        )
        env = FakeEnv(dict(self.obs))

        executed = _run(
            env,
            connection,
            model="lingbot",
            vla=make_vla(execute_horizon=2),
            task="plug",
            max_steps=4,
            control_hz=1000.0,
        )

        self.assertEqual(executed, 4)
        self.assertEqual(len(env.commands), 4)
        self.assertEqual(env.commands[0][-1], 1000.0)
        self.assertEqual(env.commands[1][-1], 0.0)
        # 每块观测请求前都要先 reset，清空服务端 streaming VAE 的缓存。
        self.assertEqual(len(connection.sent), 4)
        from apps.run_ws_control import _unpack_ndarray

        self.assertEqual(
            _unpack_ndarray(connection.sent[0]),
            {"reset": True, "prompt": "plug"},
        )
        self.assertEqual(
            _unpack_ndarray(connection.sent[2]),
            {"reset": True, "prompt": "plug"},
        )


class OpenWAMProtocolTest(unittest.TestCase):
    def setUp(self):
        self.obs = {
            "joint_positions": np.arange(7, dtype=float),
            "third_left_top_rgb": np.zeros((8, 6, 3), dtype=np.uint8),
            "wrist_rgb": np.full((8, 6, 3), 255, dtype=np.uint8),
        }

    def test_body_uses_openwam_image_slots(self):
        from apps.run_ws_control import get_model_body

        body = get_model_body(self.obs, model="openwam", task="pick the pen")

        self.assertEqual(body["type"], "obs")
        self.assertEqual(
            set(body["images"]),
            {"head_camera", "left_wrist_camera", "right_wrist_camera"},
        )
        self.assertIsNone(body["images"]["right_wrist_camera"])
        self.assertTrue(
            base64.b64decode(body["images"]["head_camera"]).startswith(b"\x89PNG")
        )
        self.assertEqual(len(body["state"]), 7)
        self.assertEqual(body["prompt"], "pick the pen")

    def test_prepare_server_pings_and_resets(self):
        from apps.run_ws_control import _prepare_server

        connection = FakeConnection(
            [
                json.dumps(
                    {
                        "type": "pong",
                        "representation": "joint",
                        "gripper_convention": "0_closed_1_open",
                    }
                ),
                json.dumps({"type": "reset_ack"}),
            ]
        )

        _prepare_server(connection, model="openwam", task="pick")

        self.assertEqual(
            [json.loads(frame) for frame in connection.sent],
            [{"type": "ping"}, {"type": "reset"}],
        )

    def test_prepare_server_rejects_eef_representation(self):
        from apps.run_ws_control import _prepare_server

        connection = FakeConnection(
            [json.dumps({"type": "pong", "representation": "eef"})]
        )

        with self.assertRaisesRegex(RuntimeError, "joint"):
            _prepare_server(connection, model="openwam", task="pick")

    def test_request_raises_on_error_frame(self):
        from apps.run_ws_control import _request

        connection = FakeConnection(
            [json.dumps({"type": "error", "code": "NO_ACTION", "message": "warmup"})]
        )

        with self.assertRaisesRegex(RuntimeError, "NO_ACTION"):
            _request(connection, {"type": "obs"}, model="openwam")

    def test_run_executes_returned_action(self):
        from apps.run_ws_control import _run

        connection = FakeConnection(
            [
                json.dumps(
                    {
                        "type": "action",
                        "action": [[0.1] * 6 + [0.9], [0.2] * 6 + [0.1]],
                    }
                )
            ]
        )
        env = FakeEnv(dict(self.obs))

        executed = _run(
            env,
            connection,
            model="openwam",
            vla=make_vla(execute_horizon=2),
            task="pick",
            max_steps=2,
            control_hz=1000.0,
        )

        self.assertEqual(executed, 2)
        self.assertEqual([command[-1] for command in env.commands], [1000.0, 0.0])

    def test_body_requests_chunk_only_when_enabled(self):
        from apps.run_ws_control import get_model_body

        chunked = get_model_body(
            self.obs, model="openwam", task="pick", openwam_chunk=True
        )
        stepped = get_model_body(
            self.obs, model="openwam", task="pick", openwam_chunk=False
        )

        self.assertIs(chunked["chunk"], True)
        self.assertNotIn("chunk", stepped)
        # 两种模式下 obs 的其余字段完全一致。
        self.assertEqual(
            {k: v for k, v in chunked.items() if k != "chunk"}, stepped
        )

    def test_parse_actions_accepts_actions_frame(self):
        from apps.run_ws_control import _parse_actions

        actions = _parse_actions(
            {
                "type": "actions",
                "actions": [[0.1] * 6 + [0.9], [0.2] * 6 + [0.1]],
                "count": 2,
            },
            model="openwam",
        )

        self.assertEqual(actions.shape, (2, 7))
        self.assertAlmostEqual(actions[1, 6], 0.1)

    def test_parse_actions_rejects_chunk_count_mismatch(self):
        from apps.run_ws_control import _parse_actions

        with self.assertRaisesRegex(ValueError, "count=3"):
            _parse_actions(
                {"type": "actions", "actions": [[0.1] * 7], "count": 3},
                model="openwam",
            )

    def test_request_accepts_actions_frame(self):
        from apps.run_ws_control import _request

        frame = {"type": "actions", "actions": [[0.1] * 7], "count": 1}
        connection = FakeConnection([json.dumps(frame)])

        self.assertEqual(_request(connection, {"type": "obs"}, model="openwam"), frame)

    def test_run_executes_returned_chunk(self):
        from apps.run_ws_control import _run

        chunk = [[0.1] * 6 + [0.9], [0.2] * 6 + [0.1], [0.3] * 6 + [0.9]]
        connection = FakeConnection(
            [json.dumps({"type": "actions", "actions": chunk, "count": 3})]
        )
        env = FakeEnv(dict(self.obs))

        executed = _run(
            env,
            connection,
            model="openwam",
            vla=make_vla(execute_horizon=3),
            task="pick",
            max_steps=3,
            control_hz=1000.0,
        )

        # 一次 obs 换回整块动作，三条都执行，夹爪按 threshold 二值化。
        self.assertEqual(executed, 3)
        self.assertEqual([command[-1] for command in env.commands], [1000.0, 0.0, 1000.0])
        self.assertIs(json.loads(connection.sent[0])["chunk"], True)
        self.assertEqual(len(connection.sent), 1)

    def test_run_can_disable_chunk_requests(self):
        from apps.run_ws_control import _run

        connection = FakeConnection(
            [json.dumps({"type": "action", "action": [0.1] * 6 + [0.9]})]
        )
        env = FakeEnv(dict(self.obs))

        executed = _run(
            env,
            connection,
            model="openwam",
            vla=make_vla(execute_horizon=1),
            task="pick",
            max_steps=1,
            control_hz=1000.0,
            openwam_chunk=False,
        )

        self.assertEqual(executed, 1)
        self.assertNotIn("chunk", json.loads(connection.sent[0]))


class RobotCommandTest(unittest.TestCase):
    def test_command_binarizes_gripper(self):
        from apps.run_ws_control import _robot_command

        command = _robot_command(np.array([0.0] * 6 + [0.9]), 0.7)

        self.assertEqual(command.shape, (7,))
        self.assertEqual(command[-1], 1000.0)

    def test_command_clamps_joint_delta(self):
        from apps.run_ws_control import _robot_command

        action = np.array([1.0] * 6 + [0.0])
        command = _robot_command(
            action,
            0.7,
            reference=np.zeros(6),
            max_joint_delta=0.1,
        )

        np.testing.assert_allclose(command[:6], np.full(6, 0.1))

    def test_command_rejects_non_finite_action(self):
        from apps.run_ws_control import _robot_command

        with self.assertRaisesRegex(ValueError, "finite"):
            _robot_command(np.array([np.nan] * 6 + [0.0]), 0.7)

    def test_command_rejects_eef_pose_as_joints(self):
        from apps.run_ws_control import _robot_command

        # A 6-D pose plus gripper would otherwise be sent straight to servoJ.
        pose = np.array([0.5, -0.2, 0.3, 3.1, -0.2, 0.1, 0.0])
        with self.assertRaisesRegex(ValueError, "joint command"):
            _robot_command(pose * 3.0, 0.7)


class ResetKeyTest(unittest.TestCase):
    def test_take_is_one_shot(self):
        from apps.run_ws_control import ResetKeyListener

        listener = ResetKeyListener()
        self.assertFalse(listener.take())
        listener._requested.set()
        self.assertTrue(listener.take())
        self.assertFalse(listener.take())
        listener.stop()  # no listener thread was started in this test

    def test_reset_keys_cover_uppercase_input(self):
        from apps.run_ws_control import RESET_KEYS

        # pynput reports 'N'/'T' when shift is held; the handler lowercases.
        self.assertEqual(
            [char.lower() in RESET_KEYS for char in ("N", "T", "n", "t")],
            [True, True, True, True],
        )

    @unittest.mock.patch("apps.run_ws_control.ResetKeyListener")
    def test_reset_request_returns_to_start_pose(self, listener_cls):
        from apps.run_ws_control import _run

        class FakeKeys:
            def __init__(self):
                self.calls = 0

            def start(self):
                pass

            def stop(self):
                pass

            def take(self):
                # 第一轮循环开始时就请求一次复位。
                self.calls += 1
                return self.calls == 1

        listener_cls.side_effect = lambda: FakeKeys()
        connection = FakeConnection([json.dumps({"action": [[0.1] * 6 + [0.0]]})])
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
            vla=make_vla(execute_horizon=1),
            task="pick",
            max_steps=1,
            control_hz=1000.0,
            reset_joints=np.full(7, 0.5),
        )

        self.assertEqual(executed, 1)
        # 100 步复位插值 + 1 条策略动作。
        self.assertEqual(len(env.commands), 101)
        np.testing.assert_allclose(env.commands[0][:6], np.zeros(6))
        np.testing.assert_allclose(env.commands[99][:6], np.full(6, 0.5))
        np.testing.assert_allclose(env.commands[-1][:6], np.full(6, 0.1))

    def test_reset_is_skipped_without_reset_joints(self):
        from apps.run_ws_control import _run

        connection = FakeConnection([json.dumps({"action": [[0.1] * 6 + [0.0]]})])
        env = FakeEnv(
            {
                "joint_positions": np.zeros(7),
                "third_left_top_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
                "wrist_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
            }
        )

        # 没有 reset_joints 时不启动监听（测试环境里也没有 pynput 显示器）。
        executed = _run(
            env,
            connection,
            model="json",
            vla=make_vla(execute_horizon=1),
            task="pick",
            max_steps=1,
            control_hz=1000.0,
        )

        self.assertEqual(executed, 1)
        self.assertEqual(len(env.commands), 1)


class FakeVideo:
    def __init__(self, frames):
        self.has_frames = frames
        self.frames = []
        self.saved = 0
        self.discarded = 0

    def add(self, obs):
        self.frames.append(obs)

    def save(self):
        self.saved += 1

    def discard(self):
        self.discarded += 1


class SaveVideoTest(unittest.TestCase):
    def test_flag_accepts_bare_and_explicit_boolean(self):
        from apps.run_ws_control import build_parser

        parser = build_parser()
        self.assertTrue(parser.parse_args(["--save-video"]).save_video)
        self.assertTrue(parser.parse_args(["--save-video", "true"]).save_video)
        self.assertFalse(parser.parse_args(["--save-video", "false"]).save_video)
        self.assertFalse(parser.parse_args([]).save_video)

    def test_run_records_a_frame_per_action(self):
        from apps.run_ws_control import _run

        connection = FakeConnection(
            [json.dumps({"action": [[0.1] * 6 + [0.0], [0.2] * 6 + [0.0]]})]
        )
        env = FakeEnv(
            {
                "joint_positions": np.zeros(7),
                "third_left_top_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
                "wrist_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
            }
        )
        video = FakeVideo(frames=[])

        executed = _run(
            env,
            connection,
            model="json",
            vla=make_vla(execute_horizon=2),
            task="pick",
            max_steps=2,
            control_hz=1000.0,
            video=video,
        )

        self.assertEqual(executed, 2)
        self.assertEqual(len(video.frames), 2)

    def test_reset_asks_before_saving(self):
        from apps.run_ws_control import _handle_reset

        env = FakeEnv({"joint_positions": np.zeros(7)})
        kept = FakeVideo(frames=[object()])
        with unittest.mock.patch("builtins.input", return_value="y"):
            _handle_reset(env, np.zeros(7), kept)

        self.assertEqual((kept.saved, kept.discarded), (1, 0))
        # 复位仍然执行：100 步插值。
        self.assertEqual(len(env.commands), 100)

        env = FakeEnv({"joint_positions": np.zeros(7)})
        dropped = FakeVideo(frames=[object()])
        with unittest.mock.patch("builtins.input", return_value="n"):
            _handle_reset(env, np.zeros(7), dropped)

        self.assertEqual((dropped.saved, dropped.discarded), (0, 1))

    def test_reset_without_frames_does_not_prompt(self):
        from apps.run_ws_control import _handle_reset

        env = FakeEnv({"joint_positions": np.zeros(7)})
        empty = FakeVideo(frames=[])

        with unittest.mock.patch("builtins.input") as ask:
            _handle_reset(env, np.zeros(7), empty)

        ask.assert_not_called()
        self.assertEqual((empty.saved, empty.discarded), (0, 0))


if __name__ == "__main__":
    unittest.main()
