import unittest

import numpy as np


class FakeConnection:
    def __init__(self, responses):
        self.responses = list(responses)
        self.sent = []
        self.closed = False

    def recv(self):
        return self.responses.pop(0)

    def send(self, payload):
        self.sent.append(payload)

    def close(self):
        self.closed = True


class VLAWebSocketClientTest(unittest.TestCase):
    def test_reads_metadata_and_sends_cloudrobo_infer_command(self):
        from ur5e_control.inference.protocol import packb, unpackb
        from ur5e_control.inference.vla_client import VLAClient

        connection = FakeConnection(
            [
                packb({"target_action_space": "abs", "action_horizon": 8}),
                packb(
                    {
                        "ok": True,
                        "request_id": "req-1",
                        "payload": {
                            "actions": np.array(
                                [[[0, 1, 2, 3, 4, 5, 6]]],
                                dtype=np.float32,
                            )
                        },
                    }
                ),
            ]
        )
        captured = {}

        def fake_factory(server_url, api_key, timeout_s):
            captured["server_url"] = server_url
            captured["api_key"] = api_key
            captured["timeout_s"] = timeout_s
            return connection

        client = VLAClient(
            server_url="ws://127.0.0.1:8000",
            api_key="secret",
            timeout_s=3.0,
            connection_factory=fake_factory,
        )

        response = client.infer(
            {
                "repo_id": "repo",
                "observation.state": np.array([0, 1, 2, 3, 4, 5, 6]),
            }
        )

        self.assertEqual(
            client.get_server_metadata(),
            {"target_action_space": "abs", "action_horizon": 8},
        )
        self.assertEqual(captured["server_url"], "ws://127.0.0.1:8000")
        self.assertEqual(captured["api_key"], "secret")
        self.assertEqual(captured["timeout_s"], 3.0)
        self.assertIn("actions", response)
        sent = unpackb(connection.sent[0])
        self.assertEqual(sent["__cloudrobovla_command__"], "infer")
        self.assertEqual(sent["obs"]["repo_id"], "repo")
        np.testing.assert_allclose(
            sent["obs"]["observation.state"],
            np.array([0, 1, 2, 3, 4, 5, 6]),
        )

    def test_server_error_response_raises_runtime_error(self):
        from ur5e_control.inference.protocol import packb
        from ur5e_control.inference.vla_client import VLAClient

        connection = FakeConnection(
            [
                packb({}),
                packb(
                    {
                        "ok": False,
                        "request_id": "req-1",
                        "error": {"message": "bad observation"},
                    }
                ),
            ]
        )

        client = VLAClient(
            server_url="ws://127.0.0.1:8000",
            connection_factory=lambda *_: connection,
        )

        with self.assertRaisesRegex(RuntimeError, "bad observation"):
            client.infer({"repo_id": "repo"})

    def test_close_closes_underlying_connection(self):
        from ur5e_control.inference.protocol import packb
        from ur5e_control.inference.vla_client import VLAClient

        connection = FakeConnection([packb({})])
        client = VLAClient(
            server_url="ws://127.0.0.1:8000",
            connection_factory=lambda *_: connection,
        )

        client.close()

        self.assertTrue(connection.closed)


if __name__ == "__main__":
    unittest.main()
