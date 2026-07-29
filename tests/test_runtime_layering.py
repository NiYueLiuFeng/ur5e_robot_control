import importlib
import unittest


class FakeVLAConnection:
    def __init__(self, responses):
        self.responses = list(responses)
        self.sent = []

    def recv(self):
        return self.responses.pop(0)

    def send(self, payload):
        self.sent.append(payload)


class RuntimeLayeringTest(unittest.TestCase):
    def test_action_dimension_validation_accepts_expected_dimension(self):
        from ur5e_control.runtime.safety import validate_action_dimension
        from ur5e_control.runtime.types import RobotAction

        action = RobotAction(values=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1000.0))

        self.assertIs(validate_action_dimension(action, expected_dim=7), action)

    def test_action_dimension_validation_rejects_wrong_dimension(self):
        from ur5e_control.runtime.safety import validate_action_dimension
        from ur5e_control.runtime.types import RobotAction

        action = RobotAction(values=(0.0, 1.0))

        with self.assertRaisesRegex(ValueError, "expected 7 values"):
            validate_action_dimension(action, expected_dim=7)

    def test_task_loader_reads_pen_pick_place_config(self):
        from ur5e_control.data_collection.tasks import load_task_file

        task = load_task_file("configs/tasks/pen_pick_place.yaml")

        self.assertEqual(
            task.name, "pick up the pen from the desk and place it into the pen holder"
        )
        self.assertEqual(task.task_type, "long_horizon_task")
        self.assertEqual(
            task.subtasks,
            (
                "pick up the pen from the desk",
                "place the pen into the pen holder",
            ),
        )

    def test_action_adapter_parses_json_response_and_validates_dimension(self):
        from ur5e_control.inference.action_adapter import ActionAdapter

        adapter = ActionAdapter(expected_dim=7)
        response = {"action": [0.0, -1.0, 1.0, -2.0, 2.0, 0.5, 800.0]}

        action = adapter.from_response(response)

        self.assertEqual(action.values, (0.0, -1.0, 1.0, -2.0, 2.0, 0.5, 800.0))

    def test_vla_client_sends_cloudrobo_infer_payload(self):
        import numpy as np

        from ur5e_control.inference.protocol import packb, unpackb
        from ur5e_control.inference.vla_client import VLAClient

        connection = FakeVLAConnection(
            [
                packb({"target_action_space": "abs"}),
                packb({"ok": True, "payload": {"actions": [[0, 1, 2, 3, 4, 5, 6]]}}),
            ]
        )

        client = VLAClient(
            server_url="ws://vla-server.example:8000",
            connection_factory=lambda *_: connection,
        )

        response = client.infer(
            {
                "repo_id": "repo",
                "task": "pick up the pen",
                "observation.state": np.array([0, 1, 2, 3, 4, 5, 6]),
            }
        )

        sent = unpackb(connection.sent[0])
        self.assertEqual(response["actions"], [[0, 1, 2, 3, 4, 5, 6]])
        self.assertEqual(sent["__cloudrobovla_command__"], "infer")
        self.assertEqual(sent["obs"]["task"], "pick up the pen")

    def test_app_modules_import_without_starting_hardware(self):
        self.assertTrue(importlib.import_module("apps.record_teleop"))
        self.assertTrue(importlib.import_module("apps.run_vla_inference"))
        self.assertTrue(importlib.import_module("apps.calibrate_safety_zone"))
        self.assertTrue(importlib.import_module("apps.check_cameras"))
        self.assertTrue(importlib.import_module("apps.calibrate_cameras"))


if __name__ == "__main__":
    unittest.main()
