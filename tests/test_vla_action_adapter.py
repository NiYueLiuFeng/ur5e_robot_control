import unittest

import numpy as np


class VLAActionAdapterTest(unittest.TestCase):
    def test_extracts_actions_chunk_from_cloudrobo_response(self):
        from ur5e_control.inference.action_adapter import ActionAdapter

        adapter = ActionAdapter(expected_dim=7)
        response = {
            "actions": np.array(
                [
                    [
                        [0.0, -1.0, 1.0, -2.0, 2.0, 0.5, 800.0],
                        [0.1, -0.9, 1.1, -1.9, 2.1, 0.6, 750.0],
                    ]
                ],
                dtype=np.float32,
            )
        }

        actions = adapter.actions_from_response(
            response,
            current_state=np.array([0.0, -1.0, 1.0, -2.0, 2.0, 0.5, 900.0]),
            action_space="abs",
            execute_horizon=2,
        )

        self.assertEqual(len(actions), 2)
        self.assertEqual(
            actions[0].values,
            (0.0, -1.0, 1.0, -2.0, 2.0, 0.5, 800.0),
        )
        self.assertEqual(
            actions[1].values,
            (
                0.10000000149011612,
                -0.8999999761581421,
                1.100000023841858,
                -1.899999976158142,
                2.0999999046325684,
                0.6000000238418579,
                750.0,
            ),
        )

    def test_delta_actions_are_relative_to_current_arm_joints(self):
        from ur5e_control.inference.action_adapter import ActionAdapter

        adapter = ActionAdapter(expected_dim=7)
        actions = adapter.actions_from_response(
            {"actions": [[0.01, -0.02, 0.03, -0.04, 0.05, -0.06, 600.0]]},
            current_state=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 900.0]),
            action_space="delta",
            execute_horizon=1,
        )

        self.assertEqual(
            actions[0].values,
            (1.01, 1.98, 3.03, 3.96, 5.05, 5.94, 600.0),
        )

    def test_max_joint_delta_limits_large_absolute_jump(self):
        from ur5e_control.inference.action_adapter import ActionAdapter

        adapter = ActionAdapter(expected_dim=7)
        actions = adapter.actions_from_response(
            {"actions": [[1.0, -1.0, 0.2, -0.2, 0.0, 0.6, 1000.0]]},
            current_state=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 500.0]),
            action_space="abs",
            execute_horizon=1,
            max_joint_delta=0.1,
        )

        self.assertEqual(
            actions[0].values,
            (0.1, -0.1, 0.1, -0.1, 0.0, 0.1, 1000.0),
        )

    def test_non_finite_action_is_rejected(self):
        from ur5e_control.inference.action_adapter import ActionAdapter

        adapter = ActionAdapter(expected_dim=7)

        with self.assertRaisesRegex(ValueError, "finite"):
            adapter.actions_from_response({"actions": [[0, 1, 2, 3, 4, 5, np.nan]]})


if __name__ == "__main__":
    unittest.main()
