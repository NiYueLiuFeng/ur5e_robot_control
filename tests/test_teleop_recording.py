import unittest


class TeleopRecordingTest(unittest.TestCase):
    def test_infer_image_shapes_from_env_obs_uses_current_camera_keys(self):
        from ur5e_control.data_collection.teleop_recording import (
            infer_image_shapes_from_env_obs,
        )

        env_obs = {
            "joint_positions": [0, 1, 2, 3, 4, 5, 6],
            "wrist_rgb": FakeImage(shape=(480, 640, 3)),
            "third_left_top_rgb": FakeImage(shape=(480, 640, 3)),
            "wrist_depth": FakeImage(shape=(480, 640)),
            "third_left_top_depth": FakeImage(shape=(480, 640, 1)),
        }

        self.assertEqual(
            infer_image_shapes_from_env_obs(env_obs),
            {
                "wrist": (480, 640, 3),
                "third_person": (480, 640, 3),
                "wrist_depth": (480, 640, 1),
                "third_person_depth": (480, 640, 1),
            },
        )

    def test_record_lerobot_step_maps_env_obs_action_and_subtask(self):
        from ur5e_control.data_collection.teleop_recording import record_lerobot_step
        from ur5e_control.runtime.types import TaskInfo

        recorder = FakeRecorder()
        task = TaskInfo(
            name="pick up the pen from the desk and place it into the pen holder",
            subtasks=("pick up the pen from the desk", "place the pen into the pen holder"),
        )
        env_obs = {
            "joint_positions": [0, 1, 2, 3, 4, 5, 6],
            "gripper_position": [6],
            "tcp_force": [0] * 6,
            "tcp_pose": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "wrist_rgb": "wrist-frame",
        }

        record_lerobot_step(
            recorder=recorder,
            env_obs=env_obs,
            action_values=[10, 11, 12, 13, 14, 15, 16],
            task=task,
            subtask="place the pen into the pen holder",
            timestamp=0.5,
        )

        self.assertEqual(recorder.calls[0][0].joint_positions, (0, 1, 2, 3, 4, 5, 6))
        self.assertEqual(
            recorder.calls[0][0].tcp_pose,
            (0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        )
        self.assertEqual(recorder.calls[0][0].images, {"wrist": "wrist-frame"})
        self.assertEqual(recorder.calls[0][1].values, (10, 11, 12, 13, 14, 15, 16))
        self.assertEqual(recorder.calls[0][2], "place the pen into the pen holder")

    def test_select_subtask_switches_between_current_task_segments(self):
        from ur5e_control.data_collection.teleop_recording import select_subtask

        subtasks = ("pick up the pen from the desk", "place the pen into the pen holder")

        self.assertEqual(select_subtask(subtasks, second_subtask_active=False), subtasks[0])
        self.assertEqual(select_subtask(subtasks, second_subtask_active=True), subtasks[1])


class FakeImage:
    def __init__(self, shape):
        self.shape = shape


class FakeRecorder:
    def __init__(self):
        self.calls = []

    def add_frame(self, observation, action, subtask=None):
        self.calls.append((observation, action, subtask))


if __name__ == "__main__":
    unittest.main()
