import unittest


class LeRobotRecordingTest(unittest.TestCase):
    def test_observation_from_env_dict_maps_current_gello_camera_keys(self):
        from ur5e_control.data_collection.observations import observation_from_env_dict

        env_obs = {
            "joint_positions": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1000.0],
            "gripper_position": [1000.0],
            "wrist_rgb": "wrist-frame",
            "third_front_rgb": "third-person-frame",
        }

        observation = observation_from_env_dict(env_obs, timestamp=12.5)

        self.assertEqual(
            observation.joint_positions,
            (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1000.0),
        )
        self.assertEqual(observation.gripper_position, (1000.0,))
        self.assertEqual(
            observation.images,
            {
                "wrist": "wrist-frame",
                "third_person": "third-person-frame",
            },
        )
        self.assertEqual(observation.timestamp, 12.5)

    def test_build_lerobot_frame_contains_state_action_images_task_and_subtask(self):
        from ur5e_control.data_collection.lerobot_v3 import build_lerobot_frame
        from ur5e_control.runtime.types import RobotAction, RobotObservation, TaskInfo

        observation = RobotObservation(
            joint_positions=(0, 1, 2, 3, 4, 5, 6),
            gripper_position=(6,),
            images={"wrist": "wrist-frame", "third_person": "third-person-frame"},
            timestamp=1.25,
        )
        action = RobotAction(values=(10, 11, 12, 13, 14, 15, 16))
        task = TaskInfo(
            name="pick up the pen from the desk and place it into the pen holder",
            subtasks=(
                "pick up the pen from the desk",
                "place the pen into the pen holder",
            ),
        )

        frame = build_lerobot_frame(
            observation=observation,
            action=action,
            task=task,
            subtask="place the pen into the pen holder",
        )

        self.assertEqual(frame["task"], task.name)
        self.assertEqual(frame["subtask"], "place the pen into the pen holder")
        self.assertEqual(frame["subtask_index"], 1)
        self.assertEqual(frame["timestamp"], 1.25)

    def test_build_lerobot_features_describes_ur5e_d455_dataset(self):
        from ur5e_control.data_collection.lerobot_v3 import build_lerobot_features

        features = build_lerobot_features(
            state_dim=7,
            action_dim=7,
            image_shapes={
                "wrist": (480, 640, 3),
                "third_person": (480, 640, 3),
            },
        )

        self.assertEqual(features["observation.state"]["shape"], (7,))
        self.assertEqual(features["action"]["shape"], (7,))
        self.assertEqual(features["observation.images.wrist"]["dtype"], "video")
        self.assertEqual(
            features["observation.images.third_person"]["shape"],
            (480, 640, 3),
        )
        self.assertEqual(features["subtask_index"]["dtype"], "int64")

    def test_lerobot_v3_recorder_writes_frames_and_finalizes_dataset(self):
        from ur5e_control.data_collection.lerobot_v3 import (
            LeRobotV3Recorder,
            LeRobotV3RecorderConfig,
        )
        from ur5e_control.runtime.types import RobotAction, RobotObservation, TaskInfo

        fake_dataset = FakeLeRobotDataset()
        factory_calls = []

        def fake_factory(**kwargs):
            factory_calls.append(kwargs)
            return fake_dataset

        recorder = LeRobotV3Recorder(
            LeRobotV3RecorderConfig(
                repo_id="chopintelligence/ur5e-pen-pick-place",
                root="/data/lerobot/ur5e-pen-pick-place",
                fps=20,
                robot_type="ur5e_pgc140_d455",
            ),
            dataset_factory=fake_factory,
        )
        task = TaskInfo(
            name="pick up the pen from the desk and place it into the pen holder",
            subtasks=(
                "pick up the pen from the desk",
                "place the pen into the pen holder",
            ),
        )
        observation = RobotObservation(
            joint_positions=(0, 1, 2, 3, 4, 5, 6),
            images={"wrist": "wrist-frame"},
            timestamp=2.0,
        )
        action = RobotAction(values=(10, 11, 12, 13, 14, 15, 16))

        recorder.start_episode(task)
        recorder.add_frame(observation, action, subtask="pick up the pen from the desk")
        episode_id = recorder.save_episode()
        recorder.finalize()

        self.assertEqual(
            factory_calls[0]["repo_id"], "chopintelligence/ur5e-pen-pick-place"
        )
        self.assertEqual(factory_calls[0]["root"], "/data/lerobot/ur5e-pen-pick-place")
        self.assertEqual(factory_calls[0]["fps"], 20)
        self.assertEqual(fake_dataset.frames[0]["task"], task.name)
        self.assertEqual(fake_dataset.frames[0]["subtask_index"], 0)
        self.assertEqual(fake_dataset.save_calls, 1)
        self.assertEqual(episode_id, "episode-0")
        self.assertTrue(fake_dataset.finalized)

    def test_async_lerobot_recorder_drains_frames_before_saving(self):
        from ur5e_control.data_collection.lerobot_v3 import AsyncLeRobotV3Recorder
        from ur5e_control.runtime.types import RobotAction, RobotObservation, TaskInfo

        inner = FakeInnerRecorder()
        recorder = AsyncLeRobotV3Recorder(inner, max_queue_size=4)
        task = TaskInfo(name="pick up the pen")
        observation = RobotObservation(
            joint_positions=(0, 1, 2, 3, 4, 5, 6),
            images={"wrist": "wrist-frame"},
        )
        action = RobotAction(values=(10, 11, 12, 13, 14, 15, 16))

        recorder.start_episode(task)
        recorder.add_frame(observation, action, subtask=None)
        recorder.add_frame(observation, action, subtask=None)
        episode_id = recorder.save_episode()
        recorder.finalize()

        self.assertEqual(episode_id, "episode-0")
        self.assertEqual(len(inner.frames), 2)
        self.assertEqual(inner.save_calls, 1)
        self.assertTrue(inner.finalized)

    def test_async_lerobot_recorder_propagates_writer_errors(self):
        from ur5e_control.data_collection.lerobot_v3 import AsyncLeRobotV3Recorder
        from ur5e_control.runtime.types import RobotAction, RobotObservation, TaskInfo

        inner = FailingInnerRecorder()
        recorder = AsyncLeRobotV3Recorder(inner, max_queue_size=4)
        recorder.start_episode(TaskInfo(name="task"))
        recorder.add_frame(
            RobotObservation(joint_positions=(0, 1, 2, 3, 4, 5, 6)),
            RobotAction(values=(10, 11, 12, 13, 14, 15, 16)),
        )

        with self.assertRaisesRegex(RuntimeError, "writer failed"):
            recorder.save_episode()

        recorder.close()


class FakeLeRobotDataset:
    def __init__(self):
        self.frames = []
        self.save_calls = 0
        self.finalized = False

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.save_calls += 1
        return "episode-0"

    def finalize(self):
        self.finalized = True


class FakeInnerRecorder:
    def __init__(self):
        self.frames = []
        self.save_calls = 0
        self.finalized = False

    def start_episode(self, task):
        self.task = task

    def add_frame(self, observation, action, subtask=None):
        self.frames.append((observation, action, subtask))

    def save_episode(self):
        self.save_calls += 1
        return "episode-0"

    def finalize(self):
        self.finalized = True


class FailingInnerRecorder(FakeInnerRecorder):
    def add_frame(self, observation, action, subtask=None):
        raise RuntimeError("writer failed")


if __name__ == "__main__":
    unittest.main()
