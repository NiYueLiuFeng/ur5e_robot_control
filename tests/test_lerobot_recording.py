import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


class LeRobotRecordingTest(unittest.TestCase):
    def test_observation_from_env_dict_maps_current_gello_camera_keys(self):
        from ur5e_control.data_collection.observations import observation_from_env_dict

        env_obs = {
            "joint_positions": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1000.0],
            "gripper_position": [1000.0],
            "tcp_force": [0.0] * 6,
            "tcp_pose": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "wrist_rgb": "wrist-frame",
            "third_left_top_rgb": "third-person-frame",
            "wrist_depth": np.zeros((480, 640), dtype=np.uint16),
            "third_left_top_depth": np.zeros((480, 640, 1), dtype=np.uint16),
        }

        observation = observation_from_env_dict(env_obs, timestamp=12.5)

        self.assertEqual(
            observation.joint_positions,
            (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1000.0),
        )
        self.assertEqual(observation.gripper_position, (1000.0,))
        self.assertEqual(observation.tcp_pose, (0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
        self.assertEqual(observation.images["wrist"], "wrist-frame")
        self.assertEqual(observation.images["third_person"], "third-person-frame")
        self.assertEqual(observation.images["wrist_depth"].shape, (480, 640, 1))
        self.assertEqual(
            observation.images["third_person_depth"].shape, (480, 640, 1)
        )
        self.assertEqual(observation.timestamp, 12.5)

    def test_infer_image_shapes_normalizes_channels_first_images(self):
        from ur5e_control.data_collection.teleop_recording import (
            infer_image_shapes_from_env_obs,
        )

        channels_first = type("Image", (), {"shape": (3, 480, 640)})()
        channels_last = type("Image", (), {"shape": (720, 1280, 3)})()

        shapes = infer_image_shapes_from_env_obs(
            {"wrist": channels_first, "external": channels_last},
            image_key_map={"wrist": "wrist", "external": "third_person"},
        )

        self.assertEqual(shapes["wrist"], (480, 640, 3))
        self.assertEqual(shapes["third_person"], (720, 1280, 3))

    def test_build_lerobot_frame_contains_state_action_images_task_and_subtask(self):
        from ur5e_control.data_collection.lerobot_v3 import build_lerobot_frame
        from ur5e_control.runtime.types import RobotAction, RobotObservation, TaskInfo

        observation = RobotObservation(
            tcp_force=(0, 0, 0, 0, 0, 0),
            tcp_pose=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
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
        np.testing.assert_allclose(
            frame["observation.tcp_pose"],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        )

    def test_build_lerobot_frame_preserves_binary_gripper_state(self):
        from ur5e_control.data_collection.lerobot_v3 import build_lerobot_frame
        from ur5e_control.runtime.types import RobotAction, RobotObservation, TaskInfo

        frame = build_lerobot_frame(
            observation=RobotObservation(
                tcp_force=(0, 0, 0, 0, 0, 0),
                tcp_pose=(0, 0, 0, 0, 0, 0),
                joint_positions=(0, 1, 2, 3, 4, 5, 1),
            ),
            action=RobotAction(values=(10, 11, 12, 13, 14, 15, 1000)),
            task=TaskInfo(name="pick up the block"),
            gripper_threadhold=700,
        )

        self.assertEqual(frame["observation.state"][-1], 1.0)
        self.assertEqual(frame["action"][-1], 1.0)

    def test_build_lerobot_features_describes_ur5e_d455_dataset(self):
        from ur5e_control.data_collection.lerobot_v3 import build_lerobot_features

        features = build_lerobot_features(
            state_dim=7,
            action_dim=7,
            image_shapes={
                "wrist": (3, 480, 640),
                "third_person": (480, 640, 3),
                "wrist_depth": (1, 480, 640),
                "third_person_depth": (480, 640, 1),
            },
        )

        self.assertEqual(features["observation.state"]["shape"], (7,))
        self.assertEqual(
            features["observation.state"]["names"],
            [
                "shoulder_pan.pos",
                "shoulder_lift.pos",
                "elbow.pos",
                "wrist_1.pos",
                "wrist_2.pos",
                "wrist_3.pos",
                "gripper.pos",
            ],
        )
        self.assertEqual(features["action"]["shape"], (7,))
        self.assertEqual(features["observation.tcp_pose"]["shape"], (6,))
        self.assertEqual(
            features["observation.tcp_pose"]["names"],
            [
                "translation.x",
                "translation.y",
                "translation.z",
                "rotation_vector.x",
                "rotation_vector.y",
                "rotation_vector.z",
            ],
        )
        self.assertEqual(
            features["action"]["names"],
            features["observation.state"]["names"],
        )
        self.assertEqual(
            features["tcp_force"]["names"],
            [
                "force.x",
                "force.y",
                "force.z",
                "torque.x",
                "torque.y",
                "torque.z",
            ],
        )
        self.assertEqual(features["observation.images.wrist"]["dtype"], "video")
        self.assertEqual(
            features["observation.images.wrist"]["shape"],
            (480, 640, 3),
        )
        self.assertEqual(
            features["observation.images.third_person"]["shape"],
            (480, 640, 3),
        )
        self.assertFalse(
            features["observation.images.wrist"]["info"]["is_depth_map"]
        )
        self.assertEqual(
            features["observation.images.wrist_depth"]["shape"],
            (480, 640, 1),
        )
        self.assertTrue(
            features["observation.images.wrist_depth"]["info"]["is_depth_map"]
        )
        self.assertTrue(
            features["observation.images.third_person_depth"]["info"]["is_depth_map"]
        )
        self.assertEqual(features["subtask_index"]["dtype"], "int64")

    def test_recompute_exact_dataset_stats_uses_all_frames(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas is not installed")

        from ur5e_control.data_collection.lerobot_v3 import (
            recompute_exact_dataset_stats,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "meta").mkdir()
            (root / "data" / "chunk-000").mkdir(parents=True)
            (root / "data" / "chunk-001").mkdir(parents=True)

            info = {
                "features": {
                    "index": {"dtype": "int64", "shape": [1], "names": None},
                    "action": {
                        "dtype": "float32",
                        "shape": [2],
                        "names": ["joint.pos", "gripper.pos"],
                    },
                    "observation.images.wrist": {
                        "dtype": "video",
                        "shape": [480, 640, 3],
                        "names": ["height", "width", "channel"],
                    },
                }
            }
            (root / "meta" / "info.json").write_text(json.dumps(info))
            image_stats = {"count": [2], "mean": [[[0.5]], [[0.5]], [[0.5]]]}
            (root / "meta" / "stats.json").write_text(
                json.dumps({"observation.images.wrist": image_stats})
            )

            pd.DataFrame(
                {
                    "index": [0, 1, 2, 3, 4],
                    "action": [[value, value * 2] for value in range(5)],
                }
            ).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
            pd.DataFrame(
                {
                    "index": [100, 101, 102, 103, 104],
                    "action": [[value, value * 2] for value in range(100, 105)],
                }
            ).to_parquet(root / "data" / "chunk-001" / "file-000.parquet")

            stats = recompute_exact_dataset_stats(root)

            self.assertAlmostEqual(stats["index"]["q01"][0], 0.09)
            self.assertAlmostEqual(stats["index"]["q99"][0], 103.91)
            self.assertEqual(stats["index"]["count"], [10])
            self.assertEqual(stats["action"]["count"], [10])
            self.assertEqual(stats["observation.images.wrist"], image_stats)
            self.assertFalse((root / "meta" / "stats.json.tmp").exists())

    def test_lerobot_v3_recorder_saves_camera_calibration_as_frame_features(self):
        from ur5e_control.data_collection.lerobot_v3 import (
            LeRobotV3Recorder,
            LeRobotV3RecorderConfig,
        )
        from ur5e_control.runtime.types import RobotAction, RobotObservation, TaskInfo

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            camera_calibration = {
                "image_features": {
                    "third_left_top": "observation.images.third_person",
                },
                "intrinsics": {
                    "third_left_top": {
                        "image_size": [640, 480],
                        "camera_matrix": [
                            [600.0, 0.0, 320.0],
                            [0.0, 600.0, 240.0],
                            [0.0, 0.0, 1.0],
                        ],
                        "distortion_coefficients": [0.1, 0.0, 0.0, 0.0, 0.0],
                    },
                },
                "extrinsics": {
                    "transforms": {
                        "third_left_top_color_T_depth": {
                            "matrix": [
                                [1.0, 0.0, 0.0, 0.1],
                                [0.0, 1.0, 0.0, 0.2],
                                [0.0, 0.0, 1.0, 0.3],
                                [0.0, 0.0, 0.0, 1.0],
                            ],
                        },
                    },
                },
            }
            dataset_root = root / "dataset"
            fake_dataset = FakeLeRobotDataset(root=dataset_root)
            factory_calls = []

            def fake_factory(**kwargs):
                factory_calls.append(kwargs)
                return fake_dataset

            recorder = LeRobotV3Recorder(
                LeRobotV3RecorderConfig(
                    repo_id="test/calibrated-dataset",
                    root=str(dataset_root),
                    camera_calibration=camera_calibration,
                    recompute_exact_stats=False,
                ),
                dataset_factory=fake_factory,
            )
            recorder.start_episode(TaskInfo(name="calibration feature test"))
            recorder.add_frame(
                RobotObservation(
                    tcp_force=(0, 0, 0, 0, 0, 0),
                    tcp_pose=(0, 0, 0, 0, 0, 0),
                    joint_positions=(0, 1, 2, 3, 4, 5, 0),
                ),
                RobotAction(values=(0, 1, 2, 3, 4, 5, 0)),
            )

            matrix_key = (
                "observation.camera_intrinsics.third_person.camera_matrix"
            )
            distortion_key = (
                "observation.camera_intrinsics.third_person."
                "distortion_coefficients"
            )
            resolution_key = (
                "observation.camera_intrinsics.third_person.resolution"
            )
            extrinsics_key = (
                "observation.camera_extrinsics."
                "third_left_top_color_T_depth.matrix"
            )
            features = factory_calls[0]["features"]
            frame = fake_dataset.frames[0]

            self.assertEqual(features[matrix_key]["shape"], (3, 3))
            self.assertEqual(features[distortion_key]["shape"], (5,))
            self.assertEqual(features[resolution_key]["shape"], (2,))
            self.assertEqual(features[extrinsics_key]["shape"], (4, 4))
            np.testing.assert_array_equal(
                frame[matrix_key],
                [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
            )
            np.testing.assert_allclose(
                frame[distortion_key],
                [0.1, 0.0, 0.0, 0.0, 0.0],
            )
            np.testing.assert_array_equal(frame[resolution_key], [640, 480])
            self.assertEqual(
                frame[extrinsics_key].shape,
                (4, 4),
            )
            self.assertFalse(
                (dataset_root / "meta" / "camera_calibration.json").exists()
            )

    def test_camera_calibration_features_pass_lerobot_stats_aggregation(self):
        try:
            from lerobot.datasets.compute_stats import (
                aggregate_stats,
                compute_episode_stats,
            )
        except ImportError:
            self.skipTest("LeRobot is not installed")

        from ur5e_control.data_collection.lerobot_v3 import (
            build_camera_calibration_features,
            build_camera_calibration_frame,
        )

        calibration = {
            "image_features": {
                "wrist": "observation.images.wrist",
            },
            "intrinsics": {
                "wrist": {
                    "image_size": [640, 480],
                    "camera_matrix": np.eye(3).tolist(),
                    "distortion_coefficients": [0.0] * 5,
                },
            },
            "extrinsics": {"transforms": {}},
        }
        features = build_camera_calibration_features(calibration)
        frame = build_camera_calibration_frame(calibration)
        episode_data = {
            key: np.stack([value, value]) for key, value in frame.items()
        }

        episode_stats = compute_episode_stats(episode_data, features)
        aggregated_stats = aggregate_stats([episode_stats, episode_stats])

        resolution_key = "observation.camera_intrinsics.wrist.resolution"
        self.assertIn(resolution_key, aggregated_stats)
        self.assertEqual(aggregated_stats[resolution_key]["min"].shape, (2,))

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
                gripper_threadhold=10,
                recompute_exact_stats=False,
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
            tcp_force=(0, 0, 0, 0, 0, 0),
            tcp_pose=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
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
        self.assertEqual(fake_dataset.frames[0]["observation.state"][-1], 0.0)
        self.assertEqual(fake_dataset.frames[0]["action"][-1], 1.0)
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
            tcp_force=(0, 0, 0, 0, 0, 0),
            tcp_pose=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
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
            RobotObservation(
                tcp_force=(0, 0, 0, 0, 0, 0),
                tcp_pose=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
                joint_positions=(0, 1, 2, 3, 4, 5, 6),
            ),
            RobotAction(values=(10, 11, 12, 13, 14, 15, 16)),
        )

        with self.assertRaisesRegex(RuntimeError, "writer failed"):
            recorder.save_episode()

        recorder.close()

    def test_require_new_dataset_root_rejects_existing_destination(self):
        from ur5e_control.data_collection.teleop.runner import (
            require_new_dataset_root,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            new_root = root / "new-dataset"
            self.assertEqual(require_new_dataset_root(new_root), new_root)

            existing_root = root / "existing-dataset"
            existing_root.mkdir()
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                require_new_dataset_root(existing_root)

    def test_remaining_frame_delay_matches_declared_fps(self):
        from ur5e_control.data_collection.teleop.runner import remaining_frame_delay

        self.assertAlmostEqual(remaining_frame_delay(0.01, fps=20), 0.04)
        self.assertEqual(remaining_frame_delay(0.10, fps=20), 0.0)
        with self.assertRaisesRegex(ValueError, "positive"):
            remaining_frame_delay(0.0, fps=0)


class FakeLeRobotDataset:
    def __init__(self, root=None):
        self.root = root
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
