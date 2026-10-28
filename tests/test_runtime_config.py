import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class RuntimeConfigTest(unittest.TestCase):
    def test_default_config_loads_ur5e_hardware_defaults(self):
        from ur5e_control.runtime.config import load_runtime_config

        config = load_runtime_config(environ={})

        self.assertEqual(config.robot.name, "ur5e")
        self.assertEqual(config.robot.ip, "192.168.56.2")
        self.assertEqual(config.robot.zmq_port, 6001)
        self.assertEqual(config.agent.gello_port, "/dev/ttyUSB1")
        self.assertEqual(config.gripper.port, "/dev/ttyUSB0")
        self.assertEqual(config.gripper.position_tolerance, 10)
        self.assertEqual(config.gripper.min_command_interval_s, 0.05)
        self.assertEqual(config.gripper.position_read_interval_s, 0.1)
        self.assertEqual(
            config.camera.realsense_ids["333422304095"],
            "wrist",
        )
        self.assertEqual(config.camera.fps, 30)
        self.assertTrue(config.camera.enable_depth)
        self.assertEqual(config.camera.depth_quality_sample_count, 15)
        self.assertEqual(config.camera.depth_max_invalid_ratio, {})
        self.assertEqual(
            config.camera.depth_options,
            {
                "wrist": {"visual_preset": "high_density"},
                "third_left_top": {"visual_preset": "high_density"},
            },
        )
        self.assertEqual(config.camera.calibration_dir, "configs/cameras")
        self.assertEqual(
            config.camera.intrinsics_files,
            {
                "third_left_top": "configs/cameras/intrinsics_third_left_top.yaml",
                "wrist": "configs/cameras/intrinsics_wrist.yaml",
            },
        )
        self.assertEqual(
            config.camera.extrinsics_file,
            "configs/cameras/extrinsics.yaml",
        )
        self.assertEqual(
            config.dataset.lerobot_root,
            ".datasets/lerobot/pick_place_block_150_train1_v3.0",
        )
        self.assertEqual(config.dataset.lerobot_fps, 15)
        self.assertTrue(config.dataset.lerobot_async_recording)
        self.assertEqual(config.dataset.lerobot_record_queue_size, 64)
        self.assertTrue(config.dataset.lerobot_repair_video_timestamps)
        self.assertEqual(config.dataset.lerobot_depth_min_m, 0.0)
        self.assertEqual(
            config.safety.joint_limits_file,
            "configs/safety/ur5e_joint_limits.yaml",
        )
        self.assertEqual(
            config.task.file,
            "configs/tasks/pick_place_purple_block.yaml",
        )
        self.assertEqual(config.vla.server_url, "ws://127.0.0.1:8000")
        self.assertEqual(config.vla.state_key, "observation.state")
        self.assertEqual(
            config.vla.image_key_map,
            {
                "third_left_top_rgb": "observation.images.third_person",
                "wrist_rgb": "observation.images.wrist",
            },
        )
        self.assertEqual(config.vla.request_timeout_s, 5.0)
        self.assertEqual(config.vla.max_joint_delta, 0.1)
        self.assertTrue(config.vla.reset_on_start)

    def test_environment_overrides_laptop_specific_values(self):
        from ur5e_control.runtime.config import load_runtime_config

        config = load_runtime_config(
            environ={
                "UR5E_GELLO_PORT": "/dev/serial/by-id/gello",
                "UR5E_PGC_PORT": "/dev/serial/by-id/pgc",
                "UR5E_PGC_POSITION_TOLERANCE": "20",
                "UR5E_PGC_MIN_COMMAND_INTERVAL_S": "0.08",
                "UR5E_PGC_POSITION_READ_INTERVAL_S": "0.2",
                "UR5E_ROBOT_IP": "192.168.1.20",
                "UR5E_CAMERA_HOSTNAME": "0.0.0.0",
                "UR5E_CAMERA_FPS": "15",
                "UR5E_CAMERA_ENABLE_DEPTH": "true",
                "UR5E_CAMERA_CALIBRATION_DIR": "/data/calibration",
                "UR5E_CAMERA_EXTRINSICS_FILE": "/data/calibration/extrinsics.yaml",
                "UR5E_AGENT_NAME": "spacemouse",
                "UR5E_SPACEMOUSE_DEVICE_PATH": "/dev/hidraw0",
                "UR5E_SPACEMOUSE_TRANSLATION_SCALE": "0.08",
                "UR5E_SPACEMOUSE_ANGLE_SCALE": "0.3",
                "UR5E_SPACEMOUSE_INVERT_CONTROL": "1,-1,1,-1,1,-1",
                "UR5E_SPACEMOUSE_ROTATION_MODE": "rpy",
                "UR5E_SPACEMOUSE_INVERT_BUTTON": "true",
                "UR5E_SPACEMOUSE_GRIPPER_OPEN": "900",
                "UR5E_SPACEMOUSE_GRIPPER_CLOSED": "100",
                "UR5E_LEROBOT_ROOT": "/data/lerobot",
                "UR5E_LEROBOT_FPS": "15",
                "UR5E_LEROBOT_ASYNC_RECORDING": "false",
                "UR5E_LEROBOT_RECORD_QUEUE_SIZE": "8",
                "UR5E_LEROBOT_REPAIR_VIDEO_TIMESTAMPS": "true",
                "UR5E_JOINT_LIMITS_FILE": "/data/limits/ur5e_joint_limits.yaml",
                "UR5E_EPISODE_COUNT": "7",
                "UR5E_VLA_SERVER_URL": "ws://vla.example:9000",
                "UR5E_VLA_API_KEY": "secret",
                "UR5E_VLA_REPO_ID": "owner/runtime-ur5e",
                "UR5E_VLA_ACTION_SPACE": "delta",
                "UR5E_VLA_EXECUTE_HORIZON": "4",
                "UR5E_VLA_REQUEST_TIMEOUT_S": "8.5",
                "UR5E_VLA_MAX_JOINT_DELTA": "0.04",
                "UR5E_VLA_RESET_ON_START": "false",
            }
        )

        self.assertEqual(config.agent.gello_port, "/dev/serial/by-id/gello")
        self.assertEqual(config.agent.name, "spacemouse")
        self.assertEqual(config.agent.spacemouse.device_path, "/dev/hidraw0")
        self.assertEqual(config.agent.spacemouse.translation_scale, 0.08)
        self.assertEqual(config.agent.spacemouse.angle_scale, 0.3)
        self.assertEqual(
            config.agent.spacemouse.invert_control,
            (1.0, -1.0, 1.0, -1.0, 1.0, -1.0),
        )
        self.assertEqual(config.agent.spacemouse.rotation_mode, "rpy")
        self.assertTrue(config.agent.spacemouse.invert_button)
        self.assertEqual(config.agent.spacemouse.gripper_open, 900)
        self.assertEqual(config.agent.spacemouse.gripper_closed, 100)
        self.assertEqual(config.gripper.port, "/dev/serial/by-id/pgc")
        self.assertEqual(config.gripper.position_tolerance, 20)
        self.assertEqual(config.gripper.min_command_interval_s, 0.08)
        self.assertEqual(config.gripper.position_read_interval_s, 0.2)
        self.assertEqual(config.robot.ip, "192.168.1.20")
        self.assertEqual(config.camera.hostname, "0.0.0.0")
        self.assertEqual(config.camera.fps, 15)
        self.assertTrue(config.camera.enable_depth)
        self.assertEqual(config.camera.calibration_dir, "/data/calibration")
        self.assertEqual(
            config.camera.extrinsics_file,
            "/data/calibration/extrinsics.yaml",
        )
        self.assertEqual(config.dataset.lerobot_root, "/data/lerobot")
        self.assertEqual(config.dataset.lerobot_fps, 15)
        self.assertFalse(config.dataset.lerobot_async_recording)
        self.assertEqual(config.dataset.lerobot_record_queue_size, 8)
        self.assertTrue(config.dataset.lerobot_repair_video_timestamps)
        self.assertEqual(
            config.safety.joint_limits_file,
            "/data/limits/ur5e_joint_limits.yaml",
        )
        self.assertEqual(config.task.episode_count, 7)
        self.assertEqual(config.vla.server_url, "ws://vla.example:9000")
        self.assertEqual(config.vla.api_key, "secret")
        self.assertEqual(config.vla.repo_id, "owner/runtime-ur5e")
        self.assertEqual(config.vla.action_space, "delta")
        self.assertEqual(config.vla.execute_horizon, 4)
        self.assertEqual(config.vla.request_timeout_s, 8.5)
        self.assertEqual(config.vla.max_joint_delta, 0.04)
        self.assertFalse(config.vla.reset_on_start)

    def test_invalid_lerobot_fps_is_rejected(self):
        from dataclasses import replace

        from ur5e_control.runtime.config import DatasetRuntimeConfig

        with self.assertRaisesRegex(ValueError, "dataset.lerobot.fps"):
            replace(DatasetRuntimeConfig(), lerobot_fps=0)

    def test_invalid_lerobot_record_queue_size_is_rejected(self):
        from dataclasses import replace

        from ur5e_control.runtime.config import DatasetRuntimeConfig

        with self.assertRaisesRegex(ValueError, "dataset.lerobot.record_queue_size"):
            replace(DatasetRuntimeConfig(), lerobot_record_queue_size=0)

    def test_invalid_vla_action_space_is_rejected(self):
        from dataclasses import replace

        from ur5e_control.runtime.config import VLARuntimeConfig

        with self.assertRaisesRegex(ValueError, "vla.action_space"):
            replace(VLARuntimeConfig(), action_space="joint_teleport")

    def test_non_ur5e_robot_name_is_rejected(self):
        from dataclasses import replace

        from ur5e_control.runtime.config import RobotRuntimeConfig

        with self.assertRaisesRegex(ValueError, "robot.name"):
            replace(RobotRuntimeConfig(), name="other_robot")

    def test_non_gello_agent_is_rejected(self):
        from dataclasses import replace

        from ur5e_control.runtime.config import AgentRuntimeConfig

        with self.assertRaisesRegex(ValueError, "agent.name"):
            replace(AgentRuntimeConfig(), name="other_agent")

    def test_spacemouse_agent_config_is_supported(self):
        from ur5e_control.runtime.config import (
            AgentRuntimeConfig,
            SpaceMouseRuntimeConfig,
        )

        config = AgentRuntimeConfig(
            name="spacemouse",
            robot="ur5",
            spacemouse=SpaceMouseRuntimeConfig(
                device_path="/dev/hidraw0",
                translation_scale=0.08,
                angle_scale=0.3,
                rotation_mode="rpy",
                invert_button=True,
                gripper_open=1000,
                gripper_closed=0,
            ),
        )

        self.assertEqual(config.name, "spacemouse")
        self.assertEqual(config.spacemouse.device_path, "/dev/hidraw0")
        self.assertEqual(config.spacemouse.translation_scale, 0.08)
        self.assertEqual(config.spacemouse.angle_scale, 0.3)
        self.assertEqual(config.spacemouse.rotation_mode, "rpy")
        self.assertTrue(config.spacemouse.invert_button)
        self.assertEqual(config.spacemouse.gripper_open, 1000)
        self.assertEqual(config.spacemouse.gripper_closed, 0)

    def test_single_configured_camera_gets_default_intrinsics_path(self):
        from ur5e_control.runtime.config import CameraRuntimeConfig

        config = CameraRuntimeConfig(realsense_ids={"333422304095": "wrist"})

        self.assertEqual(
            config.intrinsics_files,
            {"wrist": "configs/cameras/intrinsics_wrist.yaml"},
        )

    def test_invalid_camera_fps_is_rejected(self):
        from dataclasses import replace

        from ur5e_control.runtime.config import CameraRuntimeConfig

        config = CameraRuntimeConfig(
            realsense_ids={
                "333422300669": "third_front",
                "333422304095": "wrist",
            }
        )
        with self.assertRaisesRegex(ValueError, "camera.fps"):
            replace(config, fps=0)

    def test_invalid_depth_quality_limit_is_rejected(self):
        from ur5e_control.runtime.config import CameraRuntimeConfig

        with self.assertRaisesRegex(ValueError, "must be in"):
            CameraRuntimeConfig(
                enable_depth=True,
                realsense_ids={"333422304095": "wrist"},
                depth_max_invalid_ratio={"wrist": 1.1},
            )

    def test_disabling_depth_by_environment_also_disables_quality_gate(self):
        from ur5e_control.runtime.config import load_runtime_config

        config = load_runtime_config(environ={"UR5E_CAMERA_ENABLE_DEPTH": "false"})

        self.assertFalse(config.camera.enable_depth)
        self.assertEqual(config.camera.depth_max_invalid_ratio, {})
        self.assertEqual(config.camera.depth_options, {})

    def test_unknown_depth_option_is_rejected(self):
        from ur5e_control.runtime.config import CameraRuntimeConfig

        with self.assertRaisesRegex(ValueError, "unknown options"):
            CameraRuntimeConfig(
                enable_depth=True,
                realsense_ids={"333422304095": "wrist"},
                depth_options={"wrist": {"magic_filter": True}},
            )

    def test_config_without_realsense_ids_is_rejected(self):
        from ur5e_control.runtime.config import load_runtime_config

        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "hardware.yaml"
            config_path.write_text(
                """
robot:
  name: ur5e
agent:
  name: gello
camera:
  hostname: "127.0.0.1"
  fps: 30
gripper:
  type: dh_pgc_140_50
dataset:
  lerobot:
    repo_id: "test/ur5e"
    root: ".datasets/test"
    fps: 30
task:
  file: "configs/tasks/pen_pick_place.yaml"
  episode_count: 1
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "camera.realsense_ids"):
                load_runtime_config(config_path, environ={})

    def test_camera_calibration_file_names_must_match_logical_cameras(self):
        from ur5e_control.runtime.config import CameraRuntimeConfig

        with self.assertRaisesRegex(ValueError, "unknown"):
            CameraRuntimeConfig(
                realsense_ids={
                    "333422300669": "third_front",
                    "333422304095": "wrist",
                },
                intrinsics_files={
                    "wrist": "configs/cameras/intrinsics_wrist.yaml",
                    "unknown": "configs/cameras/intrinsics_unknown.yaml",
                },
            )


if __name__ == "__main__":
    unittest.main()
