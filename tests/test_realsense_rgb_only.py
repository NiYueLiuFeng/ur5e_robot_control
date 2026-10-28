import sys
import types
import unittest


class RealSenseRGBOnlyTest(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("pyrealsense2", None)

    def test_realsense_camera_enables_only_color_stream_by_default(self):
        fake_rs = FakeRealSenseModule()
        sys.modules["pyrealsense2"] = fake_rs

        from ur5e_control.hardware.cameras.realsense import RealSenseCamera

        camera = RealSenseCamera(device_id="serial", fps=30)

        self.assertEqual(
            fake_rs.last_config.enabled_streams,
            [("color", 640, 480, "bgr8", 30)],
        )
        calibration = camera.get_calibration()
        self.assertEqual(calibration["color_intrinsics"]["image_size"], [640, 480])
        self.assertNotIn("depth_intrinsics", calibration)

    def test_realsense_camera_enables_depth_when_requested(self):
        fake_rs = FakeRealSenseModule()
        sys.modules["pyrealsense2"] = fake_rs

        from ur5e_control.hardware.cameras.realsense import RealSenseCamera

        camera = RealSenseCamera(device_id="serial", fps=30, enable_depth=True)

        self.assertEqual(
            fake_rs.last_config.enabled_streams,
            [
                ("depth", 640, 480, "z16", 30),
                ("color", 640, 480, "bgr8", 30),
            ],
        )
        calibration = camera.get_calibration()
        self.assertEqual(calibration["depth_intrinsics"]["image_size"], [640, 480])
        self.assertEqual(len(calibration["color_T_depth"]), 4)
        self.assertEqual(len(calibration["depth_T_color"]), 4)

    def test_realsense_camera_applies_depth_options_before_streaming(self):
        fake_rs = FakeRealSenseModule()
        sys.modules["pyrealsense2"] = fake_rs

        from ur5e_control.hardware.cameras.realsense import RealSenseCamera

        RealSenseCamera(
            device_id="serial",
            fps=30,
            enable_depth=True,
            depth_options={"visual_preset": "high_density"},
        )

        self.assertEqual(
            fake_rs.last_pipeline.options_at_start,
            {"visual_preset": 4.0},
        )

    def test_camera_server_aggregates_live_calibration_by_logical_name(self):
        from ur5e_control.hardware.cameras.realsense import RealSenseCameraServer

        server = object.__new__(RealSenseCameraServer)
        server.realsense_ids = {"serial": "wrist"}
        server.cameras = {"serial": FakeCalibrationClient()}

        calibration = server.get_calibration()

        self.assertIn("wrist", calibration["intrinsics"])
        self.assertIn("wrist_depth", calibration["intrinsics"])
        self.assertIn(
            "wrist_color_T_depth",
            calibration["extrinsics"]["transforms"],
        )

    def test_extrinsics_rotation_is_decoded_from_column_major_order(self):
        from ur5e_control.hardware.cameras.realsense import _extrinsics_matrix

        extrinsics = types.SimpleNamespace(
            rotation=[0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            translation=[0.1, 0.2, 0.3],
        )

        matrix = _extrinsics_matrix(extrinsics)

        self.assertEqual(
            matrix,
            [
                [0.0, -1.0, 0.0, 0.1],
                [1.0, 0.0, 0.0, 0.2],
                [0.0, 0.0, 1.0, 0.3],
                [0.0, 0.0, 0.0, 1.0],
            ],
        )

    def test_depth_options_apply_visual_preset_emitter_and_laser_power(self):
        from ur5e_control.hardware.cameras.realsense import _apply_depth_options

        sensor = FakeDepthSensor()
        fake_rs = types.SimpleNamespace(
            option=types.SimpleNamespace(
                visual_preset="visual_preset",
                emitter_enabled="emitter_enabled",
                laser_power="laser_power",
            ),
            rs400_visual_preset=types.SimpleNamespace(high_density=4),
        )

        _apply_depth_options(
            sensor,
            fake_rs,
            {
                "visual_preset": "high-density",
                "emitter_enabled": False,
                "laser_power": 120,
            },
        )

        self.assertEqual(
            sensor.values,
            {
                "visual_preset": 4.0,
                "emitter_enabled": 0.0,
                "laser_power": 120.0,
            },
        )

    def test_depth_options_retry_when_device_is_temporarily_busy(self):
        from ur5e_control.hardware.cameras.realsense import _apply_depth_options

        sensor = TemporarilyBusyDepthSensor()
        fake_rs = types.SimpleNamespace(
            option=types.SimpleNamespace(visual_preset="visual_preset"),
            rs400_visual_preset=types.SimpleNamespace(high_density=4),
        )

        _apply_depth_options(
            sensor,
            fake_rs,
            {"visual_preset": "high_density"},
            retry_delay_s=0,
        )

        self.assertEqual(sensor.attempts, 3)
        self.assertEqual(sensor.values, {"visual_preset": 4.0})

    def test_pipeline_start_retries_when_device_is_temporarily_busy(self):
        from ur5e_control.hardware.cameras.realsense import _start_pipeline

        fake_rs = FakeRealSenseModule()
        fake_rs.pipeline_failures_remaining = 2
        pipeline = fake_rs.pipeline()
        config = fake_rs.config()
        live_pipeline, profile = _start_pipeline(
            fake_rs,
            "serial",
            30,
            True,
            pipeline,
            config,
            retry_delay_s=0,
        )

        self.assertIs(live_pipeline, fake_rs.last_pipeline)
        self.assertIsInstance(profile, FakePipelineProfile)
        self.assertEqual(fake_rs.pipeline_count, 3)

    def test_frame_timeout_restarts_with_a_fresh_pipeline(self):
        fake_rs = FakeRealSenseModule()
        fake_rs.frame_failures_remaining = 1
        sys.modules["pyrealsense2"] = fake_rs

        from ur5e_control.hardware.cameras.realsense import RealSenseCamera

        camera = RealSenseCamera(
            device_id="serial",
            fps=30,
            enable_depth=True,
        )
        image, depth = camera.read()

        self.assertEqual(fake_rs.pipeline_count, 2)
        self.assertEqual(image.shape, (480, 640, 3))
        self.assertEqual(depth.shape, (480, 640, 1))


class FakeRealSenseModule(types.SimpleNamespace):
    def __init__(self):
        super().__init__()
        self.stream = types.SimpleNamespace(depth="depth", color="color")
        self.format = types.SimpleNamespace(z16="z16", bgr8="bgr8")
        self.camera_info = types.SimpleNamespace(serial_number="serial_number")
        self.option = types.SimpleNamespace(
            visual_preset="visual_preset",
            emitter_enabled="emitter_enabled",
            laser_power="laser_power",
        )
        self.rs400_visual_preset = types.SimpleNamespace(high_density=4)
        self.last_config = None
        self.last_pipeline = None
        self.pipeline_count = 0
        self.pipeline_failures_remaining = 0
        self.frame_failures_remaining = 0
        self.device_option_values = {}
        self.pipeline = self._pipeline
        self.pipeline_wrapper = lambda pipeline: pipeline
        self.config = self._config

    def _pipeline(self):
        self.pipeline_count += 1
        self.last_pipeline = FakePipeline(self)
        return self.last_pipeline

    def _config(self):
        self.last_config = FakeConfig()
        return self.last_config


class FakeConfig:
    def __init__(self):
        self.enabled_device = None
        self.enabled_streams = []

    def enable_device(self, device_id):
        self.enabled_device = device_id

    def enable_stream(self, stream, width, height, fmt, fps):
        self.enabled_streams.append((stream, width, height, fmt, fps))

    def resolve(self, pipeline_wrapper):
        return FakePipelineProfile(pipeline_wrapper.depth_sensor)


class FakePipeline:
    def __init__(self, owner):
        self.owner = owner
        self.depth_sensor = FakeDepthSensor(owner.device_option_values)
        self.options_at_start = None
        self.stopped = False

    def start(self, config):
        if self.owner.pipeline_failures_remaining:
            self.owner.pipeline_failures_remaining -= 1
            raise RuntimeError(
                "get_xu(id=) failed! Last Error: Device or resource busy"
            )
        self.config = config
        self.options_at_start = dict(self.depth_sensor.values)
        return FakePipelineProfile(self.depth_sensor)

    def stop(self):
        self.stopped = True

    def wait_for_frames(self):
        if self.owner.frame_failures_remaining:
            self.owner.frame_failures_remaining -= 1
            raise RuntimeError("Frame didn't arrive within 5000")
        return FakeFrameSet()


class FakeCalibrationClient:
    def get_calibration(self):
        intrinsics = {
            "image_size": [640, 480],
            "camera_matrix": [[600, 0, 320], [0, 600, 240], [0, 0, 1]],
            "distortion_coefficients": [0, 0, 0, 0, 0],
            "model": "brown_conrady",
        }
        identity = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
        return {
            "color_intrinsics": intrinsics,
            "depth_intrinsics": intrinsics,
            "color_T_depth": identity,
            "depth_T_color": identity,
        }


class FakeDepthSensor:
    def __init__(self, values=None):
        self.values = values if values is not None else {}

    def supports(self, option):
        return True

    def set_option(self, option, value):
        self.values[option] = value


class FakeFrame:
    def __init__(self, data):
        self.data = data

    def get_data(self):
        return self.data


class FakeFrameSet:
    def get_color_frame(self):
        import numpy as np

        return FakeFrame(np.zeros((480, 640, 3), dtype=np.uint8))

    def get_depth_frame(self):
        import numpy as np

        return FakeFrame(np.zeros((480, 640), dtype=np.uint16))


class TemporarilyBusyDepthSensor(FakeDepthSensor):
    def __init__(self):
        super().__init__()
        self.attempts = 0

    def set_option(self, option, value):
        self.attempts += 1
        if self.attempts < 3:
            raise RuntimeError(
                "get_xu(ctrl=1) failed! Last Error: Device or resource busy"
            )
        super().set_option(option, value)


class FakePipelineProfile:
    def __init__(self, depth_sensor=None):
        self.depth_sensor = depth_sensor or FakeDepthSensor()

    def get_device(self):
        return types.SimpleNamespace(
            first_depth_sensor=lambda: self.depth_sensor
        )

    def get_stream(self, stream):
        return FakeVideoStreamProfile(stream)


class FakeVideoStreamProfile:
    def __init__(self, stream):
        self.stream = stream

    def as_video_stream_profile(self):
        return self

    def get_intrinsics(self):
        return types.SimpleNamespace(
            width=640,
            height=480,
            fx=600.0,
            fy=601.0,
            ppx=320.0,
            ppy=240.0,
            coeffs=[0.1, 0.0, 0.0, 0.0, 0.0],
            model="brown_conrady",
        )

    def get_extrinsics_to(self, target):
        translation = (
            [0.01, 0.02, 0.03]
            if self.stream == "depth" and target.stream == "color"
            else [-0.01, -0.02, -0.03]
        )
        return types.SimpleNamespace(
            rotation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            translation=translation,
        )


if __name__ == "__main__":
    unittest.main()
