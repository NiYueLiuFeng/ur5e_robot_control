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

        RealSenseCamera(device_id="serial", fps=30)

        self.assertEqual(
            fake_rs.last_config.enabled_streams,
            [("color", 640, 480, "bgr8", 30)],
        )

    def test_realsense_camera_enables_depth_when_requested(self):
        fake_rs = FakeRealSenseModule()
        sys.modules["pyrealsense2"] = fake_rs

        from ur5e_control.hardware.cameras.realsense import RealSenseCamera

        RealSenseCamera(device_id="serial", fps=30, enable_depth=True)

        self.assertEqual(
            fake_rs.last_config.enabled_streams,
            [
                ("depth", 640, 480, "z16", 30),
                ("color", 640, 480, "bgr8", 30),
            ],
        )


class FakeRealSenseModule(types.SimpleNamespace):
    def __init__(self):
        super().__init__()
        self.stream = types.SimpleNamespace(depth="depth", color="color")
        self.format = types.SimpleNamespace(z16="z16", bgr8="bgr8")
        self.camera_info = types.SimpleNamespace(serial_number="serial_number")
        self.last_config = None
        self.pipeline = self._pipeline
        self.config = self._config

    def _pipeline(self):
        return FakePipeline()

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


class FakePipeline:
    def start(self, config):
        self.config = config


if __name__ == "__main__":
    unittest.main()
