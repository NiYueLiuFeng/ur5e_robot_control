import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class CameraCheckRecordingTest(unittest.TestCase):
    def test_records_one_mp4_per_logical_camera_view(self):
        camera_check_module = load_camera_check_module()
        record_camera_check = camera_check_module.record_camera_check

        cameras = {
            "serial-wrist": FakeCamera("wrist"),
            "serial-third": FakeCamera("third"),
        }
        camera_name_id = {
            "serial-wrist": "wrist",
            "serial-third": "third_front",
        }
        writer_factory = FakeWriterFactory()

        with TemporaryDirectory() as tmpdir:
            outputs = record_camera_check(
                cameras=cameras,
                camera_name_id=camera_name_id,
                output_dir=Path(tmpdir),
                duration_s=1.0,
                fps=2,
                video_writer_factory=writer_factory,
                sleep=lambda seconds: None,
            )

        self.assertEqual(set(outputs), {"wrist", "third_front"})
        self.assertEqual(outputs["wrist"].name, "wrist.mp4")
        self.assertEqual(outputs["third_front"].name, "third_front.mp4")
        self.assertEqual(writer_factory.created["wrist.mp4"].frames, ["wrist", "wrist"])
        self.assertEqual(
            writer_factory.created["third_front.mp4"].frames,
            ["third", "third"],
        )


class FakeCamera:
    def __init__(self, name):
        self.name = name

    def read(self):
        return FakeImage(self.name), None


class FakeImage:
    shape = (480, 640, 3)

    def __init__(self, name):
        self.name = name


class FakeWriterFactory:
    def __init__(self):
        self.created = {}

    def __call__(self, path, fps, frame_size):
        writer = FakeWriter()
        self.created[Path(path).name] = writer
        return writer


class FakeWriter:
    def __init__(self):
        self.frames = []
        self.released = False

    def write(self, image):
        self.frames.append(image.name)

    def release(self):
        self.released = True


def load_camera_check_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "ur5e_control"
        / "hardware"
        / "cameras"
        / "check.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_camera_check_module", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
