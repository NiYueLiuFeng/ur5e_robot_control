import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

try:
    import cv2  # noqa: F401
except ModuleNotFoundError:  # optional dependency in some test environments
    cv2 = None


def make_obs(height=8, width=6):
    return {
        "third_left_top_rgb": np.zeros((height, width, 3), dtype=np.uint8),
        "wrist_rgb": np.full((height, width, 3), 255, dtype=np.uint8),
    }


@unittest.skipIf(cv2 is None, "opencv is not installed")
class EpisodeVideoRecorderTest(unittest.TestCase):
    def test_merge_views_puts_cameras_side_by_side(self):
        from ur5e_control.runtime.episode_video import merge_views

        left = np.zeros((8, 6, 3), dtype=np.uint8)
        right = np.full((8, 6, 3), 255, dtype=np.uint8)

        merged = merge_views(left, right)

        self.assertEqual(merged.shape, (8, 12, 3))
        self.assertTrue((merged[:, :6] == 0).all())
        self.assertTrue((merged[:, 6:] == 255).all())

    def test_save_and_discard_write_numbered_files(self):
        from ur5e_control.runtime.episode_video import EpisodeVideoRecorder

        with tempfile.TemporaryDirectory() as root:
            recorder = EpisodeVideoRecorder(root=root, model="json", fps=1e9)
            for _ in range(3):
                recorder.add(make_obs())

            path = recorder.save()

            self.assertIsNotNone(path)
            self.assertTrue(path.exists())
            self.assertEqual(path.name, "video_1.mp4")
            self.assertEqual(path.parent.parent, Path(root) / "json")
            self.assertRegex(path.parent.name, r"^\d{8}-\d{2}:\d{2}$")

            for _ in range(2):
                recorder.add(make_obs())
            second = recorder.save()
            self.assertEqual(second.name, "video_2.mp4")

    def test_discard_deletes_the_partial_file(self):
        from ur5e_control.runtime.episode_video import EpisodeVideoRecorder

        with tempfile.TemporaryDirectory() as root:
            recorder = EpisodeVideoRecorder(root=root, model="fastwam", fps=1e9)
            recorder.add(make_obs())
            path = recorder._path
            self.assertTrue(path.exists())

            recorder.discard()

            self.assertFalse(path.exists())
            self.assertFalse(recorder.has_frames)

    def test_writer_failure_disables_recording_without_raising(self):
        from ur5e_control.runtime.episode_video import EpisodeVideoRecorder

        with tempfile.TemporaryDirectory() as root:
            recorder = EpisodeVideoRecorder(root=root, model="json", fps=1e9)
            with unittest.mock.patch(
                "ur5e_control.runtime.episode_video._open_writer",
                side_effect=RuntimeError("no encoder"),
            ):
                recorder.add(make_obs())

            self.assertFalse(recorder.has_frames)
            self.assertIsNone(recorder.save())
            self.assertFalse(list(Path(root).rglob("*.mp4")))

    def test_save_without_frames_is_a_no_op(self):
        from ur5e_control.runtime.episode_video import EpisodeVideoRecorder

        with tempfile.TemporaryDirectory() as root:
            recorder = EpisodeVideoRecorder(root=root, model="json", fps=15.0)

            self.assertIsNone(recorder.save())
            self.assertFalse(list(Path(root).rglob("*.mp4")))


if __name__ == "__main__":
    unittest.main()
