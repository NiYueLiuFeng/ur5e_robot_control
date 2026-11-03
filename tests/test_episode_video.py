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
            recorder = EpisodeVideoRecorder(root=root, model="json", fps=1000.0)
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

    def test_partial_save_renames_and_keeps_the_index_taken(self):
        from ur5e_control.runtime.episode_video import EpisodeVideoRecorder

        with tempfile.TemporaryDirectory() as root:
            recorder = EpisodeVideoRecorder(root=root, model="json", fps=1000.0)
            recorder.add(make_obs())

            path = recorder.save(partial=True)

            # 退出时保留的录像用 _partial 后缀，避免当成操作员确认的 episode。
            self.assertEqual(path.name, "video_1_partial.mp4")
            self.assertTrue(path.exists())
            self.assertFalse((path.parent / "video_1.mp4").exists())

            # _partial 文件占用序号 1，下一段必须从 video_2 开始。
            recorder.add(make_obs())
            second = recorder.save()
            self.assertEqual(second.name, "video_2.mp4")

    def test_discard_deletes_the_file_written_during_recording(self):
        from ur5e_control.runtime.episode_video import EpisodeVideoRecorder

        with tempfile.TemporaryDirectory() as root:
            recorder = EpisodeVideoRecorder(root=root, model="fastwam", fps=1000.0)
            self.assertFalse(list(Path(root).rglob("*.mp4")))

            recorder.add(make_obs())

            # 边录边写：第一帧就已经有了目录和文件，不等 save()。
            self.assertTrue(recorder.has_frames)
            self.assertEqual(len(list(Path(root).rglob("*.mp4"))), 1)

            recorder.discard()

            self.assertFalse(recorder.has_frames)
            self.assertFalse(list(Path(root).rglob("*.mp4")))

    def test_file_exists_before_save_so_a_crash_loses_only_the_tail(self):
        from ur5e_control.runtime.episode_video import EpisodeVideoRecorder

        with tempfile.TemporaryDirectory() as root:
            recorder = EpisodeVideoRecorder(root=root, model="json", fps=1000.0)
            recorder.add(make_obs())

            # 不调用 save()：文件必须已经在盘上。
            files = list(Path(root).rglob("*.mp4"))
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].exists())

    def test_writer_failure_disables_recording_without_raising(self):
        from ur5e_control.runtime.episode_video import EpisodeVideoRecorder

        with tempfile.TemporaryDirectory() as root:
            recorder = EpisodeVideoRecorder(root=root, model="json", fps=1000.0)
            with unittest.mock.patch(
                "ur5e_control.runtime.episode_video._open_writer",
                side_effect=RuntimeError("no encoder"),
            ):
                # 编码器不可用时第一帧就要安静地关掉录像，不能抛异常。
                recorder.add(make_obs())
                self.assertFalse(recorder.has_frames)
                self.assertIsNone(recorder.save())

            self.assertFalse(list(Path(root).rglob("*.mp4")))

    def test_gaps_are_filled_so_playback_matches_wall_clock(self):
        from ur5e_control.runtime.episode_video import EpisodeVideoRecorder

        with tempfile.TemporaryDirectory() as root:
            recorder = EpisodeVideoRecorder(root=root, model="json", fps=10.0)
            # 第一次 add 在 t=100.0；100.05 太近被节流；之后间隔 0.5 s 和 1.0 s。
            with unittest.mock.patch(
                "ur5e_control.runtime.episode_video.time.monotonic",
                side_effect=[100.0, 100.05, 100.5, 101.5],
            ):
                recorder.add(make_obs())
                recorder.add(make_obs())
                recorder.add(make_obs())
                recorder.add(make_obs())

            # save 也补尾：显式给结束时刻，否则会用真实时钟算出巨大的空档。
            path = recorder.save(end=101.5)
            captured = cv2.VideoCapture(str(path))
            frames = int(captured.get(cv2.CAP_PROP_FRAME_COUNT))
            captured.release()

        # 1 + 5（0.5 s）+ 10（1.0 s）= 16 帧，与 1.5 s 的真实时长对应。
        self.assertEqual(frames, 16)

    def test_save_holds_the_last_frame_until_the_episode_ends(self):
        from ur5e_control.runtime.episode_video import EpisodeVideoRecorder

        with tempfile.TemporaryDirectory() as root:
            recorder = EpisodeVideoRecorder(root=root, model="json", fps=10.0)
            with unittest.mock.patch(
                "ur5e_control.runtime.episode_video.time.monotonic",
                return_value=100.0,
            ):
                recorder.add(make_obs())

            # 动作在 t=100 结束，之后等推理 2 s 才停止：视频也要覆盖这 2 s。
            path = recorder.save(end=102.0)
            captured = cv2.VideoCapture(str(path))
            frames = int(captured.get(cv2.CAP_PROP_FRAME_COUNT))
            captured.release()

        self.assertEqual(frames, 20)

    def test_missing_opencv_disables_recording_without_raising(self):
        from ur5e_control.runtime.episode_video import EpisodeVideoRecorder

        with tempfile.TemporaryDirectory() as root:
            recorder = EpisodeVideoRecorder(root=root, model="json", fps=1000.0)
            with unittest.mock.patch(
                "ur5e_control.runtime.episode_video.merge_views",
                side_effect=ModuleNotFoundError("No module named 'cv2'"),
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
