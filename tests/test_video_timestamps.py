from fractions import Fraction
import unittest


class VideoTimestampRepairTest(unittest.TestCase):
    def test_rebuilds_episode_boundaries_from_real_pts(self):
        from ur5e_control.data_collection.video_timestamps import (
            EpisodeTimestampSpan,
            rebuild_episode_timestamps,
        )

        # A one-tick concat gap shifts the second episode without changing its
        # internal fixed-rate cadence.
        pts = [0, 1024, 2048, 3073, 4097]
        episodes = [
            EpisodeTimestampSpan(0, 3, 0.0, 0.2),
            EpisodeTimestampSpan(1, 2, 0.2, 1 / 3),
        ]

        rebuilt = rebuild_episode_timestamps(
            pts,
            Fraction(1, 15360),
            episodes,
            fps=15,
        )

        self.assertEqual(rebuilt[0].from_timestamp, 0.0)
        self.assertAlmostEqual(rebuilt[0].to_timestamp, 0.2)
        self.assertAlmostEqual(rebuilt[1].from_timestamp, 3073 / 15360)
        self.assertAlmostEqual(rebuilt[1].to_timestamp, 3073 / 15360 + 2 / 15)

    def test_rejects_frame_count_mismatch(self):
        from ur5e_control.data_collection.video_timestamps import (
            EpisodeTimestampSpan,
            rebuild_episode_timestamps,
        )

        with self.assertRaisesRegex(ValueError, "frame count"):
            rebuild_episode_timestamps(
                [0, 1024],
                Fraction(1, 15360),
                [EpisodeTimestampSpan(0, 3, 0.0, 0.2)],
                fps=15,
            )

    def test_rejects_variable_rate_episode_that_metadata_cannot_fix(self):
        from ur5e_control.data_collection.video_timestamps import (
            EpisodeTimestampSpan,
            rebuild_episode_timestamps,
        )

        with self.assertRaisesRegex(ValueError, "re-encode"):
            rebuild_episode_timestamps(
                [0, 1024, 2100],
                Fraction(1, 15360),
                [EpisodeTimestampSpan(0, 3, 0.0, 0.2)],
                fps=15,
                tolerance_s=1e-4,
            )


if __name__ == "__main__":
    unittest.main()
