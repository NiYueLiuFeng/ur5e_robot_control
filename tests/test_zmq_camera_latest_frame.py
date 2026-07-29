import unittest


class LatestFrameCacheTest(unittest.TestCase):
    def test_returns_newest_captured_frame_and_preserves_none_depth(self):
        from ur5e_control.hardware.cameras.zmq import LatestFrameCache

        camera = FakeCamera(
            [
                ("rgb-1", None),
                ("rgb-2", None),
            ]
        )
        cache = LatestFrameCache(camera)

        cache.capture_once()
        cache.capture_once()

        self.assertEqual(cache.read_latest(timeout_s=0.0), ("rgb-2", None))
        self.assertEqual(camera.read_count, 2)

    def test_read_latest_times_out_before_first_frame(self):
        from ur5e_control.hardware.cameras.zmq import LatestFrameCache

        cache = LatestFrameCache(FakeCamera([]))

        with self.assertRaisesRegex(TimeoutError, "camera frame"):
            cache.read_latest(timeout_s=0.0)

    def test_read_latest_reports_capture_errors(self):
        from ur5e_control.hardware.cameras.zmq import LatestFrameCache

        cache = LatestFrameCache(FailingCamera())

        cache.capture_once()

        with self.assertRaisesRegex(RuntimeError, "camera failed"):
            cache.read_latest(timeout_s=0.0)


class ZMQClientCameraTest(unittest.TestCase):
    def test_read_resets_socket_after_timeout(self):
        import zmq

        from ur5e_control.hardware.cameras.zmq import ZMQClientCamera

        client = ZMQClientCamera.__new__(ZMQClientCamera)
        client._socket = TimeoutSocket(zmq.Again())
        client.reset_count = 0

        def reset_socket():
            client.reset_count += 1

        client._reset_socket = reset_socket

        with self.assertRaisesRegex(RuntimeError, "timeout"):
            client.read()

        self.assertEqual(client.reset_count, 1)


class FakeCamera:
    def __init__(self, frames):
        self.frames = list(frames)
        self.read_count = 0

    def read(self, img_size=None):
        self.read_count += 1
        return self.frames.pop(0)


class FailingCamera:
    def read(self, img_size=None):
        raise RuntimeError("camera failed")


class TimeoutSocket:
    def __init__(self, exc):
        self.exc = exc

    def send(self, _message):
        pass

    def recv(self):
        raise self.exc


if __name__ == "__main__":
    unittest.main()
