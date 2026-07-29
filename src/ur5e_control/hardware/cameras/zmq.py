import pickle
import threading
import time
from typing import Optional, Tuple

import numpy as np
import zmq

from ur5e_control.hardware.cameras.base import CameraDriver

DEFAULT_CAMERA_PORT = 5000
DEFAULT_ZMQ_TIMEOUT_MS = 600000


class LatestFrameCache:
    """Continuously capture camera frames and serve the newest one to clients."""

    def __init__(self, camera: CameraDriver, poll_interval_s: float = 0.0):
        self._camera = camera
        self._poll_interval_s = float(poll_interval_s)
        self._condition = threading.Condition()
        self._latest_frame = None
        self._error: BaseException | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def capture_once(self) -> None:
        try:
            frame = self._camera.read()
        except BaseException as exc:  # noqa: BLE001
            with self._condition:
                self._error = exc
                self._condition.notify_all()
            return
        with self._condition:
            self._error = None
            self._latest_frame = frame
            self._condition.notify_all()

    def read_latest(
        self,
        img_size: Optional[Tuple[int, int]] = None,
        timeout_s: float = 1.0,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        deadline = time.monotonic() + max(0.0, timeout_s)
        with self._condition:
            while self._latest_frame is None:
                if self._error is not None:
                    raise self._error
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0:
                    raise TimeoutError("Timed out waiting for latest camera frame")
                self._condition.wait(timeout=remaining_s)
            if self._error is not None:
                raise self._error
            image, depth = self._latest_frame
        if img_size is None:
            return image, depth
        return _resize_frame(image, depth, img_size)

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            self.capture_once()
            if self._poll_interval_s > 0:
                time.sleep(self._poll_interval_s)


class ZMQClientCamera(CameraDriver):
    """A class representing a ZMQ client for a leader robot."""

    def __init__(self, port: int = DEFAULT_CAMERA_PORT, host: str = "127.0.0.1"):
        self._context = zmq.Context()  # type: ignore
        self._port = port
        self._host = host
        self._socket = None
        self._reset_socket()

    def _reset_socket(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
        self._socket = self._context.socket(zmq.REQ)  # type: ignore
        self._socket.setsockopt(zmq.RCVTIMEO, DEFAULT_ZMQ_TIMEOUT_MS)  # type: ignore
        self._socket.setsockopt(zmq.SNDTIMEO, DEFAULT_ZMQ_TIMEOUT_MS)  # type: ignore
        self._socket.setsockopt(zmq.LINGER, 0)  # type: ignore
        self._socket.connect(f"tcp://{self._host}:{self._port}")

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        # pack the image_size and send it to the server
        send_message = pickle.dumps(img_size)
        try:
            self._socket.send(send_message)
            state_dict = pickle.loads(self._socket.recv())
            if isinstance(state_dict, dict) and "error" in state_dict:
                raise RuntimeError(state_dict["error"])
            return state_dict
        except zmq.Again as exc:  # type: ignore
            self._reset_socket()
            raise RuntimeError("ZMQ timeout while reading camera frame") from exc
        except zmq.ZMQError as exc:  # type: ignore
            self._reset_socket()
            raise RuntimeError("ZMQ camera socket error") from exc


class ZMQServerCamera:
    def __init__(
        self,
        camera: CameraDriver,
        port: int = DEFAULT_CAMERA_PORT,
        host: str = "127.0.0.1",
    ):
        self._camera = camera
        self._frame_cache = LatestFrameCache(camera)
        self._context = zmq.Context()  # type: ignore
        self._socket = self._context.socket(zmq.REP)  # type: ignore
        self._socket.setsockopt(zmq.LINGER, 0)  # type: ignore
        addr = f"tcp://{host}:{port}"
        debug_message = f"Camera Server Binding to {addr}, Camera: {camera}"
        print(debug_message)
        self._timeout_message = f"Timeout in Camera Server, Camera: {camera}"
        self._socket.bind(addr)
        self._stop_event = threading.Event()

    def serve(self) -> None:
        """Serve the leader robot state over ZMQ."""
        self._socket.setsockopt(zmq.RCVTIMEO, DEFAULT_ZMQ_TIMEOUT_MS)  # type: ignore
        self._frame_cache.start()
        try:
            while not self._stop_event.is_set():
                try:
                    message = self._socket.recv()
                    img_size = pickle.loads(message)
                    camera_read = self._frame_cache.read_latest(img_size)
                    self._socket.send(pickle.dumps(camera_read))
                except zmq.Again:  # type: ignore
                    pass
                except Exception as exc:  # noqa: BLE001
                    self._socket.send(pickle.dumps({"error": str(exc)}))
        finally:
            self._frame_cache.stop()

    def stop(self) -> None:
        """Signal the server to stop serving."""
        self._stop_event.set()
        self._frame_cache.stop()


def _resize_frame(
    image: np.ndarray,
    depth: Optional[np.ndarray],
    img_size: Tuple[int, int],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    import cv2

    resized_image = cv2.resize(image, img_size)
    resized_depth = None if depth is None else cv2.resize(depth, img_size)
    return resized_image, resized_depth
