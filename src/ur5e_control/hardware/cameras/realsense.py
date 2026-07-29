"""RealSense D455 camera adapters."""

from __future__ import annotations

import atexit
from multiprocessing import Process
import time
from typing import List, Mapping, Optional, Tuple

import numpy as np

from ur5e_control.hardware.cameras.base import CameraDriver
from ur5e_control.hardware.cameras.zmq import ZMQClientCamera, ZMQServerCamera


def get_device_ids(reset: bool = False) -> List[str]:
    import pyrealsense2 as rs

    ctx = rs.context()
    devices = ctx.query_devices()
    device_ids = []
    for dev in devices:
        if reset:
            dev.hardware_reset()
        device_ids.append(dev.get_info(rs.camera_info.serial_number))
    if reset:
        time.sleep(2)
    return device_ids


class RealSenseCamera(CameraDriver):
    def __init__(
        self,
        device_id: Optional[str] = None,
        flip: bool = False,
        fps: int = 30,
        reset_on_connect: bool = False,
        enable_depth: bool = False,
    ):
        import pyrealsense2 as rs

        self._device_id = device_id
        self._enable_depth = bool(enable_depth)
        if reset_on_connect:
            ctx = rs.context()
            for dev in ctx.query_devices():
                dev.hardware_reset()
            time.sleep(2)

        self._pipeline = rs.pipeline()
        config = rs.config()
        if device_id is not None:
            config.enable_device(device_id)
        if self._enable_depth:
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)
        self._pipeline.start(config)
        self._flip = flip

    def __repr__(self) -> str:
        return f"RealSenseCamera(device_id={self._device_id})"

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        import cv2

        frames = self._pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data()).copy()
        depth_image = None
        if self._enable_depth:
            depth_frame = frames.get_depth_frame()
            depth_image = np.asanyarray(depth_frame.get_data()).copy()
        if img_size is None:
            image = color_image[:, :, ::-1]
            depth = depth_image
        else:
            image = cv2.resize(color_image, img_size)[:, :, ::-1]
            depth = None if depth_image is None else cv2.resize(depth_image, img_size)

        if self._flip:
            image = cv2.rotate(image, cv2.ROTATE_180)
            if depth is not None:
                depth = cv2.rotate(depth, cv2.ROTATE_180)[:, :, None]
        elif depth is not None:
            depth = depth[:, :, None]

        return image, depth


class RealSenseCameraServer:
    def __init__(
        self,
        hostname: str = "127.0.0.1",
        realsense_ids: Mapping[str, str] | None = None,
        reset_cameras: bool = False,
        fps: int = 30,
        enable_depth: bool = False,
    ):
        self.hostname = hostname
        self.realsense_ids = dict(realsense_ids or {})
        self.enable_depth = bool(enable_depth)
        if not self.realsense_ids:
            raise ValueError("realsense_ids must be provided by runtime config")
        self.camera_servers = {}
        self.cameras = {}
        camera_port = 5000

        configured_ids = set(self.realsense_ids)
        detected_device_ids = get_device_ids(reset=reset_cameras)
        detected_ids = set(detected_device_ids)
        missing_ids = configured_ids.difference(detected_ids)
        if missing_ids:
            missing = ", ".join(sorted(missing_ids))
            raise RuntimeError(
                f"Configured RealSense cameras are not connected: {missing}"
            )

        for camera_id in self.realsense_ids.keys():
            # if camera_id not in self.realsense_ids:
            #     raise ValueError(
            #         f"Camera with ID {camera_id} is not configured in realsense_ids"
            #     )
            name = self.realsense_ids[camera_id]
            print(f"Launching camera {name} (ID {camera_id}) on port {camera_port}")
            self.camera_servers[camera_id] = Process(
                target=self.launch_server,
                args=(camera_port, camera_id, fps, self.enable_depth),
            )
            self.cameras[camera_id] = ZMQClientCamera(
                port=camera_port,
                host=self.hostname,
            )
            camera_port += 1

        for server in self.camera_servers.values():
            server.start()
            time.sleep(2)
        atexit.register(self.stop)
        self.num_cameras = len(self.camera_servers)

    def launch_server(self, port: int, camera_id: str, fps: int, enable_depth: bool):
        camera = RealSenseCamera(camera_id, fps=fps, enable_depth=enable_depth)
        server = ZMQServerCamera(camera, port=port, host=self.hostname)
        print(f"Starting camera server on port {port}")
        server.serve()

    def stop(self):
        for server in self.camera_servers.values():
            if server.is_alive():
                server.terminate()
                server.join(timeout=2)
