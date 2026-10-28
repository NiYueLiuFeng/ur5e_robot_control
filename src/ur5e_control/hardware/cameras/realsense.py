"""RealSense D455 camera adapters."""

from __future__ import annotations

import atexit
import gc
from multiprocessing import Process
import threading
import time
from typing import Any, List, Mapping, Optional, Tuple

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
        depth_options: Mapping[str, Any] | None = None,
    ):
        import pyrealsense2 as rs

        self._rs = rs
        self._device_id = device_id
        self._fps = int(fps)
        self._enable_depth = bool(enable_depth)
        self._pipeline_lock = threading.RLock()
        if reset_on_connect:
            ctx = rs.context()
            for dev in ctx.query_devices():
                dev.hardware_reset()
            time.sleep(2)

        if self._enable_depth and depth_options:
            # Configure the depth sensor before pipeline.start().  Some D400
            # firmware rejects XU writes (notably visual_preset) while the
            # sensor is already streaming with EBUSY.
            _configure_depth_options(
                rs,
                device_id,
                fps,
                depth_options,
            )
            # A preset write briefly reconfigures the D400 firmware.  Starting
            # the streams in the same instant can fail with get_xu(...): busy.
            time.sleep(0.25)
        pipeline, config = _create_pipeline_config(
            rs,
            device_id,
            fps,
            self._enable_depth,
        )
        self._pipeline, self._profile = _start_pipeline(
            rs,
            device_id,
            fps,
            self._enable_depth,
            pipeline,
            config,
        )
        self._update_stream_profiles()
        self._flip = flip

    def _update_stream_profiles(self) -> None:
        rs = self._rs
        color_profile = self._profile.get_stream(
            rs.stream.color
        ).as_video_stream_profile()
        self._color_profile = color_profile
        self._depth_profile = None
        if self._enable_depth:
            self._depth_profile = self._profile.get_stream(
                rs.stream.depth
            ).as_video_stream_profile()

    def _restart_pipeline(self) -> None:
        """Replace a stalled pipeline without reusing half-open UVC state."""

        old_pipeline = self._pipeline
        try:
            old_pipeline.stop()
        except RuntimeError:
            pass
        self._pipeline = None
        del old_pipeline
        gc.collect()
        time.sleep(0.5)

        pipeline, config = _create_pipeline_config(
            self._rs,
            self._device_id,
            self._fps,
            self._enable_depth,
        )
        self._pipeline, self._profile = _start_pipeline(
            self._rs,
            self._device_id,
            self._fps,
            self._enable_depth,
            pipeline,
            config,
        )
        self._update_stream_profiles()

    def _wait_for_frames(self) -> Any:
        try:
            return self._pipeline.wait_for_frames()
        except RuntimeError as exc:
            if "frame didn't arrive" not in str(exc).lower():
                raise

        print(
            f"RealSense {self._device_id or '<default>'} timed out waiting "
            "for frames; restarting its pipeline",
            flush=True,
        )
        self._restart_pipeline()
        try:
            return self._pipeline.wait_for_frames()
        except RuntimeError as exc:
            raise RuntimeError(
                f"RealSense {self._device_id or '<default>'} did not deliver "
                f"frames after one pipeline restart: {exc}"
            ) from exc

    def __repr__(self) -> str:
        return f"RealSenseCamera(device_id={self._device_id})"

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        with self._pipeline_lock:
            frames = self._wait_for_frames()
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
            import cv2

            image = cv2.resize(color_image, img_size)[:, :, ::-1]
            depth = None if depth_image is None else cv2.resize(depth_image, img_size)

        if self._flip:
            import cv2

            image = cv2.rotate(image, cv2.ROTATE_180)
            if depth is not None:
                depth = cv2.rotate(depth, cv2.ROTATE_180)[:, :, None]
        elif depth is not None:
            depth = depth[:, :, None]

        return image, depth

    def get_calibration(self) -> Mapping[str, Any]:
        """Return intrinsics and stream-to-stream extrinsics from the device."""

        with self._pipeline_lock:
            calibration: dict[str, Any] = {
                "color_intrinsics": _intrinsics_mapping(
                    self._color_profile.get_intrinsics()
                )
            }
            if self._depth_profile is not None:
                calibration["depth_intrinsics"] = _intrinsics_mapping(
                    self._depth_profile.get_intrinsics()
                )
                calibration["color_T_depth"] = _extrinsics_matrix(
                    self._depth_profile.get_extrinsics_to(self._color_profile)
                )
                calibration["depth_T_color"] = _extrinsics_matrix(
                    self._color_profile.get_extrinsics_to(self._depth_profile)
                )
        return calibration


class RealSenseCameraServer:
    def __init__(
        self,
        hostname: str = "127.0.0.1",
        realsense_ids: Mapping[str, str] | None = None,
        reset_cameras: bool = False,
        fps: int = 30,
        enable_depth: bool = False,
        depth_options: Mapping[str, Mapping[str, Any]] | None = None,
    ):
        self.hostname = hostname
        self.realsense_ids = dict(realsense_ids or {})
        self.enable_depth = bool(enable_depth)
        self.depth_options = {
            str(name): dict(options)
            for name, options in (depth_options or {}).items()
        }
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
                args=(
                    camera_port,
                    camera_id,
                    fps,
                    self.enable_depth,
                    self.depth_options.get(name),
                ),
            )
            self.cameras[camera_id] = ZMQClientCamera(
                port=camera_port,
                host=self.hostname,
                camera_name=name,
            )
            camera_port += 1

        for camera_id, server in self.camera_servers.items():
            server.start()
            camera_name = self.realsense_ids[camera_id]
            try:
                # Do not let robot setup race the camera capture thread.  A
                # successful first read also verifies that the configured
                # color/depth profile actually delivers synchronized frames.
                self.cameras[camera_id].read()
            except Exception as exc:
                self.stop()
                raise RuntimeError(
                    f"RealSense camera {camera_name} ({camera_id}) failed "
                    f"during first-frame startup: {exc}"
                ) from exc
        atexit.register(self.stop)
        self.num_cameras = len(self.camera_servers)

    def launch_server(
        self,
        port: int,
        camera_id: str,
        fps: int,
        enable_depth: bool,
        depth_options: Mapping[str, Any] | None,
    ):
        camera = RealSenseCamera(
            camera_id,
            fps=fps,
            enable_depth=enable_depth,
            depth_options=depth_options,
        )
        server = ZMQServerCamera(camera, port=port, host=self.hostname)
        print(f"Starting camera server on port {port}")
        server.serve()

    def get_calibration(self) -> dict[str, Any]:
        """Fetch active calibration from every RealSense camera process."""

        calibration: dict[str, Any] = {
            "intrinsics": {},
            "extrinsics": {"transforms": {}},
        }
        for camera_id, camera in self.cameras.items():
            camera_name = self.realsense_ids[camera_id]
            values = camera.get_calibration()
            calibration["intrinsics"][camera_name] = values["color_intrinsics"]
            if "depth_intrinsics" not in values:
                continue
            depth_name = f"{camera_name}_depth"
            calibration["intrinsics"][depth_name] = values["depth_intrinsics"]
            for transform_name in ("color_T_depth", "depth_T_color"):
                calibration["extrinsics"]["transforms"][
                    f"{camera_name}_{transform_name}"
                ] = {"matrix": values[transform_name]}
        return calibration

    def stop(self):
        for server in self.camera_servers.values():
            if server.is_alive():
                server.terminate()
                server.join(timeout=2)


def _intrinsics_mapping(intrinsics: Any) -> dict[str, Any]:
    return {
        "image_size": [int(intrinsics.width), int(intrinsics.height)],
        "camera_matrix": [
            [float(intrinsics.fx), 0.0, float(intrinsics.ppx)],
            [0.0, float(intrinsics.fy), float(intrinsics.ppy)],
            [0.0, 0.0, 1.0],
        ],
        "distortion_coefficients": [
            float(value) for value in intrinsics.coeffs
        ],
        "model": str(intrinsics.model),
    }


def _extrinsics_matrix(extrinsics: Any) -> list[list[float]]:
    # librealsense exposes rs2_extrinsics.rotation in column-major order.
    rotation = np.asarray(extrinsics.rotation, dtype=float).reshape(3, 3).T
    translation = np.asarray(extrinsics.translation, dtype=float).reshape(3)
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix.tolist()


def _create_pipeline_config(
    rs: Any,
    device_id: str | None,
    fps: int,
    enable_depth: bool,
) -> tuple[Any, Any]:
    pipeline = rs.pipeline()
    config = rs.config()
    if device_id is not None:
        config.enable_device(device_id)
    if enable_depth:
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)
    return pipeline, config


def _configure_depth_options(
    rs: Any,
    device_id: str | None,
    fps: int,
    options: Mapping[str, Any],
) -> None:
    """Configure depth options using handles separate from the live stream."""

    pipeline, config = _create_pipeline_config(rs, device_id, fps, True)
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    resolved_profile = config.resolve(pipeline_wrapper)
    sensor = resolved_profile.get_device().first_depth_sensor()
    _apply_depth_options(sensor, rs, options)
    # librealsense may otherwise keep a partially resolved UVC device alive
    # until a later garbage-collection cycle.
    del sensor, resolved_profile, pipeline_wrapper, config, pipeline
    gc.collect()


def _start_pipeline(
    rs: Any,
    device_id: str | None,
    fps: int,
    enable_depth: bool,
    pipeline: Any,
    config: Any,
    *,
    busy_retries: int = 4,
    retry_delay_s: float = 0.25,
) -> tuple[Any, Any]:
    """Start with a fresh pipeline after every transient UVC/XU failure."""

    for attempt in range(busy_retries + 1):
        try:
            return pipeline, pipeline.start(config)
        except RuntimeError as exc:
            message = str(exc).lower()
            is_transient = (
                "device or resource busy" in message
                or "uvc device is already opened" in message
            )
            if not is_transient or attempt == busy_retries:
                raise RuntimeError(
                    f"failed to start RealSense pipeline after "
                    f"{attempt + 1} attempt(s): {exc}"
                ) from exc
            try:
                pipeline.stop()
            except RuntimeError:
                pass
            del pipeline, config
            gc.collect()
            time.sleep(retry_delay_s * (attempt + 1))
            pipeline, config = _create_pipeline_config(
                rs,
                device_id,
                fps,
                enable_depth,
            )
    raise AssertionError("unreachable")


def _apply_depth_options(
    sensor: Any,
    rs: Any,
    options: Mapping[str, Any],
    *,
    busy_retries: int = 4,
    retry_delay_s: float = 0.25,
) -> None:
    """Apply supported D400 depth-sensor options with clear failures."""

    setters: list[tuple[str, Any, float]] = []
    preset_name = options.get("visual_preset")
    if preset_name is not None:
        normalized_name = str(preset_name).strip().lower().replace("-", "_")
        try:
            preset = getattr(rs.rs400_visual_preset, normalized_name)
        except AttributeError as exc:
            raise ValueError(
                f"unknown RealSense visual preset {preset_name!r}"
            ) from exc
        setters.append(("visual_preset", rs.option.visual_preset, float(preset)))
    if "emitter_enabled" in options:
        emitter_enabled = options["emitter_enabled"]
        if not isinstance(emitter_enabled, bool):
            raise ValueError("RealSense emitter_enabled must be true or false")
        setters.append(
            (
                "emitter_enabled",
                rs.option.emitter_enabled,
                float(emitter_enabled),
            )
        )
    if "laser_power" in options:
        setters.append(
            ("laser_power", rs.option.laser_power, float(options["laser_power"]))
        )

    for name, option, value in setters:
        if hasattr(sensor, "supports") and not sensor.supports(option):
            raise RuntimeError(f"RealSense depth sensor does not support {name}")
        for attempt in range(busy_retries + 1):
            try:
                sensor.set_option(option, value)
                break
            except RuntimeError as exc:
                is_busy = "device or resource busy" in str(exc).lower()
                if not is_busy or attempt == busy_retries:
                    raise RuntimeError(
                        f"failed to set RealSense depth option "
                        f"{name}={value}: {exc}"
                    ) from exc
                time.sleep(retry_delay_s * (attempt + 1))
