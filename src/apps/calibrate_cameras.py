"""Calibrate RealSense D455 cameras for the UR5e runtime."""

from __future__ import annotations

import argparse
import atexit
from multiprocessing import Process
from pathlib import Path
import time

from ur5e_control.calibration.capture import (
    camera_id_for_logical_name,
    capture_base_extrinsics_samples,
    capture_hand_eye_samples,
    capture_intrinsics_samples,
    capture_stereo_extrinsics_samples,
)
from ur5e_control.calibration.io import (
    load_extrinsics,
    load_intrinsics,
    load_yaml,
    write_extrinsics,
    write_intrinsics,
)
from ur5e_control.calibration.patterns import CharucoPattern, ChessboardPattern
from ur5e_control.calibration.solvers import (
    solve_base_extrinsics,
    solve_hand_eye,
    solve_intrinsics,
    solve_stereo_extrinsics,
)
from ur5e_control.calibration.transforms import identity_matrix
from ur5e_control.runtime.config import load_runtime_config, resolve_repo_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate UR5e RealSense cameras")
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Hardware/runtime YAML config. Defaults to UR5E_HARDWARE_CONFIG "
            "or configs/hardware/ur5e_d455.yaml."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    intrinsics = subparsers.add_parser("intrinsics")
    _add_pattern_args(intrinsics)
    _add_capture_args(intrinsics, default_samples=20)
    intrinsics.add_argument("--camera", default="wrist")
    intrinsics.add_argument("--output", default=None)
    intrinsics.set_defaults(func=run_intrinsics)

    hand_eye = subparsers.add_parser("hand-eye")
    _add_pattern_args(hand_eye)
    _add_capture_args(hand_eye, default_samples=12)
    _add_robot_args(hand_eye)
    hand_eye.add_argument("--camera", default="wrist")
    hand_eye.add_argument("--intrinsics", default=None)
    hand_eye.add_argument("--output", default=None)
    hand_eye.add_argument("--method", default="tsai")
    hand_eye.set_defaults(func=run_hand_eye)

    base = subparsers.add_parser("base-extrinsics")
    _add_pattern_args(base)
    _add_capture_args(base, default_samples=12)
    _add_robot_args(base)
    base.add_argument("--camera", default="third_front")
    base.add_argument("--intrinsics", default=None)
    base.add_argument("--tool-target-transform", default=None)
    base.add_argument("--output", default=None)
    base.set_defaults(func=run_base_extrinsics)

    stereo = subparsers.add_parser("stereo-extrinsics")
    _add_pattern_args(stereo)
    _add_capture_args(stereo, default_samples=12)
    stereo.add_argument("--reference-camera", default="wrist")
    stereo.add_argument("--camera", default="third_front")
    stereo.add_argument("--reference-intrinsics", default=None)
    stereo.add_argument("--intrinsics", default=None)
    stereo.add_argument("--output", default=None)
    stereo.set_defaults(func=run_stereo_extrinsics)
    return parser


def run_intrinsics(args) -> None:
    config = load_runtime_config(args.config)
    pattern = _pattern_from_args(args)
    camera_server = _start_camera_server(config)
    try:
        serial = camera_id_for_logical_name(config.camera.realsense_ids, args.camera)
        samples = capture_intrinsics_samples(
            camera=camera_server.cameras[serial],
            pattern=pattern,
            sample_count=args.samples,
            interval_s=args.interval,
        )
        intrinsics = solve_intrinsics(
            samples,
            camera_name=args.camera,
            serial_number=serial,
        )
        output = _intrinsics_output_path(config, args.camera, args.output)
        write_intrinsics(output, intrinsics)
    finally:
        camera_server.stop()
    print(f"Saved {args.camera} intrinsics to {output}")


def run_hand_eye(args) -> None:
    config = load_runtime_config(args.config)
    pattern = _pattern_from_args(args)
    intrinsics = load_intrinsics(
        _intrinsics_input_path(config, args.camera, args.intrinsics)
    )
    camera_server = _start_camera_server(config)
    robot_process = None
    try:
        robot_process, robot = _robot_client(config, args.use_existing_robot_server)
        serial = camera_id_for_logical_name(config.camera.realsense_ids, args.camera)
        samples = capture_hand_eye_samples(
            camera=camera_server.cameras[serial],
            robot=robot,
            pattern=pattern,
            intrinsics=intrinsics,
            sample_count=args.samples,
            interval_s=args.interval,
        )
        transform = solve_hand_eye(
            samples,
            parent_frame="tool",
            camera_frame=args.camera,
            method=args.method,
        )
        output = _extrinsics_output_path(config, args.output)
        _write_extrinsics_merged(output, transform)
    finally:
        camera_server.stop()
        _stop_robot_process(robot_process)
    print(f"Saved {transform.name} to {output}")


def run_base_extrinsics(args) -> None:
    config = load_runtime_config(args.config)
    pattern = _pattern_from_args(args)
    intrinsics = load_intrinsics(
        _intrinsics_input_path(config, args.camera, args.intrinsics)
    )
    tool_T_target = _load_optional_transform(args.tool_target_transform)
    camera_server = _start_camera_server(config)
    robot_process = None
    try:
        robot_process, robot = _robot_client(config, args.use_existing_robot_server)
        serial = camera_id_for_logical_name(config.camera.realsense_ids, args.camera)
        samples = capture_base_extrinsics_samples(
            camera=camera_server.cameras[serial],
            robot=robot,
            pattern=pattern,
            intrinsics=intrinsics,
            sample_count=args.samples,
            interval_s=args.interval,
        )
        transform = solve_base_extrinsics(
            samples,
            tool_T_target=tool_T_target,
            parent_frame="base",
            camera_frame=args.camera,
        )
        output = _extrinsics_output_path(config, args.output)
        _write_extrinsics_merged(output, transform)
    finally:
        camera_server.stop()
        _stop_robot_process(robot_process)
    print(f"Saved {transform.name} to {output}")


def run_stereo_extrinsics(args) -> None:
    config = load_runtime_config(args.config)
    pattern = _pattern_from_args(args)
    reference_intrinsics = load_intrinsics(
        _intrinsics_input_path(
            config,
            args.reference_camera,
            args.reference_intrinsics,
        )
    )
    intrinsics = load_intrinsics(
        _intrinsics_input_path(config, args.camera, args.intrinsics)
    )
    camera_server = _start_camera_server(config)
    try:
        reference_serial = camera_id_for_logical_name(
            config.camera.realsense_ids,
            args.reference_camera,
        )
        serial = camera_id_for_logical_name(config.camera.realsense_ids, args.camera)
        samples = capture_stereo_extrinsics_samples(
            reference_camera=camera_server.cameras[reference_serial],
            camera=camera_server.cameras[serial],
            pattern=pattern,
            reference_intrinsics=reference_intrinsics,
            camera_intrinsics=intrinsics,
            sample_count=args.samples,
            interval_s=args.interval,
        )
        transform = solve_stereo_extrinsics(
            samples,
            reference_camera=args.reference_camera,
            camera=args.camera,
        )
        output = _extrinsics_output_path(config, args.output)
        _write_extrinsics_merged(output, transform)
    finally:
        camera_server.stop()
    print(f"Saved {transform.name} to {output}")


def _add_capture_args(parser: argparse.ArgumentParser, default_samples: int) -> None:
    parser.add_argument("--samples", type=int, default=default_samples)
    parser.add_argument("--interval", type=float, default=1.0)


def _add_robot_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--use-existing-robot-server",
        action="store_true",
        help="Connect to an already running robot ZMQ server instead of launching one.",
    )


def _add_pattern_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pattern", choices=("charuco", "chessboard"), default="chessboard"
    )
    parser.add_argument("--chessboard-columns", type=int, default=7)
    parser.add_argument("--chessboard-rows", type=int, default=6)
    parser.add_argument("--chessboard-square-size", type=float, default=0.025)
    parser.add_argument("--charuco-squares-x", type=int, default=7)
    parser.add_argument("--charuco-squares-y", type=int, default=5)
    parser.add_argument("--charuco-square-length", type=float, default=0.04)
    parser.add_argument("--charuco-marker-length", type=float, default=0.03)
    parser.add_argument("--charuco-dictionary", default="DICT_4X4_50")


def _pattern_from_args(args):
    if args.pattern == "chessboard":
        return ChessboardPattern(
            columns=args.chessboard_columns,
            rows=args.chessboard_rows,
            square_size_m=args.chessboard_square_size,
        )
    return CharucoPattern(
        squares_x=args.charuco_squares_x,
        squares_y=args.charuco_squares_y,
        square_length_m=args.charuco_square_length,
        marker_length_m=args.charuco_marker_length,
        dictionary=args.charuco_dictionary,
    )


def _start_camera_server(config):
    from ur5e_control.hardware import RealSenseCameraServer

    return RealSenseCameraServer(
        hostname=config.camera.hostname,
        realsense_ids=config.camera.realsense_ids,
        fps=config.camera.fps,
        enable_depth=config.camera.enable_depth,
    )


def _robot_client(config, use_existing_server: bool):
    from ur5e_control.runtime.transport.robot_zmq import ZMQClientRobot

    if use_existing_server:
        return None, ZMQClientRobot(
            port=config.robot.zmq_port,
            host=config.robot.zmq_host,
        )

    process = Process(target=_launch_robot_server, args=(config,))
    atexit.register(_stop_robot_process, process)
    process.start()
    time.sleep(2)
    return process, ZMQClientRobot(
        port=config.robot.zmq_port, host=config.robot.zmq_host
    )


def _launch_robot_server(config) -> None:
    from ur5e_control.hardware import URRobot
    from ur5e_control.runtime.transport.robot_zmq import ZMQServerRobot

    robot = URRobot(
        robot_ip=config.robot.ip,
        gripper_port=config.gripper.port,
        gripper_position_tolerance=config.gripper.position_tolerance,
        gripper_min_command_interval_s=config.gripper.min_command_interval_s,
        gripper_position_read_interval_s=config.gripper.position_read_interval_s,
    )
    server = ZMQServerRobot(
        robot,
        port=config.robot.zmq_port,
        host=config.robot.zmq_host,
    )
    print(f"Starting robot server on port {config.robot.zmq_port}")
    server.serve()


def _stop_robot_process(process) -> None:
    if process is not None and process.is_alive():
        process.terminate()
        process.join(timeout=2)


def _intrinsics_input_path(config, camera: str, override: str | None) -> Path:
    if override is not None:
        return resolve_repo_path(override)
    try:
        return resolve_repo_path(config.camera.intrinsics_files[camera])
    except KeyError as exc:
        raise ValueError(
            f"no intrinsics file configured for camera '{camera}'"
        ) from exc


def _intrinsics_output_path(config, camera: str, override: str | None) -> Path:
    return _intrinsics_input_path(config, camera, override)


def _extrinsics_output_path(config, override: str | None) -> Path:
    return resolve_repo_path(override or config.camera.extrinsics_file)


def _write_extrinsics_merged(path: Path, transform) -> None:
    transforms = {}
    if path.exists():
        transforms = load_extrinsics(path)
    transforms[transform.name] = transform
    write_extrinsics(path, transforms)


def _load_optional_transform(path: str | None):
    if path is None:
        return identity_matrix()
    data = load_yaml(resolve_repo_path(path))
    if "matrix" in data:
        return data["matrix"]
    transforms = data.get("transforms")
    if isinstance(transforms, dict) and transforms:
        first = next(iter(transforms.values()))
        if "matrix" in first:
            return first["matrix"]
    raise ValueError(f"{path} must contain a transform matrix")


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.interval < 0:
        raise ValueError("--interval must be non-negative")
    args.func(args)


if __name__ == "__main__":
    main()
