"""Record a short MP4 from each configured RealSense camera."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from ur5e_control.runtime.config import (
    load_runtime_config,
    resolve_repo_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check UR5e RealSense camera streams")
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Hardware/runtime YAML config. Defaults to UR5E_HARDWARE_CONFIG "
            "or configs/hardware/ur5e_d455.yaml."
        ),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Recording duration in seconds.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Camera check FPS. Defaults to camera.fps from the runtime config.",
    )
    parser.add_argument(
        "--output-dir",
        default=".datasets/camera_check",
        help="Directory where timestamped MP4 checks are written.",
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    config = load_runtime_config(args.config)
    fps = args.fps if args.fps is not None else config.camera.fps
    if fps <= 0:
        raise ValueError("--fps must be positive")
    if args.duration <= 0:
        raise ValueError("--duration must be positive")
    output_dir = resolve_repo_path(
        Path(args.output_dir) / time.strftime("%Y%m%d-%H%M%S")
    )

    from ur5e_control.hardware import RealSenseCameraServer
    from ur5e_control.hardware.cameras.check import record_camera_check

    camera_server = RealSenseCameraServer(
        hostname=config.camera.hostname,
        realsense_ids=config.camera.realsense_ids,
        fps=fps,
        enable_depth=config.camera.enable_depth,
    )
    try:
        outputs = record_camera_check(
            cameras=camera_server.cameras,
            camera_name_id=camera_server.realsense_ids,
            output_dir=output_dir,
            duration_s=args.duration,
            fps=fps,
        )
    finally:
        camera_server.stop()

    print("Camera check videos:")
    for name, path in sorted(outputs.items()):
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
