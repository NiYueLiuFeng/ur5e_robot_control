"""Live capture helpers for camera calibration commands."""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping

from ur5e_control.calibration.patterns import (
    CharucoPattern,
    ChessboardPattern,
    detect_charuco,
    detect_chessboard,
)
from ur5e_control.calibration.solvers import estimate_target_pose
from ur5e_control.calibration.transforms import matrix_from_ur_pose_vector
from ur5e_control.calibration.types import (
    BaseExtrinsicsSample,
    CameraIntrinsics,
    HandEyeSample,
    IntrinsicsSample,
    StereoExtrinsicsSample,
)

SleepFn = Callable[[float], None]


def camera_id_for_logical_name(
    realsense_ids: Mapping[str, str],
    logical_name: str,
) -> str:
    for serial_number, name in realsense_ids.items():
        if name == logical_name:
            return serial_number
    known = ", ".join(sorted(realsense_ids.values()))
    raise ValueError(f"unknown camera '{logical_name}'. Known logical cameras: {known}")


def capture_intrinsics_samples(
    camera: Any,
    pattern: ChessboardPattern | CharucoPattern,
    sample_count: int,
    interval_s: float,
    sleep: SleepFn = time.sleep,
) -> list[IntrinsicsSample]:
    samples: list[IntrinsicsSample] = []
    attempts = 0
    max_attempts = max(sample_count * 8, sample_count)
    while len(samples) < sample_count and attempts < max_attempts:
        attempts += 1
        image, _depth = camera.read()
        detection = detect_pattern(image, pattern)
        if detection is not None:
            samples.append(detection.as_intrinsics_sample())
            print(f"Captured intrinsics sample {len(samples)}/{sample_count}")
        else:
            print("No calibration target detected; skipping frame")
        sleep(interval_s)
    if len(samples) < sample_count:
        raise RuntimeError(
            f"captured {len(samples)} valid samples, need {sample_count}"
        )
    return samples


def capture_hand_eye_samples(
    camera: Any,
    robot: Any,
    pattern: ChessboardPattern | CharucoPattern,
    intrinsics: CameraIntrinsics,
    sample_count: int,
    interval_s: float,
    sleep: SleepFn = time.sleep,
) -> list[HandEyeSample]:
    samples: list[HandEyeSample] = []
    for camera_T_target, base_T_tool in _target_and_tool_pose_stream(
        camera,
        robot,
        pattern,
        intrinsics,
        sample_count,
        interval_s,
        sleep,
    ):
        samples.append(
            HandEyeSample(
                base_T_tool=base_T_tool,
                camera_T_target=camera_T_target,
            )
        )
    return samples


def capture_base_extrinsics_samples(
    camera: Any,
    robot: Any,
    pattern: ChessboardPattern | CharucoPattern,
    intrinsics: CameraIntrinsics,
    sample_count: int,
    interval_s: float,
    sleep: SleepFn = time.sleep,
) -> list[BaseExtrinsicsSample]:
    samples: list[BaseExtrinsicsSample] = []
    for camera_T_target, base_T_tool in _target_and_tool_pose_stream(
        camera,
        robot,
        pattern,
        intrinsics,
        sample_count,
        interval_s,
        sleep,
    ):
        samples.append(
            BaseExtrinsicsSample(
                base_T_tool=base_T_tool,
                camera_T_target=camera_T_target,
            )
        )
    return samples


def capture_stereo_extrinsics_samples(
    reference_camera: Any,
    camera: Any,
    pattern: ChessboardPattern | CharucoPattern,
    reference_intrinsics: CameraIntrinsics,
    camera_intrinsics: CameraIntrinsics,
    sample_count: int,
    interval_s: float,
    sleep: SleepFn = time.sleep,
) -> list[StereoExtrinsicsSample]:
    samples: list[StereoExtrinsicsSample] = []
    attempts = 0
    max_attempts = max(sample_count * 8, sample_count)
    while len(samples) < sample_count and attempts < max_attempts:
        attempts += 1
        reference_image, _reference_depth = reference_camera.read()
        image, _depth = camera.read()
        reference_T_target = detect_target_pose(
            reference_image,
            pattern,
            reference_intrinsics,
        )
        camera_T_target = detect_target_pose(image, pattern, camera_intrinsics)
        if reference_T_target is not None and camera_T_target is not None:
            samples.append(
                StereoExtrinsicsSample(
                    reference_camera_T_target=reference_T_target,
                    camera_T_target=camera_T_target,
                )
            )
            print(f"Captured stereo sample {len(samples)}/{sample_count}")
        else:
            print("Target not visible in both cameras; skipping frame")
        sleep(interval_s)
    if len(samples) < sample_count:
        raise RuntimeError(
            f"captured {len(samples)} valid samples, need {sample_count}"
        )
    return samples


def detect_target_pose(
    image: Any,
    pattern: ChessboardPattern | CharucoPattern,
    intrinsics: CameraIntrinsics,
):
    detection = detect_pattern(image, pattern)
    if detection is None:
        return None
    return estimate_target_pose(
        object_points=detection.object_points,
        image_points=detection.image_points,
        camera_matrix=intrinsics.camera_matrix,
        distortion_coefficients=intrinsics.distortion_coefficients,
    )


def detect_pattern(image: Any, pattern: ChessboardPattern | CharucoPattern):
    if isinstance(pattern, ChessboardPattern):
        return detect_chessboard(image, pattern)
    return detect_charuco(image, pattern)


def robot_tcp_pose_matrix(robot: Any):
    observations = robot.get_observations()
    if "tcp_pose" not in observations:
        raise RuntimeError(
            "robot observations do not include 'tcp_pose'. "
            "Update URRobot.get_observations before running camera calibration."
        )
    return matrix_from_ur_pose_vector(observations["tcp_pose"])


def _target_and_tool_pose_stream(
    camera: Any,
    robot: Any,
    pattern: ChessboardPattern | CharucoPattern,
    intrinsics: CameraIntrinsics,
    sample_count: int,
    interval_s: float,
    sleep: SleepFn,
):
    captured = 0
    attempts = 0
    max_attempts = max(sample_count * 8, sample_count)
    while captured < sample_count and attempts < max_attempts:
        attempts += 1
        image, _depth = camera.read()
        camera_T_target = detect_target_pose(image, pattern, intrinsics)
        if camera_T_target is None:
            print("No calibration target detected; skipping frame")
            sleep(interval_s)
            continue
        captured += 1
        print(f"Captured robot-camera sample {captured}/{sample_count}")
        yield camera_T_target, robot_tcp_pose_matrix(robot)
        sleep(interval_s)
    if captured < sample_count:
        raise RuntimeError(f"captured {captured} valid samples, need {sample_count}")
