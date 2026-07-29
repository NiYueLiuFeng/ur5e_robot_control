"""Calibration solvers for intrinsics and extrinsics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ur5e_control.calibration.transforms import (
    average_transforms,
    compose_transforms,
    identity_matrix,
    invert_transform,
    matrix_from_rvec_tvec,
)
from ur5e_control.calibration.types import (
    BaseExtrinsicsSample,
    CalibrationTransform,
    CameraIntrinsics,
    HandEyeSample,
    IntrinsicsSample,
    StereoExtrinsicsSample,
)


def solve_base_extrinsics(
    samples: Sequence[BaseExtrinsicsSample],
    tool_T_target=None,
    parent_frame: str = "base",
    camera_frame: str = "camera",
) -> CalibrationTransform:
    """Estimate a fixed camera pose in the robot base frame."""

    _require_samples(samples, 3, "base extrinsics")
    tool_T_target = identity_matrix() if tool_T_target is None else tool_T_target
    base_T_cameras = [
        compose_transforms(
            sample.base_T_tool,
            tool_T_target,
            invert_transform(sample.camera_T_target),
        )
        for sample in samples
    ]
    return CalibrationTransform(
        name=f"{parent_frame}_T_{camera_frame}",
        parent_frame=parent_frame,
        child_frame=camera_frame,
        matrix=average_transforms(base_T_cameras),
        sample_count=len(samples),
        error=_mean_transform_error(base_T_cameras),
    )


def solve_stereo_extrinsics(
    samples: Sequence[StereoExtrinsicsSample],
    reference_camera: str,
    camera: str,
) -> CalibrationTransform:
    """Estimate the transform from `reference_camera` to `camera`."""

    _require_samples(samples, 3, "stereo extrinsics")
    reference_T_cameras = [
        compose_transforms(
            sample.reference_camera_T_target,
            invert_transform(sample.camera_T_target),
        )
        for sample in samples
    ]
    return CalibrationTransform(
        name=f"{reference_camera}_T_{camera}",
        parent_frame=reference_camera,
        child_frame=camera,
        matrix=average_transforms(reference_T_cameras),
        sample_count=len(samples),
        error=_mean_transform_error(reference_T_cameras),
    )


def solve_hand_eye(
    samples: Sequence[HandEyeSample],
    parent_frame: str = "tool",
    camera_frame: str = "camera",
    method: str = "tsai",
) -> CalibrationTransform:
    """Estimate wrist camera pose in the robot tool frame."""

    _require_samples(samples, 3, "hand-eye calibration")

    import cv2

    method_value = _hand_eye_method(cv2, method)
    rotations_gripper_to_base = [sample.base_T_tool[:3, :3] for sample in samples]
    translations_gripper_to_base = [sample.base_T_tool[:3, 3] for sample in samples]
    rotations_target_to_camera = [sample.camera_T_target[:3, :3] for sample in samples]
    translations_target_to_camera = [sample.camera_T_target[:3, 3] for sample in samples]
    rotation, translation = cv2.calibrateHandEye(
        rotations_gripper_to_base,
        translations_gripper_to_base,
        rotations_target_to_camera,
        translations_target_to_camera,
        method=method_value,
    )
    matrix = identity_matrix()
    matrix[:3, :3] = np.asarray(rotation, dtype=float)
    matrix[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return CalibrationTransform(
        name=f"{parent_frame}_T_{camera_frame}",
        parent_frame=parent_frame,
        child_frame=camera_frame,
        matrix=matrix,
        sample_count=len(samples),
    )


def solve_intrinsics(
    samples: Sequence[IntrinsicsSample],
    camera_name: str,
    serial_number: str | None = None,
    model: str = "brown_conrady",
) -> CameraIntrinsics:
    """Solve camera intrinsics from detected 2D/3D target correspondences."""

    _require_samples(samples, 3, "intrinsics calibration")
    image_size = samples[0].image_size
    for sample in samples:
        if sample.image_size != image_size:
            raise ValueError("all intrinsics samples must have the same image_size")

    import cv2

    object_points = [
        sample.object_points.astype(np.float32).reshape(-1, 1, 3)
        for sample in samples
    ]
    image_points = [
        sample.image_points.astype(np.float32).reshape(-1, 1, 2)
        for sample in samples
    ]
    error, camera_matrix, distortion, _rvecs, _tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    return CameraIntrinsics(
        camera_name=camera_name,
        serial_number=serial_number,
        image_size=image_size,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion.reshape(-1),
        reprojection_error=float(error),
        sample_count=len(samples),
        model=model,
    )


def estimate_target_pose(
    object_points,
    image_points,
    camera_matrix,
    distortion_coefficients,
) -> np.ndarray:
    """Estimate `camera_T_target` with OpenCV PnP."""

    import cv2

    ok, rvec, tvec = cv2.solvePnP(
        np.asarray(object_points, dtype=np.float32),
        np.asarray(image_points, dtype=np.float32),
        np.asarray(camera_matrix, dtype=float),
        np.asarray(distortion_coefficients, dtype=float),
    )
    if not ok:
        raise RuntimeError("cv2.solvePnP failed for calibration target")
    return matrix_from_rvec_tvec(rvec.reshape(3), tvec.reshape(3))


def _require_samples(samples: Sequence[object], minimum: int, label: str) -> None:
    if len(samples) < minimum:
        raise ValueError(f"{label} requires at least {minimum} samples")


def _hand_eye_method(cv2, method: str) -> int:
    methods = {
        "tsai": cv2.CALIB_HAND_EYE_TSAI,
        "park": cv2.CALIB_HAND_EYE_PARK,
        "horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    try:
        return methods[method.lower()]
    except KeyError as exc:
        names = ", ".join(sorted(methods))
        raise ValueError(f"hand-eye method must be one of: {names}") from exc


def _mean_transform_error(matrices: Sequence[np.ndarray]) -> float:
    if len(matrices) <= 1:
        return 0.0
    mean = average_transforms(matrices)
    translation_errors = [
        float(np.linalg.norm(matrix[:3, 3] - mean[:3, 3])) for matrix in matrices
    ]
    return float(np.mean(translation_errors))
