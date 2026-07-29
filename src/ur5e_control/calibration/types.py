"""Data structures shared by camera calibration modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _array(value, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}, got {array.shape}")
    return array


@dataclass(frozen=True)
class CalibrationTransform:
    """A named SE(3) transform between two frames."""

    name: str
    parent_frame: str
    child_frame: str
    matrix: np.ndarray
    sample_count: int
    error: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "matrix", _array(self.matrix, (4, 4), "matrix"))
        if self.sample_count < 0:
            raise ValueError("sample_count must be non-negative")
        if not self.name:
            raise ValueError("name must be set")
        if not self.parent_frame:
            raise ValueError("parent_frame must be set")
        if not self.child_frame:
            raise ValueError("child_frame must be set")


@dataclass(frozen=True)
class CameraIntrinsics:
    """OpenCV camera intrinsics and calibration quality metadata."""

    camera_name: str
    serial_number: str | None
    image_size: tuple[int, int]
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    reprojection_error: float
    sample_count: int
    model: str = "brown_conrady"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "camera_matrix",
            _array(self.camera_matrix, (3, 3), "camera_matrix"),
        )
        coefficients = np.asarray(self.distortion_coefficients, dtype=float).reshape(-1)
        object.__setattr__(self, "distortion_coefficients", coefficients)
        if len(self.image_size) != 2:
            raise ValueError("image_size must be (width, height)")
        width, height = (int(self.image_size[0]), int(self.image_size[1]))
        if width <= 0 or height <= 0:
            raise ValueError("image_size values must be positive")
        object.__setattr__(self, "image_size", (width, height))
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if self.reprojection_error < 0:
            raise ValueError("reprojection_error must be non-negative")


@dataclass(frozen=True)
class IntrinsicsSample:
    """Detected target points for one calibration image."""

    object_points: np.ndarray
    image_points: np.ndarray
    image_size: tuple[int, int]

    def __post_init__(self) -> None:
        object_points = np.asarray(self.object_points, dtype=np.float32)
        image_points = np.asarray(self.image_points, dtype=np.float32)
        if object_points.ndim != 2 or object_points.shape[1] != 3:
            raise ValueError("object_points must have shape (N, 3)")
        if image_points.ndim != 2 or image_points.shape[1] != 2:
            raise ValueError("image_points must have shape (N, 2)")
        if object_points.shape[0] != image_points.shape[0]:
            raise ValueError("object_points and image_points must have equal length")
        object.__setattr__(self, "object_points", object_points)
        object.__setattr__(self, "image_points", image_points)
        width, height = (int(self.image_size[0]), int(self.image_size[1]))
        object.__setattr__(self, "image_size", (width, height))


@dataclass(frozen=True)
class HandEyeSample:
    """One wrist hand-eye sample."""

    base_T_tool: np.ndarray
    camera_T_target: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_T_tool",
            _array(self.base_T_tool, (4, 4), "base_T_tool"),
        )
        object.__setattr__(
            self,
            "camera_T_target",
            _array(self.camera_T_target, (4, 4), "camera_T_target"),
        )


@dataclass(frozen=True)
class BaseExtrinsicsSample:
    """One fixed-camera-to-base extrinsics sample."""

    base_T_tool: np.ndarray
    camera_T_target: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_T_tool",
            _array(self.base_T_tool, (4, 4), "base_T_tool"),
        )
        object.__setattr__(
            self,
            "camera_T_target",
            _array(self.camera_T_target, (4, 4), "camera_T_target"),
        )


@dataclass(frozen=True)
class StereoExtrinsicsSample:
    """One synchronized dual-camera target observation."""

    reference_camera_T_target: np.ndarray
    camera_T_target: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reference_camera_T_target",
            _array(
                self.reference_camera_T_target,
                (4, 4),
                "reference_camera_T_target",
            ),
        )
        object.__setattr__(
            self,
            "camera_T_target",
            _array(self.camera_T_target, (4, 4), "camera_T_target"),
        )


def matrix_sequence(values: Sequence[np.ndarray], label: str) -> list[np.ndarray]:
    """Validate a sequence of homogeneous transforms."""

    matrices = [_array(value, (4, 4), label) for value in values]
    if not matrices:
        raise ValueError(f"{label} must contain at least one transform")
    return matrices
