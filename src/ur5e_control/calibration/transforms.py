"""SE(3) transform utilities used by calibration solvers."""

from __future__ import annotations

from collections.abc import Iterable
import math

import numpy as np


def identity_matrix() -> np.ndarray:
    return np.eye(4, dtype=float)


def ensure_transform_matrix(value, label: str = "transform") -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"{label} must have shape (4, 4), got {matrix.shape}")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise ValueError(f"{label} must be a homogeneous transform")
    return matrix


def compose_transforms(*transforms) -> np.ndarray:
    result = identity_matrix()
    for transform in transforms:
        result = result @ ensure_transform_matrix(transform)
    return result


def invert_transform(transform) -> np.ndarray:
    matrix = ensure_transform_matrix(transform)
    inverse = identity_matrix()
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def matrix_from_translation_quaternion(
    translation,
    quaternion_xyzw,
) -> np.ndarray:
    matrix = identity_matrix()
    matrix[:3, :3] = rotation_matrix_from_quaternion(quaternion_xyzw)
    matrix[:3, 3] = _vector(translation, 3, "translation")
    return matrix


def translation_quaternion_from_matrix(transform) -> tuple[np.ndarray, np.ndarray]:
    matrix = ensure_transform_matrix(transform)
    return matrix[:3, 3].copy(), quaternion_from_rotation_matrix(matrix[:3, :3])


def matrix_from_ur_pose_vector(pose) -> np.ndarray:
    values = _vector(pose, 6, "pose")
    return matrix_from_rvec_tvec(values[3:], values[:3])


def ur_pose_vector_from_matrix(transform) -> tuple[float, ...]:
    matrix = ensure_transform_matrix(transform)
    rotvec = rotvec_from_rotation_matrix(matrix[:3, :3])
    translation = matrix[:3, 3]
    return tuple(float(value) for value in (*translation, *rotvec))


def matrix_from_rvec_tvec(rvec, tvec) -> np.ndarray:
    matrix = identity_matrix()
    matrix[:3, :3] = rotation_matrix_from_rotvec(rvec)
    matrix[:3, 3] = _vector(tvec, 3, "tvec")
    return matrix


def rvec_tvec_from_matrix(transform) -> tuple[np.ndarray, np.ndarray]:
    matrix = ensure_transform_matrix(transform)
    return rotvec_from_rotation_matrix(matrix[:3, :3]), matrix[:3, 3].copy()


def average_transforms(transforms: Iterable[np.ndarray]) -> np.ndarray:
    matrices = [ensure_transform_matrix(transform) for transform in transforms]
    if not matrices:
        raise ValueError("transforms must contain at least one transform")

    translations = np.asarray([matrix[:3, 3] for matrix in matrices], dtype=float)
    quaternions = [quaternion_from_rotation_matrix(matrix[:3, :3]) for matrix in matrices]
    reference = quaternions[0]
    aligned = []
    for quaternion in quaternions:
        if float(np.dot(reference, quaternion)) < 0.0:
            quaternion = -quaternion
        aligned.append(quaternion)

    accumulator = np.zeros((4, 4), dtype=float)
    for quaternion in aligned:
        accumulator += np.outer(quaternion, quaternion)
    eigenvalues, eigenvectors = np.linalg.eigh(accumulator)
    quaternion = eigenvectors[:, int(np.argmax(eigenvalues))]
    if quaternion[3] < 0.0:
        quaternion = -quaternion

    return matrix_from_translation_quaternion(
        translation=np.mean(translations, axis=0),
        quaternion_xyzw=quaternion,
    )


def rotation_matrix_from_rotvec(rotvec) -> np.ndarray:
    vector = _vector(rotvec, 3, "rotvec")
    theta = float(np.linalg.norm(vector))
    if theta < 1e-12:
        return np.eye(3, dtype=float)
    axis = vector / theta
    skew = _skew(axis)
    return (
        np.eye(3, dtype=float)
        + math.sin(theta) * skew
        + (1.0 - math.cos(theta)) * (skew @ skew)
    )


def rotvec_from_rotation_matrix(rotation) -> np.ndarray:
    quaternion = quaternion_from_rotation_matrix(rotation)
    vector = quaternion[:3]
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return np.zeros(3, dtype=float)
    angle = 2.0 * math.atan2(norm, float(quaternion[3]))
    if angle > math.pi:
        angle -= 2.0 * math.pi
    return vector / norm * angle


def rotation_matrix_from_quaternion(quaternion_xyzw) -> np.ndarray:
    x, y, z, w = _normalized_quaternion(quaternion_xyzw)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def quaternion_from_rotation_matrix(rotation) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {matrix.shape}")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diagonal(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.asarray([x, y, z, w], dtype=float)
    quaternion = quaternion / np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return quaternion


def _normalized_quaternion(quaternion_xyzw) -> np.ndarray:
    quaternion = _vector(quaternion_xyzw, 4, "quaternion_xyzw")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ValueError("quaternion_xyzw must be non-zero")
    return quaternion / norm


def _vector(value, length: int, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.shape != (length,):
        raise ValueError(f"{label} must contain {length} values")
    return vector


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.asarray(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )
