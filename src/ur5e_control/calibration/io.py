"""YAML serialization for camera calibration results."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from ur5e_control.calibration.transforms import translation_quaternion_from_matrix
from ur5e_control.calibration.types import CalibrationTransform, CameraIntrinsics


def transform_to_mapping(transform: CalibrationTransform) -> dict:
    translation, quaternion = translation_quaternion_from_matrix(transform.matrix)
    data = {
        "name": transform.name,
        "parent_frame": transform.parent_frame,
        "child_frame": transform.child_frame,
        "matrix": _nested_float_list(transform.matrix),
        "translation": _float_list(translation),
        "quaternion_xyzw": _float_list(quaternion),
        "sample_count": int(transform.sample_count),
    }
    if transform.error is not None:
        data["error"] = _clean_float(transform.error)
    return data


def intrinsics_to_mapping(intrinsics: CameraIntrinsics) -> dict:
    return {
        "camera_name": intrinsics.camera_name,
        "serial_number": intrinsics.serial_number,
        "image_size": [int(intrinsics.image_size[0]), int(intrinsics.image_size[1])],
        "camera_matrix": _nested_float_list(intrinsics.camera_matrix),
        "distortion_coefficients": _float_list(intrinsics.distortion_coefficients),
        "model": intrinsics.model,
        "reprojection_error": _clean_float(intrinsics.reprojection_error),
        "sample_count": int(intrinsics.sample_count),
    }


def write_intrinsics(path: str | Path, intrinsics: CameraIntrinsics) -> Path:
    return write_yaml(path, intrinsics_to_mapping(intrinsics))


def load_intrinsics(path: str | Path) -> CameraIntrinsics:
    data = load_yaml(path)
    return CameraIntrinsics(
        camera_name=str(data["camera_name"]),
        serial_number=data.get("serial_number"),
        image_size=tuple(data["image_size"]),
        camera_matrix=np.asarray(data["camera_matrix"], dtype=float),
        distortion_coefficients=np.asarray(
            data["distortion_coefficients"],
            dtype=float,
        ),
        reprojection_error=float(data["reprojection_error"]),
        sample_count=int(data["sample_count"]),
        model=str(data.get("model", "brown_conrady")),
    )


def write_extrinsics(
    path: str | Path,
    transforms: Mapping[str, CalibrationTransform] | list[CalibrationTransform],
) -> Path:
    if isinstance(transforms, Mapping):
        transform_items = {
            name: transform_to_mapping(transform)
            for name, transform in transforms.items()
        }
    else:
        transform_items = {
            transform.name: transform_to_mapping(transform)
            for transform in transforms
        }
    return write_yaml(path, {"transforms": transform_items})


def load_extrinsics(path: str | Path) -> dict[str, CalibrationTransform]:
    data = load_yaml(path)
    raw_transforms = data.get("transforms", {})
    if not isinstance(raw_transforms, Mapping):
        raise ValueError("extrinsics YAML must contain a 'transforms' mapping")
    return {
        str(name): transform_from_mapping(mapping)
        for name, mapping in raw_transforms.items()
    }


def transform_from_mapping(mapping: Mapping) -> CalibrationTransform:
    return CalibrationTransform(
        name=str(mapping["name"]),
        parent_frame=str(mapping["parent_frame"]),
        child_frame=str(mapping["child_frame"]),
        matrix=np.asarray(mapping["matrix"], dtype=float),
        sample_count=int(mapping.get("sample_count", 0)),
        error=(
            None
            if mapping.get("error") is None
            else float(mapping["error"])
        ),
    )


def write_yaml(path: str | Path, data: Mapping) -> Path:
    import yaml

    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(dict(data), sort_keys=False),
        encoding="utf-8",
    )
    return output_path


def load_yaml(path: str | Path) -> Mapping:
    import yaml

    input_path = Path(path).expanduser()
    loaded = yaml.safe_load(input_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError(f"{input_path} must contain a YAML mapping")
    return loaded


def _nested_float_list(array) -> list:
    return [
        [_clean_float(value) for value in row]
        for row in np.asarray(array, dtype=float).tolist()
    ]


def _float_list(array) -> list[float]:
    return [_clean_float(value) for value in np.asarray(array, dtype=float).reshape(-1)]


def _clean_float(value) -> float:
    rounded = round(float(value), 12)
    if rounded == -0.0:
        return 0.0
    return rounded
