"""LeRobot v3 recording adapter.

The real LeRobot dependency is imported lazily so this repository can be tested
without robot hardware or the dataset stack installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import json
from pathlib import Path
import queue
import threading
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
import numpy as np
from ur5e_control.runtime.types import RobotAction, RobotObservation, TaskInfo

DatasetFactory = Callable[..., Any]

DEFAULT_IMAGE_SHAPES = {
    "wrist": (480, 640, 3),
    "third_person": (480, 640, 3),
}

UR5E_DOF_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow.pos",
    "wrist_1.pos",
    "wrist_2.pos",
    "wrist_3.pos",
    "gripper.pos",
)

TCP_FORCE_NAMES = (
    "force.x",
    "force.y",
    "force.z",
    "torque.x",
    "torque.y",
    "torque.z",
)

TCP_POSE_NAMES = (
    "translation.x",
    "translation.y",
    "translation.z",
    "rotation_vector.x",
    "rotation_vector.y",
    "rotation_vector.z",
)

GLOBAL_QUANTILES = {
    "q01": 0.01,
    "q10": 0.10,
    "q50": 0.50,
    "q90": 0.90,
    "q99": 0.99,
}


@dataclass(frozen=True)
class LeRobotV3RecorderConfig:
    repo_id: str
    root: Optional[str] = None
    fps: int = 20
    robot_type: str = "ur5e_pgc140_d455"
    state_dim: int = 7
    action_dim: int = 7
    image_shapes: Mapping[str, Tuple[int, int, int]] = field(
        default_factory=lambda: dict(DEFAULT_IMAGE_SHAPES)
    )
    use_videos: bool = True
    image_writer_processes: int = 0
    image_writer_threads: int = 4
    batch_encoding_size: int = 1
    streaming_encoding: bool = False
    encoder_threads: Optional[int] = None
    video_files_size_in_mb: Optional[int] = None
    data_files_size_in_mb: Optional[int] = None
    parallel_encoding: bool = True
    gripper_threadhold: int = 700
    recompute_exact_stats: bool = True
    repair_video_timestamps: bool = False
    depth_min_m: float | None = None
    camera_calibration: Mapping[str, Any] | None = None


class LeRobotV3Recorder:
    """Episode recorder backed by `LeRobotDataset.create()`."""

    def __init__(
        self,
        config: LeRobotV3RecorderConfig,
        dataset_factory: DatasetFactory = None,  # type: ignore
    ):
        self.config = config
        camera_calibration = config.camera_calibration
        self.camera_calibration = camera_calibration
        self.camera_calibration_frame = build_camera_calibration_frame(
            camera_calibration
        )
        self.dataset = _create_dataset(
            config,
            dataset_factory,
            camera_calibration=camera_calibration,
        )
        self.current_task: Optional[TaskInfo] = None

    def start_episode(self, task: TaskInfo) -> None:
        self.current_task = task

    def add_frame(
        self,
        observation: RobotObservation,
        action: RobotAction,
        subtask: Optional[str] = None,
    ) -> None:
        if self.current_task is None:
            raise RuntimeError("start_episode() must be called before add_frame()")

        frame = build_lerobot_frame(
            observation=observation,
            action=action,
            task=self.current_task,
            subtask=subtask,
            include_subtask_text=False,
            gripper_threadhold=self.config.gripper_threadhold,
            camera_calibration_frame=self.camera_calibration_frame,
        )
        self.dataset.add_frame(frame)

    def save_episode(self) -> str:
        result = _save_episode(
            self.dataset, parallel_encoding=self.config.parallel_encoding
        )
        return "" if result is None else str(result)

    def finalize(self) -> None:
        if getattr(self, "_finalized", False):
            return
        self.dataset.finalize()
        dataset_root = getattr(self.dataset, "root", None)
        if self.config.repair_video_timestamps and dataset_root is not None:
            from ur5e_control.data_collection.video_timestamps import (
                repair_lerobot_video_timestamps,
            )

            report = repair_lerobot_video_timestamps(
                dataset_root,
                apply=True,
                backup=False,
            )
            print(
                "Depth video timestamp repair: "
                f"{report.changed_episodes}/{report.episodes} episode entries "
                f"updated; max correction "
                f"{report.max_metadata_correction_s * 1000:.3f} ms"
            )
        if self.config.recompute_exact_stats:
            if dataset_root is not None:
                recompute_exact_dataset_stats(dataset_root)
        self._finalized = True


class AsyncLeRobotV3Recorder:
    """Bounded-queue wrapper that keeps LeRobot writes off the control loop."""

    _SENTINEL = object()

    def __init__(self, recorder: Any, max_queue_size: int = 64):
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        self._recorder = recorder
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max_queue_size)
        self._error: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    @property
    def config(self) -> Any:
        return getattr(self._recorder, "config", None)

    def start_episode(self, task: TaskInfo) -> None:
        self._raise_if_error()
        self._recorder.start_episode(task)

    def add_frame(
        self,
        observation: RobotObservation,
        action: RobotAction,
        subtask: Optional[str] = None,
    ) -> None:
        self._raise_if_error()
        if self._closed:
            raise RuntimeError("AsyncLeRobotV3Recorder is closed")
        self._queue.put((observation, action, subtask))

    def save_episode(self) -> str:
        self._queue.join()
        self._raise_if_error()
        return self._recorder.save_episode()

    def finalize(self) -> None:
        try:
            self._queue.join()
            self._raise_if_error()
            self.close()
            self._recorder.finalize()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(self._SENTINEL)
        self._queue.join()
        self._thread.join(timeout=2)

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._SENTINEL:
                    return
                if self._error is None:
                    observation, action, subtask = item
                    self._recorder.add_frame(observation, action, subtask=subtask)
            except BaseException as exc:  # noqa: BLE001
                self._error = exc
            finally:
                self._queue.task_done()

    def _raise_if_error(self) -> None:
        if self._error is not None:
            raise self._error


def binarize_gripper_position(value: float, threshold: float = 700) -> float:
    """Return a binary gripper value without corrupting an already-binary input."""

    numeric_value = float(value)
    if numeric_value in (0.0, 1.0):
        return numeric_value
    return float(numeric_value > threshold)


def build_lerobot_frame(
    observation: RobotObservation,
    action: RobotAction,
    task: TaskInfo,
    subtask: Optional[str] = None,
    include_subtask_text: bool = True,
    gripper_threadhold: int = 700,
    camera_calibration_frame: Mapping[str, np.ndarray] | None = None,
) -> Dict[str, Any]:
    """Build a LeRobotDataset.add_frame-compatible frame."""
    frame: Dict[str, Any] = {
        "task": task.name,
        "tcp_force": np.array(observation.tcp_force, dtype=np.float32),
        "observation.tcp_pose": np.array(observation.tcp_pose, dtype=np.float32),
        "observation.state": np.array(observation.joint_positions, dtype=np.float32),
        "action": np.array(action.values, dtype=np.float32),
    }

    frame["action"][-1] = binarize_gripper_position(
        frame["action"][-1], gripper_threadhold
    )
    frame["observation.state"][-1] = binarize_gripper_position(
        frame["observation.state"][-1], gripper_threadhold
    )

    if observation.timestamp is not None:
        frame["timestamp"] = observation.timestamp

    if subtask is not None:
        subtask_index = _subtask_index(task, subtask)
        frame["subtask_index"] = np.array([subtask_index], dtype=np.int64)
        if include_subtask_text:
            frame["subtask"] = subtask

    for camera_name, image in observation.images.items():
        frame[f"observation.images.{camera_name}"] = image

    if camera_calibration_frame:
        frame.update(
            {
                feature_name: np.asarray(value).copy()
                for feature_name, value in camera_calibration_frame.items()
            }
        )

    return frame


def build_lerobot_features(
    state_dim: int = 7,
    action_dim: int = 7,
    tcp_force_dim: int = 6,
    tcp_pose_dim: int = 6,
    image_shapes: Mapping[str, Tuple[int, int, int]] = None,  # type: ignore
    use_videos: bool = True,
    camera_calibration: Mapping[str, Any] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Return the feature schema passed to `LeRobotDataset.create()`."""

    image_shapes = image_shapes or DEFAULT_IMAGE_SHAPES
    visual_dtype = "video" if use_videos else "image"
    features: Dict[str, Dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": _feature_names(UR5E_DOF_NAMES, state_dim, "state"),
        },
        "tcp_force": {
            "dtype": "float32",
            "shape": (tcp_force_dim,),
            "names": _feature_names(TCP_FORCE_NAMES, tcp_force_dim, "tcp_force"),
        },
        "observation.tcp_pose": {
            "dtype": "float32",
            "shape": (tcp_pose_dim,),
            "names": _feature_names(TCP_POSE_NAMES, tcp_pose_dim, "tcp_pose"),
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": _feature_names(UR5E_DOF_NAMES, action_dim, "action"),
        },
        "subtask_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["subtask_index"],
        },
    }

    for camera_name, shape in image_shapes.items():
        image_shape = tuple(int(value) for value in shape)
        if len(image_shape) != 3:
            raise ValueError(
                f"visual feature {camera_name!r} must have a 3D HWC or CHW shape, "
                f"got {image_shape}"
            )
        if image_shape[0] in (1, 3) and image_shape[-1] not in (1, 3):
            image_shape = (image_shape[1], image_shape[2], image_shape[0])
        channels = image_shape[-1]
        if channels not in (1, 3):
            raise ValueError(
                f"visual feature {camera_name!r} must have 1 or 3 channels, "
                f"got shape {image_shape}"
            )
        features[f"observation.images.{camera_name}"] = {
            "dtype": visual_dtype,
            "shape": image_shape,
            "names": ["height", "width", "channel"],
            "info": {"is_depth_map": channels == 1},
        }

    features.update(build_camera_calibration_features(camera_calibration))

    return features


def recompute_exact_dataset_stats(dataset_root: str | Path) -> Dict[str, Any]:
    """Recompute global numeric stats directly from all dataset frames."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "pandas is required to recompute LeRobot dataset statistics"
        ) from exc

    root = Path(dataset_root)
    info_path = root / "meta" / "info.json"
    stats_path = root / "meta" / "stats.json"
    parquet_paths = sorted((root / "data").glob("*/*.parquet"))
    if not parquet_paths:
        raise ValueError(f"no parquet files found under {root / 'data'}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    existing_stats = (
        json.loads(stats_path.read_text(encoding="utf-8"))
        if stats_path.exists()
        else {}
    )
    stats = {
        key: value
        for key, value in existing_stats.items()
        if info["features"].get(key, {}).get("dtype") in {"image", "video"}
    }

    numeric_features = {
        key: feature
        for key, feature in info["features"].items()
        if feature["dtype"] not in {"image", "video", "string"}
    }
    for key in numeric_features:
        chunks = []
        for parquet_path in parquet_paths:
            column = pd.read_parquet(parquet_path, columns=[key])[key]
            if column.empty:
                continue
            first_value = column.iloc[0]
            if np.isscalar(first_value):
                chunks.append(column.to_numpy()[:, None])
            else:
                chunks.append(np.stack(column.to_numpy()))

        if not chunks:
            continue
        values = np.concatenate(chunks, axis=0)
        stats[key] = _compute_exact_feature_stats(values)

    temporary_path = stats_path.with_name(f"{stats_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(stats, indent=4, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(stats_path)
    return stats


def build_camera_calibration_features(
    calibration: Mapping[str, Any] | None,
) -> Dict[str, Dict[str, Any]]:
    """Build LeRobot feature definitions for camera calibration parameters."""

    return {
        feature_name: {
            "dtype": str(value.dtype),
            "shape": value.shape,
            "names": _calibration_feature_names(feature_name, value),
        }
        for feature_name, value in build_camera_calibration_frame(
            calibration
        ).items()
    }


def build_camera_calibration_frame(
    calibration: Mapping[str, Any] | None,
) -> Dict[str, np.ndarray]:
    """Build the static calibration columns added to every LeRobot frame."""

    if calibration is None:
        return {}

    frame: Dict[str, np.ndarray] = {}
    raw_intrinsics = calibration.get("intrinsics", {})
    if not isinstance(raw_intrinsics, Mapping):
        raise ValueError("camera calibration intrinsics must be a mapping")
    for camera_name, raw_values in sorted(raw_intrinsics.items()):
        if not isinstance(raw_values, Mapping):
            raise ValueError(f"intrinsics for camera {camera_name!r} must be a mapping")
        dataset_camera_name = _dataset_camera_name(calibration, str(camera_name))
        prefix = f"observation.camera_intrinsics.{dataset_camera_name}"
        frame[f"{prefix}.camera_matrix"] = _calibration_array(
            raw_values.get("camera_matrix"),
            dtype=np.float32,
            shape=(3, 3),
            label=f"{camera_name} camera_matrix",
        )
        distortion = np.asarray(
            raw_values.get("distortion_coefficients"), dtype=np.float32
        ).reshape(-1)
        if distortion.size == 0 or not np.isfinite(distortion).all():
            raise ValueError(
                f"{camera_name} distortion_coefficients must contain finite values"
            )
        frame[f"{prefix}.distortion_coefficients"] = distortion
        # LeRobot currently classifies any feature whose key contains
        # "image" as image data when validating episode statistics.  Keep the
        # source calibration name (image_size), but use a neutral dataset
        # feature name so this two-element vector is treated as numeric data.
        frame[f"{prefix}.resolution"] = _calibration_array(
            raw_values.get("image_size"),
            dtype=np.int64,
            shape=(2,),
            label=f"{camera_name} image_size",
        )

    raw_extrinsics = calibration.get("extrinsics", {})
    if not isinstance(raw_extrinsics, Mapping):
        raise ValueError("camera calibration extrinsics must be a mapping")
    transforms = raw_extrinsics.get("transforms", {})
    if not isinstance(transforms, Mapping):
        raise ValueError("camera calibration transforms must be a mapping")
    for transform_name, raw_transform in sorted(transforms.items()):
        if not isinstance(raw_transform, Mapping):
            raise ValueError(
                f"extrinsics transform {transform_name!r} must be a mapping"
            )
        frame[f"observation.camera_extrinsics.{transform_name}.matrix"] = (
            _calibration_array(
                raw_transform.get("matrix"),
                dtype=np.float32,
                shape=(4, 4),
                label=f"{transform_name} matrix",
            )
        )
    return frame


def _dataset_camera_name(
    calibration: Mapping[str, Any],
    camera_name: str,
) -> str:
    image_features = calibration.get("image_features", {})
    if not isinstance(image_features, Mapping):
        raise ValueError("camera calibration image_features must be a mapping")
    image_feature = str(
        image_features.get(camera_name, f"observation.images.{camera_name}")
    )
    prefix = "observation.images."
    if not image_feature.startswith(prefix) or len(image_feature) == len(prefix):
        raise ValueError(
            f"camera image feature {image_feature!r} must start with {prefix!r}"
        )
    return image_feature[len(prefix) :]


def _calibration_array(
    value: Any,
    *,
    dtype: Any,
    shape: tuple[int, ...],
    label: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain finite values")
    return array


def _calibration_feature_names(
    feature_name: str,
    value: np.ndarray,
) -> list[str] | None:
    if feature_name.endswith(".resolution"):
        return ["width", "height"]
    if feature_name.endswith(".distortion_coefficients"):
        return [f"coefficient_{index}" for index in range(value.shape[0])]
    return None


def _compute_exact_feature_stats(values: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    if not np.isfinite(values).all():
        raise ValueError(
            "cannot compute dataset statistics from NaN or infinite values"
        )

    stats = {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "count": [int(values.shape[0])],
    }
    stats.update(
        {
            name: np.quantile(values, probability, axis=0).tolist()
            for name, probability in GLOBAL_QUANTILES.items()
        }
    )
    return stats


def _feature_names(
    default_names: Sequence[str], dimension: int, fallback_prefix: str
) -> list[str]:
    names = list(default_names[:dimension])
    names.extend(f"{fallback_prefix}_{index}" for index in range(len(names), dimension))
    return names


def _create_dataset(
    config: LeRobotV3RecorderConfig,
    dataset_factory: DatasetFactory = None,  # type: ignore
    camera_calibration: Mapping[str, Any] | None = None,
) -> Any:
    factory = dataset_factory or _load_lerobot_dataset_factory()
    kwargs = {
        "repo_id": config.repo_id,
        "root": config.root,
        "fps": config.fps,
        "robot_type": config.robot_type,
        "features": build_lerobot_features(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            image_shapes=config.image_shapes,
            use_videos=config.use_videos,
            camera_calibration=camera_calibration,
        ),
        "use_videos": config.use_videos,
        "image_writer_processes": config.image_writer_processes,
        "image_writer_threads": config.image_writer_threads,
        "batch_encoding_size": config.batch_encoding_size,
        "streaming_encoding": config.streaming_encoding,
        "encoder_threads": config.encoder_threads,
        "video_files_size_in_mb": config.video_files_size_in_mb,
        "data_files_size_in_mb": config.data_files_size_in_mb,
    }
    if config.depth_min_m is not None:
        try:
            from lerobot.configs.video import DepthEncoderConfig
        except ImportError as exc:
            raise RuntimeError(
                "This LeRobot version does not expose DepthEncoderConfig; "
                "upgrade the depth-video-enabled LeRobot dependency or unset "
                "dataset.lerobot.depth_min_m"
            ) from exc
        kwargs["depth_encoder"] = DepthEncoderConfig(depth_min=config.depth_min_m)
    return factory(**kwargs)


def _load_lerobot_dataset_factory() -> DatasetFactory:
    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "Unable to import LeRobotDataset. LeRobot may be missing, or one of "
            "its dependencies may be incompatible. Original import error: "
            f"{exc}"
        ) from exc

    return LeRobotDataset.create


def _save_episode(dataset: Any, parallel_encoding: bool) -> Any:
    signature = inspect.signature(dataset.save_episode)
    if "parallel_encoding" in signature.parameters:
        return dataset.save_episode(parallel_encoding=parallel_encoding)
    return dataset.save_episode()


def _subtask_index(task: TaskInfo, subtask: str) -> int:
    try:
        return task.subtasks.index(subtask)
    except ValueError as exc:
        raise ValueError(f"unknown subtask {subtask!r} for task {task.name!r}") from exc
