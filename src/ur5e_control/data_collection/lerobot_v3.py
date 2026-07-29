"""LeRobot v3 recording adapter.

The real LeRobot dependency is imported lazily so this repository can be tested
without robot hardware or the dataset stack installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import queue
import threading
from typing import Any, Callable, Dict, Mapping, Optional, Tuple
import numpy as np
from ur5e_control.runtime.types import RobotAction, RobotObservation, TaskInfo

DatasetFactory = Callable[..., Any]

DEFAULT_IMAGE_SHAPES = {
    "wrist": (480, 640, 3),
    "third_person": (480, 640, 3),
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


class LeRobotV3Recorder:
    """Episode recorder backed by `LeRobotDataset.create()`."""

    def __init__(
        self,
        config: LeRobotV3RecorderConfig,
        dataset_factory: DatasetFactory = None,  # type: ignore
    ):
        self.config = config
        self.dataset = _create_dataset(config, dataset_factory)
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
        )
        self.dataset.add_frame(frame)

    def save_episode(self) -> str:
        result = _save_episode(
            self.dataset, parallel_encoding=self.config.parallel_encoding
        )
        return "" if result is None else str(result)

    def finalize(self) -> None:
        self.dataset.finalize()


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

def binarize_gripper_position(values, threshold: float=30):
    close_condition = values <= threshold
    return np.where(close_condition, 0, 1)
    
def build_lerobot_frame(
    observation: RobotObservation,
    action: RobotAction,
    task: TaskInfo,
    subtask: Optional[str] = None,
    include_subtask_text: bool = True,
    gripper_threadhold: int = 700,
) -> Dict[str, Any]:
    """Build a LeRobotDataset.add_frame-compatible frame."""
    frame: Dict[str, Any] = {
        "task": task.name,
        "tcp_force": np.array(observation.tcp_force, dtype=np.float32),
        "observation.state": np.array(observation.joint_positions, dtype=np.float32),
        "action": np.array(action.values,  dtype=np.float32),
    }

    frame["action"][-1] = binarize_gripper_position(frame["action"][-1], gripper_threadhold)
    frame["observation.state"][-1] = binarize_gripper_position(int(frame["observation.state"][-1]), gripper_threadhold)
    # print("###########action########", frame["action"])
    # print("###########observation.state########", frame["observation.state"])

    if observation.timestamp is not None:
        frame["timestamp"] = observation.timestamp

    if subtask is not None:
        subtask_index = _subtask_index(task, subtask)
        frame["subtask_index"] =  np.array([subtask_index], dtype=np.int64)
        if include_subtask_text:
            frame["subtask"] = subtask

    for camera_name, image in observation.images.items():
        frame[f"observation.images.{camera_name}"] = image

    return frame


def build_lerobot_features(
    state_dim: int = 7,
    action_dim: int = 7,
    tcp_force_dim: int = 6,
    image_shapes: Mapping[str, Tuple[int, int, int]] = None,  # type: ignore
    use_videos: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Return the feature schema passed to `LeRobotDataset.create()`."""

    image_shapes = image_shapes or DEFAULT_IMAGE_SHAPES
    visual_dtype = "video" if use_videos else "image"
    features: Dict[str, Dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["state"],
        },
        "tcp_force": {
            "dtype": "float32",
            "shape": (tcp_force_dim,),
            "names": ["force"],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["action"],
        },
        "subtask_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["subtask_index"],
        }
    }

    for camera_name, shape in image_shapes.items():
        features[f"observation.images.{camera_name}"] = {
            "dtype": visual_dtype,
            "shape": tuple(shape),
            "names": ["height", "width", "channel"],
        }

    return features


def _create_dataset(
    config: LeRobotV3RecorderConfig,
    dataset_factory: DatasetFactory = None,  # type: ignore
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
    return factory(**kwargs)


def _load_lerobot_dataset_factory() -> DatasetFactory:
    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot is not installed. Install the LeRobot package before using "
            "LeRobotV3Recorder without an injected dataset_factory."
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
