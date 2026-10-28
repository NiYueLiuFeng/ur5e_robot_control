"""Helpers for wiring teleoperation loops to dataset recorders."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ur5e_control.data_collection.observations import (
    DEFAULT_IMAGE_KEY_MAP,
    observation_from_env_dict,
)
from ur5e_control.runtime.types import RobotAction, TaskInfo


def infer_image_shapes_from_env_obs(
    env_obs: Mapping[str, Any],
    image_key_map: Mapping[str, str] = None, # type: ignore
) -> Dict[str, Tuple[int, int, int]]:
    """Infer LeRobot image feature shapes from the first environment observation."""

    image_key_map = image_key_map or DEFAULT_IMAGE_KEY_MAP
    image_shapes: Dict[str, Tuple[int, int, int]] = {}
    for source_name, target_name in image_key_map.items():
        image = env_obs.get(source_name)
        shape = getattr(image, "shape", None)
        if shape is not None and len(shape) == 2:
            image_shapes[target_name] = (int(shape[0]), int(shape[1]), 1)
        elif shape is not None and len(shape) == 3:
            image_shape = tuple(int(value) for value in shape)
            if image_shape[0] in (1, 3) and image_shape[-1] not in (1, 3):
                image_shape = (image_shape[1], image_shape[2], image_shape[0])
            image_shapes[target_name] = image_shape
    return image_shapes


def record_lerobot_step(
    recorder: Any,
    env_obs: Mapping[str, Any],
    action_values: Sequence[float],
    task: TaskInfo,
    subtask: Optional[str],
    timestamp: Optional[float] = None,
) -> None:
    """Map a live teleop step into the recorder interface."""

    observation = observation_from_env_dict(env_obs, timestamp=timestamp)
    action = RobotAction(values=tuple(action_values))
    recorder.add_frame(observation, action, subtask=subtask)


def select_subtask(subtasks: Sequence[str], second_subtask_active: bool) -> Optional[str]:
    """Return the active subtask label for the current keyboard state."""

    if not subtasks:
        return None
    if second_subtask_active and len(subtasks) > 1:
        return subtasks[1]
    return subtasks[0]

