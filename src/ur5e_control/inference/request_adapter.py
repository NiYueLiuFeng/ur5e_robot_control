"""Build CloudRobo-VLA requests from live UR5e environment observations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ur5e_control.runtime.types import TaskInfo

DEFAULT_VLA_IMAGE_KEY_MAP = {
    "third_front_rgb": "observation.images.third_front_image",
    "wrist_rgb": "observation.images.wrist_image",
}


@dataclass(frozen=True)
class VLARequestAdapter:
    repo_id: str
    state_key: str = "observation.state"
    image_key_map: Mapping[str, str] = field(
        default_factory=lambda: dict(DEFAULT_VLA_IMAGE_KEY_MAP)
    )

    def __post_init__(self) -> None:
        if not self.repo_id:
            raise ValueError("repo_id must be set")
        if not self.state_key:
            raise ValueError("state_key must be set")
        object.__setattr__(
            self,
            "image_key_map",
            {str(src): str(dst) for src, dst in self.image_key_map.items()},
        )

    def from_env_obs(
        self,
        env_obs: Mapping[str, Any],
        *,
        task: TaskInfo,
        sub_task: str | None = None,
        task_name: str | None = None,
        # rtc: Any | None = None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "repo_id": self.repo_id,
            "task": task.name if task_name is None else task_name,
            self.state_key: _state_from_env_obs(env_obs),
        }
        # if rtc is not None:
        #     self.prev_rtc_leftover = None
        #     request["rtc"] = rtc.to_dict()
        if sub_task:
            request["sub_task"] = sub_task

        for env_key, request_key in self.image_key_map.items():
            if env_key not in env_obs:
                raise KeyError(f"env observation is missing image key: {env_key}")
            request[request_key] = env_obs[env_key]

        return request


def _state_from_env_obs(env_obs: Mapping[str, Any]) -> np.ndarray:
    if "joint_positions" not in env_obs:
        raise KeyError("env observation is missing joint_positions")

    joints = np.asarray(env_obs["joint_positions"], dtype=np.float32).reshape(-1)
    if joints.size == 6 and "gripper_position" in env_obs:
        gripper = np.asarray(env_obs["gripper_position"], dtype=np.float32).reshape(-1)
        if gripper.size:
            joints = np.concatenate([joints, gripper[:1]])
    if joints.size != 7:
        raise ValueError(
            "VLA observation state must contain 7 values "
            f"(6 arm joints + gripper), got {joints.size}"
        )
    return joints
