"""Adapters from legacy environment observations into runtime observations."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ur5e_control.runtime.types import RobotObservation

DEFAULT_IMAGE_KEY_MAP = {
    "wrist_rgb": "wrist",
    "third_left_top_rgb": "third_person",
}


def observation_from_env_dict(
    env_obs: Mapping[str, Any],
    timestamp: Optional[float] = None,
    image_key_map: Mapping[str, str] = None, # type: ignore
) -> RobotObservation:
    """Convert the current GELLO env observation dict to `RobotObservation`."""

    if "joint_positions" not in env_obs:
        raise ValueError("env observation is missing 'joint_positions'")

    image_key_map = image_key_map or DEFAULT_IMAGE_KEY_MAP
    images = {
        target_name: env_obs[source_name]
        for source_name, target_name in image_key_map.items()
        if source_name in env_obs and env_obs[source_name] is not None
    }

    return RobotObservation(
        tcp_force=tuple(env_obs["tcp_force"]),
        joint_positions=tuple(env_obs["joint_positions"]),
        gripper_position=tuple(env_obs.get("gripper_position", ())),
        images=images,
        timestamp=timestamp,
    )

