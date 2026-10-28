"""Adapters from legacy environment observations into runtime observations."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ur5e_control.runtime.types import RobotObservation

DEFAULT_IMAGE_KEY_MAP = {
    "wrist_rgb": "wrist",
    "third_left_top_rgb": "third_person",
    "wrist_depth": "wrist_depth",
    "third_left_top_depth": "third_person_depth",
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
    images = {}
    for source_name, target_name in image_key_map.items():
        image = env_obs.get(source_name)
        if image is None:
            continue

        # LeRobot visual features use HWC shapes. RealSense currently returns
        # HxWx1 depth frames, but accepting HxW here keeps the recorder robust
        # to other camera adapters and resized depth frames.
        shape = getattr(image, "shape", None)
        if target_name.endswith("_depth") and shape is not None and len(shape) == 2:
            image = image[..., None]
        images[target_name] = image

    return RobotObservation(
        tcp_force=tuple(env_obs["tcp_force"]),
        tcp_pose=tuple(env_obs["tcp_pose"]),
        joint_positions=tuple(env_obs["joint_positions"]),
        gripper_position=tuple(env_obs.get("gripper_position", ())),
        images=images,
        timestamp=timestamp,
    )

