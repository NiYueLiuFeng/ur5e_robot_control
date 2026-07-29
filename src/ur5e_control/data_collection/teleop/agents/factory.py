"""Factory for teleoperation input backends."""

from __future__ import annotations

from typing import Any, Callable
from ur5e_control.runtime.config import AgentRuntimeConfig

AgentFactory = Callable[..., Any]


def create_teleop_agent(
    config: AgentRuntimeConfig,
    gello_factory: AgentFactory | None = None,
    spacemouse_factory: AgentFactory | None = None,
) -> Any:
    if config.name == "gello":
        factory = gello_factory or _load_gello_agent
        return factory(port=config.gello_port)

    if config.name == "spacemouse":
        spacemouse = config.spacemouse
        factory = spacemouse_factory or _load_spacemouse_agent
        return factory(
            robot_type=config.robot,
            device_path=spacemouse.device_path,
            translation_scale=spacemouse.translation_scale,
            angle_scale=spacemouse.angle_scale,
            invert_control=spacemouse.invert_control,
            rotation_mode=spacemouse.rotation_mode,
            invert_button=spacemouse.invert_button,
            gripper_open=spacemouse.gripper_open,
            gripper_closed=spacemouse.gripper_closed,
            verbose=config.verbose,
        )
    
    raise ValueError(f"Unsupported teleop agent {config.name!r}")


def _load_gello_agent(**kwargs: Any) -> Any:
    from ur5e_control.data_collection.teleop.gello import GelloAgent

    return GelloAgent(**kwargs)


def _load_spacemouse_agent(**kwargs: Any) -> Any:
    from ur5e_control.data_collection.teleop.spacemouse import SpaceMouseAgent

    return SpaceMouseAgent(**kwargs)