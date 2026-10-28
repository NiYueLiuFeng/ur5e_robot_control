"""Small data contracts shared by collection and inference code."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class RobotAction:
    """Robot command vector.

    Current UR5e + PGC collection uses six arm joints plus one gripper value.
    The dimension is validated by runtime safety helpers so adapters can share
    the same guardrail.
    """

    values: Tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(float(value) for value in self.values))


@dataclass(frozen=True)
class RobotObservation:
    """Observation payload passed to policies and VLA servers."""
    tcp_force: Tuple[float, ...]
    tcp_pose: Tuple[float, ...]
    joint_positions: Tuple[float, ...]
    gripper_position: Tuple[float, ...] = ()
    images: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "joint_positions",
            tuple(float(value) for value in self.joint_positions),
        )
        object.__setattr__(
            self,
            "tcp_force",
            tuple(float(value) for value in self.tcp_force),
        )
        object.__setattr__(
            self,
            "tcp_pose",
            tuple(float(value) for value in self.tcp_pose),
        )
        object.__setattr__(
            self,
            "gripper_position",
            tuple(float(value) for value in self.gripper_position),
        )
        object.__setattr__(self, "images", dict(self.images))
        if self.timestamp is not None:
            object.__setattr__(self, "timestamp", float(self.timestamp))

    def to_payload(self) -> Dict[str, Any]:
        """Return a JSON-friendly representation for API transports."""

        return {
            "joint_positions": list(self.joint_positions),
            "gripper_position": list(self.gripper_position),
            "tcp_force": list(self.tcp_force),
            "tcp_pose": list(self.tcp_pose),
            "images": {
                name: _to_jsonable_image(image) for name, image in self.images.items()
            },
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class TaskInfo:
    """Language task metadata used by teleop recording and policy inference."""

    name: str
    task_type: str = "long_horizon_task"
    subtasks: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "subtasks", tuple(str(item) for item in self.subtasks))


def _to_jsonable_image(image: Any) -> Any:
    if hasattr(image, "tolist"):
        return image.tolist()
    return image

