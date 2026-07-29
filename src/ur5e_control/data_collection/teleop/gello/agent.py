import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ur5e_control.data_collection.teleop.agents.base import Agent
from ur5e_control.data_collection.teleop.gello.dynamixel_robot import DynamixelRobot


@dataclass
class DynamixelRobotConfig:
    joint_ids: Sequence[int]
    """The GELLO joint ids, excluding the gripper."""

    joint_offsets: Sequence[float]
    """One pi/2-multiple offset per GELLO joint id."""

    joint_signs: Sequence[int]
    """The joint signs of GELLO. Each joint_id needs a sign of 1 or -1."""

    gripper_config: Tuple[int, int, int]
    """Tuple of gripper id, open degrees, and closed degrees."""

    def __post_init__(self):
        assert len(self.joint_ids) == len(self.joint_offsets)
        assert len(self.joint_ids) == len(self.joint_signs)

    def make_robot(
        self, port: str = "/dev/ttyUSB1", start_joints: Optional[np.ndarray] = None
    ) -> DynamixelRobot:
        return DynamixelRobot(
            joint_ids=self.joint_ids,
            joint_offsets=list(self.joint_offsets),
            real=True,
            joint_signs=list(self.joint_signs),
            port=port,
            gripper_config=self.gripper_config,
            start_joints=start_joints,
        )


UR5E_DYNAMIXEL_CONFIG = DynamixelRobotConfig(
    joint_ids=(1, 2, 3, 4, 5, 6),
    joint_offsets=(
        0 * np.pi / 2,
        3 * np.pi / 2,
        3 * np.pi / 2,
        3 * np.pi / 2,
        3 * np.pi / 2,
        1 * np.pi / 2,
    ),
    joint_signs=(1, 1, -1, 1, 1, 1),
    gripper_config=(7, 199, 157),
)


class GelloAgent(Agent):
    def __init__(
        self,
        port: str,
        dynamixel_config: Optional[DynamixelRobotConfig] = None,
        start_joints: Optional[np.ndarray] = None,
    ):
        # Ensure start_joints is a numpy array if provided
        if start_joints is not None and not isinstance(start_joints, np.ndarray):
            start_joints = np.array(start_joints)
        if dynamixel_config is not None:
            self._robot = dynamixel_config.make_robot(
                port=port, start_joints=start_joints
            )
        else:
            assert os.path.exists(port), port
            self._robot = UR5E_DYNAMIXEL_CONFIG.make_robot(
                port=port,
                start_joints=start_joints,
            )

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        return self._robot.get_joint_state()
