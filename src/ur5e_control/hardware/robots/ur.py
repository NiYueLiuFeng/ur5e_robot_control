"""UR5e RTDE robot adapter."""

from __future__ import annotations

from typing import Dict

import numpy as np
import time
from ur5e_control.hardware.grippers import PGCGripper


class URRobot:
    """UR5e follower robot with optional DH PGC gripper."""

    def __init__(
        self,
        robot_ip: str = "192.168.56.2",
        gripper_port: str = "/dev/ttyUSB0",
        no_gripper: bool = False,
        rtde_frequency: float = 500,
        gripper_position_tolerance: int = 10,
        gripper_min_command_interval_s: float = 0.05,
        gripper_position_read_interval_s: float = 0.1,
        pre_gripper_state: int = 1000,
    ) -> None:
        import rtde_control
        import rtde_receive

        self.robot_ip = robot_ip
        self.robot = rtde_control.RTDEControlInterface(
            robot_ip,
            frequency=rtde_frequency,
            # flags=rtde_control.RTDEControlInterface.FLAG_USE_EXT_UR_CAP,
        )
        self.r_inter = rtde_receive.RTDEReceiveInterface(robot_ip)

        self._use_gripper = not no_gripper
        self.gripper = (
            None
            if no_gripper
            else PGCGripper(
                port=gripper_port,
                position_tolerance=gripper_position_tolerance,
                min_command_interval_s=gripper_min_command_interval_s,
                position_read_interval_s=gripper_position_read_interval_s,
            )
        )
        self._free_drive = False
        self.robot.endFreedriveMode()
        self.pre_gripper_state=pre_gripper_state

    def num_dofs(self) -> int:
        return 7 if self._use_gripper else 6

    def get_joint_state(self) -> np.ndarray:
        robot_joints = self.r_inter.getActualQ()
        if not self._use_gripper:
            return np.asarray(robot_joints)
        return np.append(robot_joints, self._get_gripper_pos())

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        velocity = 0.5
        acceleration = 0.5
        dt = 1.0 / 500
        lookahead_time = 0.2
        gain = 100

        robot_joints = joint_state[:6]
        t_start = self.robot.initPeriod()

        # self.robot.moveJ(robot_joints, 3.0, 1.0)

        self.robot.servoJ(robot_joints, velocity, acceleration, dt, lookahead_time, gain)


        if self._use_gripper:
            assert self.gripper is not None
            if int(joint_state[-1]) != self.pre_gripper_state:
                self.gripper.grip(int(joint_state[-1]))
                self.pre_gripper_state = int(joint_state[-1])
        self.robot.waitPeriod(t_start)

    def freedrive_enabled(self) -> bool:
        return self._free_drive

    def set_freedrive_mode(self, enable: bool) -> None:
        if enable and not self._free_drive:
            self._free_drive = True
            self.robot.freedriveMode()
        elif not enable and self._free_drive:
            self._free_drive = False
            self.robot.endFreedriveMode()

    def get_observations(self) -> Dict[str, np.ndarray]:
        joints = self.get_joint_state()
        gripper_pos = np.array([joints[-1]]) if self._use_gripper else np.array([])
        return {
            "joint_positions": joints,
            "gripper_position": gripper_pos,
            "tcp_pose": np.asarray(self.r_inter.getActualTCPPose(), dtype=float),
            "tcp_force": np.asarray(self.r_inter.getActualTCPForce(), dtype=float),
        }

    def _get_gripper_pos(self) -> float:
        assert self.gripper is not None
        gripper_pos = self.gripper.get_gripper_position()
        if not 0 <= gripper_pos <= 1000:
            raise ValueError(
                "PGC gripper position must be in [0, 1000], "
                f"got {gripper_pos}"
            )
        print(
            f"[gripper] raw_position={gripper_pos}, "
            f"gripper_state={int(gripper_pos > 700)}",
            flush=True,
        )
        self.pre_gripper_state = gripper_pos
        return float(gripper_pos)
