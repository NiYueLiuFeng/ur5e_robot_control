"""SpaceMouse 3D mouse teleoperation agent for UR5e."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Dict, Optional, Sequence

from ur5e_control.data_collection.teleop.agents.base import Agent
from ur5e_control.data_collection.teleop.spacemouse.ur5e_model import (
    load_ur5e_physics,
)


@dataclass(frozen=True)
class SpaceMouseConfig:
    angle_scale: float = 0.24
    translation_scale: float = 0.06
    invert_control: Sequence[float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    rotation_mode: str = "euler"
    gripper_open: int = 1000
    gripper_closed: int = 0


class SpaceMouseAgent(Agent):
    def __init__(
        self,
        robot_type: str,
        device_path: Optional[str] = None,
        translation_scale: float = 0.06,
        angle_scale: float = 0.24,
        invert_control: Sequence[float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        rotation_mode: str = "euler",
        invert_button: bool = True,
        gripper_open: int = 1000,
        gripper_closed: int = 0,
        verbose: bool = True,

    ) -> None:
        if robot_type not in {"ur5", "ur5e"}:
            raise ValueError(f"Unsupported SpaceMouse robot type: {robot_type}")
        self.pre_gripper_action=1000.0 ##lwz
        self.config = SpaceMouseConfig(
            angle_scale=angle_scale,
            translation_scale=translation_scale,
            invert_control=tuple(invert_control),
            rotation_mode=rotation_mode,
            gripper_open=gripper_open,
            gripper_closed=gripper_closed,
        )
        self.last_state_lock = threading.Lock()
        self.spacemouse_latest_state = None
        self._invert_button = invert_button
        self._verbose = verbose
        self.physics = load_ur5e_physics()
        self._mouse = self._open_spacemouse(device_path)
        spacemouse_thread = threading.Thread(
            target=self._read_from_spacemouse,
            args=(self._mouse,),
            daemon=True,
        )
        spacemouse_thread.start()

    def _open_spacemouse(self, device_path: Optional[str]):
        import pyspacemouse

        target = "auto-detected device" if device_path is None else device_path
        try:
            mouse = (
                pyspacemouse.open()
                if device_path is None
                else pyspacemouse.open_by_path(device_path)
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to open SpaceMouse ({target}): {exc}. "
                "Check that its /dev/hidraw* node is accessible to this user."
            ) from exc
        if not mouse:
            raise RuntimeError(f"Failed to open SpaceMouse device: {target}")
        return mouse

    def _read_from_spacemouse(self, mouse):
        while True:
            state = mouse.read()
            with self.last_state_lock:
                self.spacemouse_latest_state = state
            time.sleep(0.001)

    def act(self, obs: Dict):
        import numpy as np
        import quaternion
        from dm_control.utils.inverse_kinematics import qpos_from_site_pose

        if MJ_TO_UR is None:
            _init_transforms()

        current_qpos = np.asarray(obs["joint_positions"][:6])
        # Binary observations use 0/1; robot commands require PGC 0/1000.
        current_gripper_position = float(obs["joint_positions"][-1]) * 1000.0
        self.physics.data.qpos[:6] = current_qpos
        self.physics.step()

        ee_rot = np.array(self.physics.named.data.site_xmat["attachment_site"]).reshape(
            3,
            3,
        )
        ee_pos = np.array(self.physics.named.data.site_xpos["attachment_site"])
        ee_rot = MJ_TO_UR[:3, :3] @ ee_rot
        ee_pos = _apply_transform(MJ_TO_UR, ee_pos)

        with self.last_state_lock:
            spacemouse_read = self.spacemouse_latest_state
        if spacemouse_read is None:
            return np.concatenate([current_qpos, [current_gripper_position]])

        spacemouse_xyz_rot = np.array(
            [
                spacemouse_read.x,
                spacemouse_read.y,
                spacemouse_read.z,
                spacemouse_read.roll,
                spacemouse_read.pitch,
                spacemouse_read.yaw,
            ]
        )
        spacemouse_xyz_rot = spacemouse_xyz_rot * np.asarray(
            self.config.invert_control
        )
        if np.max(np.abs(spacemouse_xyz_rot)) > 0.9:
            spacemouse_xyz_rot[np.abs(spacemouse_xyz_rot) < 0.6] = 0

        tx, ty, tz, roll, pitch, yaw = spacemouse_xyz_rot
        trans_transform = np.eye(4)
        trans_transform[:3, 3] = _apply_transform(
            SPACEMOUSE_TO_UR,
            np.array([tx, ty, tz]) * self.config.translation_scale,
        )
        rot_transform = _spacemouse_rotation_transform(
            roll,
            pitch,
            yaw,
            self.config.angle_scale,
            quaternion,
        )

        new_ee_pos = trans_transform[:3, 3] + ee_pos
        if self.config.rotation_mode == "rpy":
            new_ee_rot = ee_rot @ rot_transform[:3, :3]
        elif self.config.rotation_mode == "euler":
            new_ee_rot = rot_transform[:3, :3] @ ee_rot
        else:
            raise ValueError(f"Unknown rotation mode: {self.config.rotation_mode}")

        target_quat = quaternion.as_float_array(
            quaternion.from_rotation_matrix(UR_TO_MJ[:3, :3] @ new_ee_rot)
        )
        ik_result = qpos_from_site_pose(
            self.physics,
            "attachment_site",
            target_pos=_apply_transform(UR_TO_MJ, new_ee_pos),
            target_quat=target_quat,
            tol=1e-14,
            max_steps=400,
        )
        self.physics.reset()
        if not ik_result.success:
            if self._verbose:
                print("SpaceMouse IK failed; keeping current joint command")
            # return np.concatenate([current_qpos, [current_gripper_position]])
            return np.concatenate([current_qpos, [self.pre_gripper_action]])  ##lwz

        self.pre_gripper_action = self._gripper_command(
            spacemouse_read.buttons,
            self.pre_gripper_action,
            # current_gripper_position,
        )
        # print(f"########current action######{self.pre_gripper_action}###")
        return np.concatenate([ik_result.qpos[:6], [self.pre_gripper_action]])

    def _gripper_command(self, buttons, current_gripper_position: float) -> int | float:
        # if buttons[1]:
        #     return self.config.gripper_closed
        # if buttons[0]:
        #     return self.config.gripper_open
        # return -1
    
        if self._invert_button:
            if buttons[1]:
                return self.config.gripper_open
            if buttons[0]:
                return self.config.gripper_closed
        else:
            if buttons[1]:
                return self.config.gripper_closed
            if buttons[0]:
                return self.config.gripper_open
        return current_gripper_position


MJ_TO_UR = None
UR_TO_MJ = None
SPACEMOUSE_TO_UR = None
UR_TO_SPACEMOUSE = None


def _init_transforms():
    global MJ_TO_UR, UR_TO_MJ, SPACEMOUSE_TO_UR, UR_TO_SPACEMOUSE

    import numpy as np

    MJ_TO_UR = np.array(
        [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    UR_TO_MJ = np.linalg.inv(MJ_TO_UR)
    SPACEMOUSE_TO_UR = np.array(
        [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    UR_TO_SPACEMOUSE = np.linalg.inv(SPACEMOUSE_TO_UR)


def _apply_transform(mat, xyz):
    import numpy as np

    if mat is None:
        _init_transforms()
        mat = MJ_TO_UR
    if len(xyz) == 3:
        xyz = np.append(xyz, 1)
    return np.matmul(mat, xyz)[:3]


def _spacemouse_rotation_transform(roll, pitch, yaw, angle_scale, quaternion):
    import numpy as np

    if SPACEMOUSE_TO_UR is None or UR_TO_SPACEMOUSE is None:
        _init_transforms()
    rot_transform_x = np.eye(4)
    rot_transform_x[:3, :3] = quaternion.as_rotation_matrix(
        quaternion.from_rotation_vector(np.array([-pitch, 0, 0]) * angle_scale)
    )
    rot_transform_y = np.eye(4)
    rot_transform_y[:3, :3] = quaternion.as_rotation_matrix(
        quaternion.from_rotation_vector(np.array([0, roll, 0]) * angle_scale)
    )
    rot_transform_z = np.eye(4)
    rot_transform_z[:3, :3] = quaternion.as_rotation_matrix(
        quaternion.from_rotation_vector(np.array([0, 0, -yaw]) * angle_scale)
    )
    return (
        SPACEMOUSE_TO_UR
        @ rot_transform_z
        @ rot_transform_y
        @ rot_transform_x
        @ UR_TO_SPACEMOUSE
    )
