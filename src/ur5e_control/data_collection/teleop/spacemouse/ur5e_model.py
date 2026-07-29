"""Minimal MuJoCo UR5e model loader for SpaceMouse inverse kinematics."""

from __future__ import annotations

from pathlib import Path

from ur5e_control.runtime.config import REPO_ROOT

MENAGERIE_ROOT = REPO_ROOT / "third_party" / "mujoco_menagerie"
UR5E_XML = MENAGERIE_ROOT / "universal_robots_ur5e" / "ur5e.xml"
ROBOTIQ_2F85_XML = MENAGERIE_ROOT / "robotiq_2f85" / "2f85.xml"


def load_ur5e_physics():
    from dm_control import mjcf

    if not UR5E_XML.exists():
        raise RuntimeError(
            "SpaceMouse teleoperation needs the mujoco_menagerie submodule. "
            "Run `git submodule update --init --recursive` from the repository root."
        )

    arm_mjcf = mjcf.from_path(str(UR5E_XML))  
    if ROBOTIQ_2F85_XML.exists():
        gripper_mjcf = mjcf.from_path(str(ROBOTIQ_2F85_XML))
        _attach_hand_to_arm(arm_mjcf, gripper_mjcf)

    arm_mjcf.model = "UR5e"
    return mjcf.Physics.from_mjcf_model(arm_mjcf)


def _attach_hand_to_arm(arm_mjcf, hand_mjcf) -> None:
    import numpy as np
    from dm_control import mjcf

    physics = mjcf.Physics.from_mjcf_model(hand_mjcf)
    attachment_site = arm_mjcf.find("site", "attachment_site")
    if attachment_site is None:
        raise ValueError("No attachment_site found in the UR5e MJCF model")

    arm_key = arm_mjcf.find("key", "home")
    if arm_key is not None:
        hand_key = hand_mjcf.find("key", "home")
        if hand_key is None:
            arm_key.ctrl = np.concatenate([arm_key.ctrl, np.zeros(physics.model.nu)])
            arm_key.qpos = np.concatenate([arm_key.qpos, np.zeros(physics.model.nq)])
        else:
            arm_key.ctrl = np.concatenate([arm_key.ctrl, hand_key.ctrl])
            arm_key.qpos = np.concatenate([arm_key.qpos, hand_key.qpos])

    attachment_site.attach(hand_mjcf)


def menagerie_root() -> Path:
    return MENAGERIE_ROOT
