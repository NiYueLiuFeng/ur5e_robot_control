import time
from typing import Any, Dict, Optional

import numpy as np

from ur5e_control.hardware.robots.base import Robot
from ur5e_control.hardware.cameras.zmq import ZMQClientCamera
from ur5e_control.runtime.safety import JointLimits


class Rate:
    def __init__(self, rate: float):
        self.last = time.time()
        self.rate = rate

    def sleep(self) -> None:
        while self.last + 1.0 / self.rate > time.time():
            time.sleep(0.0001)
        self.last = time.time()


class RobotEnv:
    def __init__(
        self,
        robot: Robot,
        control_rate_hz: float = 100.0,
        camera_name_id: Optional[Dict[str, str]] = None,
        cameras: Optional[Dict[str, ZMQClientCamera]] = None,
        joint_limits: Optional[JointLimits] = None,
    ) -> None:
        self._robot = robot
        self._rate = Rate(control_rate_hz)
        self._camera_name_id = {} if camera_name_id is None else camera_name_id
        self.cameras = cameras if cameras is not None else {}
        self._joint_limits = joint_limits
        self._last_joint_limit_log_s = 0.0

    def robot(self) -> Robot:
        """Get the robot object.

        Returns:
            robot: the robot object.
        """
        return self._robot

    def __len__(self):
        return 0

    def step(self, joints: np.ndarray) -> Dict[str, Any]:
        """Step the environment forward.

        Args:
            joints: joint angles command to step the environment with.

        Returns:
            obs: observation from the environment.
        """
        joints = self.apply_safety(joints)
        assert len(joints) == (
            self._robot.num_dofs()
        ), f"input:{len(joints)}, robot:{self._robot.num_dofs()}"

        self._robot.command_joint_state(joints)
        self._rate.sleep()
        
        return self.get_obs()

    def apply_safety(self, joints: Any) -> np.ndarray:
        """Apply configured joint-space limits to a candidate command."""

        command = np.asarray(joints, dtype=float)
        if self._joint_limits is None:
            return command

        values, was_clamped = self._joint_limits.clamp_action(tuple(command))
        if was_clamped:
            now = time.time()
            if now - self._last_joint_limit_log_s > 1.0:
                print("Joint safety limits clamped the outgoing command")
                self._last_joint_limit_log_s = now
        return np.asarray(values, dtype=float)
    

    def binarize_gripper_position(self, values, threadhold:int = 300):
        close_condition = values < threadhold
        return np.where(close_condition, 0, 1)
    
    def get_obs(self) -> Dict[str, Any]:
        """Get observation from the environment.

        Returns:
            obs: observation from the environment.
        """
        observations = {}
        for camera_id, camera in self.cameras.items():
            name = self._camera_name_id[camera_id]
            image, depth = camera.read()
            image = np.transpose(image, (2, 0, 1))
            observations[f"{name}_rgb"] = image
            if depth is not None:
                observations[f"{name}_depth"] = depth

        robot_obs = self._robot.get_observations()
        assert "joint_positions" in robot_obs
        assert "gripper_position" in robot_obs
        # observations["joint_positions"] = robot_obs["joint_positions"]
        observations["gripper_position"] = np.array(self.binarize_gripper_position(robot_obs["gripper_position"], 700))

        observations["joint_positions"] = np.concatenate((robot_obs["joint_positions"][:6], observations["gripper_position"]), axis=0)
        observations["tcp_force"] = robot_obs["tcp_force"]

        return observations


def main() -> None:
    pass


if __name__ == "__main__":
    main()
