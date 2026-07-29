"""Joint-space safety-zone calibration for UR5e teleoperation."""

from __future__ import annotations

import atexit
from multiprocessing import Process
from pathlib import Path
import time

from ur5e_control.runtime.config import (
    RuntimeConfig,
    load_runtime_config,
    resolve_repo_path,
)
from ur5e_control.runtime.safety import JointLimits


class SafetyZoneCalibration:
    """Record min/max arm joints while teleoperating inside the desired workspace."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        output_path: str | Path | None = None,
        margin: float = 0.0,
        runtime_config: RuntimeConfig | None = None,
    ) -> None:
        import numpy as np
        from ur5e_control.data_collection.teleop.env import RobotEnv
        from ur5e_control.runtime.transport.robot_zmq import ZMQClientRobot

        self.runtime_config = runtime_config or load_runtime_config(config_path)
        self._np = np
        self.margin = float(margin)
        self.output_path = resolve_repo_path(
            output_path or self.runtime_config.safety.joint_limits_file
        )
        self.flag_save = False
        self.robot_port = self.runtime_config.robot.zmq_port
        self.hostname = self.runtime_config.robot.zmq_host
        self.robot_ip = self.runtime_config.robot.ip
        self.gripper_port = self.runtime_config.gripper.port

        robot_client = ZMQClientRobot(port=self.robot_port, host=self.hostname)
        self.env = RobotEnv(
            robot_client,
            control_rate_hz=self.runtime_config.robot.control_hz,
            cameras={},
        )
        self.listener = None

    def launch_robot_server(self) -> None:
        from ur5e_control.hardware import URRobot
        from ur5e_control.runtime.transport.robot_zmq import ZMQServerRobot

        robot = URRobot(
            robot_ip=self.robot_ip,
            gripper_port=self.gripper_port,
            gripper_position_tolerance=self.runtime_config.gripper.position_tolerance,
            gripper_min_command_interval_s=(
                self.runtime_config.gripper.min_command_interval_s
            ),
            gripper_position_read_interval_s=(
                self.runtime_config.gripper.position_read_interval_s
            ),
        )
        server = ZMQServerRobot(robot, port=self.robot_port, host=self.hostname)
        print(f"Starting robot server on port {self.robot_port}")
        server.serve()

    def start_robot_process(self) -> Process:
        process = Process(target=self.launch_robot_server)

        def kill_child_process(child_process: Process) -> None:
            print("Killing child process...")
            child_process.terminate()
            child_process.join(timeout=2)

        atexit.register(kill_child_process, process)
        process.start()
        return process

    def start_keyboard_listener(self):
        from pynput import keyboard

        def on_press(key) -> None:
            if key == keyboard.Key.esc:
                self.flag_save = True

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        self.listener = listener
        return listener

    def start(self) -> JointLimits:
        from ur5e_control.data_collection.teleop.agents.factory import (
            create_teleop_agent,
        )

        np = self._np
        self.start_robot_process()
        self.start_keyboard_listener()
        agent = create_teleop_agent(self.runtime_config.agent)
        lower = np.full(6, np.inf)
        upper = np.full(6, -np.inf)

        print("Move the robot through the desired workspace boundary.")
        print("Press ESC or Ctrl-C to save joint limits.")
        try:
            while not self.flag_save:
                obs = self.env.get_obs()
                current = np.asarray(obs["joint_positions"], dtype=float)[:6]
                lower = np.minimum(lower, current)
                upper = np.maximum(upper, current)
                action = agent.act(obs)
                self.env.step(action)
        except KeyboardInterrupt:
            print("Calibration interrupted; saving captured limits.")
        finally:
            if self.listener is not None:
                self.listener.stop()

        if np.isinf(lower).any() or np.isinf(upper).any():
            raise RuntimeError("No joint observations were captured during calibration")

        limits = JointLimits(
            lower=tuple(float(value) for value in lower),
            upper=tuple(float(value) for value in upper),
            margin=self.margin,
            enabled=True,
        )
        limits.save(self.output_path)
        print(f"Saved joint safety limits to {self.output_path}")
        return limits


def main(
    config_path: str | Path | None = None,
    output_path: str | Path | None = None,
    margin: float = 0.0,
) -> JointLimits:
    calibration = SafetyZoneCalibration(
        config_path=config_path,
        output_path=output_path,
        margin=margin,
    )
    return calibration.start()
